"""Broker service coordinating multiple balancers for XTTS streaming."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Dict, List

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from xtts_stream.api.broker.settings import BROKER_CONFIG_ENV_VAR, BrokerSettings, load_broker_settings


BROKER_HEADER = "x-xtts-broker"
CONFIG_ENV_VAR = BROKER_CONFIG_ENV_VAR
IDLE_ENDPOINT = "/workers/idle"


class BalancerUnavailable(RuntimeError):
    """Raised when no balancer can accept a request."""


@dataclass
class BalancerRegistration:
    id: str
    host: str
    port: int
    instances: int

    def websocket_url(self, voice_id: str, query_bytes: bytes) -> str:
        query = f"?{query_bytes.decode()}" if query_bytes else ""
        return f"ws://{self.host}:{self.port}/v1/text-to-speech/{voice_id}/stream-input{query}"

    def idle_endpoint(self) -> str:
        return f"http://{self.host}:{self.port}{IDLE_ENDPOINT}"


class BalancerPool:
    def __init__(self, timeout_seconds: float = 2.0) -> None:
        self._balancers: Dict[str, BalancerRegistration] = {}
        self._lock = asyncio.Lock()
        self._timeout = timeout_seconds

    async def register(self, registration: BalancerRegistration) -> None:
        async with self._lock:
            self._balancers[registration.id] = registration

    async def drop(self, balancer_id: str) -> None:
        async with self._lock:
            self._balancers.pop(balancer_id, None)

    async def list(self) -> List[BalancerRegistration]:
        async with self._lock:
            return list(self._balancers.values())

    async def _probe_idle_workers(self, balancers: List[BalancerRegistration]) -> Dict[str, int]:
        results: Dict[str, int] = {}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for balancer in balancers:
                try:
                    resp = await client.get(balancer.idle_endpoint())
                    resp.raise_for_status()
                    data = resp.json()
                    idle = int(data.get("idle_workers", 0))
                    if idle > 0:
                        results[balancer.id] = idle
                except Exception:
                    await self.drop(balancer.id)
        return results

    async def choose(self, strategy: str) -> BalancerRegistration:
        balancers = await self.list()
        idle_map = await self._probe_idle_workers(balancers)
        available = [b for b in balancers if b.id in idle_map]
        if not available:
            raise BalancerUnavailable("No balancer has idle workers")

        if strategy == "random":
            return random.choice(available)

        available.sort(key=lambda b: idle_map[b.id])
        if strategy == "deep":
            return available[0]
        if strategy == "wide":
            return available[-1]

        raise BalancerUnavailable(f"Unsupported strategy: {strategy}")


def _config_path() -> Path:
    from pathlib import Path
    import os

    if CONFIG_ENV_VAR not in os.environ:
        raise BrokerSettingsError(
            "Environment variable XTTS_BROKER_CONFIG_FILE must point to the broker configuration file."
        )
    return Path(os.environ[CONFIG_ENV_VAR]).expanduser().resolve(strict=False)


async def _proxy_stream(client_ws: WebSocket, balancer_ws_url: str) -> None:
    async with websockets.connect(balancer_ws_url, max_size=None, extra_headers={BROKER_HEADER: "1"}) as backend:
        async def client_to_balancer():
            try:
                while True:
                    msg = await client_ws.receive_text()
                    await backend.send(msg)
            except WebSocketDisconnect:
                await backend.close()
            except Exception:
                await backend.close()

        async def balancer_to_client():
            try:
                while True:
                    msg = await backend.recv()
                    if isinstance(msg, bytes):
                        await client_ws.send_bytes(msg)
                    else:
                        await client_ws.send_text(msg)
            except Exception:
                await client_ws.close()

        await asyncio.gather(client_to_balancer(), balancer_to_client())


def create_app(settings: BrokerSettings) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    pool = BalancerPool(timeout_seconds=settings.service.balancer_timeout_seconds)
    app.state.settings = settings
    app.state.pool = pool

    @app.post("/broker/register")
    async def register_balancer(payload: Dict[str, object]):
        try:
            registration = BalancerRegistration(
                id=str(payload["id"]),
                host=str(payload["host"]),
                port=int(payload["port"]),
                instances=int(payload.get("instances", 1)),
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing registration field: {exc.args[0]}") from exc
        await pool.register(registration)
        return {"status": "ok"}

    @app.get("/broker/balancers")
    async def list_balancers():
        balancers = await pool.list()
        return {"balancers": [b.__dict__ for b in balancers]}

    @app.websocket("/v1/text-to-speech/{voice_id}/stream-input")
    async def proxy_ws(ws: WebSocket, voice_id: str):
        await ws.accept()
        try:
            balancer = await pool.choose(settings.service.strategy)
        except BalancerUnavailable:
            await ws.close(code=1013, reason="No balancer with idle workers available")
            return

        query_bytes: bytes = ws.scope.get("query_string", b"")
        balancer_url = balancer.websocket_url(voice_id, query_bytes)

        try:
            await _proxy_stream(ws, balancer_url)
        except Exception:
            await pool.drop(balancer.id)
            await ws.close(code=1011, reason="Failed to proxy request")

    return app


def run_broker(settings: BrokerSettings) -> None:
    uvicorn.run(
        create_app(settings),
        host=settings.service.host,
        port=settings.service.port,
        log_level="info",
    )


if __name__ == "__main__":
    cfg_path = _config_path()
    broker_settings = load_broker_settings(cfg_path)
    run_broker(broker_settings)

