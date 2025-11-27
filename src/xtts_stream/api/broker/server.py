"""Broker service coordinating multiple balancers for XTTS streaming."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
import logging
from typing import Dict, List

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from settings import BROKER_CONFIG_ENV_VAR, BrokerSettings, BrokerSettingsError, load_broker_settings


BROKER_HEADER = "x-xtts-broker"
CONFIG_ENV_VAR = BROKER_CONFIG_ENV_VAR
IDLE_ENDPOINT = "/workers/idle"

logger = logging.getLogger(__name__)


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
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def register(self, registration: BalancerRegistration) -> None:
        async with self._lock:
            self._balancers[registration.id] = registration
            logger.info(
                "Registered balancer %s at %s:%s with %s instances",
                registration.id,
                registration.host,
                registration.port,
                registration.instances,
            )

    async def drop(self, balancer_id: str) -> None:
        async with self._lock:
            self._balancers.pop(balancer_id, None)
            logger.warning("Dropped balancer %s", balancer_id)

    async def list(self) -> List[BalancerRegistration]:
        async with self._lock:
            logger.debug("Listing %s registered balancers", len(self._balancers))
            return list(self._balancers.values())

    async def _probe_idle_workers(self, balancers: List[BalancerRegistration]) -> Dict[str, int]:
        results: Dict[str, int] = {}
        for balancer in balancers:
            logger.debug("Probing idle workers for balancer %s", balancer.id)
            try:
                resp = await self._client.get(balancer.idle_endpoint())
                resp.raise_for_status()
                data = resp.json()
                idle = int(data.get("idle_workers", 0))
                if idle > 0:
                    logger.info("Balancer %s has %s idle workers", balancer.id, idle)
                    results[balancer.id] = idle
            except Exception:
                logger.exception("Failed to probe balancer %s; dropping from pool", balancer.id)
                await self.drop(balancer.id)
        return results

    async def aclose(self) -> None:
        await self._client.aclose()

    async def choose(self, strategy: str) -> BalancerRegistration:
        balancers = await self.list()
        idle_map = await self._probe_idle_workers(balancers)
        available = [b for b in balancers if b.id in idle_map]
        if not available:
            logger.error("No available balancers after probing")
            raise BalancerUnavailable("No balancer has idle workers")

        if strategy == "random":
            choice = random.choice(available)
            logger.info("Selected balancer %s via random strategy", choice.id)
            return choice

        available.sort(key=lambda b: idle_map[b.id])
        if strategy == "deep":
            logger.info("Selected balancer %s via deep strategy", available[0].id)
            return available[0]
        if strategy == "wide":
            logger.info("Selected balancer %s via wide strategy", available[-1].id)
            return available[-1]

        raise BalancerUnavailable(f"Unsupported strategy: {strategy}")


def _config_path() -> Path:
    from pathlib import Path
    import os

    if CONFIG_ENV_VAR not in os.environ:
        logger.error("Environment variable %s is not set", CONFIG_ENV_VAR)
        raise BrokerSettingsError(
            "Environment variable XTTS_BROKER_CONFIG_FILE must point to the broker configuration file."
        )
    path = Path(os.environ[CONFIG_ENV_VAR]).expanduser().resolve(strict=False)
    logger.info("Using broker configuration file %s", path)
    return path


async def _proxy_stream(client_ws: WebSocket, balancer_ws_url: str) -> None:
    async with websockets.connect(
        balancer_ws_url,
        max_size=None,
        additional_headers={BROKER_HEADER: "1"},
    ) as backend:
        logger.info("Proxying websocket traffic to %s", balancer_ws_url)

        async def client_to_balancer():
            try:
                while True:
                    msg = await client_ws.receive_text()
                    logger.debug("Forwarding message from client to balancer")
                    await backend.send(msg)
            except WebSocketDisconnect:
                logger.info("Client websocket disconnected; closing backend connection")
                await backend.close()
            except Exception:
                logger.exception("Error while forwarding client messages; closing backend")
                await backend.close()

        async def balancer_to_client():
            try:
                while True:
                    msg = await backend.recv()
                    logger.debug("Forwarding message from balancer to client")
                    if isinstance(msg, bytes):
                        await client_ws.send_bytes(msg)
                    else:
                        await client_ws.send_text(msg)

            # Нормальное закрытие backend WS — НЕ ошибка
            except websockets.exceptions.ConnectionClosedOK:
                logger.info("Backend websocket closed cleanly; closing client websocket")
                try:
                    await client_ws.close()
                except RuntimeError:
                    # Клиент уже закрылся / close уже был отправлен — окей
                    logger.info("Client websocket already closed on backend clean close")

            # Реальные ошибки при проксировании
            except Exception:
                logger.exception("Error while forwarding balancer messages; closing client websocket")
                try:
                    await client_ws.close(code=1011)
                except RuntimeError:
                    logger.info("Client websocket already closed when handling error in balancer_to_client")

        # Если оба корутина завершаются без исключений, _proxy_stream
        # отработает штатно и наружу ничего не полетит
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

    logger.info(
        "Broker app initialized with host=%s port=%s strategy=%s",
        settings.service.host,
        settings.service.port,
        settings.service.strategy,
    )

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
            logger.error("Registration failed: missing field %s", exc.args[0])
            raise HTTPException(status_code=400, detail=f"Missing registration field: {exc.args[0]}") from exc
        logger.info(
            "Received registration request for balancer %s at %s:%s with %s instances",
            registration.id,
            registration.host,
            registration.port,
            registration.instances,
        )
        await pool.register(registration)
        return {"status": "ok"}

    @app.get("/broker/balancers")
    async def list_balancers():
        balancers = await pool.list()
        logger.info("Listing balancers; count=%s", len(balancers))
        return {"balancers": [b.__dict__ for b in balancers]}

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await pool.aclose()

    @app.websocket("/v1/text-to-speech/{voice_id}/stream-input")
    async def proxy_ws(ws: WebSocket, voice_id: str):
        await ws.accept()
        logger.info("Accepted websocket for voice_id=%s", voice_id)
        try:
            balancer = await pool.choose(settings.service.strategy)
        except BalancerUnavailable:
            logger.error("No balancer available for websocket request")
            await ws.close(code=1013, reason="No balancer with idle workers available")
            return

        query_bytes: bytes = ws.scope.get("query_string", b"")
        balancer_url = balancer.websocket_url(voice_id, query_bytes)
        logger.info("Proxying websocket request to balancer %s at %s", balancer.id, balancer_url)

        try:
            await _proxy_stream(ws, balancer_url)
        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Websocket with balancer %s closed cleanly", balancer.id)
            # Ничего не делаем, это штатная ситуация
        except Exception:
            logger.exception("Proxy failure; dropping balancer %s and closing websocket", balancer.id)
            await pool.drop(balancer.id)
            try:
                await ws.close(code=1011, reason="Failed to proxy request")
            except RuntimeError:
                logger.info("Client websocket already closed when handling proxy failure")

    return app


def run_broker(settings: BrokerSettings) -> None:
    logger.info("Starting broker server on %s:%s", settings.service.host, settings.service.port)
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

