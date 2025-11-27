"""Balancer entrypoint that proxies generation requests to worker processes."""

from __future__ import annotations

import os
import sys
import random
import socket
import asyncio
from asyncio.streams import StreamReader, StreamWriter
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager

import websockets
import httpx

import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from xtts_stream.api.service.worker import run_worker
from xtts_stream.api.service.settings import Settings, SettingsError, load_settings


CONFIG_ENV_VAR = "XTTS_CONFIG_FILE"
BALANCER_ID_ENV_VAR = "XTTS_BALANCER_ID"
DEFAULT_REGISTER_RETRIES = 5
REGISTER_ENDPOINT = "/broker/register"
BROKER_HEADER = "x-xtts-broker"


def _config_path() -> Path:
    if CONFIG_ENV_VAR not in os.environ:
        raise RuntimeError("Environment variable XTTS_CONFIG_FILE must point to the service configuration file.")
    return Path(os.environ[CONFIG_ENV_VAR]).expanduser().resolve(strict=False)


def _random_port() -> int:
    while True:
        port = random.randint(50000, 65535)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue


def _balancer_id(settings: Settings) -> str:
    env_id = os.environ.get(BALANCER_ID_ENV_VAR)
    if env_id:
        return env_id
    return f"{socket.gethostname()}:{settings.service.port}"


def _advertised_host(settings: Settings) -> str:
    if settings.service.host in {"0.0.0.0", "::"}:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    return settings.service.host


async def _register_with_broker(settings: Settings, balancer_id: str) -> None:
    broker_url = f"http://{settings.service.broker_host}:{settings.service.broker_port}{REGISTER_ENDPOINT}"
    payload = {
        "id": balancer_id,
        "host": _advertised_host(settings),
        "port": settings.service.port,
        "instances": settings.service.instances,
    }

    attempt = 0
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                resp = await client.post(broker_url, json=payload)
                resp.raise_for_status()
                return
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt >= DEFAULT_REGISTER_RETRIES:
                    raise RuntimeError(f"Failed to register balancer with broker at {broker_url}") from exc
                await asyncio.sleep(min(2**attempt, 10))


@dataclass
class WorkerHandle:
    port: int
    process: asyncio.subprocess.Process
    busy: bool = False


class WorkerPool:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.workers: list[WorkerHandle] = []
        self._lock = asyncio.Lock()

    async def _wait_for_ready(self, worker: WorkerHandle, timeout: float = 300.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            if worker.process.returncode is not None:
                raise RuntimeError("A worker process exited unexpectedly during startup")

            try:
                reader: StreamReader
                writer: StreamWriter
                reader, writer = await asyncio.open_connection("127.0.0.1", worker.port)
            except OSError:
                if loop.time() > deadline:
                    raise TimeoutError(f"Timed out waiting for worker on port {worker.port} to start")
                await asyncio.sleep(0.05)
                continue

            writer.close()
            await writer.wait_closed()
            return

    async def start(self, instances: int) -> None:
        for _ in range(instances):
            port = _random_port()
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "from xtts_stream.api.service.worker import run_worker; run_worker(%d)" % port,
                env={**os.environ, CONFIG_ENV_VAR: str(self.config_path)},
            )
            self.workers.append(WorkerHandle(port=port, process=proc))

        await asyncio.gather(*(self._wait_for_ready(worker) for worker in self.workers))

    async def acquire(self) -> WorkerHandle:
        while True:
            async with self._lock:
                for worker in self.workers:
                    if worker.process.returncode is not None:
                        raise RuntimeError("A worker process exited unexpectedly")
                    if not worker.busy:
                        worker.busy = True
                        return worker
            await asyncio.sleep(0.05)

    async def release(self, worker: WorkerHandle) -> None:
        async with self._lock:
            worker.busy = False

    async def idle_workers(self) -> int:
        async with self._lock:
            return sum(1 for worker in self.workers if not worker.busy and worker.process.returncode is None)

    async def shutdown(self) -> None:
        for worker in self.workers:
            if worker.process.returncode is None:
                worker.process.terminate()
        for worker in self.workers:
            if worker.process.returncode is None:
                try:
                    await asyncio.wait_for(worker.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    worker.process.kill()
        self.workers.clear()


def _build_worker_url(worker: WorkerHandle, voice_id: str, ws: WebSocket) -> str:
    query_bytes: bytes = ws.scope.get("query_string", b"")
    query = f"?{query_bytes.decode()}" if query_bytes else ""
    return f"ws://127.0.0.1:{worker.port}/v1/text-to-speech/{voice_id}/stream-input{query}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg_path = _config_path()
    try:
        settings = load_settings(cfg_path)
    except SettingsError as exc:
        raise RuntimeError(str(exc)) from exc

    pool = WorkerPool(cfg_path)
    await pool.start(settings.service.instances)
    balancer_id = _balancer_id(settings)

    if not settings.service.standalone_mode:
        await _register_with_broker(settings, balancer_id)

    app.state.settings = settings
    app.state.pool = pool
    app.state.balancer_id = balancer_id

    try:
        yield
    finally:
        await pool.shutdown()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/workers/idle")
async def idle_workers():
    pool: WorkerPool = app.state.pool
    idle = await pool.idle_workers()
    return {"idle_workers": idle}


@app.websocket("/v1/text-to-speech/{voice_id}/stream-input")
async def ws_balancer(ws: WebSocket, voice_id: str):
    await ws.accept()
    pool: WorkerPool = app.state.pool
    settings: Settings = app.state.settings

    if not settings.service.standalone_mode:
        broker_header = ws.headers.get(BROKER_HEADER)
        if broker_header != "1":
            await ws.close(code=1008, reason="Balancer is broker-managed; connect via Broker service")
            return

    worker = await pool.acquire()
    try:
        worker_url = _build_worker_url(worker, voice_id, ws)

        async with websockets.connect(worker_url, max_size=None) as backend:

            async def client_to_worker():
                try:
                    while True:
                        msg = await ws.receive_text()
                        await backend.send(msg)
                except websockets.exceptions.ConnectionClosedOK:
                    pass
                except WebSocketDisconnect:
                    await backend.close()
                except Exception:
                    await backend.close()

            async def worker_to_client():
                try:
                    while True:
                        msg = await backend.recv()
                        if isinstance(msg, bytes):
                            await ws.send_bytes(msg)
                        else:
                            await ws.send_text(msg)
                except websockets.exceptions.ConnectionClosedOK:
                    pass
                except Exception:
                    await ws.close()

            await asyncio.gather(client_to_worker(), worker_to_client())

    finally:
        await pool.release(worker)


def run_balancer(settings: Settings) -> None:
    uvicorn.run(app, host=settings.service.host, port=settings.service.port, log_level="info")


if __name__ == "__main__":
    cfg_path = _config_path()
    settings = load_settings(cfg_path)
    run_balancer(settings)

