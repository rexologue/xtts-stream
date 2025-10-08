"""FastAPI application exposing a streaming TTS websocket.

The websocket contract follows the ElevenLabs ``stream-input`` protocol.  The
implementation itself is model agnostic and relies on :mod:`xtts_stream.wrappers`
to provide concrete backends.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from xtts_stream.service.settings import SettingsError, load_settings
from xtts_stream.wrappers.base import StreamGenerationConfig, StreamingTTSWrapper
from xtts_stream.wrappers.xtts import XttsStreamingWrapper


CONFIG_PATH = Path(os.environ.get("XTTS_SETTINGS_FILE", "config.yaml")).resolve()

try:
    settings = load_settings(CONFIG_PATH)
except SettingsError as exc:
    raise RuntimeError(str(exc)) from exc

MAX_CONCURRENCY = settings.service.max_concurrency

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_wrapper: Optional[StreamingTTSWrapper] = None
gpu_sema = asyncio.Semaphore(MAX_CONCURRENCY)


def encode_audio_bytes(
    wrapper: Optional[StreamingTTSWrapper], frame_f32: np.ndarray, output_format: str
) -> bytes:
    if wrapper is None:
        raise RuntimeError("TTS wrapper has not been initialised")
    return wrapper.encode_audio(frame_f32, output_format)


async def stream_audio(
    wrapper: Optional[StreamingTTSWrapper],
    text: str,
    options: StreamGenerationConfig,
) -> AsyncIterator[np.ndarray]:
    if wrapper is None:
        raise RuntimeError("TTS wrapper has not been initialised")
    async for frame in wrapper.stream(text, options):
        yield frame


@dataclass
class GenContext:
    output_format: str
    sync_alignment: bool
    chunk_schedule: List[int]
    generation_options: StreamGenerationConfig

    def __post_init__(self) -> None:
        self.next_thr = 0
        self.buffer = ""
        self.lock = asyncio.Lock()
        self.t_last = time.monotonic()

    def touch(self) -> None:
        self.t_last = time.monotonic()

    def add_text(self, text: str) -> None:
        self.buffer += text
        self.touch()

    def pop_all(self) -> str:
        pending = self.buffer
        self.buffer = ""
        return pending

    def due_by_schedule(self) -> bool:
        if self.next_thr >= len(self.chunk_schedule):
            return False
        need = self.chunk_schedule[self.next_thr]
        if len(self.buffer) >= need:
            self.next_thr += 1
            return True
        return False


@app.websocket("/v1/text-to-speech/{voice_id}/stream-input")
async def ws_stream_input(ws: WebSocket, voice_id: str):  # noqa: D401
    await ws.accept()

    qp = dict(ws.query_params)
    output_format = qp.get("output_format", "pcm_24000")
    inactivity_timeout = float(qp.get("inactivity_timeout", "20"))
    sync_alignment = qp.get("sync_alignment", "false").lower() == "true"

    stream_chunk_size = int(qp.get("stream_chunk_size", "20"))
    overlap_wav_len = int(qp.get("overlap_wav_len", "512"))
    left_ctx_seconds = int(qp["left_context_seconds"]) if "left_context_seconds" in qp else None
    speed = float(qp.get("speed", "1.0"))
    language = qp.get("language")

    _ = ws.headers.get("xi-api-key")

    generation_options = StreamGenerationConfig(
        stream_chunk_size=stream_chunk_size,
        overlap_wav_len=overlap_wav_len,
        left_context_seconds=left_ctx_seconds,
        speed=speed,
        language=language,
    )
    ctx = GenContext(output_format, sync_alignment, [], generation_options)

    try:
        while True:
            if time.monotonic() - ctx.t_last > inactivity_timeout:
                await ws.close(code=1000)
                break

            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            ctx.touch()

            data = json.loads(raw)

            gen_cfg = data.get("generation_config") or {}
            if "chunk_length_schedule" in gen_cfg:
                ctx.chunk_schedule = list(map(int, gen_cfg["chunk_length_schedule"]))

            txt = data.get("text")
            flush = bool(data.get("flush"))
            try_trigger = bool(data.get("try_trigger_generation"))

            if txt == " ":
                continue

            if txt == "":
                pending = ctx.pop_all()
                if pending:
                    async with ctx.lock, gpu_sema:
                        async for frame in stream_audio(tts_wrapper, pending, ctx.generation_options):
                            payload = {
                                "audio": base64.b64encode(
                                    encode_audio_bytes(tts_wrapper, frame, ctx.output_format)
                                ).decode("ascii"),
                                "isFinal": False,
                            }
                            if ctx.sync_alignment:
                                frame_ms = (
                                    tts_wrapper.frame_duration_ms
                                    if tts_wrapper is not None
                                    else StreamingTTSWrapper.frame_duration_ms
                                )
                                payload["normalizedAlignment"] = {
                                    "charStartTimesMs": [0],
                                    "charDurationsMs": [frame_ms],
                                }
                            await ws.send_text(json.dumps(payload))

                await ws.send_text(json.dumps({"isFinal": True}))
                ctx = GenContext(output_format, sync_alignment, list(ctx.chunk_schedule), generation_options)
                continue

            if isinstance(txt, str) and txt:
                ctx.add_text(txt)

            if flush or try_trigger or ctx.due_by_schedule():
                pending = ctx.pop_all()
                if pending:
                    async with ctx.lock, gpu_sema:
                        async for frame in stream_audio(tts_wrapper, pending, ctx.generation_options):
                            payload = {
                                "audio": base64.b64encode(
                                    encode_audio_bytes(tts_wrapper, frame, ctx.output_format)
                                ).decode("ascii"),
                                "isFinal": False,
                            }
                            if ctx.sync_alignment:
                                frame_ms = (
                                    tts_wrapper.frame_duration_ms
                                    if tts_wrapper is not None
                                    else StreamingTTSWrapper.frame_duration_ms
                                )
                                payload["normalizedAlignment"] = {
                                    "charStartTimesMs": [0],
                                    "charDurationsMs": [frame_ms],
                                }
                            await ws.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        return


@app.on_event("startup")
def _startup() -> None:
    global tts_wrapper
    tts_wrapper = XttsStreamingWrapper.from_settings(settings.model)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if tts_wrapper is not None:
        await tts_wrapper.close()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.service.host,
        port=settings.service.port,
    )
