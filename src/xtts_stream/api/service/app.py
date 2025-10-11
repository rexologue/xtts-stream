"""
FastAPI app exposing a streaming TTS websocket compatible with ElevenLabs "stream-input".
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from xtts_stream.api.service.settings import SettingsError, load_settings
from xtts_stream.api.wrappers.base import StreamGenerationConfig, StreamingTTSWrapper
from xtts_stream.api.wrappers.xtts import XttsStreamingWrapper

# ======================================================================================
# Settings & init
# ======================================================================================

CONFIG_ENV_VAR = "XTTS_SETTINGS_FILE"
if CONFIG_ENV_VAR not in os.environ:
    raise RuntimeError(
        "Environment variable XTTS_SETTINGS_FILE must point to the service configuration file."
    )

CONFIG_PATH = Path(os.environ[CONFIG_ENV_VAR]).expanduser().resolve(strict=False)

try:
    settings = load_settings(CONFIG_PATH)
except SettingsError as exc:
    raise RuntimeError(str(exc)) from exc

MAX_CONCURRENCY = settings.service.max_concurrency

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ======================================================================================
# Lifespan: model init / shutdown (+ warmup)
# ======================================================================================

@asynccontextmanager
async def lifespan(_: FastAPI):
    global tts_wrapper
    tts_wrapper = XttsStreamingWrapper.from_settings(settings.model)
    logger.info("XTTS model initialised and ready for generation.")

    # --- WARMUP: прогреть графы и вокодер до старта --- 
    async def _warmup():
        try:
            opts = StreamGenerationConfig(
                stream_chunk_size=3,          # маленький чанк для быстрой первой выдачи
                overlap_wav_len=512,
                left_context_seconds=0.6,
                speed=1.0,
                language=settings.model.language,
            )
            # однократный проход до первого кадра — и выходим
            async for frame in tts_wrapper.stream("привет", opts):  # type: ignore
                if frame is not None and np.size(frame) > 0:
                    break
            logger.info("XTTS warmup done.")
        except Exception as e:
            logger.warning(f"XTTS warmup skipped with error: {e}")

    await _warmup()
    # --------------------------------------------------- 

    try:
        yield
    finally:
        if tts_wrapper is not None:
            await tts_wrapper.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_wrapper: Optional[StreamingTTSWrapper] = None
gpu_sema = asyncio.Semaphore(MAX_CONCURRENCY)

# ======================================================================================
# EL audio format parsing & pacing
# ======================================================================================

EL_PCM_RE = re.compile(r"^pcm_(\d{4,6})$")  # e.g., pcm_24000
ALLOWED_SR = {8000, 16000, 22050, 24000, 44100, 48000}
PCM_BYTES_PER_SAMPLE = 2  # 16-bit PCM
PCM_CHANNELS = 1          # mono
DEFAULT_TARGET_LEAD_MS = 20.0
LEAD_HYSTERESIS_MS = 2.0

# time-shard: верхняя граница длительности одного WS-пакета 
MAX_PACKET_MS = 60.0  # 50–80 мс — хороший диапазон                     

def parse_el_audio_format(fmt: str) -> Tuple[str, int]:
    m = EL_PCM_RE.match(fmt)
    if not m:
        raise ValueError(f"Unsupported EL_AUDIO_FORMAT '{fmt}'. Expected 'pcm_<sr>'.")
    sr = int(m.group(1))
    if sr not in ALLOWED_SR:
        raise ValueError(f"Unsupported sample rate {sr}. Allowed: {sorted(ALLOWED_SR)}.")
    return "pcm", sr

@dataclass
class Pacer:
    sample_rate: int
    target_lead_ms: float = DEFAULT_TARGET_LEAD_MS

    def __post_init__(self):
        self.bytes_per_sec = self.sample_rate * PCM_BYTES_PER_SAMPLE * PCM_CHANNELS
        self.ms_per_byte = 1000.0 / self.bytes_per_sec
        self.start_t = time.monotonic()
        self.sent_ms = 0.0

    def duration_ms_from_pcm_bytes(self, nbytes: int) -> float:
        return nbytes * self.ms_per_byte

    async def wait_before_send(self, next_frame_ms: float):
        while True:
            now_ms = (time.monotonic() - self.start_t) * 1000.0
            ahead_ms = self.sent_ms - now_ms
            if ahead_ms <= (self.target_lead_ms - LEAD_HYSTERESIS_MS):
                return
            delta_ms = max(0.0, ahead_ms - self.target_lead_ms)
            await asyncio.sleep(min(delta_ms / 1000.0, 0.050))

    def on_sent(self, frame_ms: float):
        self.sent_ms += frame_ms

# --- helpers for time-shard ---  
def _bytes_per_ms(sr: int) -> float:
    return (sr * PCM_BYTES_PER_SAMPLE * PCM_CHANNELS) / 1000.0

def _iter_time_shards(raw_bytes: bytes, sr: int, max_packet_ms: float):
    """Режем любой PCM16 mono на кусочки <= max_packet_ms по длительности."""
    b_per_ms = _bytes_per_ms(sr)
    max_bytes = max(1, int(max_packet_ms * b_per_ms))
    for i in range(0, len(raw_bytes), max_bytes):
        shard = raw_bytes[i:i + max_bytes]
        shard_ms = len(shard) / b_per_ms
        yield shard, shard_ms

# ======================================================================================
# Backend helpers
# ======================================================================================

def encode_audio_bytes(wrapper: Optional[StreamingTTSWrapper], frame_f32: np.ndarray, output_format: str) -> bytes:
    if wrapper is None:
        raise RuntimeError("TTS wrapper has not been initialised")
    return wrapper.encode_audio(frame_f32, output_format)

async def stream_audio(wrapper: Optional[StreamingTTSWrapper], text: str, options: StreamGenerationConfig) -> AsyncIterator[np.ndarray]:
    if wrapper is None:
        raise RuntimeError("TTS wrapper has not been initialised")
    async for frame in wrapper.stream(text, options):  # type: ignore
        yield frame

# ======================================================================================
# Packet aggregator (frame schedule)
# ======================================================================================

@dataclass
class PacketAggregator:
    """Aggregates raw PCM frames into WS packets according to a frame schedule."""
    schedule_frames: List[int]              # e.g., [50, 100]
    default_frames_per_packet: int = 1      
    def __post_init__(self):
        self.idx = 0
        self.cur_frames = 0
        self.buf = bytearray()
        self.cur_ms = 0.0

    def _current_threshold(self) -> int:
        if self.idx < len(self.schedule_frames):
            return self.schedule_frames[self.idx]
        return self.default_frames_per_packet

    def add_frame(self, raw_bytes: bytes, frame_ms: float) -> List[Tuple[bytes, float]]:
        """Add single PCM frame; return list of ready packets (bytes, total_ms)."""
        self.buf.extend(raw_bytes)
        self.cur_frames += 1
        self.cur_ms += frame_ms
        out: List[Tuple[bytes, float]] = []
        thr = self._current_threshold()
        if self.cur_frames >= thr:
            out.append((bytes(self.buf), self.cur_ms))
            self.buf.clear()
            self.cur_frames = 0
            self.cur_ms = 0.0
            if self.idx < len(self.schedule_frames):
                self.idx += 1
        return out

    def flush(self) -> Optional[Tuple[bytes, float]]:
        if self.cur_frames > 0:
            pkt = (bytes(self.buf), self.cur_ms)
            self.buf.clear()
            self.cur_frames = 0
            self.cur_ms = 0.0
            return pkt
        return None

# ======================================================================================
# Per-connection context
# ======================================================================================

@dataclass
class GenContext:
    output_format: str
    sync_alignment: bool
    char_schedule: List[int]
    frame_schedule: List[int]
    generation_options: StreamGenerationConfig

    def __post_init__(self) -> None:
        self.next_char_thr = 0
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

    def due_by_char_schedule(self) -> bool:
        if self.next_char_thr >= len(self.char_schedule):
            return False
        need = self.char_schedule[self.next_char_thr]
        if len(self.buffer) >= need:
            self.next_char_thr += 1
            return True
        return False

# ======================================================================================
# Redis forwarder (optional) + time-shard
# ======================================================================================

async def forward_from_redis(
    ws: WebSocket,
    pacer: Pacer,
    audio_channel: str,
    inactivity_timeout: float,
    sync_alignment: bool,
    sr: int,  
):
    try:
        import redis.asyncio as redis
    except Exception as e:
        await ws.send_text(json.dumps({"error": f"Redis mode requested but redis-py missing: {e}"}))
        await ws.close(code=1011)
        return

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    r = redis.from_url(redis_url, decode_responses=False)
    pubsub = r.pubsub()
    await pubsub.subscribe(audio_channel)
    last_activity = time.monotonic()

    try:
        while True:
            if time.monotonic() - last_activity > inactivity_timeout:
                await ws.close(code=1000)
                break

            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.2)
            if not msg:
                await asyncio.sleep(0.05)
                continue

            last_activity = time.monotonic()

            try:
                data = json.loads(msg["data"])
                b64 = data.get("audio")
                is_final = bool(data.get("isFinal", False))
                if not isinstance(b64, str):
                    continue
                raw = base64.b64decode(b64, validate=True)
            except Exception:
                raw = msg["data"]
                is_final = False

            # --- time-shard и пейсинг --- 
            for shard_bytes, shard_ms in _iter_time_shards(raw, sr, MAX_PACKET_MS):
                await pacer.wait_before_send(shard_ms)
                payload = {"audio": base64.b64encode(shard_bytes).decode("ascii"), "isFinal": False}
                if sync_alignment:
                    payload["normalizedAlignment"] = {"charStartTimesMs": [0], "charDurationsMs": [shard_ms]}
                await ws.send_text(json.dumps(payload))
                pacer.on_sent(shard_ms)

            if is_final:
                break
    finally:
        try:
            await pubsub.unsubscribe(audio_channel)
        except Exception:
            pass
        await r.close()

# ======================================================================================
# WebSocket endpoint
# ======================================================================================

@app.websocket("/v1/text-to-speech/{voice_id}/stream-input")
async def ws_stream_input(ws: WebSocket, voice_id: str):  # noqa: D401
    await ws.accept()

    # ---------- 0) Query как fallback (дефолты до init) ----------
    qp = dict(ws.query_params)

    mode: str = qp.get("mode", "generate")
    speed: float = float(qp.get("speed", "1.0"))
    language: Optional[str] = qp.get("language")
    stream_chunk_size: int = int(qp.get("stream_chunk_size", "10"))  
    overlap_wav_len: int = int(qp.get("overlap_wav_len", "512"))
    left_ctx_seconds: Optional[float] = float(qp["left_context_seconds"]) if "left_context_seconds" in qp else 1.0
    sync_alignment: bool = qp.get("sync_alignment", "false").lower() == "true"
    inactivity_timeout: float = float(qp.get("inactivity_timeout", "20"))

    output_format: str = qp.get("output_format", "pcm_24000")
    target_lead_ms: float = DEFAULT_TARGET_LEAD_MS

    char_schedule: List[int] = []
    frame_schedule: List[int] = []
    audio_channel: Optional[str] = None

    # ---------- 1) ttsInitRequest (источник истины) ----------
    pending_initial_packet = None
    try:
        raw_first = await asyncio.wait_for(ws.receive_text(), timeout=1.5)
        first = json.loads(raw_first)

        if first.get("type") == "ttsInitRequest":
            gen_cfg = first.get("generation_config") or {}
            output_format = first.get("audio_format") or first.get("EL_AUDIO_FORMAT") or output_format
            speed = float(gen_cfg.get("speed", speed))
            language = gen_cfg.get("language", language)
            target_lead_ms = float(first.get("target_lead_ms", target_lead_ms))
            char_schedule = list(map(int, gen_cfg.get("chunk_length_schedule", []) or []))
            frame_schedule = list(map(int, gen_cfg.get("chunk_length_schedule_frames", []) or []))
            audio_channel = first.get("audio_channel") or audio_channel
        else:
            pending_initial_packet = first
    except asyncio.TimeoutError:
        pending_initial_packet = None

    # ---------- 2) Валидация формата и фиксация «жёстких» ----------
    try:
        encoding, sr = parse_el_audio_format(output_format)
        assert encoding == "pcm"
    except Exception as e:
        await ws.send_text(json.dumps({"error": f"Invalid EL_AUDIO_FORMAT: {e}"}))
        await ws.close(code=1003)
        return

    pacer = Pacer(sample_rate=sr, target_lead_ms=target_lead_ms)

    # ---------- 3) Стартовые опции генерации ----------
    generation_options = StreamGenerationConfig(
        stream_chunk_size=stream_chunk_size,
        overlap_wav_len=overlap_wav_len,
        left_context_seconds=left_ctx_seconds,
        speed=speed,
        language=language,
    )

    ctx = GenContext(
        output_format=output_format,
        sync_alignment=sync_alignment,
        char_schedule=char_schedule,
        frame_schedule=frame_schedule,
        generation_options=generation_options,
    )

    # --- 4) forward ---
    if mode == "forward":
        if not audio_channel:
            await ws.send_text(json.dumps({"error": "forward mode requires 'audio_channel' in ttsInitRequest"}))
            await ws.close(code=1008)
            return
        await forward_from_redis(
            ws=ws,
            pacer=pacer,
            audio_channel=audio_channel,
            inactivity_timeout=inactivity_timeout,
            sync_alignment=sync_alignment,
            sr=sr,  
        )
        return

    # --- 5) GENERATE loop (с time-shard) ---
    try:
        pending_first_data = []
        if pending_initial_packet:
            pending_first_data.append(pending_initial_packet)

        while True:
            if time.monotonic() - ctx.t_last > inactivity_timeout:
                await ws.close(code=1000)
                break

            try:
                if pending_first_data:
                    data = pending_first_data.pop(0)
                else:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=0.5)
                    data = json.loads(raw)
            except asyncio.TimeoutError:
                continue

            ctx.touch()

            gen_cfg = data.get("generation_config") or {}
            if "chunk_length_schedule" in gen_cfg:
                ctx.char_schedule = list(map(int, gen_cfg["chunk_length_schedule"]))
                ctx.next_char_thr = 0
            if "chunk_length_schedule_frames" in gen_cfg:
                ctx.frame_schedule = list(map(int, gen_cfg["chunk_length_schedule_frames"]))

            txt = data.get("text")
            flush = bool(data.get("flush"))
            try_trigger = bool(data.get("try_trigger_generation"))

            if txt == " ":
                continue

            # End-of-turn: flush and final
            if txt == "":
                pending = ctx.pop_all()
                if pending:
                    async with ctx.lock, gpu_sema:
                        aggregator = PacketAggregator(schedule_frames=ctx.frame_schedule or [], default_frames_per_packet=1)

                        async for frame in stream_audio(tts_wrapper, pending, ctx.generation_options):
                            raw_bytes = encode_audio_bytes(tts_wrapper, frame, ctx.output_format)
                            frame_ms = pacer.duration_ms_from_pcm_bytes(len(raw_bytes))

                            for pkt_bytes, pkt_ms in aggregator.add_frame(raw_bytes, frame_ms):
                                # --- time-shard send ---  
                                for shard_bytes, shard_ms in _iter_time_shards(pkt_bytes, sr, MAX_PACKET_MS):
                                    await pacer.wait_before_send(shard_ms)
                                    payload = {"audio": base64.b64encode(shard_bytes).decode("ascii"), "isFinal": False}
                                    if ctx.sync_alignment:
                                        payload["normalizedAlignment"] = {"charStartTimesMs": [0], "charDurationsMs": [shard_ms]}
                                    await ws.send_text(json.dumps(payload))
                                    pacer.on_sent(shard_ms)

                        # tail
                        tail = aggregator.flush()
                        if tail:
                            pkt_bytes, pkt_ms = tail
                            for shard_bytes, shard_ms in _iter_time_shards(pkt_bytes, sr, MAX_PACKET_MS): 
                                await pacer.wait_before_send(shard_ms)
                                payload = {"audio": base64.b64encode(shard_bytes).decode("ascii"), "isFinal": False}
                                if ctx.sync_alignment:
                                    payload["normalizedAlignment"] = {"charStartTimesMs": [0], "charDurationsMs": [shard_ms]}
                                await ws.send_text(json.dumps(payload))
                                pacer.on_sent(shard_ms)

                await ws.send_text(json.dumps({"isFinal": True}))
                ctx = GenContext(output_format, sync_alignment, list(ctx.char_schedule), list(ctx.frame_schedule), generation_options)
                continue

            if isinstance(txt, str) and txt:
                ctx.add_text(txt)

            if flush or try_trigger or ctx.due_by_char_schedule():
                pending = ctx.pop_all()
                if pending:
                    async with ctx.lock, gpu_sema:
                        aggregator = PacketAggregator(schedule_frames=ctx.frame_schedule or [], default_frames_per_packet=1)

                        async for frame in stream_audio(tts_wrapper, pending, ctx.generation_options):
                            raw_bytes = encode_audio_bytes(tts_wrapper, frame, ctx.output_format)
                            frame_ms = pacer.duration_ms_from_pcm_bytes(len(raw_bytes))

                            for pkt_bytes, pkt_ms in aggregator.add_frame(raw_bytes, frame_ms):
                                for shard_bytes, shard_ms in _iter_time_shards(pkt_bytes, sr, MAX_PACKET_MS):  
                                    await pacer.wait_before_send(shard_ms)
                                    payload = {"audio": base64.b64encode(shard_bytes).decode("ascii"), "isFinal": False}
                                    if ctx.sync_alignment:
                                        payload["normalizedAlignment"] = {"charStartTimesMs": [0], "charDurationsMs": [shard_ms]}
                                    await ws.send_text(json.dumps(payload))
                                    pacer.on_sent(shard_ms)

                        tail = aggregator.flush()
                        if tail:
                            pkt_bytes, pkt_ms = tail
                            for shard_bytes, shard_ms in _iter_time_shards(pkt_bytes, sr, MAX_PACKET_MS): 
                                await pacer.wait_before_send(shard_ms)
                                payload = {"audio": base64.b64encode(shard_bytes).decode("ascii"), "isFinal": False}
                                if ctx.sync_alignment:
                                    payload["normalizedAlignment"] = {"charStartTimesMs": [0], "charDurationsMs": [shard_ms]}
                                await ws.send_text(json.dumps(payload))
                                pacer.on_sent(shard_ms)

    except WebSocketDisconnect:
        return

# ======================================================================================
# Entrypoint
# ======================================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.service.host,
        port=settings.service.port,
    )
