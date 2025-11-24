"""Worker process hosting a single XTTS instance."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import copy
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.config import LOGGING_CONFIG

from xtts_stream.api.service.pacing import DEFAULT_TARGET_LEAD_MS, MAX_PACKET_MS, Pacer, iter_time_shards
from xtts_stream.api.service.settings import SettingsError, load_settings
from xtts_stream.api.wrappers.base import StreamGenerationConfig, StreamingTTSWrapper
from xtts_stream.api.wrappers.xtts import XttsStreamingWrapper


CONFIG_ENV_VAR = "XTTS_CONFIG_FILE"
ALLOWED_SAMPLE_RATES = {8000, 16000, 24000, 44100}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_worker_port: int | None = None
metrics_logger: logging.Logger | None = None


class _WorkerPortFilter(logging.Filter):
    def __init__(self, port: int) -> None:
        super().__init__()
        self.port = port

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        if not hasattr(record, "worker"):
            record.worker = self.port
        return True


def _configure_worker_logging(port: int) -> None:
    filt = _WorkerPortFilter(port)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", __name__):
        logging.getLogger(name).addFilter(filt)
    logging.getLogger().addFilter(filt)


def _log_config_with_worker(port: int) -> dict:
    cfg = copy.deepcopy(LOGGING_CONFIG)
    cfg["formatters"]["default"]["fmt"] = "%(levelprefix)s [worker=%(worker)s] %(message)s"
    cfg["formatters"]["access"]["fmt"] = (
        "%(levelprefix)s [worker=%(worker)s] %(client_addr)s - \"%(request_line)s\" %(status_code)s"
    )
    return cfg


def _init_metrics_logger(log_dir: Path, port: int) -> tuple[logging.Logger, Path]:
    if log_dir.exists() and not log_dir.is_dir():
        raise RuntimeError(f"log_dir must be a directory: {log_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)

    dated = log_dir / f"metrics_{datetime.now().date().isoformat()}_{port}.log"
    handler = logging.FileHandler(dated)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger(f"xtts_stream.streaming_metrics.{port}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    logger.addHandler(handler)

    return logger, dated


def _config_path() -> Path:
    if CONFIG_ENV_VAR not in os.environ:
        raise RuntimeError("Environment variable XTTS_CONFIG_FILE must point to the service configuration file.")
    return Path(os.environ[CONFIG_ENV_VAR]).expanduser().resolve(strict=False)


def parse_el_audio_format(fmt: str) -> Tuple[str, int]:
    m = re.match(r"^pcm_(\d{4,6})$", fmt)
    if not m:
        raise ValueError(f"Unsupported EL_AUDIO_FORMAT '{fmt}'. Expected 'pcm_<sr>'.")
    sr = int(m.group(1))
    if sr not in ALLOWED_SAMPLE_RATES:
        raise ValueError(
            f"Unsupported sample rate {sr}. Allowed: {sorted(ALLOWED_SAMPLE_RATES)}."
        )
    return "pcm", sr


def pcm_from_float(frame_f32: np.ndarray) -> bytes:
    x = np.asarray(frame_f32, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16, copy=False)
    return pcm.tobytes()


def _maybe_resample_frame(frame_f32: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample ``frame_f32`` to ``dst_sr`` using torchaudio if needed."""

    if src_sr == dst_sr:
        return frame_f32

    tensor = torch.from_numpy(np.asarray(frame_f32, dtype=np.float32))
    if tensor.ndim != 1:
        tensor = tensor.reshape(-1)

    resampled = torchaudio.functional.resample(tensor.unsqueeze(0), src_sr, dst_sr)
    return resampled.squeeze(0).cpu().numpy()


@dataclass
class PacketAggregator:
    """Aggregates raw PCM frames into packets according to a frame schedule."""

    schedule_frames: List[int]
    default_frames_per_packet: int = 1

    def __post_init__(self) -> None:
        self.idx = 0
        self.cur_frames = 0
        self.buf = bytearray()
        self.cur_ms = 0.0

    def _current_threshold(self) -> int:
        if self.idx < len(self.schedule_frames):
            return self.schedule_frames[self.idx]
        return self.default_frames_per_packet

    def add_frame(self, raw_bytes: bytes, frame_ms: float) -> List[Tuple[bytes, float]]:
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    global tts_wrapper, metrics_logger
    cfg_path = _config_path()
    try:
        settings = load_settings(cfg_path)
    except SettingsError as exc:
        raise RuntimeError(str(exc)) from exc

    metrics_logger = None
    if settings.service.log_dir:
        if _worker_port is None:
            raise RuntimeError("Worker port is not initialised")
        metrics_logger, metrics_path = _init_metrics_logger(settings.service.log_dir, _worker_port)
        logger.info("Streaming metrics will be written to %s", metrics_path)

    tts_wrapper = XttsStreamingWrapper.from_settings(settings.model, metrics_logger=metrics_logger)
    logger.info("XTTS model initialised and ready for generation.")

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
gpu_sema = asyncio.Semaphore(1)


async def stream_audio(wrapper: Optional[StreamingTTSWrapper], text: str, options: StreamGenerationConfig) -> AsyncIterator[np.ndarray]:
    if wrapper is None:
        raise RuntimeError("TTS wrapper has not been initialised")
    async for frame in wrapper.stream(text, options):
        yield frame


@app.websocket("/v1/text-to-speech/{voice_id}/stream-input")
async def ws_stream_input(ws: WebSocket, voice_id: str):
    await ws.accept()

    qp = dict(ws.query_params)

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
        else:
            pending_initial_packet = first
    except asyncio.TimeoutError:
        pending_initial_packet = None

    try:
        encoding, sr = parse_el_audio_format(output_format)
        assert encoding == "pcm"
    except Exception as e:
        await ws.send_text(json.dumps({"error": f"Invalid EL_AUDIO_FORMAT: {e}"}))
        await ws.close(code=1003)
        return

    pacer = Pacer(
        sample_rate=sr,
        target_lead_ms=target_lead_ms,
        first_packet_no_wait=settings.service.first_packet_no_wait,
    )
    source_sample_rate = tts_wrapper.sample_rate if tts_wrapper is not None else sr

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

    try:
        pending_first_data: List[dict] = []
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

            if txt == "":
                pending = ctx.pop_all()
                if pending:
                    async with ctx.lock, gpu_sema:
                        aggregator = PacketAggregator(schedule_frames=ctx.frame_schedule or [], default_frames_per_packet=1)

                        async for frame in stream_audio(tts_wrapper, pending, ctx.generation_options):
                            resampled = _maybe_resample_frame(frame, source_sample_rate, sr)
                            raw_bytes = pcm_from_float(resampled)
                            frame_ms = pacer.duration_ms_from_pcm_bytes(len(raw_bytes))

                            for pkt_bytes, pkt_ms in aggregator.add_frame(raw_bytes, frame_ms):
                                for shard_bytes, shard_ms in iter_time_shards(pkt_bytes, sr, MAX_PACKET_MS):
                                    await pacer.wait_before_send(shard_ms)
                                    payload = {"audio": base64.b64encode(shard_bytes).decode("ascii"), "isFinal": False}
                                    if ctx.sync_alignment:
                                        payload["normalizedAlignment"] = {"charStartTimesMs": [0], "charDurationsMs": [shard_ms]}
                                    await ws.send_text(json.dumps(payload))
                                    pacer.on_sent(shard_ms)

                        tail = aggregator.flush()
                        if tail:
                            pkt_bytes, pkt_ms = tail
                            for shard_bytes, shard_ms in iter_time_shards(pkt_bytes, sr, MAX_PACKET_MS):
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
                            resampled = _maybe_resample_frame(frame, source_sample_rate, sr)
                            raw_bytes = pcm_from_float(resampled)
                            frame_ms = pacer.duration_ms_from_pcm_bytes(len(raw_bytes))

                            for pkt_bytes, pkt_ms in aggregator.add_frame(raw_bytes, frame_ms):
                                for shard_bytes, shard_ms in iter_time_shards(pkt_bytes, sr, MAX_PACKET_MS):
                                    await pacer.wait_before_send(shard_ms)
                                    payload = {"audio": base64.b64encode(shard_bytes).decode("ascii"), "isFinal": False}
                                    if ctx.sync_alignment:
                                        payload["normalizedAlignment"] = {"charStartTimesMs": [0], "charDurationsMs": [shard_ms]}
                                    await ws.send_text(json.dumps(payload))
                                    pacer.on_sent(shard_ms)

                        tail = aggregator.flush()
                        if tail:
                            pkt_bytes, pkt_ms = tail
                            for shard_bytes, shard_ms in iter_time_shards(pkt_bytes, sr, MAX_PACKET_MS):
                                await pacer.wait_before_send(shard_ms)
                                payload = {"audio": base64.b64encode(shard_bytes).decode("ascii"), "isFinal": False}
                                if ctx.sync_alignment:
                                    payload["normalizedAlignment"] = {"charStartTimesMs": [0], "charDurationsMs": [shard_ms]}
                                await ws.send_text(json.dumps(payload))
                                pacer.on_sent(shard_ms)

    except WebSocketDisconnect:
        return


def run_worker(port: int) -> None:
    global _worker_port
    _worker_port = port
    _configure_worker_logging(port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", log_config=_log_config_with_worker(port))


if __name__ == "__main__":
    raise RuntimeError("Use xtts_stream.api.service.balancer to launch workers.")

