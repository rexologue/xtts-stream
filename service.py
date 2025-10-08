# xtts_ws_server.py
from __future__ import annotations
import os, json, time, base64, asyncio
from dataclasses import fields
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# === твои модули ===
from xtts import Xtts
from xtts_config import XttsArgs, XttsAudioConfig, XttsConfig

# ------------------------
# Конфиг/загрузка модели
# ------------------------

def _filter_kwargs(cls, data: dict) -> dict:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}

def load_config(config_path: Path) -> XttsConfig:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_args = raw.get("model_args", {})
    audio_args = raw.get("audio", {})
    cfg_kwargs = _filter_kwargs(XttsConfig, raw)
    cfg = XttsConfig(**cfg_kwargs)
    if isinstance(model_args, dict):
        cfg.model_args = XttsArgs(**_filter_kwargs(XttsArgs, model_args))
    if isinstance(audio_args, dict):
        cfg.audio = XttsAudioConfig(**_filter_kwargs(XttsAudioConfig, audio_args))
    return cfg

XTTS_CONFIG     = Path(os.environ.get("XTTS_CONFIG", "config.json"))
XTTS_CHECKPOINT = Path(os.environ.get("XTTS_CHECKPOINT", "model.pth"))
XTTS_TOKENIZER  = os.environ.get("XTTS_TOKENIZER")  # vocab.json (опц.)
XTTS_SPEAKER    = Path(os.environ.get("XTTS_SPEAKER", "ref.wav"))  # референс-голос
XTTS_DEVICE     = os.environ.get("XTTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
XTTS_LANGUAGE   = os.environ.get("XTTS_LANGUAGE", "ru")
MAX_CONCURRENCY = int(os.environ.get("XTTS_MAX_CONCURRENCY", "1"))  # по умолчанию сериализуем

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model: Xtts = None  # type: ignore
cfg: XttsConfig = None  # type: ignore
sample_rate: int = 24000
voice_latent = None
voice_embed = None
gpu_sema = asyncio.Semaphore(MAX_CONCURRENCY)

FRAME_MS = 20
def f32_to_s16le_bytes(x: np.ndarray) -> bytes:
    y = np.clip(x, -1.0, 1.0)
    y = (y * 32767.0).astype(np.int16)
    return y.tobytes()

def encode_audio_bytes(frame_f32: np.ndarray, output_format: str) -> bytes:
    # Минимальный набор: pcm_24000 (S16LE). Расширишь при необходимости (ulaw_8000/mp3/opus).
    if output_format == "pcm_24000":
        return f32_to_s16le_bytes(frame_f32)
    return f32_to_s16le_bytes(frame_f32)

async def _gen_frames(text: str, *, stream_chunk_size: int, overlap_wav_len: int,
                      left_ctx_seconds: Optional[int], speed: float) -> AsyncIterator[np.ndarray]:
    """
    Обёртка над model.inference_stream(...), отдаёт float32[-1;1] кадры ~20мс.
    ВАЖНО: inference_stream — синхронный генератор; дергаем его из потока, чтобы не блокировать event loop.
    """
    def _make_gen():
        return model.inference_stream(
            text=text,
            language=XTTS_LANGUAGE,
            gpt_cond_latent=voice_latent,
            speaker_embedding=voice_embed,
            stream_chunk_size=stream_chunk_size,
            overlap_wav_len=overlap_wav_len,
            left_context_seconds=left_ctx_seconds,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            length_penalty=cfg.length_penalty,
            repetition_penalty=cfg.repetition_penalty,
            speed=speed,
            enable_text_splitting=False,
        )

    gen = await asyncio.to_thread(_make_gen)
    while True:
        try:
            chunk = await asyncio.to_thread(next, gen)
        except StopIteration:
            break
        if chunk is None:
            continue
        # torch.Tensor -> np.float32 mono
        if isinstance(chunk, torch.Tensor):
            arr = chunk.detach().to(torch.float32).cpu().numpy()
        else:
            arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim == 0 or arr.size == 0:
            continue
        yield arr

# ------------------------
# Состояние сеанса (буфер текста и триггеры)
# ------------------------

class GenContext:
    def __init__(self, output_format: str, sync_alignment: bool, chunk_schedule: List[int],
                 stream_chunk_size: int, overlap_wav_len: int, left_ctx_seconds: Optional[int], speed: float):
        self.output_format = output_format
        self.sync_alignment = sync_alignment
        self.chunk_schedule = chunk_schedule or [120, 160, 250, 290]
        self.next_thr = 0
        self.buffer = ""
        self.lock = asyncio.Lock()
        self.stream_chunk_size = stream_chunk_size
        self.overlap_wav_len = overlap_wav_len
        self.left_ctx_seconds = left_ctx_seconds
        self.speed = speed
        self.t_last = time.monotonic()

    def touch(self): self.t_last = time.monotonic()
    def add_text(self, t: str): self.buffer += t; self.touch()
    def pop_all(self) -> str: s = self.buffer; self.buffer=""; return s
    def due_by_schedule(self) -> bool:
        if self.next_thr >= len(self.chunk_schedule): return False
        need = self.chunk_schedule[self.next_thr]
        if len(self.buffer) >= need:
            self.next_thr += 1
            return True
        return False

# ------------------------
# WS: stream-input (ElevenLabs совместимо)
# ------------------------

@app.websocket("/v1/text-to-speech/{voice_id}/stream-input")
async def ws_stream_input(ws: WebSocket, voice_id: str):
    await ws.accept()

    qp = dict(ws.query_params)
    output_format = qp.get("output_format", "pcm_24000")
    inactivity_timeout = float(qp.get("inactivity_timeout", "20"))
    sync_alignment = qp.get("sync_alignment", "false").lower() == "true"

    # Необязательные кастом-параметры под XTTS
    stream_chunk_size = int(qp.get("stream_chunk_size", "20"))  # GPT токены/сэмплы — зависит от твоей имплементации
    overlap_wav_len   = int(qp.get("overlap_wav_len", "512"))
    left_ctx_seconds  = int(qp["left_context_seconds"]) if "left_context_seconds" in qp else None
    speed             = float(qp.get("speed", "1.0"))

    # Заголовки (как у 11labs)
    _ = ws.headers.get("xi-api-key")  # при желании — валидация
    # _ = ws.headers.get("authorization")

    ctx = GenContext(output_format, sync_alignment, [], stream_chunk_size, overlap_wav_len, left_ctx_seconds, speed)

    try:
        while True:
            # idle-таймаут
            if time.monotonic() - ctx.t_last > inactivity_timeout:
                await ws.close(code=1000)
                break

            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            ctx.touch()

            data = json.loads(raw)

            # принять/обновить расписание триггера генерации
            gen_cfg = data.get("generation_config") or {}
            if "chunk_length_schedule" in gen_cfg:
                ctx.chunk_schedule = list(map(int, gen_cfg["chunk_length_schedule"]))

            txt  = data.get("text", None)
            flush = bool(data.get("flush"))
            try_trigger = bool(data.get("try_trigger_generation"))

            # keep-alive: пробел
            if txt == " ":
                continue

            # завершение текущей реплики
            if txt == "":
                pending = ctx.pop_all()
                if pending:
                    async with ctx.lock, gpu_sema:
                        async for frame in _gen_frames(
                            pending,
                            stream_chunk_size=ctx.stream_chunk_size,
                            overlap_wav_len=ctx.overlap_wav_len,
                            left_ctx_seconds=ctx.left_ctx_seconds,
                            speed=ctx.speed,
                        ):
                            payload = {"audio": base64.b64encode(encode_audio_bytes(frame, ctx.output_format)).decode("ascii"),
                                       "isFinal": False}
                            
                            if ctx.sync_alignment:
                                payload["normalizedAlignment"] = {"charStartTimesMs": [0], "charDurationsMs": [FRAME_MS]}
                            await ws.send_text(json.dumps(payload))

                await ws.send_text(json.dumps({"isFinal": True}))
                
                # контекст не закрываем — как у 11labs, можно начать новую фразу тем же сокетом
                ctx = GenContext(output_format, sync_alignment, ctx.chunk_schedule,
                                 stream_chunk_size, overlap_wav_len, left_ctx_seconds, speed)
                continue

            # обычный текст
            if isinstance(txt, str) and txt:
                ctx.add_text(txt)

            # форс-триггеры
            if flush or try_trigger or ctx.due_by_schedule():
                pending = ctx.pop_all()
                if pending:
                    async with ctx.lock, gpu_sema:
                        async for frame in _gen_frames(
                            pending,
                            stream_chunk_size=ctx.stream_chunk_size,
                            overlap_wav_len=ctx.overlap_wav_len,
                            left_ctx_seconds=ctx.left_ctx_seconds,
                            speed=ctx.speed,
                        ):
                            payload = {"audio": base64.b64encode(encode_audio_bytes(frame, ctx.output_format)).decode("ascii"),
                                       "isFinal": False}
                            if ctx.sync_alignment:
                                payload["normalizedAlignment"] = {"charStartTimesMs": [0], "charDurationsMs": [FRAME_MS]}
                            await ws.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        return

# ------------------------
# Инициализация модели и голоса (1 раз при старте)
# ------------------------

@app.on_event("startup")
def _startup():
    global model, cfg, sample_rate, voice_latent, voice_embed

    cfg = load_config(XTTS_CONFIG)
    cfg.model_dir = str(XTTS_CHECKPOINT.parent)

    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(
        cfg,
        checkpoint_path=str(XTTS_CHECKPOINT),
        vocab_path=str(XTTS_TOKENIZER) if XTTS_TOKENIZER else None,
        speaker_file_path=None,
        use_deepspeed=True,
    )
    model.to(XTTS_DEVICE)
    model.eval()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # прогреваем и кэшируем голос (один референс для минимального примера)
    voice_settings = {
        "gpt_cond_len": cfg.gpt_cond_len,
        "gpt_cond_chunk_len": cfg.gpt_cond_chunk_len,
        "max_ref_length": cfg.max_ref_len,
        "sound_norm_refs": cfg.sound_norm_refs,
    }
    voice = model.clone_voice(speaker_wav=str(XTTS_SPEAKER), **voice_settings)
    # кэш, чтобы не дёргать clone_voice на каждый запрос
    voice_latent = voice["gpt_conditioning_latents"]
    voice_embed  = voice["speaker_embedding"]

    sample_rate = int(cfg.model_args.output_sample_rate)
    if sample_rate != 24000:
        # для простоты: ожидаем 24 кГц (как в pcm_24000). При желании добавь ресемплер с torchaudio/soxr.
        raise RuntimeError(f"Expected model output_sample_rate=24000, got {sample_rate}")

    print(f"[XTTS] model loaded on {XTTS_DEVICE}, SR={sample_rate}, voice ready from {XTTS_SPEAKER}")

if __name__ == "__main__":
    # пример: UVICORN_WORKERS=1 XTTS_MAX_CONCURRENCY=1 python xtts_ws_server.py
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
