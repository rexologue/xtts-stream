"""Streaming wrapper for Coqui XTTS without auxiliary processing."""

from __future__ import annotations

import json
import time
import asyncio
import logging
from pathlib import Path
from dataclasses import fields
from typing import AsyncIterator, Optional

import torch
import numpy as np
from ruaccent import RUAccent

from xtts_stream.api.service.settings import ModelSettings
from xtts_stream.api.wrappers.base import StreamGenerationConfig, StreamingTTSWrapper

from xtts_stream.core.xtts import StreamingMetrics, Xtts
from xtts_stream.core.xtts_config import XttsArgs, XttsAudioConfig, XttsConfig


def _filter_kwargs(cls, data: dict) -> dict:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def load_config(config_path: Path) -> XttsConfig:
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    model_args = raw.get("model_args", {})
    audio_args = raw.get("audio", {})
    cfg_kwargs = _filter_kwargs(XttsConfig, raw)

    cfg = XttsConfig(**cfg_kwargs)

    if isinstance(model_args, dict):
        cfg.model_args = XttsArgs(**_filter_kwargs(XttsArgs, model_args))

    if isinstance(audio_args, dict):
        cfg.audio = XttsAudioConfig(**_filter_kwargs(XttsAudioConfig, audio_args))

    return cfg


class XttsStreamingWrapper(StreamingTTSWrapper):
    """Implementation of :class:`StreamingTTSWrapper` for XTTS."""

    def __init__(
        self,
        *,
        cfg: XttsConfig,
        checkpoint: Path,
        tokenizer: Optional[Path],
        speaker_wav: Path,
        device: str,
        language: str,
        metrics_logger: Optional[logging.Logger] = None,
        use_accentizer: bool = True,
    ) -> None:
        self.cfg = cfg
        self.cfg.model_dir = str(checkpoint.parent)
        self.model = Xtts.init_from_config(self.cfg)

        use_deepspeed = torch.cuda.is_available() and device.lower().startswith("cuda")
        if not use_deepspeed:
            raise ValueError("CUDA IS NOT AVAILABLE!")

        self.model.load_checkpoint(
            self.cfg,
            checkpoint_path=str(checkpoint),
            vocab_path=str(tokenizer) if tokenizer else None,
            speaker_file_path=None,
            use_deepspeed=use_deepspeed,
        )
        self.model.to(device)
        self.model.eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        voice_settings = {
            "gpt_cond_len": self.cfg.gpt_cond_len,
            "gpt_cond_chunk_len": self.cfg.gpt_cond_chunk_len,
            "max_ref_length": self.cfg.max_ref_len,
            "sound_norm_refs": self.cfg.sound_norm_refs,
        }
        voice = self.model.clone_voice(speaker_wav=str(speaker_wav), **voice_settings)

        self.voice_latent = voice["gpt_conditioning_latents"]
        self.voice_embed = voice["speaker_embedding"]

        self.sample_rate = int(self.cfg.model_args.output_sample_rate)
        if self.sample_rate != 24000:
            raise RuntimeError(
                "Expected model output_sample_rate=24000, got {0}".format(self.sample_rate)
            )

        self.device = device
        self.language = language
        self.metrics_logger = metrics_logger

        if use_accentizer:
            self.accentizer = RUAccent()
            self.accentizer.load(
                omograph_model_size="turbo3.1",
                use_dictionary=True,
                tiny_mode=False,
                device=device.upper(),
            )
        else:
            self.accentizer = None

    @classmethod
    def from_settings(
        cls,
        settings: ModelSettings,
        *,
        metrics_logger: Optional[logging.Logger] = None,
    ) -> "XttsStreamingWrapper":
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = load_config(settings.config_path)
        return cls(
            cfg=cfg,
            checkpoint=settings.checkpoint_path,
            tokenizer=settings.tokenizer_path,
            speaker_wav=settings.speaker_wav,
            device=device,
            language=settings.language,
            use_accentizer=settings.enable_accentizer,
            metrics_logger=metrics_logger,
        )

    async def stream(
        self,
        text: str,
        options: StreamGenerationConfig,
    ) -> AsyncIterator[np.ndarray]:
        def _make_gen():
            if self.accentizer:
                processed = self.accentizer.process_all(text)
            else:
                processed = text

            return self.model.inference_stream(
                text=processed,
                language=options.language or self.language,
                gpt_cond_latent=self.voice_latent,
                speaker_embedding=self.voice_embed,
                stream_chunk_size=options.stream_chunk_size,
                overlap_wav_len=options.overlap_wav_len,
                left_context_seconds=options.left_context_seconds,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                length_penalty=self.cfg.length_penalty,
                repetition_penalty=self.cfg.repetition_penalty,
                speed=options.speed,
                enable_text_splitting=False,
                apply_asr=False,
            )

        generator = await asyncio.to_thread(_make_gen)

        def _next_chunk():
            try:
                return next(generator)
            except StopIteration as stop:
                return stop

        while True:
            chunk = await asyncio.to_thread(_next_chunk)
            if isinstance(chunk, StopIteration):
                metrics: Optional[StreamingMetrics] = getattr(chunk, "value", None)
                if metrics and self.metrics_logger:
                    payload = {
                        "time_to_first_token": metrics.time_to_first_token,
                        "time_to_first_audio": metrics.time_to_first_audio,
                        "real_time_factor": metrics.real_time_factor,
                        "latency": metrics.latency,
                        "timestamp": time.time(),
                    }
                    self.metrics_logger.info(json.dumps(payload, ensure_ascii=False))
                break
            
            if chunk is None:
                continue
            if isinstance(chunk, torch.Tensor):
                arr = chunk.detach().to(torch.float32).cpu().numpy()
            else:
                arr = np.asarray(chunk, dtype=np.float32)
            if arr.ndim == 0 or arr.size == 0:
                continue
            yield arr

