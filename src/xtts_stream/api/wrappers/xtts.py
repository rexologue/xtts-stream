"""Streaming wrapper for Coqui XTTS."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import torch

from xtts_stream.core.xtts import Xtts
from xtts_stream.core.xtts_config import XttsArgs, XttsAudioConfig, XttsConfig
from xtts_stream.api.service.settings import ResolvedModelSettings
from xtts_stream.api.wrappers.base import StreamGenerationConfig, StreamingTTSWrapper


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
    ) -> None:
        self.cfg = cfg
        self.cfg.model_dir = str(checkpoint.parent)
        self.model = Xtts.init_from_config(self.cfg)
        self.model.load_checkpoint(
            self.cfg,
            checkpoint_path=str(checkpoint),
            vocab_path=str(tokenizer) if tokenizer else None,
            speaker_file_path=None,
            use_deepspeed=True,
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

        self.language = language
        self.device = device

    @classmethod
    def from_environment(cls) -> "XttsStreamingWrapper":
        cfg_path = Path(os.environ.get("XTTS_CONFIG", "config.json"))
        checkpoint = Path(os.environ.get("XTTS_CHECKPOINT", "model.pth"))
        tokenizer_env = os.environ.get("XTTS_TOKENIZER")
        speaker = Path(os.environ.get("XTTS_SPEAKER", "ref.wav"))
        device = os.environ.get("XTTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        language = os.environ.get("XTTS_LANGUAGE", "ru")

        cfg = load_config(cfg_path)
        return cls(
            cfg=cfg,
            checkpoint=checkpoint,
            tokenizer=Path(tokenizer_env) if tokenizer_env else None,
            speaker_wav=speaker,
            device=device,
            language=language,
        )

    @classmethod
    def from_settings(cls, settings: ResolvedModelSettings) -> "XttsStreamingWrapper":
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
        )

    async def stream(
        self,
        text: str,
        options: StreamGenerationConfig,
    ) -> AsyncIterator[np.ndarray]:
        def _make_gen():
            return self.model.inference_stream(
                text=text,
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
            )

        generator = await asyncio.to_thread(_make_gen)
        _sentinel = object()

        def _next_chunk():
            try:
                return next(generator)
            except StopIteration:
                return _sentinel

        while True:
            chunk = await asyncio.to_thread(_next_chunk)
            if chunk is _sentinel:
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

    def encode_audio(self, frame_f32: np.ndarray, output_format: str) -> bytes:
        frame_f32 = np.clip(frame_f32, -1.0, 1.0)
        pcm = (frame_f32 * 32767.0).astype(np.int16)
        if output_format == "pcm_24000":
            return pcm.tobytes()
        # default to PCM until additional codecs are implemented
        return pcm.tobytes()
