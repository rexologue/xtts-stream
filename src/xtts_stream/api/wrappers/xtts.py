"""Streaming wrapper for Coqui XTTS without auxiliary processing."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import AsyncIterator, Callable, Dict, Optional, Tuple

import torch
import numpy as np
from ruaccent import RUAccent

from xtts_stream.api.service.settings import ModelSettings
from xtts_stream.api.wrappers.base import StreamGenerationConfig, StreamingTTSWrapper

from xtts_stream.core.xtts import StreamingMetrics, Xtts
from xtts_stream.core.xtts_config import XttsArgs, XttsAudioConfig, XttsConfig


logger = logging.getLogger(__name__)


@dataclass
class VoiceLatents:
    gpt_conditioning_latents: torch.Tensor
    speaker_embedding: torch.Tensor


class VoicePool:
    """Stores preloaded voice conditioning latents."""

    def __init__(self) -> None:
        self._voices: Dict[str, VoiceLatents] = {}
        self.default_voice_id: Optional[str] = None

    def add_voice(self, voice_id: str, latents: VoiceLatents) -> None:
        if voice_id not in self._voices:
            self._voices[voice_id] = latents
            if self.default_voice_id is None:
                self.default_voice_id = voice_id

    def resolve(self, voice_id: Optional[str]) -> Tuple[str, VoiceLatents]:
        if voice_id and voice_id in self._voices:
            return voice_id, self._voices[voice_id]

        if self.default_voice_id is None:
            raise RuntimeError("No reference voices available")

        if voice_id and voice_id not in self._voices:
            logger.warning("Voice '%s' not found, using default '%s'", voice_id, self.default_voice_id)

        return self.default_voice_id, self._voices[self.default_voice_id]

    @property
    def available_ids(self) -> Tuple[str, ...]:
        return tuple(self._voices.keys())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._voices)


def load_voices_from_directory(
    voices_dir: Path, loader: Callable[[Path], VoiceLatents], *, logger_: Optional[logging.Logger] = None
) -> VoicePool:
    pool = VoicePool()
    if not voices_dir.exists() or not voices_dir.is_dir():
        raise RuntimeError(f"voices_dir does not exist or is not a directory: {voices_dir}")

    voice_paths = sorted(voices_dir.glob("*.mp3"))
    if not voice_paths:
        raise RuntimeError(f"No .mp3 reference voices found in directory: {voices_dir}")

    for path in voice_paths:
        voice_id = path.stem
        try:
            latents = loader(path)
            pool.add_voice(voice_id, latents)
            if logger_:
                logger_.info("Loaded reference voice '%s' from %s", voice_id, path)
        except Exception:
            if logger_:
                logger_.exception("Failed to load reference voice from %s", path)
            continue

    if len(pool) == 0:
        raise RuntimeError(f"Failed to load any reference voices from {voices_dir}")

    if logger_:
        logger_.info("Loaded %d reference voices: %s", len(pool), ", ".join(pool.available_ids))

    return pool


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
        speaker_wav: Optional[Path],
        voices_dir: Optional[Path],
        device: str,
        language: str,
        metrics_logger: Optional[logging.Logger] = None,
        use_accentizer: bool = True,
    ) -> None:
        self.cfg = cfg
        self.cfg.model_dir = str(checkpoint.parent)
        self.model = Xtts.init_from_config(self.cfg, apply_asr=True)

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

        self.voice_pool = VoicePool()
        if voices_dir is not None:
            self.voice_pool = load_voices_from_directory(voices_dir, self._load_voice_latents, logger_=logger)

        if len(self.voice_pool) == 0 and speaker_wav is not None:
            latents = self._load_voice_latents(speaker_wav)
            self.voice_pool.add_voice(speaker_wav.stem, latents)

        if len(self.voice_pool) == 0:
            raise RuntimeError("No reference voices could be loaded. Check voices_dir or ref_file configuration.")

        logger.info(
            "Voice pool initialised with %d voice(s). Default: '%s'", len(self.voice_pool), self.voice_pool.default_voice_id
        )
        logger.info("Available voices: %s", ", ".join(self.voice_pool.available_ids))

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
                device="CUDA",
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
            voices_dir=settings.voices_dir,
            device=device,
            language=settings.language,
            use_accentizer=settings.enable_accentizer,
            metrics_logger=metrics_logger,
        )

    def _load_voice_latents(self, speaker_wav: Path) -> VoiceLatents:
        voice_settings = {
            "gpt_cond_len": self.cfg.gpt_cond_len,
            "gpt_cond_chunk_len": self.cfg.gpt_cond_chunk_len,
            "max_ref_length": self.cfg.max_ref_len,
            "sound_norm_refs": self.cfg.sound_norm_refs,
        }
        voice = self.model.clone_voice(speaker_wav=str(speaker_wav), **voice_settings)
        return VoiceLatents(
            gpt_conditioning_latents=voice["gpt_conditioning_latents"],
            speaker_embedding=voice["speaker_embedding"],
        )

    async def stream(
        self,
        text: str,
        options: StreamGenerationConfig,
        *,
        voice_id: Optional[str] = None,
    ) -> AsyncIterator[np.ndarray]:
        def _make_gen():
            selected_voice_id, latents = self.voice_pool.resolve(voice_id)
            logger.info("Using reference voice '%s'", selected_voice_id)

            if self.accentizer:
                processed = self.accentizer.process_all(text)
            else:
                processed = text

            return self.model.inference_stream(
                text=processed,
                language=options.language or self.language,
                gpt_cond_latent=latents.gpt_conditioning_latents,
                speaker_embedding=latents.speaker_embedding,
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
                apply_asr=True,
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

