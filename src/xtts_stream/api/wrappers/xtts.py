"""Streaming wrapper for Coqui XTTS."""

from __future__ import annotations

import os
import re
import json
import asyncio
from dataclasses import fields
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import torch

from ruaccent import RUAccent

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

        self.accentizer = RUAccent()
        self.accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False, device=device.upper())

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
                text=self.accentizer.process_all(text),
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

    
    _PCM_FMT_RE = re.compile(r"^pcm_(\d+)$")
    _have_torchaudio: Optional[bool] = None
    _have_scipy: Optional[bool] = None

    @staticmethod
    def _tpdf_dither(x: np.ndarray, lsb: float = 1.0 / 32768.0) -> np.ndarray:
        """TPDF-дизеринг: добавляем треугольный шум ±1 LSB перед квантованием в int16."""
        # два независимых равномерных → треугольное распределение
        n = x.shape[-1]
        noise = (np.random.random(n) - np.random.random(n)) * lsb
        return x + noise.astype(np.float32, copy=False)

    @classmethod
    def _check_torchaudio(cls) -> bool:
        if cls._have_torchaudio is None:
            try:
                import torchaudio  # type: ignore
                cls._have_torchaudio = True
            except Exception:
                cls._have_torchaudio = False
        return bool(cls._have_torchaudio)

    @classmethod
    def _check_scipy(cls) -> bool:
        if cls._have_scipy is None:
            try:
                import scipy.signal  # type: ignore
                cls._have_scipy = True
            except Exception:
                cls._have_scipy = False
        return bool(cls._have_scipy)

    @staticmethod
    def _resample_torchaudio(x_f32: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        import torch
        import torchaudio.functional as AF  # type: ignore
        t = torch.from_numpy(x_f32).to(torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,N]
        y = AF.resample(
            t, orig_freq=src_sr, new_freq=dst_sr,
            lowpass_filter_width=16, rolloff=0.99,
            resampling_method="sinc_interp_kaiser", beta=14.769656459379492  # ~80 dB
        )
        return y.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _resample_scipy_poly(x_f32: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        import numpy as _np
        from math import gcd
        from scipy.signal import resample_poly  # type: ignore
        g = gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        # Kaiser beta~14 → ~80 dB подавление боковых лепестков
        y = resample_poly(x_f32, up, down, window=("kaiser", 14.0))
        return _np.asarray(y, dtype=_np.float32, copy=False)

    @staticmethod
    def _design_lowpass_fir(cutoff_hz: float, sr: int, taps: int = 127) -> np.ndarray:
        """
        Классический windowed-sinc c Blackman-окном.
        cutoff_hz — частота среза (Гц), taps — нечётное число.
        """
        import numpy as _np
        assert taps % 2 == 1, "taps must be odd"
        nyq = sr * 0.5
        fc = float(cutoff_hz) / nyq  # [0..1]
        n = _np.arange(taps, dtype=_np.float64)
        m = (taps - 1) / 2.0
        # sinc low-pass
        h = _np.sinc(2 * fc * (n - m))
        # Blackman window
        w = 0.42 - 0.5 * _np.cos(2 * _np.pi * n / (taps - 1)) + 0.08 * _np.cos(4 * _np.pi * n / (taps - 1))
        h *= w
        # нормировка на единичный коэффициент усиления по DC
        h = h / _np.sum(h)
        return h.astype(_np.float32, copy=False)

    @classmethod
    def _resample_fallback(cls, x_f32: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """
        Fallback: антиалиасный downsample (FIR) + линейный rate-convert.
        Для upsample — просто линейная интерполяция.
        """
        import numpy as _np

        x = _np.asarray(x_f32, dtype=_np.float32, copy=False)
        if src_sr == dst_sr or x.size == 0:
            return x

        if dst_sr < src_sr:
            # антиалиас до новой Найквистовой частоты
            cutoff = 0.45 * dst_sr  # чуть ниже 0.5*dst_sr, чтобы оставить запас
            taps = 127 if src_sr >= 16000 else 63
            h = cls._design_lowpass_fir(cutoff_hz=cutoff, sr=src_sr, taps=taps)
            # линейная свёртка (same)
            y = _np.convolve(x, h, mode="same").astype(_np.float32, copy=False)
        else:
            y = x

        # линейное изменение скорости дискретизации
        n_src = y.shape[-1]
        n_dst = max(1, int(round(n_src * (dst_sr / float(src_sr)))))
        t_src = _np.linspace(0.0, 1.0, num=n_src, endpoint=False, dtype=_np.float64)
        t_dst = _np.linspace(0.0, 1.0, num=n_dst, endpoint=False, dtype=_np.float64)
        out = _np.interp(t_dst, t_src, y.astype(_np.float64, copy=False))
        return out.astype(_np.float32, copy=False)
    
    def encode_audio(self, frame_f32: np.ndarray, output_format: str) -> bytes:
        """
        Кодирует float32 [-1, 1] в PCM16 mono с ресемплингом под любой pcm_<sr>,
        включая 8000, 16000, 22050, 24000, 44100, 48000.
        Повторной валидации формата не делаем — app.py уже проверил.
        """
        # 1) подготовим моно float32 [-1..1]
        x = np.asarray(frame_f32, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        x = np.clip(x, -1.0, 1.0)

        # 2) извлекаем целевую частоту из 'pcm_<sr>'; если не распознали — считаем равной исходной
        m = self._PCM_FMT_RE.match(output_format.strip()) if isinstance(output_format, str) else None
        target_sr = int(m.group(1)) if m else self.sample_rate

        # 3) ресемплинг при необходимости
        if target_sr != self.sample_rate:
            try:
                if self._check_torchaudio():
                    x = self._resample_torchaudio(x, self.sample_rate, target_sr)
                elif self._check_scipy():
                    x = self._resample_scipy_poly(x, self.sample_rate, target_sr)
                else:
                    x = self._resample_fallback(x, self.sample_rate, target_sr)
            except Exception:
                # на случай любых неожиданных проблем — безопасный откат на fallback
                x = self._resample_fallback(x, self.sample_rate, target_sr)

            # страхующий клиппинг после ресемплинга
            x = np.clip(x, -1.0, 1.0)

        # 4) TPDF-дизеринг + квантование в PCM16 mono
        x = self._tpdf_dither(x)  # мягкая квантзация
        x = np.clip(x, -1.0, 1.0)
        pcm = (x * 32767.0).astype(np.int16, copy=False)

        return pcm.tobytes()

