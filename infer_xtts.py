#!/usr/bin/env python3
"""Command line interface for XTTS inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from xtts import Xtts
from xtts_config import XttsArgs, XttsAudioConfig, XttsConfig


def _filter_kwargs(cls, data: dict) -> dict:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def load_config(config_path: Path) -> XttsConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    model_args = raw.get("model_args", {})
    audio_args = raw.get("audio", {})
    config_kwargs = _filter_kwargs(XttsConfig, raw)
    config = XttsConfig(**config_kwargs)
    if isinstance(model_args, dict):
        config.model_args = XttsArgs(**_filter_kwargs(XttsArgs, model_args))
    if isinstance(audio_args, dict):
        config.audio = XttsAudioConfig(**_filter_kwargs(XttsAudioConfig, audio_args))
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XTTS inference")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (model.pth)")
    parser.add_argument("--tokenizer", type=Path, help="Path to vocab.json. Defaults to config directory")
    parser.add_argument("--speakers", type=Path, help="Optional speaker embedding archive")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", default="en", help="Language code for the text")
    parser.add_argument("--reference", type=Path, required=True, help="Reference audio for cloning")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the generated wav file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Override nucleus sampling")
    parser.add_argument("--top_k", type=int, default=None, help="Override top-k sampling")
    parser.add_argument("--length_penalty", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--split-text", action="store_true", help="Enable automatic sentence splitting")
    parser.add_argument("--stream", action="store_true", help="Enable streaming synthesis")
    parser.add_argument(
        "--stream-chunk-size",
        type=int,
        default=20,
        help="Number of GPT tokens to accumulate before emitting an audio chunk",
    )
    parser.add_argument(
        "--stream-overlap",
        type=int,
        default=512,
        help="Number of samples to overlap when cross-fading streamed chunks",
    )
    parser.add_argument(
        "--stream-ctx-seconds",
        type=int,
        default=None,
        help="How much seconds of context will be used for vocoder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.model_dir = str(args.checkpoint.parent)

    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.length_penalty is not None:
        config.length_penalty = args.length_penalty
    if args.repetition_penalty is not None:
        config.repetition_penalty = args.repetition_penalty

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(args.checkpoint),
        vocab_path=str(args.tokenizer) if args.tokenizer else None,
        speaker_file_path=str(args.speakers) if args.speakers else None,
        use_deepspeed=True
    )
    model.to(args.device)
    model.eval()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    inference_kwargs = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "length_penalty": config.length_penalty,
        "repetition_penalty": config.repetition_penalty,
        "speed": args.speed,
        "enable_text_splitting": args.split_text,
    }

    if args.stream:
        voice_settings = {
            "gpt_cond_len": config.gpt_cond_len,
            "gpt_cond_chunk_len": config.gpt_cond_chunk_len,
            "max_ref_length": config.max_ref_len,
            "sound_norm_refs": config.sound_norm_refs,
        }
        voice = model.clone_voice(speaker_wav=str(args.reference), **voice_settings)
        gpt_latent = voice["gpt_conditioning_latents"]
        speaker_embedding = voice["speaker_embedding"]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = config.model_args.output_sample_rate
        total_samples = 0
        stream = model.inference_stream(
            text=args.text,
            language=args.language,
            gpt_cond_latent=gpt_latent,
            speaker_embedding=speaker_embedding,
            stream_chunk_size=args.stream_chunk_size,
            overlap_wav_len=args.stream_overlap,
            left_context_seconds=args.stream_ctx_seconds,
            **inference_kwargs,
        )
        metrics = None
        with sf.SoundFile(
            str(args.output),
            mode="w",
            samplerate=sample_rate,
            channels=1,
            subtype="FLOAT",
        ) as stream_file:
            while True:
                try:
                    chunk = next(stream)
                except StopIteration as stop:
                    metrics = stop.value
                    break

                if chunk is None:
                    continue
                chunk_np = chunk.detach().to(torch.float32).cpu().numpy()
                if chunk_np.ndim == 0 or chunk_np.size == 0:
                    continue
                stream_file.write(chunk_np)
                total_samples += int(chunk_np.size)
                print(
                    f"Streamed {chunk_np.size} samples (total {total_samples})",
                    flush=True,
                )
        print(f"Saved streamed audio to {args.output}")
        if metrics is not None:
            print("Streaming metrics:")
            if metrics.time_to_first_token is not None:
                print(f"  Time to first token: {metrics.time_to_first_token:.3f}s")
            if metrics.time_to_first_audio is not None:
                print(f"  Time to first audio: {metrics.time_to_first_audio:.3f}s")
            if metrics.real_time_factor is not None:
                print(f"  Real-time factor: {metrics.real_time_factor:.3f}")
            print(f"  Latency (avg chunk): {metrics.latency:.3f}s")
    else:
        result = model.synthesize(
            text=args.text,
            speaker_wav=str(args.reference),
            language=args.language,
            **inference_kwargs,
        )

        wav = np.asarray(result["wav"], dtype=np.float32)
        sf.write(args.output, wav, config.model_args.output_sample_rate)
        print(f"Saved audio to {args.output}")


if __name__ == "__main__":
    main()
