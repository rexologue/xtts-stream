#!/usr/bin/env python3
"""Generate streaming XTTS outputs for an LJSpeech-style dataset and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch

from xtts import StreamingMetrics, Xtts
from xtts_config import XttsArgs, XttsAudioConfig, XttsConfig


def _filter_kwargs(cls, data: dict) -> dict:
    allowed = {field.name for field in fields(cls)}
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
    parser = argparse.ArgumentParser(description="Stream XTTS synthesis for an LJSpeech dataset")
    parser.add_argument("dataset", type=Path, help="Path to LJSpeech-style dataset directory")
    parser.add_argument("--config", type=Path, required=True, help="Path to XTTS config.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to XTTS checkpoint (model.pth)")
    parser.add_argument("--tokenizer", type=Path, help="Path to vocab.json. Defaults to config directory")
    parser.add_argument("--speakers", type=Path, help="Optional speaker embedding archive")
    parser.add_argument("--language", default="en", help="Language code for the text")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Override nucleus sampling")
    parser.add_argument("--top_k", type=int, default=None, help="Override top-k sampling")
    parser.add_argument("--length_penalty", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--split-text", action="store_true", help="Enable automatic sentence splitting")
    parser.add_argument("--stream-chunk-size", type=int, default=20, help="Number of GPT tokens per chunk")
    parser.add_argument("--stream-overlap", type=int, default=512, help="Sample overlap for crossfade")
    parser.add_argument("--stream-ctx-seconds", type=int, default=None, help="Optional left context in seconds")
    parser.add_argument("--output", type=Path, required=True, help="Directory to store generated results")
    return parser.parse_args()


@dataclass
class SampleResult:
    identifier: str
    text: str
    metrics: StreamingMetrics | None


def read_metadata(metadata_path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if not row:
                continue
            identifier = row[0].strip()
            if not identifier:
                continue
            text = row[-1].strip()
            entries.append((identifier, text))
    return entries


def ensure_output_dir(path: Path) -> tuple[Path, Path]:
    path.mkdir(parents=True, exist_ok=True)
    wav_dir = path / "wavs"
    wav_dir.mkdir(exist_ok=True)
    return path, wav_dir


def format_metric(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else ""


def write_results_csv(destination: Path, sample_results: Iterable[SampleResult]) -> None:
    rows: list[list[str]] = []
    ttft_values: list[float] = []
    ttfa_values: list[float] = []
    rtf_values: list[float] = []
    latency_values: list[float] = []

    for result in sample_results:
        metrics = result.metrics
        if metrics is not None:
            if metrics.time_to_first_token is not None:
                ttft_values.append(metrics.time_to_first_token)
            if metrics.time_to_first_audio is not None:
                ttfa_values.append(metrics.time_to_first_audio)
            if metrics.real_time_factor is not None:
                rtf_values.append(metrics.real_time_factor)
            latency_values.append(metrics.latency)
            row = [
                result.identifier,
                result.text,
                format_metric(metrics.time_to_first_token),
                format_metric(metrics.time_to_first_audio),
                format_metric(metrics.real_time_factor),
                format_metric(metrics.latency),
            ]
        else:
            row = [result.identifier, result.text, "", "", "", ""]
        rows.append(row)

    def average(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    avg_row = [
        "average",
        "",
        format_metric(average(ttft_values)),
        format_metric(average(ttfa_values)),
        format_metric(average(rtf_values)),
        format_metric(average(latency_values)),
    ]

    output_path = destination / "results.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "ttft", "ttfa", "rtf", "latency"])
        writer.writerows(rows)
        writer.writerow(avg_row)


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset
    metadata_path = dataset_dir / "metadata.csv"
    reference_wav = dataset_dir / "ref" / "ref.wav"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
    if not reference_wav.is_file():
        raise FileNotFoundError(f"Reference audio not found at {reference_wav}")

    entries = read_metadata(metadata_path)
    if not entries:
        raise RuntimeError("No entries found in metadata.csv")

    output_dir, wav_dir = ensure_output_dir(args.output)

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
        use_deepspeed=True,
    )
    model.to(args.device)
    model.eval()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    voice_settings = {
        "gpt_cond_len": config.gpt_cond_len,
        "gpt_cond_chunk_len": config.gpt_cond_chunk_len,
        "max_ref_length": config.max_ref_len,
        "sound_norm_refs": config.sound_norm_refs,
    }
    voice = model.clone_voice(speaker_wav=str(reference_wav), **voice_settings)
    gpt_latent = voice["gpt_conditioning_latents"]
    speaker_embedding = voice["speaker_embedding"]

    inference_kwargs = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "length_penalty": config.length_penalty,
        "repetition_penalty": config.repetition_penalty,
        "speed": args.speed,
        "enable_text_splitting": args.split_text,
    }

    sample_rate = config.model_args.output_sample_rate
    results: list[SampleResult] = []

    for identifier, text in entries:
        output_wav = wav_dir / f"{identifier}.wav"
        print(f"Generating {identifier}: {text}")
        stream = model.inference_stream(
            text=text,
            language=args.language,
            gpt_cond_latent=gpt_latent,
            speaker_embedding=speaker_embedding,
            stream_chunk_size=args.stream_chunk_size,
            overlap_wav_len=args.stream_overlap,
            left_context_seconds=args.stream_ctx_seconds,
            **inference_kwargs,
        )
        metrics: StreamingMetrics | None = None
        with sf.SoundFile(
            str(output_wav),
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
                stream_file.write(np.squeeze(chunk_np))
        results.append(SampleResult(identifier=identifier, text=text, metrics=metrics))

    write_results_csv(output_dir, results)
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
