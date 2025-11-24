#!/usr/bin/env python3
"""Launch multiple streaming TTS requests in parallel for latency testing."""

import argparse
import asyncio
import base64
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import websockets


def build_uri(
    host: str,
    port: int,
    voice_id: str,
    sr: int,
    stream_chunk_size: int,
    left_context_seconds: float,
    overlap_wav_len: int,
    speed: float,
    language: Optional[str],
) -> str:
    query = {
        "output_format": f"pcm_{sr}",
        "stream_chunk_size": str(stream_chunk_size),
        "left_context_seconds": str(left_context_seconds),
        "overlap_wav_len": str(overlap_wav_len),
        "speed": str(speed),
    }
    if language:
        query["language"] = language
    query_str = "&".join(f"{k}={v}" for k, v in query.items())
    return f"ws://{host}:{port}/v1/text-to-speech/{voice_id}/stream-input?{query_str}"


async def stream_once(
    *,
    host: str,
    port: int,
    voice_id: str,
    text: str,
    sr: int,
    stream_chunk_size: int,
    left_context_seconds: float,
    overlap_wav_len: int,
    speed: float,
    language: Optional[str],
    target_lead_ms: float,
) -> tuple[float, Optional[float]]:
    """Return total latency and TTFA for a single streaming request."""

    uri = build_uri(
        host=host,
        port=port,
        voice_id=voice_id,
        sr=sr,
        stream_chunk_size=stream_chunk_size,
        left_context_seconds=left_context_seconds,
        overlap_wav_len=overlap_wav_len,
        speed=speed,
        language=language,
    )

    start = time.perf_counter()
    ttfa_ms: Optional[float] = None

    async with websockets.connect(uri, max_size=None) as ws:
        init_message = {
            "type": "ttsInitRequest",
            "audio_format": f"pcm_{sr}",
            "generation_config": {
                "speed": speed,
                "language": language,
            },
            "target_lead_ms": target_lead_ms,
        }
        await ws.send(json.dumps(init_message))

        await ws.send(json.dumps({"text": text}))
        await ws.send(json.dumps({"flush": True}))
        flush_sent_at = time.perf_counter()
        await ws.send(json.dumps({"text": ""}))

        while True:
            response = json.loads(await ws.recv())

            if "audio" in response:
                if ttfa_ms is None:
                    ttfa_ms = (time.perf_counter() - flush_sent_at) * 1000.0
                # No need to store the audio for benchmarking.
                _ = base64.b64decode(response["audio"])

            if response.get("isFinal") is True:
                break

    latency_ms = (time.perf_counter() - start) * 1000.0
    return latency_ms, ttfa_ms


def _worker(params: dict) -> dict:
    try:
        latency_ms, ttfa_ms = asyncio.run(stream_once(**params))
        return {"latency_ms": latency_ms, "ttfa_ms": ttfa_ms, "error": None}
    except Exception as exc:  # pragma: no cover - benchmark helper
        return {"latency_ms": None, "ttfa_ms": None, "error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N streaming queries in parallel")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=60215, help="Server port")
    parser.add_argument("--voice-id", default="VOICE123", help="Voice identifier")
    parser.add_argument("--text", default="Hello! This is a streaming TTS test.")
    parser.add_argument("--sr", type=int, default=24000, help="Sample rate")
    parser.add_argument("--stream_chunk_size", type=int, default=10)
    parser.add_argument("--left_context_seconds", type=float, default=1.0)
    parser.add_argument("--overlap_wav_len", type=int, default=512)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--language", default=None)
    parser.add_argument("--target_lead_ms", type=float, default=20.0)
    parser.add_argument("--queries", type=int, default=4, help="Number of parallel requests")
    parser.add_argument(
        "--metrics-file",
        default="metrics.csv",
        help="CSV file to store latency and TTFA results",
    )
    return parser.parse_args()


def write_results(path: str, results: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "latency_ms", "ttfa_ms", "error"])
        for idx, item in enumerate(results):
            writer.writerow([idx, item.get("latency_ms"), item.get("ttfa_ms"), item.get("error")])


def print_summary(results: list[dict]) -> None:
    successes = [r for r in results if r.get("error") is None]
    for idx, item in enumerate(results):
        if item.get("error"):
            print(f"[{idx}] ERROR: {item['error']}")
        else:
            ttfa_display = f", TTFA: {item['ttfa_ms']:.1f} ms" if item.get("ttfa_ms") is not None else ""
            print(f"[{idx}] Latency: {item['latency_ms']:.1f} ms{ttfa_display}")

    if successes:
        latencies = [r["latency_ms"] for r in successes if r.get("latency_ms") is not None]
        ttfas = [r["ttfa_ms"] for r in successes if r.get("ttfa_ms") is not None]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"Average latency: {avg_latency:.1f} ms")
        if ttfas:
            avg_ttfa = sum(ttfas) / len(ttfas)
            print(f"Average TTFA: {avg_ttfa:.1f} ms")


def main() -> None:
    args = parse_args()
    params = dict(
        host=args.host,
        port=args.port,
        voice_id=args.voice_id,
        text=args.text,
        sr=args.sr,
        stream_chunk_size=args.stream_chunk_size,
        left_context_seconds=args.left_context_seconds,
        overlap_wav_len=args.overlap_wav_len,
        speed=args.speed,
        language=args.language,
        target_lead_ms=args.target_lead_ms,
    )

    with ProcessPoolExecutor(max_workers=args.queries) as executor:
        futures = [executor.submit(_worker, params) for _ in range(args.queries)]
        results = [future.result() for future in futures]

    write_results(args.metrics_file, results)
    print(f"Saved metrics to {args.metrics_file}")
    print_summary(results)


if __name__ == "__main__":
    main()
