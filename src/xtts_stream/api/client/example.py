#!/usr/bin/env python3
"""Minimal example client for the ElevenLabs-compatible websocket API."""

import argparse
import asyncio
import base64
import json
import time
import wave
import shutil
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
    play: bool,
) -> tuple[bytes, float, Optional[float]]:
    """Send a single TTS request and return audio, total latency, and TTFA."""

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

    audio = bytearray()
    start = time.perf_counter()
    ttfa_ms: Optional[float] = None

    ffplay_process = None
    ffplay_stdin_closed = False
    if play:
        if shutil.which("ffplay") is None:
            raise RuntimeError("ffplay not found; install FFmpeg or disable --play")
        ffplay_process = await asyncio.create_subprocess_exec(
            "ffplay",
            "-autoexit",
            "-nodisp",
            "-f",
            f"pcm_s16le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

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
                chunk = base64.b64decode(response["audio"])
                audio.extend(chunk)
                if ffplay_process and ffplay_process.stdin and not ffplay_stdin_closed:
                    try:
                        ffplay_process.stdin.write(chunk)
                        await ffplay_process.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        # ffplay may exit early (e.g., if it cannot play the stream);
                        # keep streaming without playback instead of crashing.
                        ffplay_stdin_closed = True

            if response.get("isFinal") is True:
                break

    latency_ms = (time.perf_counter() - start) * 1000.0
    if ffplay_process:
        if ffplay_process.stdin:
            ffplay_process.stdin.close()
            await ffplay_process.stdin.wait_closed()
        await ffplay_process.wait()
    return bytes(audio), latency_ms, ttfa_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single streaming TTS request")
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
    parser.add_argument("--outfile", default="out.wav", help="Path to save WAV output")
    parser.add_argument("--play", action="store_true", help="Play audio with ffplay while streaming")
    parser.add_argument("--no-save", action="store_true", help="Do not write the resulting WAV file")
    return parser.parse_args()


def write_wav(path: str, audio: bytes, sr: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio)


def main() -> None:
    args = parse_args()
    audio, latency_ms, ttfa_ms = asyncio.run(
        stream_once(
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
            play=args.play,
        )
    )

    if not args.no_save:
        write_wav(args.outfile, audio, args.sr)
        print(f"Saved audio to {args.outfile}")
    print(f"Total latency: {latency_ms:.1f} ms")
    if ttfa_ms is not None:
        print(f"TTFA: {ttfa_ms:.1f} ms")


if __name__ == "__main__":
    main()
