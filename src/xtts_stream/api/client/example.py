#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example.py — клиент для ElevenLabs-совместимого WS TTS
По умолчанию БЕЗ расписания по кадрам (frames-schedule выключено).
Опционально умеет live-воспроизведение через ffplay.

Deps:
    pip install websockets
    # для --play нужен ffmpeg (ffplay в PATH)
"""

import argparse
import asyncio
import base64
import json
import time
import wave
import shutil
import subprocess
from typing import Optional, List

import websockets


def build_uri(host: str, port: int, voice_id: str, sr: int, sync_alignment: bool,
              stream_chunk_size: int, left_context_seconds: float, overlap_wav_len: int, speed: float,
              language: Optional[str], mode: str) -> str:
    q = {
        "output_format": f"pcm_{sr}",
        "sync_alignment": "true" if sync_alignment else "false",
        "stream_chunk_size": str(stream_chunk_size),
        "left_context_seconds": str(left_context_seconds),  # важно: правильный ключ
        "overlap_wav_len": str(overlap_wav_len),
        "speed": str(speed),
        "mode": mode,  # "generate" | "forward"
    }
    if language:
        q["language"] = language
    qstr = "&".join(f"{k}={v}" for k, v in q.items())
    return f"ws://{host}:{port}/v1/text-to-speech/{voice_id}/stream-input?{qstr}"


def ms_from_bytes(nbytes: int, sr: int, bytes_per_sample: int = 2, channels: int = 1) -> float:
    return (nbytes * 1000.0) / (sr * bytes_per_sample * channels)


async def run_client(args: argparse.Namespace) -> None:
    uri = build_uri(
        host=args.host,
        port=args.port,
        voice_id=args.voice_id,
        sr=args.sr,
        sync_alignment=args.sync_alignment,
        stream_chunk_size=args.stream_chunk_size,
        left_context_seconds=args.left_context_seconds,
        overlap_wav_len=args.overlap_wav_len,
        speed=args.speed,
        language=args.language,
        mode=args.mode,
    )

    print(f"Runing the following URI: {uri}")

    # WAV-выход (по желанию)
    wf = None
    if not args.no_save:
        wf = wave.open(args.outfile, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(args.sr)

    # Live-воспроизведение (опционально)
    ffplay_proc = None
    if args.play:
        if shutil.which("ffplay") is None:
            raise RuntimeError("ffplay не найден в PATH. Установите FFmpeg или уберите --play.")
        ff_cmd = ["ffplay", "-autoexit", "-nodisp", "-f", "s16le", "-ac", "1", "-ar", str(args.sr), "-i", "-"]
        ffplay_proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, bufsize=0)

    ttft_ms = None
    ttfa_ms = None
    t_connect = None
    t_first_trigger = None
    received_audio_ms = 0.0
    packets = 0

    # Псевдо-стрим текста
    chunks: List[str] = []
    if args.text:
        text = args.text.strip()
        if args.chunk_chars > 0 and len(text) > args.chunk_chars:
            for i in range(0, len(text), args.chunk_chars):
                chunks.append(text[i:i + args.chunk_chars])
        else:
            chunks = [text]

    async with websockets.connect(uri, max_size=None) as ws:
        t_connect = time.perf_counter()

        # 1) ttsInitRequest — frames-schedule шлём ТОЛЬКО если задано явно
        gen_cfg = {
            "chunk_length_schedule": [len(c) for c in chunks] if args.use_char_schedule else [],
            "speed": args.speed,
            "language": args.language,
        }
        if args.frames_schedule:
            gen_cfg["chunk_length_schedule_frames"] = [int(x) for x in args.frames_schedule.split(",")]

        init_msg = {
            "type": "ttsInitRequest",
            "audio_format": f"pcm_{args.sr}",
            "generation_config": gen_cfg,
            "target_lead_ms": args.target_lead_ms,
            # "audio_channel": "sess:123:audio"  # если mode=forward
        }
        await ws.send(json.dumps(init_msg))

        # 2) Шлём текст порциями
        for part in chunks:
            await ws.send(json.dumps({"text": part}))
            if args.sleep_between_chunks_ms > 0:
                await asyncio.sleep(args.sleep_between_chunks_ms / 1000.0)

        # 3) Просим флаш
        await ws.send(json.dumps({"flush": True}))
        t_first_trigger = time.perf_counter()

        # 4) Маркер конца батча
        await ws.send(json.dumps({"text": ""}))

        # 5) Читаем ответы
        while True:
            msg = await ws.recv()
            data = json.loads(msg)

            if "audio" in data:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_connect) * 1000.0
                if ttfa_ms is None and t_first_trigger is not None:
                    ttfa_ms = (time.perf_counter() - t_first_trigger) * 1000.0

                pcm = base64.b64decode(data["audio"])

                if wf is not None:
                    wf.writeframes(pcm)
                if ffplay_proc is not None and ffplay_proc.stdin:
                    try:
                        ffplay_proc.stdin.write(pcm)
                    except BrokenPipeError:
                        ffplay_proc = None

                received_audio_ms += ms_from_bytes(len(pcm), args.sr)
                packets += 1

            if data.get("isFinal") is True:
                break

    if wf is not None:
        wf.close()

    if ffplay_proc is not None:
        try:
            if ffplay_proc.stdin:
                ffplay_proc.stdin.flush()
                ffplay_proc.stdin.close()
        except Exception:
            pass
        try:
            ffplay_proc.wait(timeout=5)
        except Exception:
            ffplay_proc.terminate()

    print("---- STATS ----")
    if ttft_ms is not None:
        print(f"TTFT (ms): {ttft_ms:.1f}")
    if ttfa_ms is not None:
        print(f"TTFA (ms): {ttfa_ms:.1f}")
    print(f"Packets received: {packets}")
    print(f"Audio received (ms): {received_audio_ms:.1f} (~{received_audio_ms/1000.0:.2f}s)")
    if not args.no_save:
        print(f"Output file: {args.outfile}")
    if args.play:
        print("Playback: ffplay")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ElevenLabs-compatible WS client (no frame schedule by default)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=60215)
    p.add_argument("--voice-id", default="VOICE123")
    p.add_argument("--sr", type=int, default=24000)
    p.add_argument("--text", default="Привет! Это тест потокового TTS.")

    # вывод/воспроизведение
    p.add_argument("--outfile", default="out.wav")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--play", action="store_true", help="Play via ffplay")

    # расписания и поведение
    p.add_argument("--frames-schedule", default="", help="e.g. '50,100'. Пусто — не отправлять.")
    p.add_argument("--use-char-schedule", action="store_true", help="Trigger generation by char thresholds")
    p.add_argument("--chunk-chars", type=int, default=32)
    p.add_argument("--sleep-between-chunks-ms", type=int, default=0)

    # генерация
    p.add_argument("--sync-alignment", action="store_true")
    p.add_argument("--stream_chunk_size", type=int, default=20)
    p.add_argument("--left_context_seconds", type=float, default=1.0)
    p.add_argument("--overlap_wav_len", type=int, default=512)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--language", default=None)
    p.add_argument("--target_lead_ms", type=float, default=20.0)
    p.add_argument("--mode", default="generate", choices=["generate", "forward"])
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_client(parse_args()))
