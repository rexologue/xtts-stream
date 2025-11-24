"""Shared pacing helpers for streaming responses."""

from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass
from typing import Iterator, Tuple


PCM_BYTES_PER_SAMPLE = 2
PCM_CHANNELS = 1
DEFAULT_TARGET_LEAD_MS = 20.0
LEAD_HYSTERESIS_MS = 2.0
MAX_PACKET_MS = 60.0


@dataclass
class Pacer:
    """Throttle outgoing packets so that audio lead stays within bounds."""

    sample_rate: int
    target_lead_ms: float = DEFAULT_TARGET_LEAD_MS

    def __post_init__(self) -> None:
        self.bytes_per_sec = self.sample_rate * PCM_BYTES_PER_SAMPLE * PCM_CHANNELS
        self.ms_per_byte = 1000.0 / self.bytes_per_sec
        self.start_t = time.monotonic()
        self.sent_ms = 0.0

    def duration_ms_from_pcm_bytes(self, nbytes: int) -> float:
        return nbytes * self.ms_per_byte

    async def wait_before_send(self, next_frame_ms: float) -> None:
        while True:
            now_ms = (time.monotonic() - self.start_t) * 1000.0
            ahead_ms = self.sent_ms - now_ms
            if ahead_ms <= (self.target_lead_ms - LEAD_HYSTERESIS_MS):
                return
            delta_ms = max(0.0, ahead_ms - self.target_lead_ms)
            await asyncio.sleep(min(delta_ms / 1000.0, 0.050))

    def on_sent(self, frame_ms: float) -> None:
        self.sent_ms += frame_ms


def _bytes_per_ms(sr: int) -> float:
    return (sr * PCM_BYTES_PER_SAMPLE * PCM_CHANNELS) / 1000.0


def iter_time_shards(raw_bytes: bytes, sr: int, max_packet_ms: float = MAX_PACKET_MS) -> Iterator[Tuple[bytes, float]]:
    """Slice PCM16 mono bytes into packets limited by ``max_packet_ms`` duration."""

    b_per_ms = _bytes_per_ms(sr)
    max_bytes = max(1, int(max_packet_ms * b_per_ms))
    for i in range(0, len(raw_bytes), max_bytes):
        shard = raw_bytes[i : i + max_bytes]
        shard_ms = len(shard) / b_per_ms
        yield shard, shard_ms

