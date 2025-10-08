"""Base abstractions for streaming TTS backends.

This module defines a minimal interface that mimics the semantics of the
ElevenLabs streaming API. Concrete wrappers are expected to translate between
model specific calls and the websocket layer exposed by :mod:`xtts_stream.service`.

Creating a new wrapper
======================

To add support for another model you only need to implement the
:class:`StreamingTTSWrapper` interface:

1. Subclass :class:`StreamingTTSWrapper` and implement :meth:`stream` and
   :meth:`encode_audio`.
2. Honour the :class:`StreamGenerationConfig` options – they mirror the query
   parameters understood by the websocket handler.
3. Expose a convenience constructor (for example ``from_env``) that instantiates
   the wrapper from configuration files, checkpoints and voice references.
4. Update :mod:`xtts_stream.service.app` so it imports and instantiates your new
   wrapper class. The websocket layer itself is model agnostic and only relies on
   the base interface.

By following those steps you can plug in any backend that is capable of
producing floating point audio frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np


@dataclass
class StreamGenerationConfig:
    """Model agnostic generation parameters for streaming synthesis."""

    stream_chunk_size: int
    overlap_wav_len: int
    left_context_seconds: Optional[int]
    speed: float
    language: Optional[str] = None


class StreamingTTSWrapper:
    """Minimal protocol required by the websocket service."""

    #: Duration of a frame in milliseconds. Concrete models may override this
    #: property. The ElevenLabs protocol expects 20ms PCM packets.
    frame_duration_ms: int = 20

    #: Output sample rate used by :meth:`encode_audio`.
    sample_rate: int

    async def stream(self, text: str, options: StreamGenerationConfig) -> AsyncIterator[np.ndarray]:
        """Yield floating point audio frames for ``text``.

        Implementations are free to use blocking model APIs – the helper can
        always off-load work to threads. The iterator must yield mono float32
        numpy arrays in the ``[-1.0, 1.0]`` range.
        """

        raise NotImplementedError

    def encode_audio(self, frame_f32: np.ndarray, output_format: str) -> bytes:
        """Encode a single frame into ``output_format``.

        The websocket layer only needs PCM support but models can implement
        additional formats (``opus``, ``ulaw`` …) if required.
        """

        raise NotImplementedError

    async def close(self) -> None:
        """Release any resources held by the wrapper."""

        return None
