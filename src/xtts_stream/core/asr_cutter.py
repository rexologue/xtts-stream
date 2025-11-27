"""Streaming ASR-based cutter used to trim noisy XTTS tails.

The :class:`AsrCutter` class mirrors the lightweight ASR worker that
``Xtts.inference_stream`` builds ad-hoc, but packages it as a reusable component.
It keeps all heavy signal processing inside a dedicated thread so the main TTS
loop only enqueues raw tensors and immediately continues generation.
"""

from __future__ import annotations

import threading
from queue import Queue, Empty
from typing import Any, Optional, Tuple

import numpy as np
import torch

from .tone_utils import TONE_SR, normalize_text, prepare_for_tone

SHORT_SEQ_THRESHOLD = 50
SEQ_RECONSTRUCT_THRESHOLD = 0.8


class AsrCutter:
    """Asynchronous ASR worker for trimming trailing noise.

    Parameters
    ----------
    asr_model:
        Streaming ASR pipeline (e.g. ``StreamingCTCPipeline``) that exposes a
        ``forward(feed, state) -> tuple[list[phrase], state]`` API, where each
        phrase provides ``text`` and ``end_time`` attributes.
    target_text:
        The original text that XTTS is asked to speak. For very short inputs the
        cutter pads the text to ``SHORT_SEQ_THRESHOLD`` characters to improve
        end-of-utterance detection, replicating the inline streaming logic.
    output_sample_rate:
        Sample rate of the XTTS waveform delivered to listeners. Defaults to
        24 kHz.
    seq_reconstruct_threshold:
        Fraction of normalized characters that must be observed by ASR before
        we consider the generation finished.
    safety_tail:
        Extra seconds kept after the ASR end timestamp to avoid cutting the
        final phonemes too aggressively.
    short_seq_threshold:
        Maximum text length that should trigger ASR-based trimming. The class is
        still constructed if the text is longer, but ``should_trim`` will be
        ``False`` and ASR results will never request an early stop.
    """

    def __init__(
        self,
        asr_model: Any,
        target_text: str = "",
        output_sample_rate: int = 24000,
        seq_reconstruct_threshold: float = SEQ_RECONSTRUCT_THRESHOLD,
        safety_tail: float = 0.2,
        short_seq_threshold: int = SHORT_SEQ_THRESHOLD,
    ) -> None:
        self.asr_model = asr_model
        self.output_sample_rate = output_sample_rate
        self.seq_reconstruct_threshold = seq_reconstruct_threshold
        self.safety_tail = safety_tail
        self.short_seq_threshold = short_seq_threshold

        self._chunk_samples = int(getattr(self.asr_model, "CHUNK_SIZE", TONE_SR * 0.3))
        self._task_queue: "Queue[tuple[torch.Tensor, int, int] | None | dict[str, Any]]" = Queue()
        self._result_queue: "Queue[tuple[Optional[int], bool]]" = Queue()
        self._emitted_samples_total = 0
        self._session_id = 0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stop_event = threading.Event()
        self._thread.start()
        # Initialize the first session so the worker has a baseline config.
        self.start_session(target_text)

    def start_session(self, target_text: str) -> None:
        """Reconfigure the cutter for a new utterance without recreating the thread."""

        padded_text = target_text
        if len(padded_text) < self.short_seq_threshold:
            padded_text = padded_text + " " * (self.short_seq_threshold - len(padded_text)) + padded_text

        normalized = normalize_text(padded_text)
        target_chars = int(self.seq_reconstruct_threshold * len(normalized))
        should_trim = len(target_text) <= self.short_seq_threshold

        # reset emitted counter for relative cut calculations
        self._emitted_samples_total = 0
        self._session_id += 1

        # Flush stale results from any previous run
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Empty:
                break

        self._task_queue.put(
            {
                "type": "session",
                "target_chars": target_chars,
                "should_trim": should_trim,
                "session_id": self._session_id,
            }
        )

    def push_audio_chunk(self, chunk: torch.Tensor, emitted_samples_total: Optional[int] = None) -> None:
        """Enqueue a newly generated XTTS chunk for ASR analysis.

        Heavy conversions (tensor -> NumPy, float32 -> PCM16, resampling) are
        performed entirely inside the ASR thread. The caller can optionally
        provide the absolute number of samples that were already emitted before
        this chunk; otherwise the cutter keeps its own running counter.
        """

        if chunk is None:
            return
        chunk_samples_out = int(chunk.numel())
        if chunk_samples_out == 0:
            return

        emitted_so_far = (
            int(emitted_samples_total)
            if emitted_samples_total is not None
            else self._emitted_samples_total
        )
        if emitted_samples_total is None:
            self._emitted_samples_total += chunk_samples_out

        self._task_queue.put((chunk, chunk_samples_out, emitted_so_far, self._session_id))

    def get_result(self, block: bool = False, timeout: Optional[float] = None) -> Optional[Tuple[Optional[int], bool]]:
        """Retrieve ASR decisions without blocking the generator loop.

        Returns ``(cut_samples, should_stop)`` if a chunk has been processed or
        ``None`` if the result queue is empty. ``cut_samples`` is the number of
        samples from the last pushed chunk that should be kept; ``None`` means
        no trimming is requested. ``should_stop`` becomes ``True`` when ASR has
        detected the end of the utterance.
        """

        try:
            return self._result_queue.get(block=block, timeout=timeout)
        except Empty:
            return None

    def close(self) -> None:
        """Gracefully stop the worker thread and flush pending tasks."""

        self._stop_event.set()
        self._task_queue.put(None)
        self._thread.join()
        # Drain to unblock any waiting consumer
        self._result_queue.put((None, False))

    # ==== Internal implementation ==== #
    def _worker(self) -> None:
        buffer_int32 = np.empty(0, dtype=np.int32)
        streaming_state: Any = None
        chars_seen = 0
        reached_end = False
        target_chars = 0
        should_trim = False
        active_session = 0

        while not self._stop_event.is_set():
            task = self._task_queue.get()
            if task is None:
                break

            if isinstance(task, dict) and task.get("type") == "session":
                target_chars = int(task.get("target_chars", 0))
                should_trim = bool(task.get("should_trim", False))
                active_session = int(task.get("session_id", active_session))
                buffer_int32 = np.empty(0, dtype=np.int32)
                streaming_state = None
                chars_seen = 0
                reached_end = False
                continue

            if not isinstance(task, tuple) or len(task) != 4:
                continue

            chunk, chunk_samples_out, emitted_so_far, session_id = task

            if session_id != active_session:
                # drop outdated chunk from previous session
                continue

            pcm_chunk = prepare_for_tone(chunk, sr=self.output_sample_rate)
            buffer_int32 = np.concatenate([buffer_int32, pcm_chunk])
            cut_samples_rel: Optional[int] = None

            while buffer_int32.shape[0] >= self._chunk_samples and not reached_end:
                feed = buffer_int32[: self._chunk_samples]
                buffer_int32 = buffer_int32[self._chunk_samples :]

                new_phrases, streaming_state = self.asr_model.forward(feed, streaming_state)
                if not new_phrases:
                    continue

                for phrase in new_phrases:
                    normalized_text = normalize_text(getattr(phrase, "text", ""))
                    if not normalized_text:
                        continue
                    chars_seen += len(normalized_text)

                    if should_trim and target_chars > 0 and chars_seen >= target_chars:
                        end_abs_sec = float(getattr(phrase, "end_time", 0.0) or 0.0) + self.safety_tail
                        keep_samples_abs = int(round(end_abs_sec * self.output_sample_rate))
                        keep_samples_rel = keep_samples_abs - emitted_so_far
                        cut_samples_rel = max(0, min(keep_samples_rel, chunk_samples_out))
                        reached_end = True
                        break

            self._result_queue.put((cut_samples_rel, reached_end))

        self._result_queue.put((None, False))


__all__ = ["AsrCutter"]


def _example_usage() -> None:
    """Minimal example showing how to plug ``AsrCutter`` into streaming XTTS.

    >>> cutter = AsrCutter(asr_model, target_text="Привет!", output_sample_rate=24000)
    >>> for chunk in xtts_generator():
    ...     cutter.push_audio_chunk(chunk)
    ...     result = cutter.get_result()
    ...     if result:
    ...         cut_samples, stop = result
    ...         if cut_samples is not None:
    ...             chunk = chunk[:cut_samples]
    ...         play(chunk)
    ...         if stop:
    ...             break
    >>> cutter.close()
    """

    # This function is intentionally not executed; it only documents the API.
    return None
