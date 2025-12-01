"""Unit tests for voice pool loading and routing."""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from xtts_stream.api.wrappers.xtts import VoiceLatents, VoicePool, load_voices_from_directory


def _dummy_latents(seed: int) -> VoiceLatents:
    tensor = torch.tensor([seed], dtype=torch.float32)
    return VoiceLatents(gpt_conditioning_latents=tensor, speaker_embedding=tensor)


class VoicePoolTests(unittest.TestCase):
    def test_loads_multiple_voices_from_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            files = [base / "alpha.mp3", base / "beta.mp3"]
            for path in files:
                path.write_bytes(b"")

            counter = {"calls": 0}

            def loader(path: Path) -> VoiceLatents:
                counter["calls"] += 1
                return _dummy_latents(counter["calls"])

            pool = load_voices_from_directory(base, loader, logger_=logging.getLogger(__name__))

            self.assertEqual(2, len(pool))
            self.assertEqual(("alpha", "beta"), pool.available_ids)
            self.assertEqual("alpha", pool.default_voice_id)

    def test_resolves_requested_voice(self) -> None:
        pool = VoicePool()
        pool.add_voice("one", _dummy_latents(1))
        pool.add_voice("two", _dummy_latents(2))

        selected_id, latents = pool.resolve("two")
        self.assertEqual("two", selected_id)
        self.assertEqual(2.0, latents.gpt_conditioning_latents.item())

    def test_falls_back_to_default_when_missing(self) -> None:
        pool = VoicePool()
        pool.add_voice("primary", _dummy_latents(5))

        selected_id, latents = pool.resolve("unknown")
        self.assertEqual("primary", selected_id)
        self.assertEqual(5.0, latents.speaker_embedding.item())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
