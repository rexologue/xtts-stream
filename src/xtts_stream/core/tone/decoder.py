"""Module with different decoder implementations for the ASR pipeline."""

from __future__ import annotations

import logging
from enum import Enum
from itertools import groupby
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from pyctcdecode.decoder import BeamSearchDecoderCTC as _BeamSearchDecoderCTC
from pyctcdecode.decoder import build_ctcdecoder
from typing_extensions import Self

logging.getLogger("pyctcdecode").setLevel(logging.ERROR)


LABELS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "


class DecoderType(Enum):
    """Enumeration of supported decoding strategies for CTC output."""

    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"


class GreedyCTCDecoder:
    """Implements simple greedy decoding for CTC outputs.

    Stateless decoder that selects the most probable symbol at each frame,
    collapses repeats, and removes blank tokens. It does not support batching.
    """

    def forward(self, logprobs: npt.NDArray[np.float32]) -> str:
        """Decode log-probabilities using a greedy algorithm.

        Args:
            logprobs (npt.NDArray[np.float32]): Log-probabilities for each time frame.

        Returns:
            str: The decoded text transcription as a string.

        """
        if not isinstance(logprobs, np.ndarray):
            raise TypeError(f"Incorrect 'logprobs' type: expected np.ndarray, but got {type(logprobs)}")
        if logprobs.shape[1:] != (35,):
            raise ValueError(f"Shape of 'logprobs' must be (L, 35), but got {logprobs.shape}")
        if logprobs.dtype != np.float32:
            raise ValueError(f"Incorrect dtype of 'logprobs': expected np.float32, but got {logprobs.dtype}")

        tokens: list[int] = logprobs.argmax(axis=-1).tolist()  # gready select tokens
        tokens = [token for token, _ in groupby(tokens)]  # remove repetitions
        return "".join([LABELS[token] for token in tokens if token < len(LABELS)]).strip()


class BeamSearchCTCDecoder:
    """Beam search decoder for CTC outputs.

    Uses a provided beam search decoder (optionally with a language model).
    Stateless. Batching is not supported; accepts any input length.
    """

    _decoder: _BeamSearchDecoderCTC

    @classmethod
    def from_hugging_face(cls) -> Self:
        """Load and initialize the decoder model from Hugging Face Hub.

        Downloads the model if not present locally

        Returns:
            Self: An instance of BeamSearchCTCDecoder ready for inference.

        """
        model_path = cls.download_from_hugging_face()
        return cls.from_local(model_path)

    @classmethod
    def download_from_hugging_face(cls) -> str:
        """Download the decoder model from Hugging Face Hub.

        Returns:
            str: Path to the downloaded decoder model file.

        """
        return hf_hub_download(
            "t-tech/T-one",
            "kenlm.bin",
        )

    @classmethod
    def from_local(cls, model_path: str | Path) -> Self:
        """Initialize the decoder from a local binary file.

        Args:
            model_path (str | Path): Path to the binary model file.

        Returns:
            Self: An instance of BeamSearchCTCDecoder ready for inference.

        """
        decoder = build_ctcdecoder(labels=list(LABELS), kenlm_model_path=str(model_path), alpha=0.4, beta=0.9)
        return cls(decoder)

    def __init__(self, decoder: _BeamSearchDecoderCTC) -> None:
        """Create instance of BeamSearchCTCDecoder using internal decoder."""
        self._decoder = decoder

    def forward(self, logprobs: npt.NDArray[np.float32]) -> str:
        """Decode log-probabilities using beam search decoding.

        Applies beam search (optionally with language model scoring) to produce more accurate transcriptions.

        Args:
            logprobs (npt.NDArray[np.float32]): Log-probabilities for each time frame.

        Returns:
            str: Decoded transcription as a string.

        """
        if not isinstance(logprobs, np.ndarray):
            raise TypeError(f"Incorrect 'logprobs' type: expected np.ndarray, but got {type(logprobs)}")
        if logprobs.shape[1:] != (35,):
            raise ValueError(f"Shape of 'logprobs' must be (L, 35), but got {logprobs.shape}")
        if logprobs.dtype != np.float32:
            raise ValueError(f"Incorrect dtype of 'logprobs': expected np.float32, but got {logprobs.dtype}")
        return self._decoder.decode(logprobs, beam_width=200)  # type: ignore[arg-type]
