"""Module for feature extraction from raw audio waveforms."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from typing_extensions import Self


class FilterbankFeatures(torch.nn.Module):
    """Module for computing log-mel filterbank features.

    Inspired by `torchaudio.compliance.kaldi.fbank`, but with modified forward_basis calculation,
    padded batching support, streaming adaptation and suitable for ONNX/TensorRT export
    """

    LOG_ZERO_GUARD_VALUE = 2**-24
    forward_basis: Tensor
    filterbanks: Tensor

    @classmethod
    def build(cls) -> Self:
        """Returns a FilterbankFeatures instance with default parameters for Tone model."""
        return cls(sample_rate=8000, window_size=0.02, window_stride=0.01, n_fft=160, n_mels=64)

    @property
    def state_size(self) -> int:
        """Returns the state size required for streaming."""
        return self._state_size

    def __init__(
        self,
        sample_rate: int,
        window_size: float,
        window_stride: float,
        n_fft: int,
        n_mels: int,
        preemphasis_coefficient: float = 0.97,
    ) -> None:
        """Initializes the FilterbankFeatures.

        Args:
            sample_rate (int): Input sample rate.
            window_size (float): Window size in seconds.
            window_stride (float): Window stride in seconds.
            n_fft (int): Number of FFT bins.
            n_mels (int): Number of mel filterbanks.
            preemphasis_coefficient (float, optional): Pre-emphasis coefficient. Defaults to 0.97.

        """
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = int(window_size * sample_rate)
        self.hop_length = int(window_stride * sample_rate)
        self.n_fft = n_fft
        self._state_size = self.n_fft - self.hop_length

        window_tensor = torch.hann_window(self.win_length, periodic=False)
        forward_basis = self._compute_forward_basis(n_fft, window_tensor, preemphasis_coefficient)
        self.register_buffer("forward_basis", forward_basis, persistent=False)
        filterbanks = self._compute_filterbanks(n_fft, sample_rate, n_mels)
        self.register_buffer("filterbanks", filterbanks, persistent=False)

    @staticmethod
    def _compute_forward_basis(n_fft: int, window: Tensor, preemphasis_coefficient: float) -> Tensor:
        window_size = window.size(-1)
        fourier_basis = torch.fft.fft(torch.eye(n_fft, device=window.device, dtype=torch.float64), norm="backward")
        fourier_basis = fourier_basis[: n_fft // 2 + 1]
        forward_basis = torch.cat((fourier_basis.real, fourier_basis.imag), dim=0).T
        forward_basis *= window[:, None].to(forward_basis.dtype)

        if preemphasis_coefficient != 0:
            preemphasis_matrix = forward_basis.new_ones(window_size).diag(0)
            preemphasis_matrix -= forward_basis.new_full((window_size - 1,), preemphasis_coefficient).diag(1)
            preemphasis_matrix[0, 0] -= preemphasis_coefficient
            forward_basis = preemphasis_matrix @ forward_basis

        return forward_basis.T[:, None, :].float().contiguous()

    @staticmethod
    def _compute_filterbanks(n_fft: int, sample_rate: int, n_mels: int) -> Tensor:
        filterbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        return filterbanks.T.contiguous()

    def _forward(self, waveform: Tensor) -> Tensor:
        # Autocast is disabled to preserve numerical precision
        with torch.autocast(waveform.device.type, enabled=False):
            spectrum = F.conv1d(waveform[:, None].float(), self.forward_basis.float(), stride=self.hop_length)
            spectrum = spectrum.view(spectrum.size(0), 2, self.forward_basis.size(0) // 2, -1).square().sum(dim=1)
            mel_energies = self.filterbanks.to(spectrum.dtype) @ spectrum
            mel_energies = torch.log(mel_energies + self.LOG_ZERO_GUARD_VALUE)
        return mel_energies.to(waveform.dtype)

    def forward(self, waveform: Tensor, waveform_lens: Tensor) -> tuple[Tensor, Tensor]:
        """Processes a batch of padded audio sequences.

        Args:
            waveform (Tensor): Batch of audio waveforms, shape (batch, time).
            waveform_lens (Tensor): Sequence lengths before padding.

        Returns:
            Tuple[Tensor, Tensor]: (Mel log-energies, new sequence lengths).

        """
        waveform = F.pad(waveform, (self._state_size, 0, 0, 0))
        return self._forward(waveform), waveform_lens // self.hop_length

    def forward_streaming(self, waveform: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Processes audio in streaming mode, maintaining and returning state.

        Args:
            waveform (Tensor): Input audio chunk, shape (batch, chunk_length).
            state (Tensor or None, optional): State tensor for left context. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: (Mel log-energies for the chunk, next state tensor).

        """
        if state is None:
            state = waveform.new_zeros(waveform.size(0), self._state_size)
        waveform = torch.cat([state, waveform], dim=1)
        state_next = waveform[:, -self._state_size :]
        return self._forward(waveform), state_next
