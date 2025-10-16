"""Modules for building the T-one model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from tone.nn.modules.conformer_blocks import EncoderState


class RMSNorm(nn.Module):
    """A Root Mean Square Layer Normalization module.

    This layer normalizes the input by its root mean square, scaling it by a
    learnable weight parameter. It is a simplified alternative to `nn.LayerNorm`.

    Args:
        d (int): The feature dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for
            numerical stability. Defaults to 1e-8.

    """

    def __init__(self, d: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.d = d
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass for RMSNorm.

        Args:
            x (torch.Tensor): The input tensor of shape (..., d).

        Returns:
            torch.Tensor: The normalized and scaled tensor, with the same dtype
                as the input.

        """
        dtype = x.dtype
        # Cast to float32 for stability in norm calculation
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float()
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
            rms_x = norm_x * d_x ** (-1.0 / 2)
            x_normed = x / (rms_x + self.eps)

            return (self.weight * x_normed).to(dtype)


class RotaryPositionalEmbeddings(nn.Module):
    """Applies Rotary Positional Embeddings (RoPE) to a subset of features.

    Rotary encoding transforms pairs of features by rotating them in a 2D plane
    by an angle that depends on the token's position. This injects relative
    positional information into the model.

    Args:
        d (int): The feature dimension to which RoPE is applied. Must be even.
        base (int, optional): The base frequency used to compute rotation angles.
            Defaults to 10_000.

    """

    def __init__(self, d: int, base: int = 10_000) -> None:
        super().__init__()
        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Applies RoPE to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, T, D), where B is
                batch size, H is heads, T is sequence length, and D is model dim.
            offset (int, optional): The starting position offset, used for
                recurrent state management (e.g., KV caching). Defaults to 0.

        Returns:
            torch.Tensor: The output tensor with RoPE applied, of the same shape
                as the input.

        """
        # (B, H, T, D) -> (T, B, H, D)
        x = x.permute(2, 0, 1, 3)

        # Split features: apply RoPE only to the first `d` features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        x_rope = self._apply_rotary_pos_emb(x_rope, offset)
        x = torch.cat((x_rope, x_pass), dim=-1)

        # (T, B, H, D) -> (B, H, T, D)
        return x.permute(1, 2, 0, 3)

    def _apply_rotary_pos_emb(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        """Applies the rotation using pre-computed sin/cos tables.

        Args:
            x (torch.Tensor): The input tensor slice to rotate.
            offset (int): The starting position offset.

        Returns:
            torch.Tensor: The rotated tensor.

        """
        self._build_state(x, offset)
        cos = self.cos_cached[: x.shape[0]]
        sin = self.sin_cached[: x.shape[0]]
        return (x * cos) + (self._rotate_half(x) * sin)

    def _build_state(self, x: torch.Tensor, offset: int = 0) -> None:
        """Builds and caches the sine and cosine rotation matrices.

        Args:
            x (torch.Tensor): A tensor to determine the device and sequence length.
            offset (int): The starting position offset for calculation.

        """
        # Return if cache is already built and sufficient
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        # Compute the inverse frequencies according to the original RoPE implementation
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.d, 2, device=x.device, dtype=torch.float32) / self.d))

        # Calculate positional_ids with offset to handle key value state
        position_ids = torch.arange(-offset, x.shape[0] - offset, device=x.device, dtype=torch.float32)
        freqs = torch.einsum("n,d->nd", position_ids, inv_freq)
        embeds = torch.cat([freqs, freqs], dim=1)

        self.cos_cached = embeds.cos()[:, None, None, :]
        self.sin_cached = embeds.sin()[:, None, None, :]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates the last dimension of the input tensor by 90 degrees.

        This is achieved by splitting the last dimension in half, negating the
        second half, and swapping their positions.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The rotated tensor.

        """
        x1 = x[..., : self.d // 2]
        x2 = x[..., self.d // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttention(nn.Module):
    """A Multi-Head Attention (MHA) layer.

    We assume d_v always equals d_k.

    Args:
        n_head (int): The number of attention heads.
        n_feat (int): The total feature dimension, must be divisible by `n_head`.
        dropout_rate (float): The dropout probability.
        recompute_scores (bool, optional): If True, recomputes query and key
            projections. If False, assumes scores are provided externally.
            Defaults to True.

    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        recompute_scores: bool = True,
    ) -> None:
        super().__init__()
        assert n_feat % n_head == 0, "n_feat must be divisible by n_head"
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.recompute_scores = recompute_scores

        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

        if self.recompute_scores:
            self.linear_q = nn.Linear(n_feat, n_feat)
            self.linear_k = nn.Linear(n_feat, n_feat)
            self.q_ln = nn.LayerNorm(self.d_k)
            self.k_ln = nn.LayerNorm(self.d_k)
        else:
            self.linear_q = None
            self.linear_k = None
            self.q_ln = None
            self.k_ln = None

    def forward_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Transforms query, key, and value tensors for attention.

        Args:
            query (torch.Tensor): Query tensor of shape (B, T_q, D).
            key (torch.Tensor): Key tensor of shape (B, T_kv, D).
            value (torch.Tensor): Value tensor of shape (B, T_kv, D).

        Returns:
            A tuple containing:
            - q (torch.Tensor | None): Transformed query (B, H, T_q, d_k).
            - k (torch.Tensor | None): Transformed key (B, H, T_kv, d_k).
            - v (torch.Tensor): Transformed value (B, H, T_kv, d_k).

        """
        n_batch = query.size(0)
        q: torch.Tensor | None = None
        k: torch.Tensor | None = None

        if self.recompute_scores:
            q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
            k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
            q = self.q_ln(q)
            k = self.k_ln(k)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Computes the attention-weighted context vector.

        Args:
            value (torch.Tensor): Value tensor of shape (B, T_kv, d_k).
            scores (torch.Tensor): Attention scores of shape (B, T_q, T_kv).
            mask (torch.Tensor | None): Boolean attention mask of shape (B, T_q, T_kv).

        Returns:
            torch.Tensor: The output context vector of shape (B, T_q, D).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -10000)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)

    def update_state(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query: torch.Tensor,
        state: EncoderState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Updates the Key-Value state for recurrent inference.

        This method implements a sliding window attention cache.

        Args:
            key (torch.Tensor): The new key tensor for the current step.
            value (torch.Tensor): The new value tensor for the current step.
            query (torch.Tensor): The new query tensor for the current step.
            state (EncoderState): A state object, expected to have an `mhsa` attribute
                which holds the previous K-V cache.

        Returns:
            A tuple containing the updated key, value, and query tensors.

        """
        if getattr(state, "mhsa", None) is not None:
            key = value = torch.cat([state.mhsa, key], dim=1)
            q_keep_size = query.shape[1]
            # Update state with a sliding window
            state.mhsa = torch.cat(
                [state.mhsa[:, q_keep_size:, :], query[:, :q_keep_size, :]],
                dim=1,
            )
        return key, value, query


class CausalConv1D(nn.Module):
    """A causal 1D convolution layer.

    This module wraps `nn.Conv1d` and applies appropriate left padding to ensure
    that the output at time `t` only depends on inputs up to time `t`,
    enforcing causality. It also supports stateful, streaming inference.

    All arguments are the same as `nn.Conv1d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # Padding is handled manually for causality
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_state(
        self,
        x: torch.Tensor,
        state: EncoderState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepares input and updates state for streaming inference.

        Args:
            x (torch.Tensor): The input for the current step.
            state (EncoderState | None, optional): The convolutional state from
                the previous step. Defaults to None.

        Returns:
            A tuple containing:
            - The prepared input tensor (padded or concatenated with state).
            - The next state tensor for the following step.

        """
        if state is None:
            # First step: apply full left padding
            padded_x = nn.functional.pad(x, pad=(self.left_padding, 0))
            next_state = None
        else:
            # Subsequent steps: use state as a prefix
            padded_x = torch.cat([state, x], dim=-1)
            # The next state is the tail of the input, to be used as a prefix
            next_state = padded_x[:, :, -state.size(dim=-1) :]

        return padded_x, next_state

    def forward(
        self,
        x: torch.Tensor,
        state: EncoderState | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, EncoderState]:
        """Performs a forward pass.

        Can operate in two modes:
        1. Non-streaming (`state=None`): Applies causal padding and returns output.
        2. Streaming (`state` is provided): Uses and updates the state, returning
           both the output and the next state.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
            state (EncoderState | None, optional): The convolutional state from
                the previous step. Defaults to None.

        Returns:
            If `state` is None, returns the output tensor.
            If `state` is provided, returns a tuple (output_tensor, next_state).

        """
        padded_x, next_state = self.update_state(x, state=state)
        output = self.conv(padded_x)
        if state is None:
            return output
        return output, next_state
