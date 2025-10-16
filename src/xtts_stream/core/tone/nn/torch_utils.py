"""Module containing utility functions for PyTorch's data types and execution contexts."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def avoid_float16_autocast_context() -> torch.cuda.amp.autocast_mode.autocast | nullcontext:
    """If the current autocast context is float16, cast it to bfloat16 if available or float32.

    This utility is designed to wrap code blocks containing operations that are
    numerically unstable in `torch.float16` (e.g., softmax, layer normalization,
    or certain loss calculations).

    Usage:
        with avoid_float16_autocast_context():
            # Operations inside this block will run in bf16/fp32 if the
            # outer context was fp16.
            stable_output = torch.nn.functional.softmax(x, dim=-1)

    Returns:
        A `torch.cuda.amp.autocast` or `contextlib.nullcontext` instance.

    """
    if torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.float16:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return torch.cuda.amp.autocast(dtype=torch.float32)

        if torch.cuda.is_bf16_supported():
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        return torch.cuda.amp.autocast(dtype=torch.float32)
    return nullcontext()


def cast_tensor(
    x: torch.Tensor,
    from_dtype: torch.dtype = torch.float16,
    to_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Conditionally casts a tensor to a new dtype.

    This function checks if the input tensor `x` has the specified `from_dtype`.
    If it does, the tensor is cast to `to_dtype`. Otherwise, the original
    tensor is returned unmodified.

    Args:
        x (torch.Tensor): The input tensor to potentially cast.
        from_dtype (torch.dtype, optional): The data type to check for.
            Defaults to `torch.float16`.
        to_dtype (torch.dtype, optional): The target data type to cast to.
            Defaults to `torch.float32`.

    Returns:
        torch.Tensor: The casted tensor, or the original tensor if no cast
            was performed.

    """
    return x.to(dtype=to_dtype) if x.dtype == from_dtype else x


def cast_all(
    x: torch.Tensor | dict | tuple,
    from_dtype: torch.dtype = torch.float16,
    to_dtype: torch.dtype = torch.float32,
) -> torch.Tensor | dict | tuple:
    """Recursively finds and casts all tensors within a nested structure.

    This function traverses nested dictionaries and tuples to find every
    `torch.Tensor`. For each tensor found, it applies the conditional casting
    logic from `cast_tensor`, changing its dtype from `from_dtype` to `to_dtype`.

    Args:
        x (torch.Tensor | dict | tuple): The input, which can be a single
            tensor or a nested structure of dictionaries and tuples containing
            tensors.
        from_dtype (torch.dtype, optional): The data type to check for in tensors.
            Defaults to `torch.float16`.
        to_dtype (torch.dtype, optional): The target data type to cast tensors to.
            Defaults to `torch.float32`.

    Returns:
        torch.Tensor | dict | tuple: A new object with the same structure as the
            input, but with all contained tensors conditionally casted.

    """
    if isinstance(x, torch.Tensor):
        return cast_tensor(x, from_dtype=from_dtype, to_dtype=to_dtype)
    if isinstance(x, dict):
        new_dict = {}
        for k in x:
            new_dict[k] = cast_all(x[k], from_dtype=from_dtype, to_dtype=to_dtype)
        return new_dict
    return tuple(cast_all(y, from_dtype=from_dtype, to_dtype=to_dtype) for y in x)
