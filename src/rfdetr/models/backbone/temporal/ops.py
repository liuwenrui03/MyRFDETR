# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Temporal operator implementations for backbone feature tensors.

All operators accept and return tensors of shape ``[B, T, C, H, W]``.
For strict backward compatibility, every operator is implemented as a no-op when ``T == 1``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class IdentityTemporal(nn.Module):
    """Identity temporal operator."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return input unchanged."""
        return x

    def reset_state(self) -> None:
        """Reset internal state (no-op for identity)."""


class TSMOnlineTemporal(nn.Module):
    """Temporal Shift Module with optional online state caching.

    During training, this module uses bidirectional temporal shift across the input sequence.
    During inference, it can reuse a cached previous-frame tensor to emulate online processing.
    """

    def __init__(self, shift_div: int = 8, online_inference: bool = True) -> None:
        """Initialize TSM operator.

        Args:
            shift_div: Fraction denominator for channels participating in shifts.
            online_inference: If ``True``, eval-mode forward caches previous-frame features.
        """
        super().__init__()
        self.shift_div = max(1, int(shift_div))
        self.online_inference = online_inference
        self._cached_prev: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Reset cached temporal state."""
        self._cached_prev = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal shift while preserving shape.

        Args:
            x: Input tensor with shape ``[B, T, C, H, W]``.

        Returns:
            Tensor with the same shape as input.
        """
        if x.dim() != 5:
            raise ValueError(f"TSMOnlineTemporal expects 5D input [B,T,C,H,W], got shape {tuple(x.shape)}")

        bsz, num_frames, channels, _, _ = x.shape
        if num_frames <= 1:
            return x

        fold = channels // self.shift_div
        if fold <= 0:
            return x

        out = x.clone()

        if self.training or not self.online_inference:
            # Left shift first fold, right shift second fold.
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, -1, :fold] = 0

            out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
            out[:, 0, fold : 2 * fold] = 0
            return out

        # Online eval path: previous frame from cache for backward-shift channels.
        prev = self._cached_prev
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, -1, :fold] = 0

        if prev is None or prev.shape != x[:, 0].shape:
            out[:, 0, fold : 2 * fold] = 0
        else:
            out[:, 0, fold : 2 * fold] = prev[:, fold : 2 * fold]

        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
        self._cached_prev = x[:, -1].detach()
        return out


class Conv3DTemporal(nn.Module):
    """Depthwise 3D temporal operator over per-stage feature maps."""

    def __init__(self, kernel_t: int = 3) -> None:
        """Initialize a lazy depthwise 3D conv operator.

        Args:
            kernel_t: Temporal kernel size; values <=1 behave like identity.
        """
        super().__init__()
        self.kernel_t = int(kernel_t)
        self._conv: nn.Conv3d | None = None

    def reset_state(self) -> None:
        """Reset internal state (no-op for stateless convolution)."""

    def _build_if_needed(self, channels: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._conv is not None:
            return

        kernel_t = max(1, self.kernel_t)
        self._conv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_t, 1, 1),
            padding=(kernel_t // 2, 0, 0),
            groups=channels,
            bias=False,
        ).to(device=device, dtype=dtype)

        # Identity init along temporal axis.
        with torch.no_grad():
            self._conv.weight.zero_()
            self._conv.weight[:, :, kernel_t // 2, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise temporal 3D convolution.

        Args:
            x: Input tensor with shape ``[B, T, C, H, W]``.

        Returns:
            Tensor with identical shape.
        """
        if x.dim() != 5:
            raise ValueError(f"Conv3DTemporal expects 5D input [B,T,C,H,W], got shape {tuple(x.shape)}")

        bsz, num_frames, channels, height, width = x.shape
        if num_frames <= 1 or self.kernel_t <= 1:
            return x

        self._build_if_needed(channels=channels, device=x.device, dtype=x.dtype)
        assert self._conv is not None

        x_c_first = x.permute(0, 2, 1, 3, 4)
        y = self._conv(x_c_first)
        return y.permute(0, 2, 1, 3, 4).reshape(bsz, num_frames, channels, height, width)


class TempAttnTemporal(nn.Module):
    """Lightweight temporal attention over frame axis.

    This module is parameter-free and performs multi-head temporal self-attention
    per spatial location using scaled dot-product attention.
    """

    def __init__(self, num_heads: int = 4) -> None:
        """Initialize temporal attention operator.

        Args:
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.num_heads = max(1, int(num_heads))

    def reset_state(self) -> None:
        """Reset internal state (no-op for stateless attention)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal self-attention per spatial location.

        Args:
            x: Input tensor with shape ``[B, T, C, H, W]``.

        Returns:
            Tensor with identical shape.
        """
        if x.dim() != 5:
            raise ValueError(f"TempAttnTemporal expects 5D input [B,T,C,H,W], got shape {tuple(x.shape)}")

        bsz, num_frames, channels, height, width = x.shape
        if num_frames <= 1:
            return x

        if channels % self.num_heads != 0:
            # Graceful fallback to preserve shape contracts when head partitioning is impossible.
            return x

        head_dim = channels // self.num_heads
        # [B*H*W, T, C] -> [B*H*W, num_heads, T, head_dim]
        tokens = x.permute(0, 3, 4, 1, 2).reshape(bsz * height * width, num_frames, channels)
        q = tokens.reshape(bsz * height * width, num_frames, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = q
        v = q

        attn_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(bsz * height * width, num_frames, channels)
        out = x + attn_out.reshape(bsz, height, width, num_frames, channels).permute(0, 3, 4, 1, 2)
        return out
