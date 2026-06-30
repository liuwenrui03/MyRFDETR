# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for temporal backbone operators and T=1 compatibility contracts."""

from __future__ import annotations

import pytest
import torch

from rfdetr.models.backbone.temporal import build_temporal_op
from rfdetr.models.backbone.temporal.ops import Conv3DTemporal, IdentityTemporal, TempAttnTemporal, TSMOnlineTemporal


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(IdentityTemporal(), id="identity"),
        pytest.param(TSMOnlineTemporal(shift_div=8), id="tsm_online"),
        pytest.param(Conv3DTemporal(kernel_t=3), id="conv3d"),
        pytest.param(TempAttnTemporal(num_heads=4), id="temp_attn"),
    ],
)
def test_temporal_op_preserves_shape(op: torch.nn.Module) -> None:
    """All temporal ops must preserve [B,T,C,H,W] tensor shape."""
    x = torch.randn(2, 4, 16, 8, 8)
    y = op(x)
    assert y.shape == x.shape


@pytest.mark.parametrize(
    "mode,expected_type",
    [
        pytest.param("identity", IdentityTemporal, id="identity"),
        pytest.param("tsm_online", TSMOnlineTemporal, id="tsm_online"),
        pytest.param("conv3d", Conv3DTemporal, id="conv3d"),
        pytest.param("temp_attn", TempAttnTemporal, id="temp_attn"),
    ],
)
def test_build_temporal_op_returns_expected_impl(mode: str, expected_type: type[torch.nn.Module]) -> None:
    """Temporal op builder should map each mode string to the correct module class."""
    module = build_temporal_op(mode)
    assert isinstance(module, expected_type)


def test_temporal_op_t1_is_noop() -> None:
    """Temporal ops must be no-op when T=1 for strict backward compatibility."""
    x = torch.randn(2, 1, 16, 8, 8)
    for op in [TSMOnlineTemporal(), Conv3DTemporal(kernel_t=3), TempAttnTemporal()]:
        y = op(x)
        assert torch.equal(y, x)


def test_tsm_online_reset_state() -> None:
    """TSM online cache reset should clear previous-frame state without changing output shape."""
    op = TSMOnlineTemporal(online_inference=True)
    op.eval()
    x = torch.randn(2, 3, 16, 8, 8)
    y1 = op(x)
    assert y1.shape == x.shape
    op.reset_state()
    y2 = op(x)
    assert y2.shape == x.shape


def test_conv3d_identity_init_center_weight() -> None:
    """Conv3DTemporal should initialize temporal kernel as identity at center index."""
    op = Conv3DTemporal(kernel_t=3)
    x = torch.randn(1, 4, 8, 4, 4)
    _ = op(x)

    assert op._conv is not None
    weight = op._conv.weight.detach()
    center_idx = 1
    assert torch.allclose(weight[:, :, center_idx, 0, 0], torch.ones_like(weight[:, :, center_idx, 0, 0]))
    assert torch.allclose(weight[:, :, 0, 0, 0], torch.zeros_like(weight[:, :, 0, 0, 0]))
    assert torch.allclose(weight[:, :, 2, 0, 0], torch.zeros_like(weight[:, :, 2, 0, 0]))


def test_temp_attn_multi_head_partition() -> None:
    """TempAttnTemporal should preserve shape when channels are divisible by num_heads."""
    op = TempAttnTemporal(num_heads=4)
    x = torch.randn(2, 3, 16, 6, 6)
    y = op(x)
    assert y.shape == x.shape


def test_temp_attn_non_divisible_channels_fallback_noop() -> None:
    """TempAttnTemporal should gracefully fallback when channels are not divisible by num_heads."""
    op = TempAttnTemporal(num_heads=4)
    x = torch.randn(2, 3, 10, 6, 6)
    y = op(x)
    assert torch.equal(y, x)


def test_build_temporal_op_rejects_unknown_mode() -> None:
    """Builder should raise a clear error for unsupported temporal op mode."""
    with pytest.raises(ValueError, match="Unsupported temporal mode"):
        _ = build_temporal_op("unknown_mode")
