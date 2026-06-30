# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Temporal operator factory for backbone feature sequences."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from rfdetr.models.backbone.temporal.ops import Conv3DTemporal, IdentityTemporal, TempAttnTemporal, TSMOnlineTemporal


def build_temporal_op(mode: str = "identity", op_kwargs: Mapping[str, Any] | None = None) -> nn.Module:
    """Build a temporal operator module.

    Args:
        mode: Temporal op mode. Supported values: ``identity``, ``tsm_online``, ``conv3d``, ``temp_attn``.
        op_kwargs: Optional keyword arguments forwarded to the selected temporal op constructor.

    Returns:
        Temporal operator module with ``forward(x: [B,T,C,H,W]) -> [B,T,C,H,W]``.

    Raises:
        ValueError: If ``mode`` is unsupported.
    """
    kwargs = dict(op_kwargs or {})
    mode_norm = mode.lower()

    if mode_norm == "identity":
        return IdentityTemporal()
    if mode_norm == "tsm_online":
        return TSMOnlineTemporal(**kwargs)
    if mode_norm == "conv3d":
        return Conv3DTemporal(**kwargs)
    if mode_norm == "temp_attn":
        return TempAttnTemporal(**kwargs)

    raise ValueError(
        f"Unsupported temporal mode: {mode}. Expected one of: identity, tsm_online, conv3d, temp_attn."
    )
