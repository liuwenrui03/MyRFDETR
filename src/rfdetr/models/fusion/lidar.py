# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LiDAR placeholder branch and shallow fusion for PR-4."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rfdetr.models.backbone.temporal import build_temporal_op


class LiDARBranch(nn.Module):
    """LiDAR feature branch placeholder with temporal-op compatibility.

    This branch provides a stable interface while the concrete LiDAR encoder is
    not yet integrated. It accepts either:
    - BEV-like tensor: ``[B, T, C, H, W]``
    - Token-like tensor: ``[B, T, N, C]``

    and returns two multi-scale placeholders:
    - shallow: ``[B, C, H, W]``
    - deep: ``[B, C]``
    """

    def __init__(self, temporal_mode: str = "identity", temporal_op_kwargs: dict | None = None) -> None:
        super().__init__()
        self.temporal_op = build_temporal_op(mode=temporal_mode, op_kwargs=temporal_op_kwargs or {})

    def forward(self, lidar_input: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run placeholder LiDAR branch.

        Args:
            lidar_input: Optional LiDAR tensor.

        Returns:
            Tuple ``(shallow, deep)`` where each element is ``None`` when input is missing.
        """
        if lidar_input is None:
            return None, None

        if lidar_input.dim() == 5:
            # BEV-like [B,T,C,H,W]
            temporal = self.temporal_op(lidar_input)
            shallow = temporal[:, -1]
            deep = temporal.mean(dim=(1, 3, 4))
            return shallow, deep

        if lidar_input.dim() == 4:
            # Token-like [B,T,N,C] -> [B,T,C,N,1] for temporal op compatibility.
            bsz, num_frames, num_tokens, channels = lidar_input.shape
            bev_proxy = lidar_input.permute(0, 1, 3, 2).reshape(bsz, num_frames, channels, num_tokens, 1)
            temporal = self.temporal_op(bev_proxy)
            shallow = temporal[:, -1]
            deep = temporal.mean(dim=(1, 3, 4))
            return shallow, deep

        raise ValueError(
            f"Unsupported lidar_input shape: {tuple(lidar_input.shape)}. Expected [B,T,C,H,W] or [B,T,N,C]."
        )


class LiDARFusion(nn.Module):
    """Shallow LiDAR-to-image fusion with stage-wise gated residual injection."""

    def __init__(self, stages_shallow: Sequence[int] | None = None, gate_init: float = 0.0) -> None:
        super().__init__()
        self.stages_shallow = set(stages_shallow or [])
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.proj: nn.ModuleDict[str, nn.Conv2d] = nn.ModuleDict()

    def _ensure_proj(self, stage: int, in_channels: int, out_channels: int, device, dtype) -> nn.Conv2d:
        key = str(stage)
        if key not in self.proj:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False).to(device=device, dtype=dtype)
            with torch.no_grad():
                conv.weight.zero_()
            self.proj[key] = conv
        return self.proj[key]

    def inject_shallow(
        self,
        raw_feats: list[torch.Tensor],
        lidar_shallow: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Inject LiDAR shallow feature into selected image stages.

        Args:
            raw_feats: Image backbone features (2D), each ``[B, C, H, W]``.
            lidar_shallow: LiDAR shallow feature ``[B, C_l, H_l, W_l]``.

        Returns:
            Updated feature list.
        """
        if lidar_shallow is None:
            return raw_feats

        updated: list[torch.Tensor] = []
        gate = torch.tanh(self.gate)

        for idx, feat in enumerate(raw_feats):
            if idx not in self.stages_shallow:
                updated.append(feat)
                continue

            lidar_resized = torch.nn.functional.interpolate(
                lidar_shallow,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            proj = self._ensure_proj(
                stage=idx,
                in_channels=lidar_resized.shape[1],
                out_channels=feat.shape[1],
                device=feat.device,
                dtype=feat.dtype,
            )
            fused = feat + gate * proj(lidar_resized)
            updated.append(fused)

        return updated
