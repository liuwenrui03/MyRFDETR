# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for LiDAR placeholder branch and shallow fusion contracts."""

from __future__ import annotations

import torch

from rfdetr.models.fusion.lidar import LiDARBranch, LiDARFusion


def test_lidar_branch_bev_input_shapes() -> None:
    """LiDARBranch should produce shallow/deep outputs for BEV-like inputs."""
    branch = LiDARBranch(temporal_mode="identity")
    x = torch.randn(2, 3, 8, 10, 12)  # [B,T,C,H,W]
    shallow, deep = branch(x)

    assert shallow is not None and deep is not None
    assert shallow.shape == (2, 8, 10, 12)
    assert deep.shape == (2, 8)


def test_lidar_branch_token_input_shapes() -> None:
    """LiDARBranch should accept token-like inputs and return compatible outputs."""
    branch = LiDARBranch(temporal_mode="identity")
    x = torch.randn(2, 4, 16, 6)  # [B,T,N,C]
    shallow, deep = branch(x)

    assert shallow is not None and deep is not None
    assert shallow.shape == (2, 6, 16, 1)
    assert deep.shape == (2, 6)


def test_lidar_branch_none_input_returns_none_tuple() -> None:
    """LiDARBranch should no-op when lidar input is missing."""
    branch = LiDARBranch(temporal_mode="identity")
    shallow, deep = branch(None)
    assert shallow is None and deep is None


def test_lidar_fusion_gate_zero_equivalence() -> None:
    """LiDARFusion with gate_init=0 should preserve selected stage features exactly."""
    fusion = LiDARFusion(stages_shallow=[0], gate_init=0.0)
    feats = [torch.randn(2, 16, 8, 8), torch.randn(2, 16, 4, 4)]
    lidar = torch.randn(2, 12, 6, 6)

    out = fusion.inject_shallow(feats, lidar)
    assert torch.equal(out[0], feats[0])
    assert torch.equal(out[1], feats[1])


def test_lidar_fusion_none_input_noop() -> None:
    """LiDARFusion should no-op when lidar shallow feature is absent."""
    fusion = LiDARFusion(stages_shallow=[0], gate_init=0.3)
    feats = [torch.randn(2, 16, 8, 8)]
    out = fusion.inject_shallow(feats, None)
    assert torch.equal(out[0], feats[0])
