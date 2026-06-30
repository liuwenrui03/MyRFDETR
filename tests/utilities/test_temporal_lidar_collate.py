# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for temporal/lidar input protocol in tensor collation."""

from __future__ import annotations

import torch

from rfdetr.utilities.tensors import NestedTensor, make_collate_fn


def _target() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "image_id": torch.tensor([0], dtype=torch.int64),
    }


def test_collate_single_frame_compatibility() -> None:
    """Default collate must keep legacy [B,C,H,W] + [B,H,W] behavior."""
    collate = make_collate_fn(block_size=None, num_frames=1)
    batch = [
        (torch.randn(3, 12, 10), _target()),
        (torch.randn(3, 8, 6), _target()),
    ]

    samples, targets = collate(batch)
    assert isinstance(samples, NestedTensor)
    assert samples.tensors.dim() == 4
    assert samples.mask is not None and samples.mask.dim() == 3
    assert len(targets) == 2


def test_collate_temporal_expands_single_frame_input() -> None:
    """Temporal collate should broadcast [C,H,W] samples to [B,T,C,H,W]."""
    collate = make_collate_fn(block_size=None, num_frames=3)
    batch = [
        (torch.randn(3, 8, 8), _target()),
        (torch.randn(3, 8, 8), _target()),
    ]

    samples, _ = collate(batch)
    assert samples.tensors.shape[:3] == (2, 3, 3)
    assert samples.mask is not None and samples.mask.shape[:2] == (2, 3)


def test_collate_temporal_accepts_prestacked_frames() -> None:
    """Temporal collate should accept per-sample pre-stacked [T,C,H,W] tensors."""
    collate = make_collate_fn(block_size=None, num_frames=2)
    batch = [
        (torch.randn(2, 3, 8, 8), _target()),
        (torch.randn(2, 3, 8, 8), _target()),
    ]

    samples, _ = collate(batch)
    assert samples.tensors.shape == (2, 2, 3, 8, 8)


def test_collate_with_optional_lidar_payload() -> None:
    """Collate should stack optional lidar payload as NestedTensor.lidar when present for all samples."""
    collate = make_collate_fn(block_size=None, num_frames=1)
    batch = [
        (torch.randn(3, 8, 8), _target(), torch.randn(4, 6, 6)),
        (torch.randn(3, 8, 8), _target(), torch.randn(4, 6, 6)),
    ]

    samples, _ = collate(batch)
    assert samples.lidar is not None
    assert samples.lidar.shape == (2, 4, 6, 6)
