# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for DINO-reference branch and injector contracts."""

from __future__ import annotations

import torch
import pytest

from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.fusion.dino_ref import DinoRefBranch, DinoRefInjector


def test_dino_ref_branch_output_shape() -> None:
    """DinoRefBranch should return [B, N, D] token embeddings for temporal image input."""
    branch = DinoRefBranch(keyframe_stride=2, aggregator="ema", embedding_dim=64, num_tokens=16)
    x = torch.randn(2, 4, 3, 16, 16)
    y = branch(x)
    assert y.shape == (2, 16, 64)


def test_dino_ref_branch_keyframe_selection_includes_last_frame() -> None:
    """Keyframe selection should always include the sequence tail frame."""
    indices = DinoRefBranch._select_keyframes(num_frames=5, stride=2)
    assert indices == [0, 2, 4]


def test_dino_ref_branch_attn_pool_output_shape() -> None:
    """attn_pool aggregator should produce the same token shape contract as other aggregators."""
    branch = DinoRefBranch(keyframe_stride=1, aggregator="attn_pool", embedding_dim=32, num_tokens=9)
    x = torch.randn(2, 3, 24, 16, 16)
    y = branch(x)
    assert y.shape == (2, 9, 32)


def test_dino_ref_branch_accepts_encoder_feature_channels() -> None:
    """DinoRefBranch should support non-RGB encoder feature channels on real-token path."""
    branch = DinoRefBranch(keyframe_stride=2, aggregator="ema", embedding_dim=48, num_tokens=16)
    x = torch.randn(2, 4, 96, 12, 12)
    y = branch(x)
    assert y.shape == (2, 16, 48)


def test_dino_ref_branch_accepts_unpooled_patch_token_sequence() -> None:
    """DinoRefBranch should accept [B,T,N,C] unpooled patch tokens from backbone internals."""
    branch = DinoRefBranch(keyframe_stride=1, aggregator="mean", embedding_dim=40, num_tokens=25)
    x = torch.randn(2, 3, 25, 96)
    y = branch(x)
    assert y.shape == (2, 25, 40)


def test_dino_ref_injector_gate_zero_cross_attn_equivalence() -> None:
    """With gate_init=0 and cross_attn mode, injection should preserve features exactly."""
    injector = DinoRefInjector(fusion_mode="cross_attn", gate_init=0.0, stages=[0])
    feats = [torch.randn(2, 32, 8, 8), torch.randn(2, 32, 4, 4)]
    ref = torch.randn(2, 8, 32)

    out = injector.inject(feats, ref)
    assert torch.equal(out[0], feats[0])
    assert torch.equal(out[1], feats[1])


def test_dino_ref_injector_cat_identity_projection_default() -> None:
    """Cat mode should default to identity-preserving 1x1 projection initialization."""
    injector = DinoRefInjector(fusion_mode="cat", gate_init=0.0, stages=[0])
    feats = [torch.randn(2, 16, 8, 8)]
    ref = torch.randn(2, 4, 16)

    out = injector.inject(feats, ref)
    assert out[0].shape == feats[0].shape


def test_backbone_select_dino_ref_tokens_deepest_source() -> None:
    """Backbone token-source policy 'deepest' should return deepest stage patch tokens."""
    backbone = object.__new__(Backbone)
    backbone.dino_ref_token_source = "deepest"
    backbone.dino_ref_token_stage_idx = -1

    stage0 = torch.randn(2, 3, 32, 8, 8)
    stage1 = torch.randn(2, 3, 32, 4, 4)
    selected = Backbone._select_dino_ref_tokens(backbone, [stage0, stage1])
    assert selected.shape == (2, 3, 16, 32)


def test_backbone_select_dino_ref_tokens_stage_idx_source() -> None:
    """Backbone token-source policy 'stage_idx' should return tokens from the configured stage."""
    backbone = object.__new__(Backbone)
    backbone.dino_ref_token_source = "stage_idx"
    backbone.dino_ref_token_stage_idx = 0

    stage0 = torch.randn(2, 3, 24, 6, 6)
    stage1 = torch.randn(2, 3, 24, 3, 3)
    selected = Backbone._select_dino_ref_tokens(backbone, [stage0, stage1])
    assert selected.shape == (2, 3, 36, 24)


def test_backbone_select_dino_ref_tokens_all_stages_source() -> None:
    """Backbone token-source policy 'all_stages' should concat stage tokens along N dimension."""
    backbone = object.__new__(Backbone)
    backbone.dino_ref_token_source = "all_stages"
    backbone.dino_ref_token_stage_idx = -1

    stage0 = torch.randn(2, 2, 16, 4, 4)
    stage1 = torch.randn(2, 2, 16, 2, 2)
    selected = Backbone._select_dino_ref_tokens(backbone, [stage0, stage1])
    assert selected.shape == (2, 2, 20, 16)


def test_backbone_select_dino_ref_tokens_stage_idx_out_of_range_raises() -> None:
    """Backbone should raise when stage_idx source points outside available temporal stages."""
    backbone = object.__new__(Backbone)
    backbone.dino_ref_token_source = "stage_idx"
    backbone.dino_ref_token_stage_idx = 5

    stage0 = torch.randn(2, 2, 16, 4, 4)
    with pytest.raises(ValueError, match="dino_ref_token_stage_idx out of range"):
        Backbone._select_dino_ref_tokens(backbone, [stage0])


def test_dino_ref_cross_attn_selected_stage_only() -> None:
    """Cross-attn mode should affect only configured stages when gate is non-zero."""
    injector = DinoRefInjector(fusion_mode="cross_attn", gate_init=1.0, stages=[1])
    feats = [torch.randn(2, 16, 8, 8), torch.randn(2, 16, 4, 4)]
    ref = torch.randn(2, 6, 16)

    out = injector.inject(feats, ref)
    assert torch.equal(out[0], feats[0])
    assert out[1].shape == feats[1].shape

