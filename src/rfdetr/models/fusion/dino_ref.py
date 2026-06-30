# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""DINO reference branch and feature injector (PR-3 foundation)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class DinoRefBranch(nn.Module):
    """DINO-reference branch with keyframe-aware temporal aggregation.

    Real-token path:
    - Input supports encoder feature maps ``[B, T, C, H, W]`` or patch-token
      sequences ``[B, T, N, C]``.
    - Keyframes are sampled over ``T``.
    - Tokens are projected to ``D`` and temporally aggregated.
    - Output reference tokens are ``[B, N, D]``.
    """

    def __init__(
        self,
        keyframe_stride: int = 2,
        aggregator: str = "ema",
        embedding_dim: int = 256,
        num_tokens: int = 16,
        ema_decay: float = 0.5,
    ) -> None:
        """Initialize DINO-ref branch.

        Args:
            keyframe_stride: Use every ``keyframe_stride`` frame as keyframe.
            aggregator: Temporal aggregation mode: ``mean``, ``ema``, ``attn_pool``.
            embedding_dim: Output embedding dimension.
            num_tokens: Number of reference tokens per sample.
            ema_decay: EMA decay factor used when ``aggregator='ema'``.
        """
        super().__init__()
        self.keyframe_stride = max(1, int(keyframe_stride))
        self.aggregator = aggregator
        self.embedding_dim = int(embedding_dim)
        self.num_tokens = max(1, int(num_tokens))
        self.ema_decay = float(ema_decay)

        self.proj = nn.LazyLinear(self.embedding_dim)
        self.attn_score = nn.Linear(self.embedding_dim, 1, bias=False)


    @staticmethod
    def _select_keyframes(num_frames: int, stride: int) -> list[int]:
        """Return keyframe indices for the current sequence length."""
        indices = list(range(0, max(1, num_frames), max(1, stride)))
        if (num_frames - 1) not in indices:
            indices.append(num_frames - 1)
        return sorted(set(indices))

    def _aggregate(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate frame embeddings of shape ``[B, Tk, N, C]`` to ``[B, N, C]``."""
        if frame_embeddings.shape[1] == 1:
            return frame_embeddings[:, 0]

        if self.aggregator == "mean":
            return frame_embeddings.mean(dim=1)
        if self.aggregator == "ema":
            ema = frame_embeddings[:, 0]
            for idx in range(1, frame_embeddings.shape[1]):
                ema = self.ema_decay * ema + (1.0 - self.ema_decay) * frame_embeddings[:, idx]
            return ema
        if self.aggregator == "attn_pool":
            # frame_embeddings: [B, Tk, N, C]
            scores = self.attn_score(frame_embeddings).squeeze(-1)  # [B, Tk, N]
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, Tk, N, 1]
            return (weights * frame_embeddings).sum(dim=1)

        raise ValueError(f"Unsupported dino_ref aggregator: {self.aggregator}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce token-level reference embeddings.

        Args:
            x: Encoder representation, either ``[B, T, C, H, W]`` or
                unpooled patch tokens ``[B, T, N, C]``.

        Returns:
            Tensor of shape ``[B, N, embedding_dim]``.
        """
        if x.dim() not in {4, 5}:
            raise ValueError(f"DinoRefBranch expects [B,T,C,H,W] or [B,T,N,C], got {tuple(x.shape)}")

        bsz, num_frames = x.shape[0], x.shape[1]
        keyframe_indices = self._select_keyframes(num_frames=num_frames, stride=self.keyframe_stride)

        if x.dim() == 5:
            # Convert deepest encoder feature map to unpooled patch-token sequence.
            keyframes = x[:, keyframe_indices]  # [B, Tk, C, H, W]
            token_seq = keyframes.flatten(3).permute(0, 1, 3, 2)  # [B, Tk, N, C]
        else:
            token_seq = x[:, keyframe_indices]  # [B, Tk, N, C]

        # Optional token count normalization for configurable N.
        if token_seq.shape[2] != self.num_tokens:
            desired_side = int(self.num_tokens**0.5)
            desired_side = max(1, desired_side)
            if desired_side * desired_side != self.num_tokens:
                desired_side += 1
            desired_tokens = desired_side * desired_side

            current_tokens = token_seq.shape[2]
            if current_tokens < desired_tokens:
                pad = torch.zeros(
                    token_seq.shape[0],
                    token_seq.shape[1],
                    desired_tokens - current_tokens,
                    token_seq.shape[3],
                    device=token_seq.device,
                    dtype=token_seq.dtype,
                )
                token_seq = torch.cat([token_seq, pad], dim=2)
            else:
                token_seq = token_seq[:, :, :desired_tokens, :]

        frame_embeddings = self.proj(token_seq)  # [B, Tk, N, D]
        return self._aggregate(frame_embeddings)


class DinoRefInjector(nn.Module):
    """Inject DINO-reference embeddings into backbone features.

    Supports ``fusion_mode``:
    - ``none``: no-op
    - ``cross_attn``: add gated projected reference residual
    - ``cat``: concatenate and 1x1-project back to original channels
    - ``both``: apply cross_attn then cat
    """

    def __init__(
        self,
        fusion_mode: str = "cross_attn",
        gate_init: float = 0.0,
        stages: Sequence[int] | None = None,
    ) -> None:
        """Initialize injector.

        Args:
            fusion_mode: ``none``, ``cross_attn``, ``cat``, ``both``.
            gate_init: Initial residual gate value.
            stages: Stage indices eligible for injection.
        """
        super().__init__()
        self.fusion_mode = fusion_mode
        self.stages = set(stages or [])
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.cat_proj: nn.ModuleDict[str, nn.Conv2d] = nn.ModuleDict()
        self.cross_attn: nn.ModuleDict[str, nn.MultiheadAttention] = nn.ModuleDict()

    def _ensure_cat_proj(self, stage: int, channels: int, device: torch.device, dtype: torch.dtype) -> nn.Conv2d:
        key = str(stage)
        if key not in self.cat_proj:
            conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False).to(device=device, dtype=dtype)
            with torch.no_grad():
                conv.weight.zero_()
                eye = torch.eye(channels, device=device, dtype=dtype).view(channels, channels, 1, 1)
                conv.weight[:, :channels] = eye
            self.cat_proj[key] = conv
        return self.cat_proj[key]

    def _ensure_cross_attn(
        self,
        stage: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.MultiheadAttention:
        key = str(stage)
        if key not in self.cross_attn:
            num_heads = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else 1)
            mha = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=num_heads,
                dropout=0.0,
                batch_first=True,
            ).to(device=device, dtype=dtype)
            self.cross_attn[key] = mha
        return self.cross_attn[key]

    def _inject_cross_attn(self, stage: int, feat: torch.Tensor, ref_tokens: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = feat.shape
        query = feat.flatten(2).transpose(1, 2)  # [B, HW, C]

        # Align token channel dimension to stage channels with zero-pad/truncate.
        if ref_tokens.shape[-1] < channels:
            pad = torch.zeros(
                (ref_tokens.shape[0], ref_tokens.shape[1], channels - ref_tokens.shape[-1]),
                device=ref_tokens.device,
                dtype=ref_tokens.dtype,
            )
            key_value = torch.cat([ref_tokens, pad], dim=-1)
        else:
            key_value = ref_tokens[..., :channels]

        mha = self._ensure_cross_attn(stage=stage, channels=channels, device=feat.device, dtype=feat.dtype)
        attn_out, _ = mha(query, key_value, key_value, need_weights=False)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, channels, height, width)
        return feat + torch.tanh(self.gate) * attn_out

    def _inject_cat(self, stage: int, feat: torch.Tensor, ref_tokens: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = feat.shape
        ref_vec = ref_tokens.mean(dim=1)
        if ref_vec.shape[1] < channels:
            pad = torch.zeros((bsz, channels - ref_vec.shape[1]), device=ref_vec.device, dtype=ref_vec.dtype)
            ref_vec = torch.cat([ref_vec, pad], dim=1)
        ref = ref_vec[:, :channels].view(bsz, channels, 1, 1).expand(-1, -1, height, width)
        cat_feat = torch.cat([feat, ref], dim=1)
        proj = self._ensure_cat_proj(stage=stage, channels=channels, device=feat.device, dtype=feat.dtype)
        return proj(cat_feat)

    def inject(self, raw_feats: list[torch.Tensor], ref_tokens: torch.Tensor) -> list[torch.Tensor]:
        """Inject reference tokens into selected backbone stages.

        Args:
            raw_feats: List of 2D backbone feature maps, each ``[B, C, H, W]``.
            ref_tokens: Reference tokens ``[B, N, D]``.

        Returns:
            Updated feature list.
        """
        if self.fusion_mode == "none":
            return raw_feats

        updated: list[torch.Tensor] = []
        for idx, feat in enumerate(raw_feats):
            if idx not in self.stages:
                updated.append(feat)
                continue

            out = feat
            if self.fusion_mode in {"cross_attn", "both"}:
                out = self._inject_cross_attn(stage=idx, feat=out, ref_tokens=ref_tokens)
            if self.fusion_mode in {"cat", "both"}:
                out = self._inject_cat(stage=idx, feat=out, ref_tokens=ref_tokens)
            updated.append(out)

        return updated
