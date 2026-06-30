# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Backbone modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov3_convnext import DinoV3ConvNext
from rfdetr.models.backbone.dinov3_vit import DinoV3ViT
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone.temporal import build_temporal_op
from rfdetr.models.fusion import DinoRefBranch, DinoRefInjector, LiDARBranch, LiDARFusion
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import NestedTensor

logger = get_logger()

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """RF-DETR backbone wrapper with projector heads."""

    def __init__(
        self,
        name: str,
        pretrained_encoder: str = None,
        window_block_indexes: list = None,
        drop_path=0.0,
        out_channels=256,
        out_feature_indexes: list = None,
        projector_scale: list = None,
        use_cls_token: bool = False,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple[int, int] = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,
        positional_encoding_size: int = 0,
        dual_projector: bool = False,
        temporal_mode: str = "identity",
        temporal_op_kwargs: dict | None = None,
        temporal_aggregator: str = "last",
        dino_ref_enable: bool = False,
        dino_ref_keyframe_stride: int = 2,
        dino_ref_aggregator: str = "ema",
        dino_ref_fusion: str = "cross_attn",
        dino_ref_token_source: str = "deepest",
        dino_ref_token_stage_idx: int = -1,
        dino_ref_stages: list[int] | None = None,
        dino_ref_gate_init: float = 0.0,
        lidar_enable: bool = False,
        lidar_temporal_mode: str = "identity",
        lidar_temporal_op_kwargs: dict | None = None,
        lidar_fusion_shallow_stages: list[int] | None = None,
        lidar_gate_init: float = 0.0,
    ):
        super().__init__()
        self.encoder_name = name
        self.temporal_mode = temporal_mode
        self.temporal_op_kwargs = dict(temporal_op_kwargs or {})
        self.temporal_aggregator = temporal_aggregator
        self.dino_ref_enable = dino_ref_enable
        self.dino_ref_token_source = dino_ref_token_source
        self.dino_ref_token_stage_idx = dino_ref_token_stage_idx
        self.lidar_enable = lidar_enable

        if name.startswith("dinov2"):
            self.encoder = self._build_dinov2_encoder(
                name=name,
                out_feature_indexes=out_feature_indexes,
                target_shape=target_shape,
                gradient_checkpointing=gradient_checkpointing,
                load_dinov2_weights=load_dinov2_weights,
                patch_size=patch_size,
                num_windows=num_windows,
                positional_encoding_size=positional_encoding_size,
                drop_path=drop_path,
            )
        elif name.startswith("dinov3_vit"):
            size = name.split("_")[-1]
            self.encoder = DinoV3ViT(
                size=size,
                out_feature_indexes=out_feature_indexes,
                shape=target_shape,
                gradient_checkpointing=gradient_checkpointing,
                load_dinov2_weights=load_dinov2_weights,
                patch_size=patch_size,
                positional_encoding_size=positional_encoding_size,
                drop_path_rate=drop_path,
            )
        elif name.startswith("dinov3_convnext"):
            size = name.split("_")[-1]
            self.encoder = DinoV3ConvNext(
                size=size,
                out_feature_indexes=out_feature_indexes,
                shape=target_shape,
                gradient_checkpointing=gradient_checkpointing,
                load_dinov2_weights=load_dinov2_weights,
                drop_path_rate=drop_path,
            )
        else:
            raise ValueError(
                f"Unsupported backbone encoder: {name}. "
                "Expected prefixes: dinov2, dinov3_vit, dinov3_convnext."
            )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        assert sorted(self.projector_scale) == self.projector_scale, (
            "only support projector scale P3/P4/P5/P6 in ascending order."
        )
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
        self.cross_attn_projector = (
            MultiScaleProjector(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                scale_factors=scale_factors,
                layer_norm=layer_norm,
                rms_norm=rms_norm,
            )
            if dual_projector
            else None
        )

        num_temporal_stages = len(self.encoder._out_feature_channels)
        self.temporal_ops = nn.ModuleList(
            [build_temporal_op(mode=self.temporal_mode, op_kwargs=self.temporal_op_kwargs) for _ in range(num_temporal_stages)]
        )

        self.dino_ref_branch = (
            DinoRefBranch(
                keyframe_stride=dino_ref_keyframe_stride,
                aggregator=dino_ref_aggregator,
                embedding_dim=out_channels,
            )
            if self.dino_ref_enable
            else None
        )
        self.dino_ref_injector = (
            DinoRefInjector(
                fusion_mode=dino_ref_fusion,
                gate_init=dino_ref_gate_init,
                stages=dino_ref_stages if dino_ref_stages is not None else [],
            )
            if self.dino_ref_enable
            else None
        )
        self.lidar_branch = (
            LiDARBranch(
                temporal_mode=lidar_temporal_mode,
                temporal_op_kwargs=lidar_temporal_op_kwargs,
            )
            if self.lidar_enable
            else None
        )
        self.lidar_fusion = (
            LiDARFusion(
                stages_shallow=lidar_fusion_shallow_stages if lidar_fusion_shallow_stages is not None else [],
                gate_init=lidar_gate_init,
            )
            if self.lidar_enable
            else None
        )

        self._export = False

    @staticmethod
    def _build_dinov2_encoder(
        name: str,
        out_feature_indexes: list,
        target_shape: tuple[int, int],
        gradient_checkpointing: bool,
        load_dinov2_weights: bool,
        patch_size: int,
        num_windows: int,
        positional_encoding_size: int,
        drop_path: float,
    ) -> DinoV2:
        name_parts = name.split("_")
        if name_parts[0] != "dinov2":
            raise ValueError(f"Invalid DINOv2 encoder name: {name}")

        use_registers = False
        if "registers" in name_parts:
            use_registers = True
            name_parts.remove("registers")

        use_windowed_attn = False
        if "windowed" in name_parts:
            use_windowed_attn = True
            name_parts.remove("windowed")

        if len(name_parts) != 2:
            raise ValueError("DINOv2 encoder name should be in format: dinov2_[registers]_[windowed]_<size>.")

        return DinoV2(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=use_registers,
            use_windowed_attn=use_windowed_attn,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            drop_path_rate=drop_path,
        )

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if not hasattr(self.encoder, "merge_and_unload"):
            return

        try:
            from peft import PeftModel
        except ModuleNotFoundError:
            logger.warning("peft is not installed; skipping LoRA weight merging during export.")
            return
        except ImportError as exc:
            logger.warning("Failed to import PeftModel from peft during export: %s", exc)
            raise

        if isinstance(self.encoder, PeftModel):
            logger.info("Merging and unloading LoRA weights")
            self.encoder = self.encoder.merge_and_unload()

    def _aggregate_temporal_feature(self, feat: torch.Tensor) -> torch.Tensor:
        """Aggregate temporal feature tensor ``[B, T, C, H, W]`` to ``[B, C, H, W]``."""
        if feat.dim() != 5:
            raise ValueError(f"Expected temporal feature with 5 dims [B,T,C,H,W], got {tuple(feat.shape)}")

        if feat.shape[1] <= 1:
            return feat[:, 0]

        if self.temporal_aggregator == "last":
            return feat[:, -1]
        if self.temporal_aggregator == "mean":
            return feat.mean(dim=1)
        if self.temporal_aggregator == "attn_pool":
            # PR-1 fallback: keep deterministic behavior until a learned pooling module is introduced.
            return feat[:, -1]

        raise ValueError(
            f"Unsupported temporal_aggregator: {self.temporal_aggregator}. Expected one of: last, mean, attn_pool."
        )

    def _aggregate_temporal_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Aggregate mask ``[B, T, H, W]`` to ``[B, H, W]`` using temporal policy."""
        if mask.dim() != 4:
            raise ValueError(f"Expected temporal mask with 4 dims [B,T,H,W], got {tuple(mask.shape)}")

        if mask.shape[1] <= 1:
            return mask[:, 0]

        if self.temporal_aggregator in {"last", "attn_pool"}:
            return mask[:, -1]
        if self.temporal_aggregator == "mean":
            return mask.float().mean(dim=1).ge(0.5)

        raise ValueError(
            f"Unsupported temporal_aggregator: {self.temporal_aggregator}. Expected one of: last, mean, attn_pool."
        )

    def _select_dino_ref_tokens(self, raw_feats_temporal: list[torch.Tensor]) -> torch.Tensor:
        """Select DINO-ref token source and return tokens as ``[B, T, N, C]``.

        Supported policies:
            - ``deepest``: use the deepest stage only.
            - ``stage_idx``: use a specific stage index (supports negative indexing).
            - ``all_stages``: concatenate tokens from all stages along token dimension.
        """
        if not raw_feats_temporal:
            raise ValueError("raw_feats_temporal must contain at least one stage.")

        def _to_tokens(feat: torch.Tensor) -> torch.Tensor:
            if feat.dim() != 5:
                raise ValueError(
                    f"Expected temporal feature with 5 dims [B,T,C,H,W], got {tuple(feat.shape)}"
                )
            return feat.flatten(3).permute(0, 1, 3, 2)

        if self.dino_ref_token_source == "deepest":
            return _to_tokens(raw_feats_temporal[-1])

        if self.dino_ref_token_source == "stage_idx":
            num_stages = len(raw_feats_temporal)
            stage_idx = self.dino_ref_token_stage_idx
            if stage_idx < 0:
                stage_idx += num_stages
            if stage_idx < 0 or stage_idx >= num_stages:
                raise ValueError(
                    f"dino_ref_token_stage_idx out of range: {self.dino_ref_token_stage_idx} for {num_stages} stages."
                )
            return _to_tokens(raw_feats_temporal[stage_idx])

        if self.dino_ref_token_source == "all_stages":
            return torch.cat([_to_tokens(feat) for feat in raw_feats_temporal], dim=2)

        raise ValueError(
            "Unsupported dino_ref_token_source: "
            f"{self.dino_ref_token_source}. Expected one of: deepest, stage_idx, all_stages."
        )

    def forward(self, tensor_list: NestedTensor):
        """Run backbone and project multi-scale features."""
        x = tensor_list.tensors
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Backbone expects input [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")

        bsz, num_frames, channels, height, width = x.shape
        x_2d = x.reshape(bsz * num_frames, channels, height, width)

        raw_feats_flat = self.encoder(x_2d)
        raw_feats_temporal = [
            feat.reshape(bsz, num_frames, feat.shape[1], feat.shape[2], feat.shape[3])
            for feat in raw_feats_flat
        ]
        raw_feats_temporal = [op(feat) for op, feat in zip(self.temporal_ops, raw_feats_temporal)]
        raw_feats = [self._aggregate_temporal_feature(feat) for feat in raw_feats_temporal]

        if self.dino_ref_branch is not None and self.dino_ref_injector is not None:
            selected_tokens = self._select_dino_ref_tokens(raw_feats_temporal)
            ref_tokens = self.dino_ref_branch(selected_tokens)
            raw_feats = self.dino_ref_injector.inject(raw_feats, ref_tokens)

        if self.lidar_branch is not None and self.lidar_fusion is not None:
            lidar_input = getattr(tensor_list, "lidar", None)
            lidar_shallow, _ = self.lidar_branch(lidar_input)
            raw_feats = self.lidar_fusion.inject_shallow(raw_feats, lidar_shallow)

        feats = self.projector(raw_feats)
        out = []

        m = tensor_list.mask
        if m is None:
            raise ValueError("NestedTensor mask is required for backbone forward.")
        if m.dim() == 3:
            m_2d = m
        elif m.dim() == 4:
            m_2d = self._aggregate_temporal_mask(m)
        else:
            raise ValueError(f"NestedTensor mask must be [B,H,W] or [B,T,H,W], got {tuple(m.shape)}")

        for feat in feats:
            mask = F.interpolate(m_2d[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))

        cross_attn_out = None
        if self.cross_attn_projector is not None:
            cross_attn_out = []
            cross_attn_feats = self.cross_attn_projector(raw_feats)
            for feat in cross_attn_feats:
                mask = F.interpolate(m_2d[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
                cross_attn_out.append(NestedTensor(feat, mask))

        return out, cross_attn_out

    def forward_export(self, tensors: torch.Tensor):
        raw_feats = self.encoder(tensors)
        feats = self.projector(raw_feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            b, _, h, w = feat.shape
            out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)

        cross_attn_feats = None
        if self.cross_attn_projector is not None:
            cross_attn_feats = list(self.cross_attn_projector(raw_feats))

        return out_feats, out_masks, cross_attn_feats

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}

        is_dinov2 = str(self.encoder_name).startswith("dinov2")
        num_layers = args.out_feature_indexes[-1] + 1

        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                if is_dinov2:
                    lr = (
                        args.lr_encoder
                        * get_dinov2_lr_decay_rate(
                            n,
                            lr_decay_rate=args.lr_vit_layer_decay,
                            num_layers=num_layers,
                        )
                        * args.lr_component_decay**2
                    )
                    wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
                else:
                    lr = args.lr_encoder * args.lr_component_decay**2
                    wd = args.weight_decay
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_dinov2_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """Calculate lr decay rate for different ViT blocks.

    Args:
        name: Parameter name.
        lr_decay_rate: Base lr decay rate.
        num_layers: Number of ViT blocks.

    Returns:
        Lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate
