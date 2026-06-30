# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoBackbone

from rfdetr.utilities.logger import get_logger

logger = get_logger()

_SIZE_TO_WIDTH = {
    "tiny": 192,
    "small": 384,
}


class DinoV3ViT(nn.Module):
    """DINOv3 ViT backbone wrapper.

    Supported sizes:
    - tiny: ``facebook/dinov3-vittiny14-pretrain-lvd1689m``
    - small: ``facebook/dinov3-vits14-pretrain-lvd1689m``
    """

    _MODEL_NAME_BY_SIZE = {
        "tiny": "facebook/dinov3-vittiny14-pretrain-lvd1689m",
        "small": "facebook/dinov3-vits14-pretrain-lvd1689m",
    }

    def __init__(
        self,
        shape: tuple[int, int] = (640, 640),
        out_feature_indexes: list[int] | None = None,
        size: str = "small",
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        positional_encoding_size: int = 37,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if out_feature_indexes is None:
            out_feature_indexes = [2, 4, 5, 9]
        if size not in self._MODEL_NAME_BY_SIZE:
            raise ValueError(f"Unsupported DINOv3 ViT size: {size}")

        self.shape = shape
        self.patch_size = patch_size

        model_name = self._MODEL_NAME_BY_SIZE[size]
        out_features = [f"stage{i}" for i in out_feature_indexes]

        if not load_dinov2_weights:
            logger.warning(
                "DINOv3 ViT is configured without pretrained backbone weights. "
                "Falling back to random initialization from model config."
            )
            config = AutoBackbone.from_pretrained(model_name).config
            config.out_features = out_features
            if hasattr(config, "drop_path_rate"):
                config.drop_path_rate = drop_path_rate
            if hasattr(config, "image_size"):
                implied_resolution = positional_encoding_size * patch_size
                config.image_size = implied_resolution
            if hasattr(config, "patch_size"):
                config.patch_size = patch_size
            self.encoder = AutoBackbone.from_config(config)
        else:
            self.encoder = AutoBackbone.from_pretrained(
                model_name,
                out_features=out_features,
                return_dict=False,
            )
            if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()

        self._out_feature_channels = [_SIZE_TO_WIDTH[size]] * len(out_feature_indexes)
        self._export = False

    def export(self) -> None:
        self._export = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[2] % self.patch_size != 0 or x.shape[3] % self.patch_size != 0:
            raise ValueError(
                f"DINOv3 ViT backbone requires input shape divisible by patch_size={self.patch_size}, "
                f"but got {tuple(x.shape)}"
            )
        outputs = self.encoder(x)
        return list(outputs[0])
