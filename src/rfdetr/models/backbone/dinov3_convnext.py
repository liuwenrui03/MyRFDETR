# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoBackbone, DINOv3ConvNextConfig

from rfdetr.utilities.logger import get_logger

logger = get_logger()


class DinoV3ConvNext(nn.Module):
    """DINOv3 ConvNext backbone wrapper.

    Supported sizes:
    - tiny: ``facebook/dinov3-convnext-tiny-pretrain-lvd1689m``
    - small: ``facebook/dinov3-convnext-small-pretrain-lvd1689m``
    """

    _MODEL_NAME_BY_SIZE = {
        "tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    }

    def __init__(
        self,
        shape: tuple[int, int] = (640, 640),
        out_feature_indexes: list[int] | None = None,
        size: str = "tiny",
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if out_feature_indexes is None:
            out_feature_indexes = [2, 3, 4]
        if size not in self._MODEL_NAME_BY_SIZE:
            raise ValueError(f"Unsupported DINOv3 ConvNext size: {size}")

        self.shape = shape
        model_name = self._MODEL_NAME_BY_SIZE[size]

        stage_names = [f"stage{i}" for i in out_feature_indexes]

        if not load_dinov2_weights:
            logger.warning(
                "DINOv3 ConvNext is configured without pretrained backbone weights. "
                "Falling back to random initialization from model config."
            )
            config = DINOv3ConvNextConfig(out_features=stage_names)
            config.drop_path_rate = drop_path_rate
            self.encoder = AutoBackbone.from_config(config)
        else:
            self.encoder = AutoBackbone.from_pretrained(
                model_name,
                out_features=stage_names,
                return_dict=False,
            )
            if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()

        hidden_sizes = list(getattr(self.encoder.config, "hidden_sizes", [96, 192, 384, 768]))
        self._out_feature_channels = [hidden_sizes[i - 1] for i in out_feature_indexes]
        self._export = False

    def export(self) -> None:
        self._export = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.encoder(x)
        return list(outputs[0])
