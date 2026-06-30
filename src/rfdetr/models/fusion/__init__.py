# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Fusion modules for multimodal extensions."""

from rfdetr.models.fusion.dino_ref import DinoRefBranch, DinoRefInjector
from rfdetr.models.fusion.lidar import LiDARBranch, LiDARFusion

__all__ = ["DinoRefBranch", "DinoRefInjector", "LiDARBranch", "LiDARFusion"]
