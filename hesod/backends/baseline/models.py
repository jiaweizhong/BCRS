"""torchvision detection model builders: COCO-pretrained Faster R-CNN and
RetinaNet, fine-tuned for this project's classes. Matches this project's own
existing convention of fine-tuning from a pretrained checkpoint
(`weights/pretrained/yolov5m.pt` for every other arm) rather than training
from random init.
"""

from __future__ import annotations

from functools import partial

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

MODEL_NAMES = ("fasterrcnn", "retinanet")


def build_model(
    name: str, num_classes: int, *, min_size: int = 800, max_size: int = 1333
) -> nn.Module:
    """`num_classes` excludes background; torchvision adds it internally.
    `min_size`/`max_size` control the aspect-ratio-preserving resize
    torchvision applies internally -- setting them equal resizes so the
    LONGER image side equals that value (see HESOD-Experiment-Plan.md SS9.3
    for why this isn't pixel-identical to the YOLOv5 letterbox convention
    used by every other arm).
    """
    if name == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            min_size=min_size,
            max_size=max_size,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        return model
    if name == "retinanet":
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            min_size=min_size,
            max_size=max_size,
        )
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels,
            num_anchors,
            num_classes + 1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
        return model
    raise ValueError(f"Unknown model {name!r}, expected one of {MODEL_NAMES}")
