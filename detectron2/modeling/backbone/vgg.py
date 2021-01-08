# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
import torchvision.models.vgg16 as vgg16

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = ["VGG", "build_vgg16_backbone"]


class VGG(Backbone):
    def __init__(self, cfg, input_shape=None, pretrained=True, freeze_at=3):
        super().__init__()
        backbone = vgg16(pretrained=pretrained)
        self.features = backbone.features
        if freeze_at >= 1:
            for p in self.features[:freeze_at].parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = {}
        for layer in self.features[:23]:
            x = layer(x)
        for layer in self.features[24:]:
            x = layer(x)
        outputs["output"] = x
        return outputs

    def output_shape(self):
        pass


@BACKBONE_REGISTRY.register()
def build_vgg16_backbone(cfg, input_shape):
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    return VGG(cfg, input_shape, pretrained=pretrained, freeze_at=freeze_at)
