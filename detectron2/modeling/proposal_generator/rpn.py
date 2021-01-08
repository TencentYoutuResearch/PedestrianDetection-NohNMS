# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from .build import PROPOSAL_GENERATOR_REGISTRY
from .rpn_outputs import RPNOutputs, find_top_rpn_proposals, find_top_rpn_proposals_all_level

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
"""
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_cell_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            in_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


@RPN_HEAD_REGISTRY.register()
class SeparateRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the hidden representation
        self.cls_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.reg_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_cell_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            in_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for l in [self.cls_conv, self.reg_conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, cls_features, reg_features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for cls_x, reg_x in zip(cls_features, reg_features):
            cls_t = F.relu(self.cls_conv(cls_x))
            reg_t = F.relu(self.reg_conv(reg_x))
            pred_objectness_logits.append(self.objectness_logits(cls_t))
            pred_anchor_deltas.append(self.anchor_deltas(reg_t))
        return pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len        = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features             = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh              = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image    = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction       = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta          = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight             = cfg.MODEL.RPN.LOSS_WEIGHT
        self.ignore_ioa              = cfg.MODEL.RPN.IGNORE_IOA
        self.allow_oob               = cfg.MODEL.ALLOW_BOX_OUT_OF_BOUNDARY
        self.top_proposals_all_level = cfg.MODEL.RPN.TOP_PROPOSALS_ALL_LEVEL
        self.update_matches          = cfg.MODEL.RPN.UPDATE_MATCHES
        self.get_gt_per_level        = cfg.MODEL.RPN.GET_GT_PER_LEVEL
        self.ignore_ambiguous_sample = cfg.MODEL.RPN.IGNORE_AMBIGUOUS_SAMPLE
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS,
            cfg.MODEL.RPN.IOU_LABELS,
            allow_low_quality_matches=True,
            update_matches=self.update_matches,
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])

    def forward(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        # gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        gt_boxes = (
            [x.gt_boxes[x.gt_classes != -1] for x in gt_instances]
            if gt_instances is not None
            else None
        )
        ignore_gt_boxes = (
            [x.gt_boxes[x.gt_classes == -1] for x in gt_instances]
            if gt_instances is not None
            else None
        )
        del gt_instances
        if isinstance(features, dict):
            features = [features[f] for f in self.in_features]
            pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
            anchors = self.anchor_generator(features)
        else:
            cls_features, reg_features = features
            cls_features = [cls_features[f] for f in self.in_features]
            reg_features = [reg_features[f] for f in self.in_features]
            pred_objectness_logits, pred_anchor_deltas = self.rpn_head(cls_features, reg_features)
            anchors = self.anchor_generator(cls_features)
        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
            ignore_gt_boxes,
            ignore_ioa=self.ignore_ioa,
            get_gt_per_level=self.get_gt_per_level,
            ignore_ambiguous_sample=self.ignore_ambiguous_sample,
        )

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}
        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            if self.top_proposals_all_level:
                proposals = find_top_rpn_proposals_all_level(
                    outputs.predict_proposals(),
                    outputs.predict_objectness_logits(),
                    images,
                    self.nms_thresh,
                    self.pre_nms_topk[self.training],
                    self.post_nms_topk[self.training],
                    self.min_box_side_len,
                    self.training,
                    allow_oob=self.allow_oob,
                )
            else:
                proposals = find_top_rpn_proposals(
                    outputs.predict_proposals(),
                    outputs.predict_objectness_logits(),
                    images,
                    self.nms_thresh,
                    self.pre_nms_topk[self.training],
                    self.post_nms_topk[self.training],
                    self.min_box_side_len,
                    self.training,
                    allow_oob=self.allow_oob,
                )

        return proposals, losses
