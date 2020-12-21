# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
import math
from copy import deepcopy
from fvcore.nn import smooth_l1_loss
from torch import nn
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from typing import Dict, List

from detectron2.layers import batched_nms, cat, batched_noh_nms, ShapeSpec
from detectron2.structures import Boxes, Instances, calculate_iog, calculate_iou

from .fast_rcnn import FastRCNNOutputs

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference_with_overlap(boxes, scores, overlap_boxes, overlap_probs, image_shapes, score_thresh, nms_thresh, topk_per_image, allow_oob=False):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """

    result_per_image = [
        fast_rcnn_inference_single_image_with_overlap(
            boxes_per_image, scores_per_image, overlap_boxes_per_image, overlap_probs_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, allow_oob
        )
        for scores_per_image, boxes_per_image, overlap_boxes_per_image, overlap_probs_per_image, image_shape in zip(scores, boxes, overlap_boxes, overlap_probs, image_shapes)
    ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image_with_overlap(
    boxes, scores, overlap_boxes, overlap_probs, image_shape, score_thresh, nms_thresh, topk_per_image, allow_oob=False
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        overlap_boxes = overlap_boxes[valid_mask]
        overlap_probs = overlap_probs[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    if not allow_oob:
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        assert overlap_boxes.size(1) == 4, "overlap boxes prediction has no category, but: {}".format(overlap_boxes.size())
        overlap_boxes = Boxes(overlap_boxes)
        overlap_boxes.clip(image_shape)
        overlap_boxes = overlap_boxes.tensor
    else:
        boxes = boxes.view(-1, num_bbox_reg_classes, 4)

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
        overlap_boxes = overlap_boxes[filter_inds[:,0]]
    else:
        boxes = boxes[filter_mask]
        overlap_boxes = overlap_boxes[filter_inds[:,0]]
    scores = scores[filter_mask]
    overlap_probs = overlap_probs[filter_mask]

    # Apply per-class NMS
    self_defined_nms_on = True #False
    if self_defined_nms_on:
        boxes = np.ascontiguousarray(boxes.cpu())
        scores = np.ascontiguousarray(scores.cpu())
        overlap_probs = np.ascontiguousarray(overlap_probs.cpu())
        overlap_boxes = np.ascontiguousarray(overlap_boxes.cpu())

        keep = batched_noh_nms(boxes, scores, overlap_probs, overlap_boxes, Nt=nms_thresh, thresh=0.01, method=3)

        boxes = torch.from_numpy(boxes).cuda()
        scores = torch.from_numpy(scores).cuda()
        overlap_probs = torch.from_numpy(overlap_probs).cuda()
        overlap_boxes = torch.from_numpy(overlap_boxes).cuda()
        keep = keep[scores[keep].argsort(descending=True)]
    else:
        from torchvision.ops import nms
        keep = nms(boxes, scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, overlap_boxes, overlap_probs, filter_inds = boxes[keep], scores[keep], overlap_boxes[keep], overlap_probs[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.overlap_boxes = Boxes(overlap_boxes)
    result.overlap_probs = overlap_probs
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class OverlapFastRCNNOutputs(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta,
        pred_overlap_deltas=None, pred_overlap_prob=None, overlap_configs=dict(), giou=False, allow_oob=False
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta

        self.pred_overlap_deltas = pred_overlap_deltas
        self.pred_overlap_prob = pred_overlap_prob

        assert isinstance(overlap_configs, dict), 'overlap configs must be dict, {}'.format(type(overlap_configs))
        self.overlap_iou_threshold = overlap_configs.get('overlap_iou_threshold', 0.3)
        self.loss_overlap_reg_coeff = overlap_configs.get('loss_overlap_reg_coeff', 0.1)
        self.uniform_reg_divisor = overlap_configs.get('uniform_reg_divisor', False)
        self.cls_box_beta = overlap_configs.get('cls_box_beta', 0.1)

        self.giou = giou
        self.allow_oob = allow_oob

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        if proposals[0].has("overlap_iou"):
            self.overlap_iou = cat([p.overlap_iou for p in proposals], dim=0)
            self.overlap_gt_boxes = box_type.cat([p.overlap_gt_boxes for p in proposals])

    def overlap_losses(self):
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        self.pred_overlap_deltas = self.pred_overlap_deltas[fg_inds]
        self.pred_overlap_prob = self.pred_overlap_prob[fg_inds]

        self.overlap_iou = self.overlap_iou[fg_inds]
        # self.all_overlap_gt_boxes = self.overlap_gt_boxes
        self.overlap_gt_boxes = self.overlap_gt_boxes[fg_inds]

        loss_overlap_reg = self.overlap_smooth_l1_loss(fg_inds)
        loss_overlap_prob = self.overlap_prob_loss()

        overlap_loss_dict = {
            "loss_overlap_reg": loss_overlap_reg,
            "loss_overlap_prob": loss_overlap_prob
        }
        return overlap_loss_dict

    def overlap_smooth_l1_loss(self, fg_inds):
        overlap_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor[fg_inds], self.overlap_gt_boxes.tensor
        )
        trained_idx = torch.nonzero(self.overlap_iou > self.overlap_iou_threshold).squeeze(1)
        loss_overlap_reg = smooth_l1_loss(
            self.pred_overlap_deltas[trained_idx],
            overlap_deltas[trained_idx],
            self.smooth_l1_beta,
            reduction="sum",
        )
        if self.uniform_reg_divisor:
            return loss_overlap_reg / (self.gt_classes.numel() + 1e-6)
        else:
            return loss_overlap_reg / (trained_idx.size(0) + 1e-6) * self.loss_overlap_reg_coeff
    
    def overlap_prob_loss(self):
        self.pred_overlap_prob = self.pred_overlap_prob
        loss_overlap_prob = smooth_l1_loss(
            self.pred_overlap_prob[:, 0], # logit --> sigmoid
            self.overlap_iou,
            self.cls_box_beta,
            reduction="sum",
        )

        return loss_overlap_prob / (self.pred_overlap_prob.size(0) + 1e-6)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        loss_dict = {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss() if not self.giou else self.giou_loss(),
        }
        loss_dict.update(self.overlap_losses())
        return loss_dict

    def predict_overlap_probs(self):
        self.pred_overlap_prob = self.pred_overlap_prob
        overlap_probs = self.pred_overlap_prob.split(self.num_preds_per_image, dim=0)
        return overlap_probs

    def predict_overlap_boxes(self):
        num_pred = len(self.pred_overlap_deltas)
        B = self.proposals.tensor.shape[1]
        K = self.pred_overlap_deltas.shape[1] // B
        overlap_boxes = self.box2box_transform.apply_deltas(
            self.pred_overlap_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return overlap_boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        overlap_boxes = self.predict_overlap_boxes()
        overlap_probs = self.predict_overlap_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference_with_overlap(
            boxes, scores, overlap_boxes, overlap_probs, image_shapes, score_thresh, nms_thresh, topk_per_image, allow_oob=self.allow_oob
        )


class OverlapOutputLayers(nn.Module):
    """
    Two linear layers for predicting overlap:
      (1) proposal-to-overlap box regression deltas
      (2) overlap confidence
    """

    def __init__(self, input_size, sigmoid_on=True, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(OverlapOutputLayers, self).__init__()
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        self.sigmoid_on = sigmoid_on

        self.overlap_prob = nn.Linear(input_size, 1)
        self.overlap_pred = nn.Linear(input_size, box_dim)

        nn.init.normal_(self.overlap_prob.weight, std=0.01)
        nn.init.normal_(self.overlap_pred.weight, std=0.001)
        for l in [self.overlap_prob, self.overlap_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        probs = self.overlap_prob(x)
        if self.sigmoid_on:
            probs = torch.sigmoid(probs)
        overlap_deltas = self.overlap_pred(x)
        return probs, overlap_deltas
