# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat
import numpy as np


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


# Note: this function (nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def nms_rotated(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the rotated boxes according
    to their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Note that RotatedBox (5, 3, 4, 2, -90) covers exactly the same region as
    RotatedBox (5, 3, 4, 2, 90) does, and their IoU will be 1. However, they
    can be representing completely different objects in certain tasks, e.g., OCR.

    As for the question of whether rotated-NMS should treat them as faraway boxes
    even though their IOU is 1, it depends on the application and/or ground truth annotation.

    As an extreme example, consider a single character v and the square box around it.

    If the angle is 0 degree, the object (text) would be read as 'v';

    If the angle is 90 degrees, the object (text) would become '>';

    If the angle is 180 degrees, the object (text) would become '^';

    If the angle is 270/-90 degrees, the object (text) would become '<'

    All of these cases have IoU of 1 to each other, and rotated NMS that only
    uses IoU as criterion would only keep one of them with the highest score -
    which, practically, still makes sense in most cases because typically
    only one of theses orientations is the correct one. Also, it does not matter
    as much if the box is only used to classify the object (instead of transcribing
    them with a sequential OCR recognition model) later.

    On the other hand, when we use IoU to filter proposals that are close to the
    ground truth during training, we should definitely take the angle into account if
    we know the ground truth is labeled with the strictly correct orientation (as in,
    upside-down words are annotated with -180 degrees even though they can be covered
    with a 0/90/-90 degree box, etc.)

    The way the original dataset is annotated also matters. For example, if the dataset
    is a 4-point polygon dataset that does not enforce ordering of vertices/orientation,
    we can estimate a minimum rotated bounding box to this polygon, but there's no way
    we can tell the correct angle with 100% confidence (as shown above, there could be 4 different
    rotated boxes, with angles differed by 90 degrees to each other, covering the exactly
    same region). In that case we have to just use IoU to determine the box
    proximity (as many detection benchmarks (even for text) do) unless there're other
    assumptions we can make (like width is always larger than height, or the object is not
    rotated by more than 90 degrees CCW/CW, etc.)

    In summary, not considering angles in rotated NMS seems to be a good option for now,
    but we should be aware of its implications.

    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        scores (Tensor[N]): Scores for each one of the rotated boxes
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    """
    from detectron2 import _C

    return _C.nms_rotated(boxes, scores, iou_threshold)


# Note: this function (batched_nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def batched_nms_rotated(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        iou_threshold (float):
           discards all overlapping boxes
           with IoU < iou_threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    assert boxes.shape[-1] == 5

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # Strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # Note that batched_nms in torchvision/ops/boxes.py only uses max_coordinate,
    # which won't handle negative coordinates correctly.
    # Here by using min_coordinate we can make sure the negative coordinates are
    # correctly handled.
    max_coordinate = (
        torch.max(boxes[:, 0], boxes[:, 1]) + torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).max()
    min_coordinate = (
        torch.min(boxes[:, 0], boxes[:, 1]) - torch.min(boxes[:, 2], boxes[:, 3]) / 2
    ).min()
    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()  # avoid modifying the original values in boxes
    boxes_for_nms[:, :2] += offsets[:, None]
    keep = nms_rotated(boxes_for_nms, scores, iou_threshold)
    return keep

def batched_noh_nms(boxes, scores, overlap_probs, overlap_boxes, sigma=0.5, Nt=0.5, thresh=0.001, method=0):
    """
    scores: sorted scores input.
    """
    N = boxes.shape[0]
    overlap_flags = np.zeros(N)
    for i in range(N):
        maxscore = scores[i]
        maxpos = i

        pos = i + 1
        while pos < N:
            if maxscore < scores[pos]:
                maxscore = scores[pos]
                maxpos = pos
            pos = pos + 1
        
        # swap max box
        tx1 = boxes[maxpos][0]
        ty1 = boxes[maxpos][1]
        tx2 = boxes[maxpos][2]
        ty2 = boxes[maxpos][3]

        ix1 = boxes[i][0]
        iy1 = boxes[i][1]
        ix2 = boxes[i][2]
        iy2 = boxes[i][3]

        boxes[i][0] = tx1
        boxes[i][1] = ty1
        boxes[i][2] = tx2
        boxes[i][3] = ty2

        boxes[maxpos][0] = ix1
        boxes[maxpos][1] = iy1
        boxes[maxpos][2] = ix2
        boxes[maxpos][3] = iy2

        # swap overlap box
        tx1 = overlap_boxes[maxpos][0]
        ty1 = overlap_boxes[maxpos][1]
        tx2 = overlap_boxes[maxpos][2]
        ty2 = overlap_boxes[maxpos][3]

        ix1 = overlap_boxes[i][0]
        iy1 = overlap_boxes[i][1]
        ix2 = overlap_boxes[i][2]
        iy2 = overlap_boxes[i][3]

        overlap_boxes[i][0] = tx1
        overlap_boxes[i][1] = ty1
        overlap_boxes[i][2] = tx2
        overlap_boxes[i][3] = ty2

        overlap_boxes[maxpos][0] = ix1
        overlap_boxes[maxpos][1] = iy1
        overlap_boxes[maxpos][2] = ix2
        overlap_boxes[maxpos][3] = iy2

        scores[maxpos] = scores[i]
        scores[i] = maxscore

        temp_flag = overlap_flags[maxpos]
        overlap_flags[maxpos] = overlap_flags[i]
        overlap_flags[i] = temp_flag

        temp_prob = overlap_probs[i]
        temp_max_prob = overlap_probs[maxpos]
        overlap_probs[maxpos] = temp_prob
        overlap_probs[i] = temp_max_prob

        pos = i + 1
        
        area_i = (boxes[i,2] - boxes[i,0] + 1) * (boxes[i,3] - boxes[i,1] + 1)
        if overlap_probs[i] > 0.3:
            overlap_x = overlap_boxes[i][0]
            overlap_y = overlap_boxes[i][1]
            overlap_w = overlap_boxes[i][2] - overlap_boxes[i][0]
            overlap_h = overlap_boxes[i][3] - overlap_boxes[i][1]

            box_x = boxes[i][0]
            box_y = boxes[i][1]
            box_w = boxes[i][2] - boxes[i][0]
            box_h = boxes[i][3] - boxes[i][1]
            pred_deltas_overlap = np.array([
                (overlap_x - box_x) / box_w,
                (overlap_y - box_y) / box_h,
                np.log(overlap_w / box_w),
                np.log(overlap_h / box_h),
            ])
        while pos < N:
            area_pos = (boxes[pos,2] - boxes[pos,0] + 1) * (boxes[pos,3] - boxes[pos,1] + 1)
            iw = (min(boxes[pos,2], boxes[i,2]) - max(boxes[pos,0], boxes[i, 0]))
            if iw > 0:
                ih = (min(boxes[pos,3], boxes[i,3]) - max(boxes[pos,1], boxes[i, 1]))
                if ih > 0:
                    ua = float(area_pos + area_i - ih * iw)
                    ov = iw * ih / ua

                    if method == 1: #linear（soft nms)
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: #gaussian (soft nms)
                        if ov > Nt:
                            weight = np.exp(-(ov * ov) / sigma)
                        else:
                            weight = 1
                    elif method == 0: #regular nms
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1
                    elif method == 3: #noh nms
                        if ov > Nt:
                            if overlap_probs[i] > 0.3 and overlap_flags[i] == 0:
                                prop_x = boxes[pos][0]
                                prop_y = boxes[pos][1]
                                prop_w = boxes[pos][2] - boxes[pos][0]
                                prop_h = boxes[pos][3] - boxes[pos][1]

                                proposal_deltas = np.array([
                                    (prop_x - box_x) / box_w,
                                    (prop_y - box_y) / box_h,
                                    np.log(prop_w / box_w),
                                    np.log(prop_h / box_h),
                                ])
                                upper = np.sum(np.power(proposal_deltas - pred_deltas_overlap, 2))
                                weight = np.exp( -upper/ (2.0 * np.power(0.2, 2)))
                                if weight >= 0.6:
                                    overlap_flags[pos] = 1
                            else:
                                weight = 0
                        else:
                            weight = 1
                    elif method == 4: #Adaptive NMS
                        Nm = max(overlap_probs[i], Nt)
                        if ov > Nm:
                            weight = 0
                        else:
                            weight = 1
                    else:
                        raise NotImplementedError("method {} is not implemented".format(method))
                    
                    scores[pos] = weight * scores[pos]

                    if scores[pos] < thresh:
                        x1 = boxes[N-1][0]
                        y1 = boxes[N-1][1]
                        x2 = boxes[N-1][2]
                        y2 = boxes[N-1][3]
                        boxes[pos][0] = x1
                        boxes[pos][1] = y1
                        boxes[pos][2] = x2
                        boxes[pos][3] = y2

                        x1 = overlap_boxes[N-1][0]
                        y1 = overlap_boxes[N-1][1]
                        x2 = overlap_boxes[N-1][2]
                        y2 = overlap_boxes[N-1][3]
                        overlap_boxes[pos][0] = x1
                        overlap_boxes[pos][1] = y1
                        overlap_boxes[pos][2] = x2
                        overlap_boxes[pos][3] = y2

                        temp_flag = overlap_flags[N-1]
                        overlap_flags[N-1] = overlap_flags[pos]
                        overlap_flags[pos] = temp_flag

                        s = scores[N-1]
                        scores[pos] = s

                        s = overlap_probs[N-1]
                        overlap_probs[pos] = s
                        N = N - 1
                        pos = pos -1
            pos = pos + 1

    keep = torch.arange(N)
    return keep
