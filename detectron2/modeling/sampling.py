# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

__all__ = ["subsample_labels"]


def subsample_labels(labels, num_samples, positive_fraction, bg_label):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = torch.nonzero((labels != -1) & (labels != bg_label)).squeeze(1)
    negative = torch.nonzero(labels == bg_label).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx

def bernoulli_subsample_labels(labels, num_samples, positive_fraction, bg_label):
    fg_mask = labels == 0
    bg_mask = labels == 1
    pos_max = int(num_samples * positive_fraction)
    fg_inds_mask = _bernoulli_sample_masks(fg_mask, pos_max, True)
    neg_max = num_samples - fg_inds_mask.sum()
    bg_inds_mask = _bernoulli_sample_masks(bg_mask, neg_max, True)
    pos_idx = torch.where(fg_inds_mask==1)[0]
    neg_idx = torch.where(bg_inds_mask==1)[0]
    return pos_idx, neg_idx


def _bernoulli_sample_masks(masks, num_samples, sample_value):
    """ Using the bernoulli sampling method"""
    positive = torch.nonzero(masks.eq(sample_value)).squeeze(1)
    num_mask = len(positive)
    num_samples = int(num_samples)
    num_final_samples = min(num_mask, num_samples)
    num_final_negative = num_mask - num_final_samples
    # here, we use the bernoulli probability to sample the anchors
    perm = torch.randperm(num_mask, device=masks.device)[:num_final_negative]
    negative = positive[perm]
    masks[negative] = not sample_value
    return masks