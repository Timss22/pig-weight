"""Evaluation metrics for the segmentation stage: Dice coefficient and IoU.

These are the *evaluation* metrics (hard, thresholded overlap) used for
reporting Dice/IoU against ground-truth masks — distinct from the
differentiable SoftDiceLoss in losses.py used for training.

Both functions share the same contract:
  - `pred`: (B, 1, H, W) or (B, H, W) tensor of either raw logits or
    probabilities in [0, 1] (see `pred_is_logits`).
  - `target`: same shape as `pred` (after squeezing the channel dim if
    needed), binary {0, 1} float or int tensor.
  - Predictions are binarized at `threshold` (applied to probabilities,
    i.e. after an internal sigmoid if `pred_is_logits=True`).
  - Returns a scalar torch.Tensor: the metric averaged over the batch
    (mean of per-sample scores), unless `reduction="none"` is given, in
    which case a per-sample tensor of shape (B,) is returned.
  - `eps` avoids division by zero for empty-mask edge cases (both pred and
    target all-background): with eps, that case scores close to 1.0
    (perfect agreement on "nothing here"), which is the conventional
    Dice/IoU convention — verify this matches what tests/test_metrics.py
    expects.
"""

from __future__ import annotations

from typing import Literal

import torch


def _prepare_binary_masks(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    pred_is_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared shape-normalization + binarization used by both metrics below.

    Not a TODO — this helper is provided so both dice_coefficient and
    iou_score can share identical preprocessing. Squeezes a channel dim of
    size 1 if present, applies sigmoid if `pred_is_logits`, thresholds
    `pred` to {0, 1}, and casts `target` to float.
    """
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    if pred_is_logits:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    return pred_binary, target_binary


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    pred_is_logits: bool = True,
    eps: float = 1e-7,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Dice coefficient: 2*|pred ∩ target| / (|pred| + |target|).

    See module docstring for the shared shape/threshold contract.
    Known cases this should satisfy (see tests/test_metrics.py):
      - identical pred/target (perfect overlap) -> 1.0
      - fully disjoint pred/target (zero overlap), both non-empty -> 0.0
      - both pred and target entirely empty (all-background) -> ~1.0 (via eps)
    """
    pred_binary, target_binary = _prepare_binary_masks(pred, target, threshold, pred_is_logits)
    # TODO(timotius): compute per-sample Dice from pred_binary/target_binary
    # (flatten spatial dims, compute intersection/union per sample, combine
    # with eps per the module docstring's formula), then apply `reduction`.
    raise NotImplementedError("TODO(timotius): implement dice_coefficient")


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    pred_is_logits: bool = True,
    eps: float = 1e-7,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Intersection-over-Union: |pred ∩ target| / |pred ∪ target|.

    See module docstring for the shared shape/threshold contract.
    Known cases this should satisfy (see tests/test_metrics.py):
      - identical pred/target (perfect overlap) -> 1.0
      - fully disjoint pred/target (zero overlap), both non-empty -> 0.0
      - both pred and target entirely empty (all-background) -> ~1.0 (via eps)
    """
    pred_binary, target_binary = _prepare_binary_masks(pred, target, threshold, pred_is_logits)
    # TODO(timotius): compute per-sample IoU from pred_binary/target_binary
    # (flatten spatial dims, compute intersection/union per sample, combine
    # with eps per the module docstring's formula), then apply `reduction`.
    raise NotImplementedError("TODO(timotius): implement iou_score")
