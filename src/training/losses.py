"""Loss functions for the segmentation stage.

DiceBCELoss combines a differentiable soft-Dice loss with BCE-with-logits,
weighted per configs/segmentation.yaml's training.loss_dice_weight /
training.loss_bce_weight. Both terms operate on logits directly (no
separate sigmoid call needed by the caller).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Differentiable soft-Dice loss for binary segmentation.

    Unlike the Dice *metric* in metrics.py (which thresholds predictions
    into hard 0/1 masks for evaluation), this loss uses sigmoid
    probabilities directly so it's differentiable and usable for training.

    Input: `logits` and `targets`, both (B, 1, H, W) (or broadcastable).
    `targets` must be binary {0, 1} float tensors.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.flatten(start_dim=1)
        targets_flat = targets.flatten(start_dim=1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Weighted sum of SoftDiceLoss and BCEWithLogitsLoss.

    total_loss = dice_weight * dice_loss + bce_weight * bce_loss

    Both sub-losses take raw logits — the model should NOT apply sigmoid
    before this loss (see src/models/unet.py: PigUNet.forward returns
    logits).
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = SoftDiceLoss(eps=eps)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce


def build_loss(config: dict) -> DiceBCELoss:
    """Construct DiceBCELoss from a config dict (the `training` section)."""
    train_cfg = config["training"]
    return DiceBCELoss(
        dice_weight=train_cfg.get("loss_dice_weight", 0.5),
        bce_weight=train_cfg.get("loss_bce_weight", 0.5),
    )
