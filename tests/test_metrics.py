"""Known-case tests for src/training/metrics.py's dice_coefficient and iou_score.

These test the metric MATH left as a TODO in src/training/metrics.py. They
will fail with NotImplementedError until that math is implemented; that's
expected. Run them against your own implementation as you build it out.

All tests pass pred_is_logits=False and feed already-binary {0.0, 1.0}
values directly as "probabilities" — this isolates the intersection/union
math from sigmoid/threshold behavior, which is straightforward and not
the point of these tests.
"""

from __future__ import annotations

import pytest
import torch

from src.training.metrics import dice_coefficient, iou_score


def _mask(pattern: list[list[float]]) -> torch.Tensor:
    """(H, W) list -> (1, 1, H, W) float tensor, matching the metrics' expected shape."""
    return torch.tensor(pattern, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


class TestPerfectOverlap:
    def test_dice_is_one(self):
        mask = _mask([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        assert dice_coefficient(mask, mask, pred_is_logits=False).item() == pytest.approx(1.0, abs=1e-5)

    def test_iou_is_one(self):
        mask = _mask([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        assert iou_score(mask, mask, pred_is_logits=False).item() == pytest.approx(1.0, abs=1e-5)


class TestZeroOverlap:
    """Disjoint, both non-empty: left half vs. right half of a 4x4 grid."""

    pred = _mask([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])
    target = _mask([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])

    def test_dice_is_zero(self):
        assert dice_coefficient(self.pred, self.target, pred_is_logits=False).item() == pytest.approx(0.0, abs=1e-5)

    def test_iou_is_zero(self):
        assert iou_score(self.pred, self.target, pred_is_logits=False).item() == pytest.approx(0.0, abs=1e-5)


class TestBothEmpty:
    """Both pred and target all-background: conventionally scores as perfect agreement, via eps."""

    empty = _mask([[0, 0], [0, 0]])

    def test_dice_near_one(self):
        assert dice_coefficient(self.empty, self.empty, pred_is_logits=False).item() == pytest.approx(1.0, abs=1e-3)

    def test_iou_near_one(self):
        assert iou_score(self.empty, self.empty, pred_is_logits=False).item() == pytest.approx(1.0, abs=1e-3)


class TestPartialOverlap:
    """pred = top row (4px), target = left column (4px); overlap = 1px at (0,0).

    intersection = 1, |pred| = 4, |target| = 4, union = |pred|+|target|-intersection = 7.
    dice = 2*1 / (4+4) = 0.25
    iou  = 1 / 7 ≈ 0.142857
    """

    pred = _mask([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    target = _mask([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

    def test_dice_matches_hand_computed_value(self):
        assert dice_coefficient(self.pred, self.target, pred_is_logits=False).item() == pytest.approx(0.25, abs=1e-4)

    def test_iou_matches_hand_computed_value(self):
        assert iou_score(self.pred, self.target, pred_is_logits=False).item() == pytest.approx(1 / 7, abs=1e-4)


class TestBatchReduction:
    """Batch of 2 known-different samples: perfect overlap + zero overlap."""

    perfect = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
    disjoint_pred = torch.tensor([[1, 0], [0, 0]], dtype=torch.float32)
    disjoint_target = torch.tensor([[0, 1], [0, 0]], dtype=torch.float32)

    pred_batch = torch.stack([perfect, disjoint_pred]).unsqueeze(1)  # (2, 1, 2, 2)
    target_batch = torch.stack([perfect, disjoint_target]).unsqueeze(1)

    def test_mean_reduction_averages_batch(self):
        # sample 0: dice=1.0, sample 1: dice=0.0 -> mean = 0.5
        result = dice_coefficient(self.pred_batch, self.target_batch, pred_is_logits=False, reduction="mean")
        assert result.item() == pytest.approx(0.5, abs=1e-4)

    def test_none_reduction_returns_per_sample(self):
        result = dice_coefficient(self.pred_batch, self.target_batch, pred_is_logits=False, reduction="none")
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(1.0, abs=1e-4)
        assert result[1].item() == pytest.approx(0.0, abs=1e-4)


class TestLogitsInput:
    """pred_is_logits=True (the default) should apply sigmoid before thresholding."""

    def test_large_positive_logit_treated_as_foreground(self):
        # sigmoid(10) ~= 0.99995, well above threshold=0.5
        pred_logits = _mask([[10, 10], [-10, -10]])
        target = _mask([[1, 1], [0, 0]])
        assert dice_coefficient(pred_logits, target, pred_is_logits=True).item() == pytest.approx(1.0, abs=1e-3)
