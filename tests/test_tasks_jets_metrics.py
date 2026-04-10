"""Unit tests for :mod:`colliderml.tasks.jets.metrics`."""

from __future__ import annotations

import math

import numpy as np

from colliderml.tasks.jets.metrics import rejection_at_efficiency, roc_auc


def test_roc_auc_perfect_classifier() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert roc_auc(y_true, y_score) == 1.0


def test_roc_auc_inverse_classifier_is_zero() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert roc_auc(y_true, y_score) == 0.0


def test_roc_auc_random_classifier_near_half() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000)
    y_score = rng.random(size=2000)
    auc = roc_auc(y_true, y_score)
    assert 0.4 < auc < 0.6


def test_roc_auc_single_class_returns_zero() -> None:
    # Degenerate inputs shouldn't blow up — the metric contract says
    # "return 0.0" when one side of the split is empty.
    assert roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) == 0.0
    assert roc_auc([0, 0, 0], [0.1, 0.2, 0.3]) == 0.0
    assert roc_auc([], []) == 0.0


def test_rejection_at_efficiency_perfect_separation() -> None:
    y_true = [0] * 100 + [1] * 100
    y_score = [0.1] * 100 + [0.9] * 100
    # All signal passes at threshold 0.9, zero background passes → inf.
    result = rejection_at_efficiency(y_true, y_score, target_eff=1.0)
    assert math.isinf(result)


def test_rejection_at_efficiency_half_overlap() -> None:
    y_true = [0] * 100 + [1] * 100
    bg_scores = list(np.linspace(0.0, 1.0, 100))
    sig_scores = list(np.linspace(0.5, 1.5, 100))  # half above max background
    result = rejection_at_efficiency(y_true, bg_scores + sig_scores, target_eff=0.5)
    # At 50% signal efficiency the threshold is ~1.0, half the background
    # should pass → rejection ~2x.
    assert result > 1.0


def test_rejection_at_efficiency_empty_signal() -> None:
    assert rejection_at_efficiency([0, 0, 0], [0.1, 0.2, 0.3], target_eff=0.5) == 0.0
