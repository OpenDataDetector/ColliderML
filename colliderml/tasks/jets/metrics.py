"""Jet classification metrics: ROC AUC and background rejection at fixed efficiency.

Minimal dependencies: just numpy. The AUC is computed via a trapezoidal
rule so we don't pull in scikit-learn.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

_ArrayLike = Union[np.ndarray, Sequence[float]]


def roc_auc(y_true: _ArrayLike, y_score: _ArrayLike) -> float:
    """Area under the ROC curve for a binary classifier.

    Args:
        y_true: Binary labels (0 or 1) or booleans.
        y_score: Classifier output; higher means more likely positive.

    Returns:
        AUC in ``[0, 1]``. Returns ``0.0`` when either class is empty.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)
    if len(y_true_arr) == 0:
        return 0.0
    order = np.argsort(-y_score_arr)
    y_sorted = y_true_arr[order]
    n_pos = float(y_sorted.sum())
    n_neg = float(len(y_sorted) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    # `np.trapezoid` is the numpy 2.x name; `np.trapz` is the removed
    # 1.x alias. Use whichever exists — we can't use `getattr(..., default)`
    # because the default argument is evaluated eagerly.
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(tpr, fpr))
    else:
        area = float(np.trapz(tpr, fpr))  # type: ignore[attr-defined]
    return round(area, 6)


def rejection_at_efficiency(
    y_true: _ArrayLike,
    y_score: _ArrayLike,
    target_eff: float,
) -> float:
    """Background rejection (``1 / FPR``) at a fixed signal efficiency.

    Args:
        y_true: Binary labels.
        y_score: Classifier score.
        target_eff: Signal efficiency working point in ``[0, 1]``.

    Returns:
        ``1 / FPR`` at the threshold that yields ``target_eff`` signal
        efficiency. Returns ``inf`` when no background passes.
    """
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_score_arr = np.asarray(y_score, dtype=float)
    sig_scores = np.sort(y_score_arr[y_true_arr])[::-1]
    bg_scores = y_score_arr[~y_true_arr]
    if len(sig_scores) == 0:
        return 0.0
    idx = int(len(sig_scores) * target_eff)
    idx = min(idx, len(sig_scores) - 1)
    threshold = float(sig_scores[idx])
    fpr = float((bg_scores >= threshold).mean())
    if fpr == 0.0:
        return float("inf")
    return round(1.0 / fpr, 3)


__all__ = ["roc_auc", "rejection_at_efficiency"]
