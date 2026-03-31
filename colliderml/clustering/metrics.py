"""Clustering performance metrics for calorimeter reconstruction.

All primary metrics are energy-weighted: a high-energy particle that is missed
matters more than a low-energy one.  Metrics are computed per-event and then
averaged (weighted by event energy) to avoid the inflation that occurs when
merging all events into one label vector.

Metric suite
------------
* **Efficiency** – fraction of truth energy that was matched to a reco cluster.
* **Purity** – fraction of reco energy that was matched to a truth cluster.
* **Energy resolution** – σ_eff of E_reco / E_truth for matched pairs.
* **Splitting rate** – energy-weighted fraction of truth clusters with
  significant energy in >1 reco cluster.
* **Merging rate** – energy-weighted fraction of reco clusters receiving
  significant energy from >1 truth cluster.
* **Fake rate** – energy-weighted fraction of reco clusters with no truth match.
* **Weighted V-score** – information-theoretic cell-assignment quality
  (homogeneity × completeness), energy-weighted and per-event averaged.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sp

from colliderml.clustering.matching import build_overlap_matrix, greedy_match


# ---------------------------------------------------------------------------
# Public: top-level evaluation
# ---------------------------------------------------------------------------


def evaluate_clustering(
    truth_clusters: pd.DataFrame,
    pred_labels: pd.DataFrame,
    *,
    match_threshold: float = 0.5,
    split_threshold: float = 0.1,
) -> Dict:
    """Full clustering evaluation pipeline.

    Builds the overlap matrix, performs greedy matching, and computes all
    metrics in one call.

    Args:
        truth_clusters: From :func:`build_truth_clusters`.  Required columns:
            ``event_id``, ``cell_index``, ``primary_ancestor_id``, ``energy``.
        pred_labels: Predicted cluster assignments.  Required columns:
            ``event_id``, ``cell_index``, ``cluster_id``.
        match_threshold: EIOU threshold for greedy matching (default 0.5).
        split_threshold: EIOU threshold for counting a truth cluster as
            contributing to a reco cluster in the splitting/merging metrics
            (default 0.1).

    Returns:
        Dictionary with keys:
            efficiency, purity, energy_resolution, splitting_rate,
            merging_rate, fake_rate, v_score, homogeneity, completeness,
            n_events, per_event (DataFrame).
    """
    overlaps = build_overlap_matrix(truth_clusters, pred_labels)
    matches = greedy_match(overlaps, threshold=match_threshold)

    eff = energy_weighted_efficiency(matches)
    pur = energy_weighted_purity(matches)
    e_res = energy_resolution(matches)
    sr = splitting_rate(overlaps, threshold=split_threshold)
    mr = merging_rate(overlaps, threshold=split_threshold)
    fr = fake_rate(matches)

    hom, comp, vs = _per_event_v_score(truth_clusters, pred_labels)

    # Per-event breakdown
    per_event = _per_event_summary(matches, overlaps, truth_clusters, pred_labels, match_threshold, split_threshold)

    return {
        "efficiency": eff,
        "purity": pur,
        "energy_resolution": e_res,
        "splitting_rate": sr,
        "merging_rate": mr,
        "fake_rate": fr,
        "v_score": vs,
        "homogeneity": hom,
        "completeness": comp,
        "n_events": truth_clusters["event_id"].nunique(),
        "per_event": per_event,
    }


# ---------------------------------------------------------------------------
# Object-level metrics (from matching)
# ---------------------------------------------------------------------------


def energy_weighted_efficiency(matches: pd.DataFrame) -> float:
    """Fraction of truth energy that was successfully matched.

    ``efficiency = Σ E_truth[matched] / Σ E_truth[all truth]``
    """
    truth_rows = matches[matches["truth_energy"] > 0]
    if truth_rows.empty:
        return 0.0
    total = truth_rows["truth_energy"].sum()
    matched = truth_rows.loc[truth_rows["matched"], "truth_energy"].sum()
    return float(matched / total) if total > 0 else 0.0


def energy_weighted_purity(matches: pd.DataFrame) -> float:
    """Fraction of reco energy that was successfully matched.

    ``purity = Σ E_reco[matched] / Σ E_reco[all reco]``
    """
    reco_rows = matches[matches["reco_energy"] > 0]
    if reco_rows.empty:
        return 0.0
    total = reco_rows["reco_energy"].sum()
    matched = reco_rows.loc[reco_rows["matched"], "reco_energy"].sum()
    return float(matched / total) if total > 0 else 0.0


def energy_resolution(matches: pd.DataFrame) -> float:
    """σ_eff of E_reco / E_truth for matched pairs, weighted by E_truth.

    σ_eff is half the width of the minimum interval containing 68.3% of
    the (energy-weighted) distribution — more robust than a Gaussian fit.
    """
    matched = matches[matches["matched"] & (matches["truth_energy"] > 0)]
    if len(matched) < 2:
        return float("nan")
    response = (matched["reco_energy"] / matched["truth_energy"]).to_numpy()
    weights = matched["truth_energy"].to_numpy()
    return float(sigma_eff(response, weights=weights))


def splitting_rate(
    overlaps: pd.DataFrame,
    *,
    threshold: float = 0.1,
) -> float:
    """Energy-weighted fraction of truth clusters split across >1 reco cluster.

    A truth cluster is "split" if it has EIOU ≥ *threshold* with more than
    one reco cluster.
    """
    sig = overlaps[overlaps["eiou"] >= threshold]
    counts = sig.groupby(["event_id", "primary_ancestor_id"]).agg(
        n_reco=("cluster_id", "nunique"),
        truth_energy=("truth_energy", "first"),
    )
    if counts.empty:
        return 0.0
    total_e = counts["truth_energy"].sum()
    split_e = counts.loc[counts["n_reco"] > 1, "truth_energy"].sum()
    return float(split_e / total_e) if total_e > 0 else 0.0


def merging_rate(
    overlaps: pd.DataFrame,
    *,
    threshold: float = 0.1,
) -> float:
    """Energy-weighted fraction of reco clusters merging >1 truth cluster.

    A reco cluster is "merged" if it has EIOU ≥ *threshold* with more than
    one truth cluster.
    """
    sig = overlaps[overlaps["eiou"] >= threshold]
    counts = sig.groupby(["event_id", "cluster_id"]).agg(
        n_truth=("primary_ancestor_id", "nunique"),
        reco_energy=("reco_energy", "first"),
    )
    if counts.empty:
        return 0.0
    total_e = counts["reco_energy"].sum()
    merge_e = counts.loc[counts["n_truth"] > 1, "reco_energy"].sum()
    return float(merge_e / total_e) if total_e > 0 else 0.0


def fake_rate(matches: pd.DataFrame) -> float:
    """Energy-weighted fraction of reco clusters with no truth match."""
    reco_rows = matches[matches["reco_energy"] > 0]
    if reco_rows.empty:
        return 0.0
    total_e = reco_rows["reco_energy"].sum()
    fake_e = reco_rows.loc[~reco_rows["matched"], "reco_energy"].sum()
    return float(fake_e / total_e) if total_e > 0 else 0.0


# ---------------------------------------------------------------------------
# sigma_eff — robust resolution estimator
# ---------------------------------------------------------------------------


def sigma_eff(
    values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    fraction: float = 0.683,
) -> float:
    """Half-width of the minimum interval containing *fraction* of the distribution.

    For a Gaussian this recovers σ.  For non-Gaussian tails it is more robust
    than a standard-deviation or Gaussian fit.

    Args:
        values: 1-D array of values (e.g. E_reco / E_truth).
        weights: Optional energy weights.  If None, uniform weights.
        fraction: Target fraction of the distribution (default 68.3%).

    Returns:
        Half-width of the narrowest interval.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan")

    order = np.argsort(values)
    values = values[order]

    if weights is None:
        # Unweighted: sliding window of size ceil(n * fraction)
        n = len(values)
        window = int(np.ceil(n * fraction))
        if window >= n:
            return float((values[-1] - values[0]) / 2)
        widths = values[window:] - values[: n - window]
        return float(widths.min() / 2)

    weights = np.asarray(weights, dtype=float)[order]
    cum_w = np.cumsum(weights)
    total_w = cum_w[-1]
    target = fraction * total_w

    best = float("inf")
    j = 0
    for i in range(len(values)):
        # Advance j until cumulative weight from i to j >= target
        while j < len(values) and (cum_w[j] - (cum_w[i - 1] if i > 0 else 0)) < target:
            j += 1
        if j >= len(values):
            break
        width = values[j] - values[i]
        if width < best:
            best = width

    return float(best / 2) if np.isfinite(best) else float("nan")


# ---------------------------------------------------------------------------
# V-score (information-theoretic, energy-weighted, per-event)
# ---------------------------------------------------------------------------


def weighted_v_score(
    truth_labels: np.ndarray,
    pred_labels: np.ndarray,
    weights: np.ndarray,
    *,
    beta: float = 1.0,
) -> Tuple[float, float, float]:
    """Energy-weighted V-score for a single set of labels.

    Args:
        truth_labels: 1-D integer array of truth class per cell.
        pred_labels: 1-D integer array of predicted cluster per cell.
        weights: 1-D energy weights per cell.
        beta: Weight of homogeneity vs completeness (default 1.0 = harmonic mean).

    Returns:
        (homogeneity, completeness, v_measure)
    """
    truth_labels = np.asarray(truth_labels)
    pred_labels = np.asarray(pred_labels)
    weights = np.asarray(weights, dtype=float)

    if len(truth_labels) == 0:
        return 1.0, 1.0, 1.0

    entropy_c = _weighted_entropy(truth_labels, weights)
    entropy_k = _weighted_entropy(pred_labels, weights)

    contingency = _weighted_contingency(truth_labels, pred_labels, weights)
    mi = _mutual_info_from_contingency(contingency)

    hom = mi / entropy_c if entropy_c > 0 else 1.0
    comp = mi / entropy_k if entropy_k > 0 else 1.0

    if hom + comp == 0.0:
        return 0.0, 0.0, 0.0

    v = (1 + beta) * hom * comp / (beta * hom + comp)
    return float(hom), float(comp), float(v)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _weighted_entropy(labels: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Shannon entropy of a label vector."""
    _, inv = np.unique(labels, return_inverse=True)
    counts = np.bincount(inv, weights=weights)
    counts = counts[counts > 0]
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(-np.sum(p * np.log(p)))


def _weighted_contingency(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    weights: np.ndarray,
) -> sp.csr_matrix:
    """Build a weighted contingency matrix (sparse CSR)."""
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    contingency = sp.coo_matrix(
        (weights, (class_idx, cluster_idx)),
        shape=(len(classes), len(clusters)),
        dtype=np.float64,
    ).tocsr()
    contingency.sum_duplicates()
    return contingency


def _mutual_info_from_contingency(contingency: sp.csr_matrix) -> float:
    """Mutual information from a (possibly weighted) contingency matrix."""
    nzx, nzy, nz_val = sp.find(contingency)
    if len(nz_val) == 0:
        return 0.0

    total = contingency.sum()
    if total == 0:
        return 0.0

    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    if pi.size <= 1 or pj.size <= 1:
        return 0.0

    log_nz = np.log(nz_val)
    nz_norm = nz_val / total
    outer = pi.take(nzx) * pj.take(nzy)
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())

    mi = nz_norm * (log_nz - np.log(total)) + nz_norm * log_outer
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return float(np.clip(mi.sum(), 0.0, None))


def _per_event_v_score(
    truth_clusters: pd.DataFrame,
    pred_labels: pd.DataFrame,
) -> Tuple[float, float, float]:
    """Compute energy-weighted V-score per event, then average across events.

    Each cell gets a hard truth label (the ancestor that deposited the most
    energy in that cell) for V-score purposes.
    """
    # Assign dominant truth label per cell
    dominant = (
        truth_clusters.sort_values("energy", ascending=False)
        .groupby(["event_id", "cell_index"], as_index=False)
        .first()[["event_id", "cell_index", "primary_ancestor_id", "total_energy"]]
    )
    merged = dominant.merge(
        pred_labels[["event_id", "cell_index", "cluster_id"]],
        on=["event_id", "cell_index"],
        how="inner",
    )

    if merged.empty:
        return 0.0, 0.0, 0.0

    hom_sum, comp_sum, total_energy = 0.0, 0.0, 0.0
    for _, event_df in merged.groupby("event_id"):
        e_weights = event_df["total_energy"].to_numpy().astype(float)
        event_energy = e_weights.sum()
        if event_energy <= 0:
            continue
        h, c, _ = weighted_v_score(
            event_df["primary_ancestor_id"].to_numpy(),
            event_df["cluster_id"].to_numpy(),
            e_weights,
        )
        hom_sum += h * event_energy
        comp_sum += c * event_energy
        total_energy += event_energy

    if total_energy <= 0:
        return 0.0, 0.0, 0.0

    hom = hom_sum / total_energy
    comp = comp_sum / total_energy
    if hom + comp == 0:
        return 0.0, 0.0, 0.0
    v = 2 * hom * comp / (hom + comp)
    return float(hom), float(comp), float(v)


def _per_event_summary(
    matches: pd.DataFrame,
    overlaps: pd.DataFrame,
    truth_clusters: pd.DataFrame,
    pred_labels: pd.DataFrame,
    match_threshold: float,
    split_threshold: float,
) -> pd.DataFrame:
    """Build a per-event metrics summary table."""
    events = sorted(truth_clusters["event_id"].unique())
    rows = []
    for eid in events:
        m_evt = matches[matches["event_id"] == eid]
        o_evt = overlaps[overlaps["event_id"] == eid]

        eff = energy_weighted_efficiency(m_evt)
        pur = energy_weighted_purity(m_evt)
        e_res = energy_resolution(m_evt)
        sr = splitting_rate(o_evt, threshold=split_threshold)
        mr = merging_rate(o_evt, threshold=split_threshold)
        fr = fake_rate(m_evt)

        rows.append(
            {
                "event_id": eid,
                "efficiency": eff,
                "purity": pur,
                "energy_resolution": e_res,
                "splitting_rate": sr,
                "merging_rate": mr,
                "fake_rate": fr,
            }
        )
    return pd.DataFrame(rows)
