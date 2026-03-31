"""Overlap matrix, Energy-weighted IoU, and greedy matching for cluster evaluation.

Given soft truth clusters (from :func:`build_truth_clusters`) and predicted
cluster labels, this module computes:

* The energy overlap matrix O[truth, reco] — how much energy from each truth
  cluster ended up in each predicted cluster.
* EIOU (Energy-weighted Intersection over Union) for every (truth, reco) pair.
* A greedy one-to-one matching that pairs truth and reco clusters by
  descending EIOU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_overlap_matrix(
    truth_clusters: pd.DataFrame,
    pred_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Build the energy overlap matrix between truth and predicted clusters.

    Args:
        truth_clusters: From :func:`build_truth_clusters`.  Required columns:
            ``event_id``, ``cell_index``, ``primary_ancestor_id``, ``energy``.
        pred_labels: Predicted cluster assignments.  Required columns:
            ``event_id``, ``cell_index``, ``cluster_id``.

    Returns:
        DataFrame with one row per non-zero (event, truth, reco) overlap::

            event_id, primary_ancestor_id, cluster_id,
            overlap_energy, truth_energy, reco_energy, eiou
    """
    # ------------------------------------------------------------------
    # 1. Join truth contributions with predicted labels on (event, cell)
    # ------------------------------------------------------------------
    merged = truth_clusters[["event_id", "cell_index", "primary_ancestor_id", "energy"]].merge(
        pred_labels[["event_id", "cell_index", "cluster_id"]],
        on=["event_id", "cell_index"],
        how="inner",
    )

    # ------------------------------------------------------------------
    # 2. Sum energy per (event, ancestor, cluster) → overlap matrix
    # ------------------------------------------------------------------
    overlap = (
        merged.groupby(["event_id", "primary_ancestor_id", "cluster_id"], as_index=False)["energy"]
        .sum()
        .rename(columns={"energy": "overlap_energy"})
    )

    # ------------------------------------------------------------------
    # 3. Compute marginal energies
    # ------------------------------------------------------------------
    truth_energy = (
        overlap.groupby(["event_id", "primary_ancestor_id"], as_index=False)["overlap_energy"]
        .sum()
        .rename(columns={"overlap_energy": "truth_energy"})
    )
    reco_energy = (
        overlap.groupby(["event_id", "cluster_id"], as_index=False)["overlap_energy"]
        .sum()
        .rename(columns={"overlap_energy": "reco_energy"})
    )

    overlap = overlap.merge(truth_energy, on=["event_id", "primary_ancestor_id"], how="left")
    overlap = overlap.merge(reco_energy, on=["event_id", "cluster_id"], how="left")

    # ------------------------------------------------------------------
    # 4. EIOU = overlap / (truth + reco - overlap)
    # ------------------------------------------------------------------
    denom = overlap["truth_energy"] + overlap["reco_energy"] - overlap["overlap_energy"]
    overlap["eiou"] = np.where(denom > 0, overlap["overlap_energy"] / denom, 0.0)

    return overlap


def greedy_match(
    overlaps: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Greedy one-to-one matching of truth ↔ reco clusters by EIOU descending.

    For each event, pairs are sorted by EIOU.  Walking from highest to lowest,
    a pair is matched if neither side has been claimed yet and EIOU ≥ threshold.

    Args:
        overlaps: From :func:`build_overlap_matrix`.  Required columns:
            ``event_id``, ``primary_ancestor_id``, ``cluster_id``, ``eiou``,
            ``truth_energy``, ``reco_energy``, ``overlap_energy``.
        threshold: Minimum EIOU to accept a match (default 0.5).

    Returns:
        DataFrame with every truth and reco cluster appearing at most once::

            event_id, primary_ancestor_id, cluster_id, eiou,
            truth_energy, reco_energy, overlap_energy, matched (bool)

        Unmatched truth clusters have ``cluster_id = NaN, matched = False``.
        Unmatched reco clusters have ``primary_ancestor_id = NaN, matched = False``.
    """
    matched_rows = []

    for event_id, event_df in overlaps.groupby("event_id"):
        # Sort candidate pairs by EIOU descending
        candidates = event_df.sort_values("eiou", ascending=False)

        matched_truth: set = set()
        matched_reco: set = set()

        for _, row in candidates.iterrows():
            tid = row["primary_ancestor_id"]
            rid = row["cluster_id"]
            if tid in matched_truth or rid in matched_reco:
                continue
            if row["eiou"] < threshold:
                break  # sorted descending, so no more above threshold
            matched_truth.add(tid)
            matched_reco.add(rid)
            matched_rows.append(
                {
                    "event_id": event_id,
                    "primary_ancestor_id": tid,
                    "cluster_id": rid,
                    "eiou": row["eiou"],
                    "truth_energy": row["truth_energy"],
                    "reco_energy": row["reco_energy"],
                    "overlap_energy": row["overlap_energy"],
                    "matched": True,
                }
            )

        # Unmatched truth clusters
        all_truth = event_df.groupby("primary_ancestor_id", as_index=False)["truth_energy"].first()
        for _, row in all_truth.iterrows():
            if row["primary_ancestor_id"] not in matched_truth:
                matched_rows.append(
                    {
                        "event_id": event_id,
                        "primary_ancestor_id": row["primary_ancestor_id"],
                        "cluster_id": np.nan,
                        "eiou": 0.0,
                        "truth_energy": row["truth_energy"],
                        "reco_energy": 0.0,
                        "overlap_energy": 0.0,
                        "matched": False,
                    }
                )

        # Unmatched reco clusters
        all_reco = event_df.groupby("cluster_id", as_index=False)["reco_energy"].first()
        for _, row in all_reco.iterrows():
            if row["cluster_id"] not in matched_reco:
                matched_rows.append(
                    {
                        "event_id": event_id,
                        "primary_ancestor_id": np.nan,
                        "cluster_id": row["cluster_id"],
                        "eiou": 0.0,
                        "truth_energy": 0.0,
                        "reco_energy": row["reco_energy"],
                        "overlap_energy": 0.0,
                        "matched": False,
                    }
                )

    if not matched_rows:
        return pd.DataFrame(
            columns=[
                "event_id", "primary_ancestor_id", "cluster_id", "eiou",
                "truth_energy", "reco_energy", "overlap_energy", "matched",
            ]
        )

    return pd.DataFrame(matched_rows)
