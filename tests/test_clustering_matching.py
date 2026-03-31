"""Unit tests for clustering overlap matrix, EIOU, and greedy matching (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_perfect_case():
    """Two truth clusters perfectly captured by two reco clusters."""
    truth_clusters = pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "cell_index": [0, 1, 2],
            "primary_ancestor_id": [10, 10, 20],
            "energy": [3.0, 2.0, 5.0],
            "total_energy": [3.0, 2.0, 5.0],
        }
    )
    pred_labels = pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "cell_index": [0, 1, 2],
            "cluster_id": [0, 0, 1],
        }
    )
    return truth_clusters, pred_labels


def _make_merged_case():
    """Two truth clusters both assigned to one reco cluster."""
    truth_clusters = pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "cell_index": [0, 1, 2],
            "primary_ancestor_id": [10, 10, 20],
            "energy": [3.0, 2.0, 5.0],
            "total_energy": [3.0, 2.0, 5.0],
        }
    )
    pred_labels = pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "cell_index": [0, 1, 2],
            "cluster_id": [0, 0, 0],  # all in one cluster
        }
    )
    return truth_clusters, pred_labels


def _make_split_case():
    """One truth cluster split across two reco clusters."""
    truth_clusters = pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "cell_index": [0, 1, 2],
            "primary_ancestor_id": [10, 10, 10],
            "energy": [3.0, 2.0, 5.0],
            "total_energy": [3.0, 2.0, 5.0],
        }
    )
    pred_labels = pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "cell_index": [0, 1, 2],
            "cluster_id": [0, 0, 1],  # cell 2 in different cluster
        }
    )
    return truth_clusters, pred_labels


def test_overlap_matrix_perfect() -> None:
    from colliderml.clustering.matching import build_overlap_matrix

    truth, pred = _make_perfect_case()
    overlaps = build_overlap_matrix(truth, pred)

    assert len(overlaps) == 2  # two (truth, reco) pairs
    # Truth 10 → Reco 0: overlap = 5.0, truth = 5.0, reco = 5.0 → EIOU = 1.0
    row_10 = overlaps[overlaps["primary_ancestor_id"] == 10]
    assert len(row_10) == 1
    assert abs(row_10.iloc[0]["eiou"] - 1.0) < 1e-6

    # Truth 20 → Reco 1: EIOU = 1.0
    row_20 = overlaps[overlaps["primary_ancestor_id"] == 20]
    assert abs(row_20.iloc[0]["eiou"] - 1.0) < 1e-6


def test_overlap_matrix_merged() -> None:
    from colliderml.clustering.matching import build_overlap_matrix

    truth, pred = _make_merged_case()
    overlaps = build_overlap_matrix(truth, pred)

    # Both truth clusters map to reco cluster 0
    assert len(overlaps) == 2
    for _, row in overlaps.iterrows():
        assert row["cluster_id"] == 0
        # EIOU < 1 because reco_energy = 10 but truth_energy = 5 for each
        assert row["eiou"] < 1.0


def test_overlap_matrix_split() -> None:
    from colliderml.clustering.matching import build_overlap_matrix

    truth, pred = _make_split_case()
    overlaps = build_overlap_matrix(truth, pred)

    # One truth cluster, two reco clusters
    assert len(overlaps) == 2
    assert overlaps["primary_ancestor_id"].nunique() == 1
    assert overlaps["cluster_id"].nunique() == 2


def test_greedy_match_perfect() -> None:
    from colliderml.clustering.matching import build_overlap_matrix, greedy_match

    truth, pred = _make_perfect_case()
    overlaps = build_overlap_matrix(truth, pred)
    matches = greedy_match(overlaps, threshold=0.5)

    matched = matches[matches["matched"]]
    assert len(matched) == 2
    assert set(matched["primary_ancestor_id"]) == {10, 20}


def test_greedy_match_merged_partial() -> None:
    from colliderml.clustering.matching import build_overlap_matrix, greedy_match

    truth, pred = _make_merged_case()
    overlaps = build_overlap_matrix(truth, pred)
    matches = greedy_match(overlaps, threshold=0.5)

    # Only one truth can match the single reco cluster (both EIOU = 0.5)
    matched = matches[matches["matched"]]
    assert len(matched) == 1
    assert matched.iloc[0]["primary_ancestor_id"] in (10, 20)


def test_greedy_match_unmatched_truth() -> None:
    from colliderml.clustering.matching import build_overlap_matrix, greedy_match

    truth, pred = _make_merged_case()
    overlaps = build_overlap_matrix(truth, pred)
    matches = greedy_match(overlaps, threshold=0.5)

    unmatched_truth = matches[~matches["matched"] & matches["truth_energy"].gt(0)]
    assert len(unmatched_truth) == 1


def test_greedy_match_high_threshold_no_matches() -> None:
    from colliderml.clustering.matching import build_overlap_matrix, greedy_match

    truth, pred = _make_merged_case()
    overlaps = build_overlap_matrix(truth, pred)
    matches = greedy_match(overlaps, threshold=0.99)

    matched = matches[matches["matched"]]
    assert len(matched) == 0


def test_greedy_match_ordering() -> None:
    """Higher EIOU pairs should be matched first."""
    from colliderml.clustering.matching import build_overlap_matrix, greedy_match

    truth, pred = _make_perfect_case()
    overlaps = build_overlap_matrix(truth, pred)
    matches = greedy_match(overlaps, threshold=0.0)

    # All should be matched with EIOU = 1.0
    matched = matches[matches["matched"]]
    assert len(matched) == 2
    assert all(matched["eiou"] > 0.99)
