"""Unit tests for clustering metrics (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# sigma_eff
# ---------------------------------------------------------------------------


def test_sigma_eff_gaussian() -> None:
    """sigma_eff of a Gaussian sample should approximate the true sigma."""
    from colliderml.clustering.metrics import sigma_eff

    rng = np.random.default_rng(42)
    true_sigma = 2.0
    values = rng.normal(loc=0, scale=true_sigma, size=100_000)
    result = sigma_eff(values)
    # Should be within 5% of true sigma for large sample
    assert abs(result - true_sigma) / true_sigma < 0.05


def test_sigma_eff_uniform() -> None:
    """For uniform [a, b], minimum 68.3% interval has width 0.683*(b-a), so sigma_eff ~ 0.3415*(b-a)."""
    from colliderml.clustering.metrics import sigma_eff

    values = np.linspace(0, 10, 10001)
    result = sigma_eff(values)
    expected = 0.683 * 10 / 2  # half-width of 68.3% of range
    assert abs(result - expected) / expected < 0.01


def test_sigma_eff_weighted() -> None:
    """Weighted sigma_eff should narrow when weight concentrates on center."""
    from colliderml.clustering.metrics import sigma_eff

    rng = np.random.default_rng(42)
    values = rng.normal(loc=0, scale=2.0, size=10_000)
    uniform_result = sigma_eff(values)
    # Weight center values more heavily
    weights = np.exp(-values**2 / 2)
    weighted_result = sigma_eff(values, weights=weights)
    assert weighted_result < uniform_result


def test_sigma_eff_degenerate() -> None:
    from colliderml.clustering.metrics import sigma_eff

    assert np.isnan(sigma_eff(np.array([1.0])))
    assert sigma_eff(np.array([1.0, 1.0])) == 0.0


# ---------------------------------------------------------------------------
# V-score
# ---------------------------------------------------------------------------


def test_v_score_perfect() -> None:
    from colliderml.clustering.metrics import weighted_v_score

    truth = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    weights = np.ones(6)

    hom, comp, v = weighted_v_score(truth, pred, weights)
    assert abs(v - 1.0) < 1e-10
    assert abs(hom - 1.0) < 1e-10
    assert abs(comp - 1.0) < 1e-10


def test_v_score_random_is_low() -> None:
    from colliderml.clustering.metrics import weighted_v_score

    rng = np.random.default_rng(42)
    n = 1000
    truth = rng.integers(0, 10, size=n)
    pred = rng.integers(0, 10, size=n)
    weights = np.ones(n)

    _, _, v = weighted_v_score(truth, pred, weights)
    assert v < 0.2  # random should be close to 0


def test_v_score_permuted_labels_still_perfect() -> None:
    """V-score is label-agnostic — permuting cluster IDs shouldn't matter."""
    from colliderml.clustering.metrics import weighted_v_score

    truth = np.array([0, 0, 1, 1])
    pred = np.array([5, 5, 9, 9])  # different IDs, same partition
    weights = np.ones(4)

    _, _, v = weighted_v_score(truth, pred, weights)
    assert abs(v - 1.0) < 1e-10


def test_v_score_energy_weighting() -> None:
    """High-energy cells should dominate the V-score."""
    from colliderml.clustering.metrics import weighted_v_score

    # 6 cells: truth has 3 classes, pred swaps the last two (class 1 and 2)
    truth = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 2, 2, 1, 1])  # classes 1 and 2 are swapped — V=1 (partition identical)

    # Now make pred actually wrong: merge class 1 into class 0
    truth2 = np.array([0, 0, 1, 1, 2, 2])
    pred2 = np.array([0, 0, 0, 0, 2, 2])  # class 1 cells assigned to class 0

    # With equal weights, the error matters
    weights_equal = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # With skewed weights where class 1 cells are negligible, the error barely matters
    weights_skewed = np.array([100.0, 100.0, 0.01, 0.01, 100.0, 100.0])

    _, _, v_equal = weighted_v_score(truth2, pred2, weights_equal)
    _, _, v_skewed = weighted_v_score(truth2, pred2, weights_skewed)

    # Skewed weights should give higher V because the "wrong" cells are negligible
    assert v_skewed > v_equal


# ---------------------------------------------------------------------------
# Object-level metrics
# ---------------------------------------------------------------------------


def _make_perfect_matches():
    """All truth matched, no fakes."""
    return pd.DataFrame(
        {
            "event_id": [1, 1],
            "primary_ancestor_id": [10, 20],
            "cluster_id": [0, 1],
            "eiou": [1.0, 1.0],
            "truth_energy": [5.0, 3.0],
            "reco_energy": [5.0, 3.0],
            "overlap_energy": [5.0, 3.0],
            "matched": [True, True],
        }
    )


def _make_one_missed():
    """One truth matched, one missed. One reco is fake."""
    return pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "primary_ancestor_id": [10, 20, np.nan],
            "cluster_id": [0, np.nan, 2],
            "eiou": [0.9, 0.0, 0.0],
            "truth_energy": [5.0, 3.0, 0.0],
            "reco_energy": [5.5, 0.0, 1.0],
            "overlap_energy": [4.5, 0.0, 0.0],
            "matched": [True, False, False],
        }
    )


def test_efficiency_perfect() -> None:
    from colliderml.clustering.metrics import energy_weighted_efficiency

    assert abs(energy_weighted_efficiency(_make_perfect_matches()) - 1.0) < 1e-10


def test_efficiency_one_missed() -> None:
    from colliderml.clustering.metrics import energy_weighted_efficiency

    matches = _make_one_missed()
    eff = energy_weighted_efficiency(matches)
    # 5.0 matched out of 8.0 total truth energy
    assert abs(eff - 5.0 / 8.0) < 1e-10


def test_purity_perfect() -> None:
    from colliderml.clustering.metrics import energy_weighted_purity

    assert abs(energy_weighted_purity(_make_perfect_matches()) - 1.0) < 1e-10


def test_purity_with_fake() -> None:
    from colliderml.clustering.metrics import energy_weighted_purity

    matches = _make_one_missed()
    pur = energy_weighted_purity(matches)
    # 5.5 matched out of 6.5 total reco energy
    assert abs(pur - 5.5 / 6.5) < 1e-6


def test_fake_rate_perfect() -> None:
    from colliderml.clustering.metrics import fake_rate

    assert abs(fake_rate(_make_perfect_matches())) < 1e-10


def test_fake_rate_with_fake() -> None:
    from colliderml.clustering.metrics import fake_rate

    fr = fake_rate(_make_one_missed())
    # 1.0 fake reco energy out of 6.5 total reco
    assert abs(fr - 1.0 / 6.5) < 1e-6


def test_energy_resolution_perfect() -> None:
    from colliderml.clustering.metrics import energy_resolution

    matches = _make_perfect_matches()
    # E_reco/E_truth = 1.0 for both → sigma_eff = 0
    e_res = energy_resolution(matches)
    assert abs(e_res) < 1e-10


def test_splitting_rate_none() -> None:
    """No splitting when each truth maps to exactly one reco."""
    from colliderml.clustering.metrics import splitting_rate

    overlaps = pd.DataFrame(
        {
            "event_id": [1, 1],
            "primary_ancestor_id": [10, 20],
            "cluster_id": [0, 1],
            "overlap_energy": [5.0, 3.0],
            "truth_energy": [5.0, 3.0],
            "reco_energy": [5.0, 3.0],
            "eiou": [1.0, 1.0],
        }
    )
    assert abs(splitting_rate(overlaps)) < 1e-10


def test_splitting_rate_present() -> None:
    """Truth cluster 10 appears in two reco clusters → splitting > 0."""
    from colliderml.clustering.metrics import splitting_rate

    overlaps = pd.DataFrame(
        {
            "event_id": [1, 1],
            "primary_ancestor_id": [10, 10],
            "cluster_id": [0, 1],
            "overlap_energy": [3.0, 2.0],
            "truth_energy": [5.0, 5.0],
            "reco_energy": [3.0, 2.0],
            "eiou": [0.5, 0.4],
        }
    )
    sr = splitting_rate(overlaps, threshold=0.1)
    assert sr > 0.99  # only truth cluster, and it's split


def test_merging_rate_present() -> None:
    """Reco cluster 0 receives from two truth → merging > 0."""
    from colliderml.clustering.metrics import merging_rate

    overlaps = pd.DataFrame(
        {
            "event_id": [1, 1],
            "primary_ancestor_id": [10, 20],
            "cluster_id": [0, 0],
            "overlap_energy": [3.0, 5.0],
            "truth_energy": [3.0, 5.0],
            "reco_energy": [8.0, 8.0],
            "eiou": [0.3, 0.5],
        }
    )
    mr = merging_rate(overlaps, threshold=0.1)
    assert mr > 0.99  # only reco cluster, and it's merged


# ---------------------------------------------------------------------------
# evaluate_clustering end-to-end
# ---------------------------------------------------------------------------


def test_evaluate_clustering_perfect() -> None:
    from colliderml.clustering.metrics import evaluate_clustering

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
    result = evaluate_clustering(truth_clusters, pred_labels)

    assert abs(result["efficiency"] - 1.0) < 1e-6
    assert abs(result["purity"] - 1.0) < 1e-6
    assert abs(result["fake_rate"]) < 1e-6
    assert abs(result["splitting_rate"]) < 1e-6
    assert result["v_score"] > 0.99
    assert result["n_events"] == 1
