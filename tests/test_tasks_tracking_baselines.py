"""Unit tests for :mod:`colliderml.tasks.tracking.baselines.oracle`."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from colliderml.tasks.tracking.baselines.oracle import (
    noised_oracle_predictions,
    perfect_oracle_predictions,
)
from colliderml.tasks.tracking.metrics import (
    duplicate_rate,
    fake_rate,
    trackml_weighted_efficiency,
)


def _make_truth_hits(n_events: int = 2, hits_per_track: int = 8, n_tracks: int = 5) -> pa.Table:
    """Synthetic truth: ``n_events`` events, each with ``n_tracks`` tracks of ``hits_per_track`` hits."""
    events, hits, pids = [], [], []
    next_hit = 0
    for e in range(n_events):
        for p in range(n_tracks):
            for _ in range(hits_per_track):
                events.append(e)
                hits.append(next_hit)
                pids.append(p + 1)  # particle_id > 0
                next_hit += 1
    return pa.table(
        {"event_id": pa.array(events), "hit_id": pa.array(hits), "particle_id": pa.array(pids)}
    )


def test_perfect_oracle_scores_one() -> None:
    truth = _make_truth_hits()
    preds = perfect_oracle_predictions(truth)
    assert preds.column_names == ["event_id", "hit_id", "track_id"]
    assert trackml_weighted_efficiency(preds, truth) == pytest.approx(1.0)
    assert fake_rate(preds, truth) == pytest.approx(0.0)
    assert duplicate_rate(preds, truth) == pytest.approx(0.0)


def test_noised_zero_equals_perfect() -> None:
    truth = _make_truth_hits()
    preds = noised_oracle_predictions(truth, split_fraction=0.0, merge_fraction=0.0, seed=0)
    # Same rows, same track_id assignment as the perfect oracle.
    assert preds.column("track_id").to_pylist() == truth.column("particle_id").to_pylist()
    assert trackml_weighted_efficiency(preds, truth) == pytest.approx(1.0)


def test_splits_only_degrade_efficiency() -> None:
    truth = _make_truth_hits(n_events=3, hits_per_track=10, n_tracks=6)
    preds = noised_oracle_predictions(truth, split_fraction=0.5, merge_fraction=0.0, seed=42)
    eff = trackml_weighted_efficiency(preds, truth)
    # Every split breaks exactly one particle's coverage, so efficiency must drop strictly.
    assert 0.0 <= eff < 1.0


def test_merges_only_degrade_efficiency() -> None:
    truth = _make_truth_hits(n_events=3, hits_per_track=10, n_tracks=6)
    preds = noised_oracle_predictions(truth, split_fraction=0.0, merge_fraction=0.5, seed=42)
    eff = trackml_weighted_efficiency(preds, truth)
    assert 0.0 <= eff < 1.0


def test_more_noise_means_lower_efficiency() -> None:
    truth = _make_truth_hits(n_events=4, hits_per_track=10, n_tracks=8)
    low_noise = trackml_weighted_efficiency(
        noised_oracle_predictions(truth, split_fraction=0.1, merge_fraction=0.1, seed=1), truth
    )
    high_noise = trackml_weighted_efficiency(
        noised_oracle_predictions(truth, split_fraction=0.5, merge_fraction=0.5, seed=1), truth
    )
    assert high_noise < low_noise


def test_seeded_noise_is_deterministic() -> None:
    truth = _make_truth_hits()
    a = noised_oracle_predictions(truth, split_fraction=0.3, merge_fraction=0.3, seed=7)
    b = noised_oracle_predictions(truth, split_fraction=0.3, merge_fraction=0.3, seed=7)
    assert a.column("track_id").to_pylist() == b.column("track_id").to_pylist()


def test_invalid_fractions_raise() -> None:
    truth = _make_truth_hits()
    with pytest.raises(ValueError):
        noised_oracle_predictions(truth, split_fraction=1.5)
    with pytest.raises(ValueError):
        noised_oracle_predictions(truth, merge_fraction=-0.1)


def test_majority_particle_id_accepted() -> None:
    """Truth tables may use ``majority_particle_id`` instead of ``particle_id``."""
    truth = _make_truth_hits()
    renamed = truth.rename_columns(["event_id", "hit_id", "majority_particle_id"])
    preds = perfect_oracle_predictions(renamed)
    assert trackml_weighted_efficiency(preds, renamed) == pytest.approx(1.0)
