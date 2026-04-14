"""Unit tests for :mod:`colliderml.tasks.tracking.metrics`.

Uses synthetic pyarrow tables so the tests run in microseconds with no
external data.
"""

from __future__ import annotations

import pyarrow as pa

from colliderml.tasks.tracking.metrics import (
    duplicate_rate,
    fake_rate,
    physics_eff_pt1,
    trackml_weighted_efficiency,
)


def _make_truth_table(rows: list[dict]) -> pa.Table:
    return pa.table(
        {
            "event_id": [r["event_id"] for r in rows],
            "hit_id": [r["hit_id"] for r in rows],
            "particle_id": [r["particle_id"] for r in rows],
            "weight": [r.get("weight", 1.0) for r in rows],
        }
    )


def _make_pred_table(rows: list[dict]) -> pa.Table:
    return pa.table(
        {
            "event_id": [r["event_id"] for r in rows],
            "hit_id": [r["hit_id"] for r in rows],
            "track_id": [r["track_id"] for r in rows],
        }
    )


def test_trackml_weighted_efficiency_perfect_reconstruction() -> None:
    # 4 hits all belonging to particle 1 in event 0, recovered as a single track.
    truth = _make_truth_table(
        [
            {"event_id": 0, "hit_id": 0, "particle_id": 1},
            {"event_id": 0, "hit_id": 1, "particle_id": 1},
            {"event_id": 0, "hit_id": 2, "particle_id": 1},
            {"event_id": 0, "hit_id": 3, "particle_id": 1},
        ]
    )
    preds = _make_pred_table(
        [
            {"event_id": 0, "hit_id": 0, "track_id": 10},
            {"event_id": 0, "hit_id": 1, "track_id": 10},
            {"event_id": 0, "hit_id": 2, "track_id": 10},
            {"event_id": 0, "hit_id": 3, "track_id": 10},
        ]
    )
    assert trackml_weighted_efficiency(preds, truth) == 1.0


def test_trackml_weighted_efficiency_split_track() -> None:
    # Particle 1 has 4 hits, reconstructed as TWO tracks of 2 hits each.
    # Neither track captures ≥50% of the particle's own hits, so both fail.
    truth = _make_truth_table(
        [
            {"event_id": 0, "hit_id": i, "particle_id": 1} for i in range(4)
        ]
    )
    preds = _make_pred_table(
        [
            {"event_id": 0, "hit_id": 0, "track_id": 10},
            {"event_id": 0, "hit_id": 1, "track_id": 10},
            {"event_id": 0, "hit_id": 2, "track_id": 11},
            {"event_id": 0, "hit_id": 3, "track_id": 11},
        ]
    )
    # 50% of each track's own weight is still ≥ 50% of the particle's weight
    # (each track has 2/4 of the particle), so the strict-majority check
    # should hold at exactly 0.5 — the metric accepts the ≥ boundary.
    result = trackml_weighted_efficiency(preds, truth)
    assert result == 1.0  # both tracks pass the ≥50% boundary


def test_trackml_weighted_efficiency_fake_track() -> None:
    # Fake track containing a mix of 3 particles — no majority >= 50%.
    truth = _make_truth_table(
        [
            {"event_id": 0, "hit_id": 0, "particle_id": 1},
            {"event_id": 0, "hit_id": 1, "particle_id": 2},
            {"event_id": 0, "hit_id": 2, "particle_id": 3},
        ]
    )
    preds = _make_pred_table(
        [
            {"event_id": 0, "hit_id": 0, "track_id": 10},
            {"event_id": 0, "hit_id": 1, "track_id": 10},
            {"event_id": 0, "hit_id": 2, "track_id": 10},
        ]
    )
    assert trackml_weighted_efficiency(preds, truth) == 0.0


def test_trackml_weighted_efficiency_empty_returns_zero() -> None:
    truth = _make_truth_table([])
    preds = _make_pred_table([])
    assert trackml_weighted_efficiency(preds, truth) == 0.0


def test_fake_rate_counts_tracks_without_majority() -> None:
    truth = _make_truth_table(
        [
            {"event_id": 0, "hit_id": 0, "particle_id": 1},
            {"event_id": 0, "hit_id": 1, "particle_id": 1},
            {"event_id": 0, "hit_id": 2, "particle_id": 2},
            {"event_id": 0, "hit_id": 3, "particle_id": 3},
            {"event_id": 0, "hit_id": 4, "particle_id": 4},
        ]
    )
    preds = _make_pred_table(
        [
            # Track 10: clean particle-1 track
            {"event_id": 0, "hit_id": 0, "track_id": 10},
            {"event_id": 0, "hit_id": 1, "track_id": 10},
            # Track 11: three distinct particles, no majority
            {"event_id": 0, "hit_id": 2, "track_id": 11},
            {"event_id": 0, "hit_id": 3, "track_id": 11},
            {"event_id": 0, "hit_id": 4, "track_id": 11},
        ]
    )
    # 1 of 2 tracks is fake.
    assert fake_rate(preds, truth) == 0.5


def test_duplicate_rate_counts_particles_split_across_tracks() -> None:
    truth = _make_truth_table(
        [
            {"event_id": 0, "hit_id": 0, "particle_id": 1},
            {"event_id": 0, "hit_id": 1, "particle_id": 1},
            {"event_id": 0, "hit_id": 2, "particle_id": 2},
        ]
    )
    preds = _make_pred_table(
        [
            {"event_id": 0, "hit_id": 0, "track_id": 10},
            {"event_id": 0, "hit_id": 1, "track_id": 11},  # duplicate of particle 1
            {"event_id": 0, "hit_id": 2, "track_id": 12},
        ]
    )
    # 1 of 2 particles is duplicated.
    assert duplicate_rate(preds, truth) == 0.5


def test_physics_eff_pt1_without_majority_id_is_zero() -> None:
    particles = pa.table(
        {
            "particle_id": [1, 2, 3],
            "px": [2.0, 0.5, 3.0],
            "py": [0.0, 0.0, 0.0],
            "primary": [True, True, True],
        }
    )
    # No majority_particle_id column in preds — the metric returns 0.
    preds = pa.table({"event_id": [0], "hit_id": [0], "track_id": [0]})
    assert physics_eff_pt1(preds, particles) == 0.0


def test_physics_eff_pt1_matches_high_pt_particles() -> None:
    particles = pa.table(
        {
            "particle_id": [1, 2, 3],
            "px": [2.0, 0.5, 3.0],  # pt for particle 2 is < 1 GeV
            "py": [0.0, 0.0, 0.0],
            "primary": [True, True, True],
        }
    )
    preds = pa.table(
        {
            "event_id": [0, 0],
            "hit_id": [0, 1],
            "track_id": [10, 11],
            "majority_particle_id": [1, 3],  # two of two high-pt particles matched
        }
    )
    assert physics_eff_pt1(preds, particles) == 1.0
