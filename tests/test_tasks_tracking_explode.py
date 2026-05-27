"""End-to-end tests for tracking task's list-per-event → flat path.

The bug these tests guard against: ``tracking.load_eval_inputs()`` and
``tracking._load_truth()`` used to return list-per-event tables straight
from the loader, but every downstream consumer (metrics, baselines, the
prediction submission contract) expects a flat row-per-hit table with an
explicit ``hit_id`` column. ``tracking.score()`` was crashing with
``KeyError: 'hit_id'`` on real eval data.

These tests exercise the explosion path with synthetic list-per-event
tables so the regression cannot reappear silently.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import polars as pl
import pyarrow as pa
import pytest

import colliderml
from colliderml import tasks as colliderml_tasks
from colliderml.tasks.tracking.baselines import perfect_oracle_predictions
from colliderml.tasks.tracking.task import TrackingTask


# ---------------------------------------------------------------------------
# Fixture: synthetic list-per-event tracker_hits + particles, the same
# shape colliderml.load() returns from disk.
# ---------------------------------------------------------------------------
def _list_per_event_hits(n_events: int = 3, hits_per_event: int = 4) -> pa.Table:
    """Build a list-per-event tracker_hits table (rows = events)."""
    rows = []
    for ev in range(n_events):
        rows.append(
            {
                "event_id": ev,
                "x": [float(ev * 10 + i) for i in range(hits_per_event)],
                "y": [float(-(ev * 10 + i)) for i in range(hits_per_event)],
                "z": [0.0] * hits_per_event,
                "particle_id": [ev * 100 + (i // 2) for i in range(hits_per_event)],
                "layer_id": list(range(hits_per_event)),
            }
        )
    return pl.DataFrame(rows).to_arrow()


def _list_per_event_particles(n_events: int = 3, parts_per_event: int = 2) -> pa.Table:
    rows = []
    for ev in range(n_events):
        rows.append(
            {
                "event_id": ev,
                "particle_id": [ev * 100 + i for i in range(parts_per_event)],
                "px": [2.0] * parts_per_event,
                "py": [0.5] * parts_per_event,
                "pz": [0.0] * parts_per_event,
                "primary": [True] * parts_per_event,
            }
        )
    return pl.DataFrame(rows).to_arrow()


@pytest.fixture
def patched_loader(monkeypatch: pytest.MonkeyPatch):
    """Make tracking.load(...) return synthetic list-per-event tables."""
    fake_hits = _list_per_event_hits()
    fake_particles = _list_per_event_particles()

    def fake_load_task_data(
        dataset: str,
        *,
        tables: List[str],
        max_events: Optional[int] = None,
        event_range: Optional[Tuple[int, int]] = None,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, pa.Table]:
        out: Dict[str, pa.Table] = {}
        if "tracker_hits" in tables:
            out["tracker_hits"] = fake_hits
        if "particles" in tables:
            out["particles"] = fake_particles
        return out

    # Patch where BenchmarkTask.load looks it up.
    from colliderml.tasks import _loading as _tasks_loading

    monkeypatch.setattr(_tasks_loading, "load_task_data", fake_load_task_data)
    # _base.load() does a deferred import of load_task_data inside the
    # function body, so also patch the symbol at the import site.
    from colliderml.tasks import _base as _tasks_base

    if hasattr(_tasks_base, "load_task_data"):
        monkeypatch.setattr(_tasks_base, "load_task_data", fake_load_task_data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_load_truth_hits_is_flat_with_hit_id(patched_loader) -> None:
    task = TrackingTask()
    truth = task.load_truth_hits(event_range=(0, 3))

    assert "hit_id" in truth.column_names
    assert "event_id" in truth.column_names
    # 3 events * 4 hits/event = 12 rows
    assert truth.num_rows == 12

    # hit_id should be the per-event row index (0..3 within each event).
    rows = truth.to_pylist()
    by_event: Dict[int, List[int]] = {}
    for r in rows:
        by_event.setdefault(int(r["event_id"]), []).append(int(r["hit_id"]))
    for ev, ids in by_event.items():
        assert ids == [0, 1, 2, 3], f"event {ev} hit_ids = {ids}"


def test_load_eval_inputs_returns_flat_tracker_hits(patched_loader) -> None:
    task = TrackingTask()
    inputs = task.load_eval_inputs()

    assert "tracker_hits" in inputs
    hits = inputs["tracker_hits"]
    # The bug: this used to be 3 rows (one per event). Must be 12 (one per hit).
    assert hits.num_rows == 12
    assert "hit_id" in hits.column_names


def test_score_runs_end_to_end_on_perfect_oracle(patched_loader) -> None:
    """The whole point: score() must succeed on real-shape data."""
    task = TrackingTask()
    truth = task.load_truth_hits(event_range=(0, 3))
    preds = perfect_oracle_predictions(truth)

    # Before the fix, this call raised KeyError: 'hit_id'.
    scores = task.score(preds)
    assert scores["trackml_eff"] == pytest.approx(1.0)
    assert scores["fake_rate"] == pytest.approx(0.0)
    assert scores["dup_rate"] == pytest.approx(0.0)


def test_event_id_preserved_in_explosion(patched_loader) -> None:
    task = TrackingTask()
    truth = task.load_truth_hits(event_range=(0, 3))
    event_ids = sorted(set(int(e) for e in truth.column("event_id").to_pylist()))
    assert event_ids == [0, 1, 2]


def test_load_truth_hits_default_range_uses_eval_range(patched_loader) -> None:
    task = TrackingTask()
    # No event_range passed → falls through to the eval range; the fake
    # loader returns the same data regardless, but the call should not
    # raise and must still produce a flat table.
    truth = task.load_truth_hits()
    assert truth.num_rows == 12
    assert "hit_id" in truth.column_names
