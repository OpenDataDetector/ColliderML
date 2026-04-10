"""Tests for :mod:`colliderml.tasks` registry and runner surface.

These tests never touch the network or the filesystem — the registry
API is pure, and ``evaluate()`` is exercised by monkey-patching each
task's ``score`` method.
"""

from __future__ import annotations

from typing import Dict

import pyarrow as pa
import pytest

import colliderml.tasks as tasks_mod
from colliderml.tasks import BenchmarkTask, evaluate, get, list_tasks, register


EXPECTED_TASK_NAMES = {
    "tracking",
    "jets",
    "anomaly",
    "tracking_latency",
    "tracking_small",
    "data_loading",
}


def test_list_tasks_returns_all_registered_tasks() -> None:
    names = list_tasks()
    assert set(names) == EXPECTED_TASK_NAMES
    assert names == sorted(names), "list_tasks() must return a sorted list"


def test_get_returns_fresh_instance_each_call() -> None:
    a = get("tracking")
    b = get("tracking")
    assert isinstance(a, BenchmarkTask)
    assert a is not b, "get() should instantiate a fresh task each call"
    assert type(a) is type(b)


def test_get_unknown_name_lists_available_tasks() -> None:
    with pytest.raises(ValueError) as excinfo:
        get("not-a-task")
    message = str(excinfo.value)
    assert "not-a-task" in message
    # All registered names should appear so users can self-correct.
    for name in EXPECTED_TASK_NAMES:
        assert name in message


def test_register_rejects_duplicate_names() -> None:
    class Dupe(BenchmarkTask):
        name = "tracking"  # already registered
        dataset = "ttbar_pu200"
        eval_event_range = (0, 1)
        inputs = ["tracker_hits"]
        metrics: list = []

        def load_eval_inputs(self) -> Dict[str, pa.Table]:
            return {}

        def validate_predictions(self, preds: pa.Table) -> None:
            pass

        def score(self, preds: pa.Table) -> Dict[str, float]:
            return {}

    with pytest.raises(ValueError, match="already registered"):
        register(Dupe)


def test_register_requires_name_attribute() -> None:
    class Nameless(BenchmarkTask):
        name = ""  # empty string — falsy
        dataset = "ttbar_pu200"
        eval_event_range = (0, 1)
        inputs = ["tracker_hits"]
        metrics: list = []

        def load_eval_inputs(self) -> Dict[str, pa.Table]:
            return {}

        def validate_predictions(self, preds: pa.Table) -> None:
            pass

        def score(self, preds: pa.Table) -> Dict[str, float]:
            return {}

    with pytest.raises(ValueError, match="missing a class-level 'name'"):
        register(Nameless)


def test_evaluate_routes_to_registered_task(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake the validate/score path so we exercise routing without needing
    # real data.
    fake_table = pa.table({"event_id": [0], "hit_id": [0], "track_id": [0]})
    captured: Dict[str, object] = {}

    def fake_validate(self: BenchmarkTask, preds: pa.Table) -> None:
        captured["validated"] = True

    def fake_score(self: BenchmarkTask, preds: pa.Table) -> Dict[str, float]:
        captured["scored"] = True
        return {"trackml_eff": 0.42}

    from colliderml.tasks.tracking.task import TrackingTask

    monkeypatch.setattr(TrackingTask, "validate_predictions", fake_validate)
    monkeypatch.setattr(TrackingTask, "score", fake_score)

    result = evaluate("tracking", fake_table)
    assert result == {"trackml_eff": 0.42}
    assert captured == {"validated": True, "scored": True}


def test_evaluate_rejects_unsupported_input_type() -> None:
    with pytest.raises(TypeError, match="predictions must be"):
        evaluate("tracking", 42)


def test_is_better_respects_direction() -> None:
    task = get("tracking")
    assert task.is_better("trackml_eff", new=0.8, current=0.7)
    assert not task.is_better("trackml_eff", new=0.6, current=0.7)
    assert task.is_better("fake_rate", new=0.1, current=0.2)  # lower is better
    assert not task.is_better("fake_rate", new=0.3, current=0.2)
