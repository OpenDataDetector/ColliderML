"""Tests for the env-var overrides on TrackingTask config.

``COLLIDERML_TRACKING_DATASET`` and ``COLLIDERML_TRACKING_EVAL_RANGE``
let tutorial / small-machine deploys swap the production PU=200 + 10k
eval split for a smaller PU=0 + 1k eval split without touching code.

These tests assert:
- Default values are unchanged when neither env var is set.
- Each env var overrides exactly its corresponding field.
- Malformed values fall back to the default (rather than crashing).
- TrackingSmallModelTask honors the same overrides.
"""

from __future__ import annotations

import pytest

from colliderml.tasks.systems.smallmodel import TrackingSmallModelTask
from colliderml.tasks.tracking.task import TrackingTask, _parse_event_range_env


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_default_dataset_is_ttbar_pu200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLLIDERML_TRACKING_DATASET", raising=False)
    monkeypatch.delenv("COLLIDERML_TRACKING_EVAL_RANGE", raising=False)
    task = TrackingTask()
    assert task.dataset == "ttbar_pu200"
    assert task.eval_event_range == (90_000, 100_000)


def test_smallmodel_inherits_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLLIDERML_TRACKING_DATASET", raising=False)
    monkeypatch.delenv("COLLIDERML_TRACKING_EVAL_RANGE", raising=False)
    task = TrackingSmallModelTask()
    assert task.dataset == "ttbar_pu200"
    assert task.eval_event_range == (90_000, 100_000)


# ---------------------------------------------------------------------------
# Overrides applied
# ---------------------------------------------------------------------------
def test_dataset_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLLIDERML_TRACKING_DATASET", "ttbar_pu0")
    monkeypatch.delenv("COLLIDERML_TRACKING_EVAL_RANGE", raising=False)
    task = TrackingTask()
    assert task.dataset == "ttbar_pu0"
    # Eval range stays at the default.
    assert task.eval_event_range == (90_000, 100_000)


def test_eval_range_env_override_colon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLLIDERML_TRACKING_DATASET", raising=False)
    monkeypatch.setenv("COLLIDERML_TRACKING_EVAL_RANGE", "0:1000")
    task = TrackingTask()
    assert task.eval_event_range == (0, 1000)
    assert task.dataset == "ttbar_pu200"  # untouched


def test_eval_range_env_override_comma(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLLIDERML_TRACKING_EVAL_RANGE", "10,42")
    task = TrackingTask()
    assert task.eval_event_range == (10, 42)


def test_both_overrides_together(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLLIDERML_TRACKING_DATASET", "ttbar_pu0")
    monkeypatch.setenv("COLLIDERML_TRACKING_EVAL_RANGE", "0:1000")
    task = TrackingTask()
    assert task.dataset == "ttbar_pu0"
    assert task.eval_event_range == (0, 1000)


def test_smallmodel_honors_both_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLLIDERML_TRACKING_DATASET", "ttbar_pu0")
    monkeypatch.setenv("COLLIDERML_TRACKING_EVAL_RANGE", "0:500")
    task = TrackingSmallModelTask()
    assert task.dataset == "ttbar_pu0"
    assert task.eval_event_range == (0, 500)


# ---------------------------------------------------------------------------
# Malformed values fall back to the default
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_value",
    [
        "not-a-range",
        "100",          # only one number
        "100:50",       # end <= start
        "100:100",      # empty range
        ":",
        "",             # empty (also covered by the strip-to-empty check)
        "abc:def",
        "-10:10",       # negative start
    ],
)
def test_malformed_eval_range_falls_back(
    monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    monkeypatch.setenv("COLLIDERML_TRACKING_EVAL_RANGE", bad_value)
    task = TrackingTask()
    assert task.eval_event_range == (90_000, 100_000)


def test_parse_helper_accepts_whitespace() -> None:
    assert _parse_event_range_env("  10 : 42 ") == (10, 42)
    assert _parse_event_range_env(" 10 , 42 ") == (10, 42)


# ---------------------------------------------------------------------------
# Class-level defaults survive instantiation
# ---------------------------------------------------------------------------
def test_class_level_defaults_unchanged_after_instance_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An override on one instance must not leak onto the class."""
    monkeypatch.setenv("COLLIDERML_TRACKING_DATASET", "ttbar_pu0")
    monkeypatch.setenv("COLLIDERML_TRACKING_EVAL_RANGE", "0:1000")
    overridden = TrackingTask()
    assert overridden.dataset == "ttbar_pu0"

    # New instance with envs cleared must see the class-level defaults.
    monkeypatch.delenv("COLLIDERML_TRACKING_DATASET")
    monkeypatch.delenv("COLLIDERML_TRACKING_EVAL_RANGE")
    plain = TrackingTask()
    assert plain.dataset == "ttbar_pu200"
    assert plain.eval_event_range == (90_000, 100_000)
    # The class itself wasn't mutated.
    assert TrackingTask.dataset == "ttbar_pu200"
    assert TrackingTask.eval_event_range == (90_000, 100_000)
