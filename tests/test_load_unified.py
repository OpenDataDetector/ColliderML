"""Tests for the top-level :func:`colliderml.load` convenience."""

from __future__ import annotations

from typing import Dict, List

import polars as pl
import pytest

import colliderml
from colliderml import _load as load_mod


@pytest.fixture
def stub_loader(monkeypatch: pytest.MonkeyPatch):
    """Replace the core loader and cache-ensurance so tests are pure-Python."""
    calls: Dict[str, object] = {}

    def fake_ensure(channel: str, pileup: str, tables: List[str], *, dataset_id: str, revision=None) -> None:
        calls["ensure"] = (channel, pileup, list(tables), dataset_id, revision)

    stub_frame = pl.DataFrame(
        {
            "event_id": list(range(10)),
            "hit_id": list(range(10)),
        }
    )

    def fake_load_tables(cfg) -> Dict[str, pl.DataFrame]:
        calls["cfg"] = cfg
        return {"tracker_hits": stub_frame.clone(), "particles": stub_frame.clone()}

    monkeypatch.setattr(load_mod, "_ensure_local", fake_ensure)
    monkeypatch.setattr(load_mod, "_core_load_tables", fake_load_tables)
    return calls


def test_load_default_tables(stub_loader) -> None:
    frames = colliderml.load("ttbar_pu0")
    # Both stub tables should be returned.
    assert set(frames.keys()) == {"tracker_hits", "particles"}
    # The internal cache pre-flight should have asked for the default object set.
    ensure = stub_loader["ensure"]
    assert ensure[0] == "ttbar"
    assert ensure[1] == "pu0"
    assert ensure[2] == colliderml.DEFAULT_OBJECT_TABLES


def test_load_custom_tables(stub_loader) -> None:
    frames = colliderml.load("higgs_portal_pu200", tables=["tracker_hits"])
    assert "tracker_hits" in frames
    assert stub_loader["ensure"][0] == "higgs_portal"
    assert stub_loader["ensure"][1] == "pu200"
    assert stub_loader["ensure"][2] == ["tracker_hits"]


def test_load_event_range_filters(stub_loader) -> None:
    frames = colliderml.load("ttbar_pu0", tables=["tracker_hits"], event_range=(2, 7))
    # stub returns events 0..9; we keep 2..7 → 5 rows.
    assert frames["tracker_hits"].height == 5
    kept = set(frames["tracker_hits"].get_column("event_id").to_list())
    assert kept == {2, 3, 4, 5, 6}


def test_load_max_events_trims(stub_loader) -> None:
    frames = colliderml.load("ttbar_pu0", tables=["tracker_hits"], max_events=3)
    assert frames["tracker_hits"].height == 3


def test_load_max_events_applied_after_event_range(stub_loader) -> None:
    frames = colliderml.load(
        "ttbar_pu0",
        tables=["tracker_hits"],
        event_range=(0, 8),
        max_events=4,
    )
    # event_range narrows to 8 rows, max_events trims to 4.
    assert frames["tracker_hits"].height == 4


def test_load_invalid_dataset_name_raises() -> None:
    with pytest.raises(ValueError, match="Cannot parse dataset name"):
        colliderml.load("ttbar")


def test_load_event_range_end_before_start_raises(stub_loader) -> None:
    with pytest.raises(ValueError, match="end .* must be >= start"):
        colliderml.load("ttbar_pu0", tables=["tracker_hits"], event_range=(10, 5))


def test_load_auto_download_false_skips_ensure(
    monkeypatch: pytest.MonkeyPatch, stub_loader
) -> None:
    calls = stub_loader

    def fake_ensure_that_shouldnt_run(*args, **kwargs) -> None:
        raise AssertionError("auto_download=False should skip _ensure_local")

    monkeypatch.setattr(load_mod, "_ensure_local", fake_ensure_that_shouldnt_run)
    frames = colliderml.load("ttbar_pu0", tables=["tracker_hits"], auto_download=False)
    assert "tracker_hits" in frames


def test_top_level_exports_present() -> None:
    # The v0.4.0 public contract — regression guard for accidental removal.
    assert hasattr(colliderml, "load")
    assert hasattr(colliderml, "simulate")     # resolved via __getattr__
    assert hasattr(colliderml, "balance")      # resolved via __getattr__
    assert hasattr(colliderml, "tasks")        # resolved via __getattr__
    assert hasattr(colliderml, "remote")       # resolved via __getattr__
    assert colliderml.__version__.startswith("0.4.")


def test_top_level_getattr_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="no attribute"):
        colliderml.this_does_not_exist  # noqa: B018
