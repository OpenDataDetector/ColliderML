"""Unit tests for :mod:`colliderml.tasks._loading`.

Covers the dataset-name parser and the shape of :func:`load_task_data`'s
output via a monkey-patched core loader (so we don't touch disk or the
network).
"""

from __future__ import annotations

from typing import Dict

import polars as pl
import pytest

from colliderml.tasks import _loading
from colliderml.tasks._loading import load_task_data, parse_dataset_name


class TestParseDatasetName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("ttbar_pu0", ("ttbar", "pu0")),
            ("ttbar_pu200", ("ttbar", "pu200")),
            ("higgs_portal_pu10", ("higgs_portal", "pu10")),
            ("susy_gmsb_pu0", ("susy_gmsb", "pu0")),
            ("hidden_valley_pu200", ("hidden_valley", "pu200")),
            ("single_muon_pu0", ("single_muon", "pu0")),
        ],
    )
    def test_valid_names(self, name: str, expected: tuple) -> None:
        assert parse_dataset_name(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "ttbar",  # no pileup segment
            "ttbar_pileup200",  # wrong prefix
            "_pu0",  # empty channel
            "ttbar_pu",  # no digit
            "",
        ],
    )
    def test_invalid_names_raise(self, name: str) -> None:
        with pytest.raises(ValueError, match="Cannot parse dataset name"):
            parse_dataset_name(name)


def test_load_task_data_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the cache-ensurance and loader so we exercise only the
    # translation logic in load_task_data itself.
    calls: Dict[str, object] = {}

    def fake_ensure(channel: str, pileup: str, tables, *, dataset_id: str) -> None:
        calls["ensure"] = (channel, pileup, list(tables), dataset_id)

    stub_frame = pl.DataFrame(
        {
            "event_id": [0, 1, 2, 3, 4],
            "hit_id": [0, 1, 2, 3, 4],
        }
    )

    def fake_load_tables(cfg) -> Dict[str, pl.DataFrame]:
        calls["cfg"] = cfg
        return {"tracker_hits": stub_frame}

    monkeypatch.setattr(_loading, "_ensure_downloaded", fake_ensure)
    monkeypatch.setattr(_loading, "load_tables", fake_load_tables)

    tables = load_task_data(
        "ttbar_pu200",
        tables=["tracker_hits"],
        event_range=(1, 4),
    )

    assert "tracker_hits" in tables
    assert tables["tracker_hits"].num_rows == 3  # events 1, 2, 3
    assert tables["tracker_hits"].column_names == ["event_id", "hit_id"]
    # Round-trip back to polars to assert the event_ids survive.
    kept_events = set(tables["tracker_hits"].column("event_id").to_pylist())
    assert kept_events == {1, 2, 3}

    # Ensure routing metadata reached the stubs.
    assert calls["ensure"][0] == "ttbar"
    assert calls["ensure"][1] == "pu200"
    assert calls["ensure"][2] == ["tracker_hits"]


def test_load_task_data_max_events_trims(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_frame = pl.DataFrame(
        {
            "event_id": list(range(10)),
            "hit_id": list(range(10)),
        }
    )

    def fake_ensure(*args, **kwargs) -> None:
        pass

    def fake_load_tables(cfg) -> Dict[str, pl.DataFrame]:
        return {"tracker_hits": stub_frame}

    monkeypatch.setattr(_loading, "_ensure_downloaded", fake_ensure)
    monkeypatch.setattr(_loading, "load_tables", fake_load_tables)

    tables = load_task_data("ttbar_pu0", tables=["tracker_hits"], max_events=3)
    assert tables["tracker_hits"].num_rows == 3


def test_load_task_data_event_range_end_before_start_raises() -> None:
    with pytest.raises(ValueError, match="end .* must be >= start"):
        load_task_data("ttbar_pu0", tables=["tracker_hits"], event_range=(5, 2))
