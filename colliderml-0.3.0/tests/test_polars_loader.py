"""Unit tests for the Polars-based local loader.

These tests create tiny Parquet shards locally and verify:
- Path resolution logic matches the CLI download layout
- max_events is applied consistently across loaded tables
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture()
def local_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "colliderml_cache"
    base.mkdir()
    monkeypatch.setenv("COLLIDERML_DATA_DIR", str(base))
    return base


def _write_shard(base: Path, dataset_id: str, config: str, split: str, df: pl.DataFrame) -> None:
    ds_dir = base / dataset_id.replace("/", "__") / config / "data" / config
    ds_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(ds_dir / f"{split}-00000-of-00001.parquet")


def test_loader_eager_and_max_events(local_dataset: Path) -> None:
    from colliderml.core.loader import load_tables

    dataset_id = "CERN/ColliderML-Release-1"
    cfg_particles = "ttbar_pu0_particles"
    cfg_tracks = "ttbar_pu0_tracks"

    particles_df = pl.DataFrame({"event_id": [1, 2, 3], "particle_id": [[10], [20, 21], [30]]})
    tracks_df = pl.DataFrame({"event_id": [1, 2, 3], "track_id": [[7], [8], [9]]})

    _write_shard(local_dataset, dataset_id, cfg_particles, "train", particles_df)
    _write_shard(local_dataset, dataset_id, cfg_tracks, "train", tracks_df)

    tables = load_tables(
        {
            "dataset_id": dataset_id,
            "channels": "ttbar",
            "pileup": "pu0",
            "objects": ["particles", "tracks"],
            "split": "train",
            "lazy": False,
            "max_events": 2,
        }
    )
    assert "particles" in tables
    assert "tracks" in tables

    # max_events=2 should filter both tables to events 1,2.
    p = tables["particles"]
    t = tables["tracks"]
    assert isinstance(p, pl.DataFrame)
    assert isinstance(t, pl.DataFrame)
    assert p.select("event_id").to_series().to_list() == [1, 2]
    assert t.select("event_id").to_series().to_list() == [1, 2]


def test_loader_lazy(local_dataset: Path) -> None:
    from colliderml.core.loader import load_tables

    dataset_id = "CERN/ColliderML-Release-1"
    cfg_particles = "ttbar_pu0_particles"
    df = pl.DataFrame({"event_id": [1, 2], "particle_id": [[10], [20]]})
    _write_shard(local_dataset, dataset_id, cfg_particles, "train", df)

    tables = load_tables(
        {
            "dataset_id": dataset_id,
            "channels": "ttbar",
            "pileup": "pu0",
            "objects": ["particles"],
            "split": "train",
            "lazy": True,
        }
    )
    assert isinstance(tables["particles"], pl.LazyFrame)


