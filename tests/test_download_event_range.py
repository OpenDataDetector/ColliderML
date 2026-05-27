"""Tests for shard-bounded downloads via ``max_events`` and ``event_range``.

The bugs these tests guard against:

* ``colliderml.load(max_events=N)`` and ``task.load(event_range=...)``
  used to drop the constraint on the way to ``DownloadSpec``, then fall
  through to ``snapshot_download`` which fetches the full ~700 GB
  high-pileup tracker config.
* The events-per-shard heuristic used to assume 100 events/shard
  uniformly — wrong for PU=0 (which ships ~1000 events/shard), causing
  a 10× over-download. The fix probes the first shard's parquet metadata
  to learn the real count.
* ``_list_remote_shards`` used to call ``list_repo_files`` against the
  whole dataset (~21 000 files paginated 1000 at a time) per call, even
  when we only wanted shards under one config. The fix uses
  ``list_repo_tree`` scoped to ``data/{config}`` and memoises the result
  per process.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

import colliderml.core.hf_download as hf_download
from colliderml.core.hf_download import (
    DownloadSpec,
    _shards_for_event_range,
    _shards_for_max_events,
    download_config,
)


def _fake_shards(num: int, config: str = "ttbar_pu0_particles") -> List[str]:
    width = max(5, len(str(num)))
    return [
        f"data/{config}/train-{i:0{width}d}-of-{num:0{width}d}.parquet"
        for i in range(num)
    ]


@pytest.fixture(autouse=True)
def _clear_lru_caches():
    """The shard listing + events-per-shard probe are memoised per process,
    which is the correct behavior in production but would leak between
    tests. Reset them before each test so fixtures are independent."""
    hf_download._list_remote_shards.cache_clear()
    hf_download._probe_events_per_shard.cache_clear()
    yield
    hf_download._list_remote_shards.cache_clear()
    hf_download._probe_events_per_shard.cache_clear()


# ---------------------------------------------------------------------------
# Unit tests for the slice helpers (no network).
# ---------------------------------------------------------------------------
class TestEventRangeSlice:
    def test_pu200_eval_slice_one_shard(self) -> None:
        # PU=200: 1000 shards * 100 events/shard. The exact case the
        # tutorial hits.
        shards = _fake_shards(1000)
        result = _shards_for_event_range(shards, (90_000, 90_050), events_per_shard=100)
        assert result == [shards[900]]

    def test_pu0_event_range_is_one_shard_not_ten(self) -> None:
        # PU=0: 1000 shards * 1000 events/shard. Before the probe fix
        # this was being computed as 100 events/shard and would download
        # 10 shards (10× over-fetch).
        shards = _fake_shards(1000)
        result = _shards_for_event_range(shards, (0, 1000), events_per_shard=1000)
        assert result == [shards[0]]

    def test_range_spanning_shard_boundary(self) -> None:
        shards = _fake_shards(1000)
        result = _shards_for_event_range(shards, (90_050, 90_150), events_per_shard=100)
        assert result == shards[900:902]

    def test_range_on_low_cardinality_config(self) -> None:
        # 100 shards * 1000 events/shard.
        shards = _fake_shards(100)
        result = _shards_for_event_range(shards, (0, 50), events_per_shard=1000)
        assert result == [shards[0]]

    def test_empty_range(self) -> None:
        shards = _fake_shards(10)
        assert _shards_for_event_range(shards, (5, 5), events_per_shard=100) == []

    def test_range_past_total_returns_empty(self) -> None:
        shards = _fake_shards(10)
        result = _shards_for_event_range(shards, (200_000, 210_000), events_per_shard=100)
        assert result == []

    def test_no_shards(self) -> None:
        assert _shards_for_event_range([], (0, 50), events_per_shard=100) == []


class TestMaxEventsSlice:
    def test_max_events_on_pu200_config(self) -> None:
        shards = _fake_shards(1000)
        assert _shards_for_max_events(shards, 200, events_per_shard=100) == shards[:2]

    def test_max_events_on_pu0_config(self) -> None:
        # 1000 events/shard, asking for 200 events → first shard only.
        shards = _fake_shards(1000)
        assert _shards_for_max_events(shards, 200, events_per_shard=1000) == shards[:1]

    def test_max_events_minimum_one_shard(self) -> None:
        shards = _fake_shards(1000)
        assert _shards_for_max_events(shards, 1, events_per_shard=100) == shards[:1]

    def test_max_events_caps_at_available(self) -> None:
        shards = _fake_shards(10)
        result = _shards_for_max_events(shards, 1_000_000, events_per_shard=1000)
        assert result == shards


# ---------------------------------------------------------------------------
# Integration: download_config respects event_range and uses hf_hub_download
# (NOT snapshot_download) when bounded.
# ---------------------------------------------------------------------------
@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "colliderml_cache"
    d.mkdir()
    monkeypatch.setenv("COLLIDERML_DATA_DIR", str(d))
    return d


def _patch_hf(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_shards: int,
    config: str,
    events_per_shard: int,
):
    """Replace HF entry points with deterministic in-memory fakes."""
    shards = _fake_shards(num_shards, config=config)
    calls = {"snapshot": [], "hub": [], "list_repo_tree": 0, "probe": 0}

    class _FakeEntry:
        def __init__(self, path: str) -> None:
            self.path = path

    def fake_list_repo_tree(
        *,
        repo_id,
        repo_type,
        revision=None,
        path_in_repo,
        recursive=False,
    ):
        # The fix uses list_repo_tree scoped to one config — assert
        # we're scoping correctly.
        assert path_in_repo == f"data/{config}", (
            f"_list_remote_shards should scope to data/{config}, got {path_in_repo}"
        )
        calls["list_repo_tree"] += 1
        return iter([_FakeEntry(s) for s in shards])

    def fake_snapshot_download(**kwargs):
        calls["snapshot"].append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        for fp in shards:
            p = local_dir / fp
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"PARQUET")
        return str(local_dir)

    def fake_hf_hub_download(*, repo_id, repo_type, revision, filename, local_dir, **kwargs):
        calls["hub"].append(filename)
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"PARQUET")
        return str(p)

    def fake_probe(dataset_id, cfg, first_shard_path, revision):
        calls["probe"] += 1
        return events_per_shard

    monkeypatch.setattr(hf_download, "list_repo_tree", fake_list_repo_tree)
    monkeypatch.setattr(hf_download, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hf_download, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(hf_download, "_probe_events_per_shard", lambda *a, **kw: events_per_shard)
    return calls, shards


def test_pu200_event_range_downloads_only_relevant_shard(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PU=200, 1000 shards × 100 events/shard. event_range=(90000, 90050) → shard 900 only."""
    calls, shards = _patch_hf(
        monkeypatch,
        num_shards=1000,
        config="ttbar_pu200_tracker_hits",
        events_per_shard=100,
    )

    spec = DownloadSpec(
        dataset_id="CERN/ColliderML-Release-1",
        config="ttbar_pu200_tracker_hits",
        event_range=(90_000, 90_050),
    )
    result = download_config(spec)

    assert calls["snapshot"] == [], "snapshot_download triggered despite event_range"
    assert calls["hub"] == [shards[900]]
    assert result.shards == [shards[900]]


def test_pu0_event_range_uses_real_shard_count(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PU=0 ships 1000 events/shard. event_range=(0, 1000) needs exactly 1 shard,
    not 10 (which was the bug before the parquet-metadata probe was added)."""
    calls, shards = _patch_hf(
        monkeypatch,
        num_shards=1000,
        config="ttbar_pu0_tracker_hits",
        events_per_shard=1000,
    )

    spec = DownloadSpec(
        dataset_id="CERN/ColliderML-Release-1",
        config="ttbar_pu0_tracker_hits",
        event_range=(0, 1000),
    )
    result = download_config(spec)
    assert calls["hub"] == [shards[0]], (
        f"PU=0 event_range=(0,1000) should pull 1 shard, got {len(calls['hub'])}"
    )
    assert result.shards == [shards[0]]


def test_max_events_downloads_only_relevant_shards(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, shards = _patch_hf(
        monkeypatch,
        num_shards=1000,
        config="ttbar_pu200_tracker_hits",
        events_per_shard=100,
    )

    spec = DownloadSpec(
        dataset_id="CERN/ColliderML-Release-1",
        config="ttbar_pu200_tracker_hits",
        max_events=200,
    )
    download_config(spec)

    assert calls["snapshot"] == []
    assert calls["hub"] == shards[:2]


def test_unbounded_download_uses_snapshot(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _patch_hf(
        monkeypatch,
        num_shards=10,
        config="ttbar_pu0_particles",
        events_per_shard=1000,
    )

    spec = DownloadSpec(
        dataset_id="CERN/ColliderML-Release-1",
        config="ttbar_pu0_particles",
    )
    download_config(spec)
    assert len(calls["snapshot"]) == 1
    assert calls["hub"] == []


def test_event_range_and_max_events_together_rejected(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_hf(
        monkeypatch,
        num_shards=10,
        config="ttbar_pu0_particles",
        events_per_shard=1000,
    )
    spec = DownloadSpec(
        config="ttbar_pu0_particles", max_events=10, event_range=(0, 10)
    )
    with pytest.raises(ValueError, match="at most one"):
        download_config(spec)


def test_event_range_skips_shards_already_on_disk(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, shards = _patch_hf(
        monkeypatch,
        num_shards=1000,
        config="ttbar_pu200_tracker_hits",
        events_per_shard=100,
    )

    config_dir = tmp_data_dir / "CERN__ColliderML-Release-1" / "ttbar_pu200_tracker_hits"
    cached = config_dir / shards[900]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"PARQUET")

    spec = DownloadSpec(
        dataset_id="CERN/ColliderML-Release-1",
        config="ttbar_pu200_tracker_hits",
        event_range=(90_000, 90_050),
    )
    download_config(spec)
    assert calls["hub"] == []


def test_list_repo_tree_is_scoped_to_one_config(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Don't paginate the whole 21k-file dataset when we want one config's shards."""
    calls, _ = _patch_hf(
        monkeypatch,
        num_shards=1000,
        config="ttbar_pu0_tracker_hits",
        events_per_shard=1000,
    )

    spec = DownloadSpec(
        dataset_id="CERN/ColliderML-Release-1",
        config="ttbar_pu0_tracker_hits",
        event_range=(0, 1000),
    )
    download_config(spec)
    # The fake_list_repo_tree asserts path_in_repo == "data/{config}",
    # so the test will fail if we revert to the dataset-wide listing.
    assert calls["list_repo_tree"] >= 1


def test_list_remote_shards_is_memoised(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated download_config() calls for the same config should only
    list the repo tree once per (dataset, config, split, revision)."""
    calls, _ = _patch_hf(
        monkeypatch,
        num_shards=1000,
        config="ttbar_pu0_tracker_hits",
        events_per_shard=1000,
    )

    spec = DownloadSpec(
        dataset_id="CERN/ColliderML-Release-1",
        config="ttbar_pu0_tracker_hits",
        event_range=(0, 1000),
    )
    download_config(spec)
    download_config(spec)
    download_config(spec)
    assert calls["list_repo_tree"] == 1
