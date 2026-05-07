"""Tests for shard-bounded downloads via ``max_events`` and ``event_range``.

The bug these tests guard against: ``colliderml.load(max_events=N)`` and
``task.load(event_range=...)`` used to drop the constraint on the way to
``DownloadSpec``, then fall through to ``snapshot_download`` which fetches
the full ~700 GB high-pileup tracker config. The fix threads both args
through and translates them into a shard slice using a uniform-shard
heuristic.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from colliderml.core.hf_download import (
    DEFAULT_TOTAL_EVENTS_PER_CONFIG,
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


# ---------------------------------------------------------------------------
# Unit tests for the slice helpers (no network).
# ---------------------------------------------------------------------------
class TestEventRangeSlice:
    def test_first_50_events_on_1000_shard_config(self) -> None:
        # 1000 shards * 100 events/shard = 100k events.
        shards = _fake_shards(1000)
        result = _shards_for_event_range(shards, (0, 50))
        assert result == [shards[0]]

    def test_eval_split_on_1000_shard_config(self) -> None:
        # The exact case the tutorial hits: 50 events starting at 90_000.
        # With 100 events/shard that's shard 900 only.
        shards = _fake_shards(1000)
        result = _shards_for_event_range(shards, (90_000, 90_050))
        assert result == [shards[900]]

    def test_range_spanning_shard_boundary(self) -> None:
        shards = _fake_shards(1000)
        # 90_050..90_150 spans shards 900 and 901.
        result = _shards_for_event_range(shards, (90_050, 90_150))
        assert result == shards[900:902]

    def test_range_on_low_cardinality_config(self) -> None:
        # 100 shards * 1000 events/shard.
        shards = _fake_shards(100)
        result = _shards_for_event_range(shards, (0, 50))
        assert result == [shards[0]]

    def test_empty_range(self) -> None:
        shards = _fake_shards(10)
        assert _shards_for_event_range(shards, (5, 5)) == []

    def test_range_clamps_to_available_shards(self) -> None:
        shards = _fake_shards(10)
        # Heuristic: 100k / 10 = 10k events/shard. Ask for events past total.
        result = _shards_for_event_range(shards, (200_000, 210_000))
        assert result == []

    def test_no_shards(self) -> None:
        assert _shards_for_event_range([], (0, 50)) == []


class TestMaxEventsSlice:
    def test_max_events_on_1000_shard_config(self) -> None:
        shards = _fake_shards(1000)  # 100 events/shard
        # 200 events → first 2 shards.
        assert _shards_for_max_events(shards, 200) == shards[:2]

    def test_max_events_on_100_shard_config(self) -> None:
        shards = _fake_shards(100)  # 1000 events/shard
        # 200 events → first shard only.
        assert _shards_for_max_events(shards, 200) == shards[:1]

    def test_max_events_minimum_one_shard(self) -> None:
        shards = _fake_shards(1000)
        assert _shards_for_max_events(shards, 1) == shards[:1]

    def test_max_events_caps_at_available(self) -> None:
        shards = _fake_shards(10)
        # 10 shards * 10k = 100k events. Asking for 10× the total → caps.
        assert _shards_for_max_events(shards, 1_000_000) == shards


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
    num_shards: int = 1000,
    config: str = "ttbar_pu200_tracker_hits",
):
    """Replace HF entry points with deterministic in-memory fakes."""
    shards = _fake_shards(num_shards, config=config)
    repo_files = ["README.md", *shards]
    calls = {"snapshot": [], "hub": [], "list_repo_files": 0}

    def fake_list_repo_files(*, repo_id, repo_type, revision=None):
        calls["list_repo_files"] += 1
        return repo_files

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

    monkeypatch.setattr("colliderml.core.hf_download.list_repo_files", fake_list_repo_files)
    monkeypatch.setattr("colliderml.core.hf_download.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("colliderml.core.hf_download.hf_hub_download", fake_hf_hub_download)
    return calls, shards


def test_event_range_downloads_only_relevant_shards(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, shards = _patch_hf(monkeypatch, num_shards=1000)

    spec = DownloadSpec(
        dataset_id="CERN/ColliderML-Release-1",
        config="ttbar_pu200_tracker_hits",
        event_range=(90_000, 90_050),
    )
    result = download_config(spec)

    # The whole point: snapshot_download must NOT be called when a bound is set.
    assert calls["snapshot"] == [], "snapshot_download triggered despite event_range"

    # Exactly one shard pulled, and it is shard 900 (events 90_000..90_099).
    assert calls["hub"] == [shards[900]]
    assert result.shards == [shards[900]]


def test_max_events_downloads_only_relevant_shards(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, shards = _patch_hf(monkeypatch, num_shards=1000)

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
    calls, _ = _patch_hf(monkeypatch, num_shards=10, config="ttbar_pu0_particles")

    spec = DownloadSpec(
        dataset_id="CERN/ColliderML-Release-1",
        config="ttbar_pu0_particles",
    )
    download_config(spec)

    # No bound → snapshot_download is the right tool.
    assert len(calls["snapshot"]) == 1
    assert calls["hub"] == []


def test_event_range_and_max_events_together_rejected(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_hf(monkeypatch, num_shards=10)
    spec = DownloadSpec(
        config="ttbar_pu0_particles", max_events=10, event_range=(0, 10)
    )
    with pytest.raises(ValueError, match="at most one"):
        download_config(spec)


def test_event_range_skips_shards_already_on_disk(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, shards = _patch_hf(monkeypatch, num_shards=1000)

    # Pretend shard 900 is already cached.
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

    # Already-present shard must be a no-op against the network.
    assert calls["hub"] == []


def test_release_1_total_events_constant_is_100k() -> None:
    # Sanity: future bumps to this constant must be conscious.
    assert DEFAULT_TOTAL_EVENTS_PER_CONFIG == 100_000
