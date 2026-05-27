"""HuggingFace dataset download utilities for ColliderML.

This module is intentionally focused on downloading Parquet shards to disk.
All loading/processing should be done by the Polars backend elsewhere.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    list_repo_tree,
    snapshot_download,
)

logger = logging.getLogger(__name__)


DEFAULT_DATASET_ID = "CERN/ColliderML-Release-1"
DEFAULT_SPLIT = "train"
DATA_DIR_ENV = "COLLIDERML_DATA_DIR"

#: Last-resort fallback when we can't probe the actual events-per-shard
#: from parquet metadata (e.g. the network is down or the dataset doesn't
#: follow Release-1 conventions). Real per-shard counts vary by pileup
#: level — PU=0 ships ~1000 events/shard, PU=200 ships ~100. The real
#: count is read from the first shard's footer in :func:`_probe_events_per_shard`.
DEFAULT_EVENTS_PER_SHARD = 1000


def default_data_dir() -> Path:
    """Return the default ColliderML cache directory.

    Uses `$COLLIDERML_DATA_DIR` if set, otherwise falls back to
    `~/.cache/colliderml`.

    Returns:
        Path: Base directory where datasets should be stored.
    """
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".cache" / "colliderml").resolve()


def _sanitize_dataset_id(dataset_id: str) -> str:
    """Sanitize a HF dataset id for use as a local directory name."""
    return dataset_id.replace("/", "__")


def local_dataset_root(data_dir: Path, dataset_id: str) -> Path:
    """Compute the local root directory for a dataset."""
    return data_dir / _sanitize_dataset_id(dataset_id)


def local_config_dir(data_dir: Path, dataset_id: str, config: str) -> Path:
    """Compute the local directory for a dataset config."""
    return local_dataset_root(data_dir, dataset_id) / config


def discover_remote_configs(dataset_id: str, revision: Optional[str] = None) -> list[str]:
    """Discover available config names for the consolidated dataset.

    This inspects repo files and extracts unique `data/{config}/...` prefixes.

    Args:
        dataset_id: HuggingFace dataset repo id.
        revision: Optional revision/tag/commit.

    Returns:
        List of config names.
    """
    files = list_repo_files(repo_id=dataset_id, repo_type="dataset", revision=revision)
    configs: set[str] = set()
    for fp in files:
        m = re.match(r"^data/([^/]+)/", fp)
        if m:
            configs.add(m.group(1))
    return sorted(configs)


@functools.lru_cache(maxsize=256)
def _list_remote_shards(
    dataset_id: str, config: str, split: str, revision: Optional[str]
) -> tuple[str, ...]:
    """List shard parquet files for a given config+split.

    Uses :func:`list_repo_tree` scoped to ``data/{config}`` so we only
    paginate the ~1000 files in this config — not the ~21 000 files in
    the whole consolidated dataset. The result is memoised per
    (dataset_id, config, split, revision) because the shard list only
    changes when the dataset revision changes, and we ``download_config``
    enough times in a single process to make repeated full-repo listings
    a real performance cliff.

    Returns a tuple (rather than a list) so it can be safely cached by
    ``lru_cache``.
    """
    prefix = f"data/{config}/"
    shards: list[str] = []
    for entry in list_repo_tree(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo=f"data/{config}",
        recursive=False,
    ):
        path = entry.path
        if (
            path.startswith(prefix)
            and path.endswith(".parquet")
            and path[len(prefix):].startswith(f"{split}-")
        ):
            shards.append(path)
    return tuple(sorted(shards))


@functools.lru_cache(maxsize=256)
def _probe_events_per_shard(
    dataset_id: str,
    config: str,
    first_shard_path: str,
    revision: Optional[str],
) -> int:
    """Read the first shard's parquet metadata to learn its row count.

    Parquet's metadata footer is a few tens of KB and at the END of the
    file, so reading just the metadata over HTTP costs one or two range
    requests — not a full multi-MB / multi-GB shard download. We use
    HuggingFace's fsspec implementation so the auth and URL resolution
    are transparent.

    The result is cached per (dataset_id, config, revision); within a
    single process we only probe each config once.

    If the probe fails for any reason (network, missing fsspec, weird
    parquet), fall back to :data:`DEFAULT_EVENTS_PER_SHARD` (1000) —
    callers will over-fetch by a constant factor in that case, but the
    download still completes.
    """
    try:
        import pyarrow.parquet as pq
        import fsspec  # type: ignore[import-untyped]

        fs = fsspec.filesystem("hf")
        rev = revision or "main"
        fs_path = f"datasets/{dataset_id}@{rev}/{first_shard_path}"
        with fs.open(fs_path, "rb") as f:
            pf = pq.ParquetFile(f)
            n = int(pf.metadata.num_rows)
        if n <= 0:
            raise ValueError(f"non-positive row count {n}")
        return n
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "Could not probe events-per-shard for %s/%s (%s); falling back to %d",
            dataset_id, config, exc, DEFAULT_EVENTS_PER_SHARD,
        )
        return DEFAULT_EVENTS_PER_SHARD


@dataclass(frozen=True)
class DownloadSpec:
    """Specification for downloading a dataset configuration.

    At most one of ``max_events`` and ``event_range`` should be set. When
    both are ``None`` the entire config+split is fetched.
    """

    dataset_id: str = DEFAULT_DATASET_ID
    config: str = "ttbar_pu0_particles"
    split: str = DEFAULT_SPLIT
    revision: Optional[str] = None
    max_events: Optional[int] = None
    event_range: Optional[Tuple[int, int]] = None


def _shards_for_event_range(
    shards: Sequence[str],
    event_range: Tuple[int, int],
    events_per_shard: int,
) -> list[str]:
    """Pick the subset of sorted shards covering events in ``[start, end)``."""
    if not shards or events_per_shard <= 0:
        return []
    start, end = int(event_range[0]), int(event_range[1])
    if end <= start:
        return []
    first = max(0, start // events_per_shard)
    last = min(len(shards), (end + events_per_shard - 1) // events_per_shard)
    return list(shards[first:last])


def _shards_for_max_events(
    shards: Sequence[str],
    max_events: int,
    events_per_shard: int,
) -> list[str]:
    """Pick the smallest prefix of sorted shards covering ``max_events`` events."""
    if not shards or events_per_shard <= 0:
        return []
    count = max(1, (int(max_events) + events_per_shard - 1) // events_per_shard)
    return list(shards[: min(count, len(shards))])


@dataclass(frozen=True)
class DownloadResult:
    """Result metadata for a download operation."""

    local_dir: Path
    dataset_id: str
    config: str
    split: str
    revision: Optional[str]
    shards: list[str]
    timestamp_unix: int


def download_config(
    spec: DownloadSpec,
    out_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> DownloadResult:
    """Download Parquet shards for a dataset config into a local directory.

    Behavior:
    - If neither ``spec.max_events`` nor ``spec.event_range`` is set:
      download every shard of the config+split.
    - If ``spec.event_range = (start, end)`` is set: download only the
      shards covering that half-open event range. The events-per-shard
      count is probed from the first shard's parquet footer (via
      :func:`_probe_events_per_shard`) so the slice is correct for
      both PU=0 (~1000 events/shard) and PU=200 (~100 events/shard).
      The Polars loader filters precisely on event id after the
      download.
    - If ``spec.max_events`` is set (and ``event_range`` is not):
      download enough leading shards to cover at least ``max_events``
      events, using the same probed events-per-shard.

    Args:
        spec: Download specification.
        out_dir: Base output directory. Defaults to ``default_data_dir()``.
        force: If True, re-download files even if they already exist.

    Returns:
        DownloadResult: metadata including local path and downloaded shards.

    Raises:
        ValueError: If both ``max_events`` and ``event_range`` are set.
        FileNotFoundError: If the config has no shards on the remote.
    """
    if spec.max_events is not None and spec.event_range is not None:
        raise ValueError("DownloadSpec accepts at most one of max_events and event_range.")

    data_dir = (out_dir or default_data_dir()).expanduser().resolve()
    config_dir = local_config_dir(data_dir, spec.dataset_id, spec.config)
    config_dir.mkdir(parents=True, exist_ok=True)

    shards = list(_list_remote_shards(spec.dataset_id, spec.config, spec.split, spec.revision))
    if not shards:
        raise FileNotFoundError(
            f"No shards found for dataset={spec.dataset_id} config={spec.config} split={spec.split}"
        )

    shard_paths: list[str]
    if spec.event_range is not None or spec.max_events is not None:
        # Probe the actual events-per-shard so we don't over-fetch on
        # low-pileup configs (PU=0 ships 1000 events/shard, PU=200 ships
        # 100; assuming one number for both burns disk).
        events_per_shard = _probe_events_per_shard(
            spec.dataset_id, spec.config, shards[0], spec.revision
        )
        if spec.event_range is not None:
            shard_paths = _shards_for_event_range(shards, spec.event_range, events_per_shard)
        else:
            shard_paths = _shards_for_max_events(shards, spec.max_events, events_per_shard)
    else:
        # Unbounded: snapshot_download is faster than per-file when we
        # really do want every shard.
        allow_patterns = [f"data/{spec.config}/{spec.split}-*.parquet"]
        snapshot_download(
            repo_id=spec.dataset_id,
            repo_type="dataset",
            revision=spec.revision,
            local_dir=str(config_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            force_download=force,
        )
        shard_paths = shards

    if spec.event_range is not None or spec.max_events is not None:
        for fp in shard_paths:
            local_file = config_dir / fp
            local_file.parent.mkdir(parents=True, exist_ok=True)
            if local_file.exists() and not force:
                continue
            hf_hub_download(
                repo_id=spec.dataset_id,
                repo_type="dataset",
                revision=spec.revision,
                filename=fp,
                local_dir=str(config_dir),
                local_dir_use_symlinks=False,
                force_download=force,
            )

    result = DownloadResult(
        local_dir=config_dir,
        dataset_id=spec.dataset_id,
        config=spec.config,
        split=spec.split,
        revision=spec.revision,
        shards=shard_paths,
        timestamp_unix=int(time.time()),
    )
    _write_metadata(result)
    return result


def _write_metadata(result: DownloadResult) -> None:
    """Write `metadata.json` for a downloaded config."""
    meta_path = result.local_dir / "metadata.json"
    payload = {
        "dataset_id": result.dataset_id,
        "config": result.config,
        "split": result.split,
        "revision": result.revision,
        "shards": result.shards,
        "timestamp_unix": result.timestamp_unix,
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def list_local_configs(data_dir: Optional[Path], dataset_id: str) -> list[str]:
    """List locally downloaded configs under the cache root."""
    base = local_dataset_root((data_dir or default_data_dir()).expanduser().resolve(), dataset_id)
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


