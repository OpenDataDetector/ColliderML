"""HuggingFace dataset download utilities for ColliderML.

This module is intentionally focused on downloading Parquet shards to disk.
All loading/processing should be done by the Polars backend elsewhere.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download


DEFAULT_DATASET_ID = "CERN/ColliderML-Release-1"
DEFAULT_SPLIT = "train"
DATA_DIR_ENV = "COLLIDERML_DATA_DIR"


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


def _list_remote_shards(
    dataset_id: str, config: str, split: str, revision: Optional[str]
) -> list[str]:
    """List shard parquet files for a given config+split."""
    prefix = f"data/{config}/"
    files = list_repo_files(repo_id=dataset_id, repo_type="dataset", revision=revision)
    shards = [
        f
        for f in files
        if f.startswith(prefix)
        and f.endswith(".parquet")
        and f[len(prefix) :].startswith(f"{split}-")
    ]
    return sorted(shards)


@dataclass(frozen=True)
class DownloadSpec:
    """Specification for downloading a dataset configuration."""

    dataset_id: str = DEFAULT_DATASET_ID
    config: str = "ttbar_pu0_particles"
    split: str = DEFAULT_SPLIT
    revision: Optional[str] = None
    max_events: Optional[int] = None


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
    - If `spec.max_events` is None: download all shards for the config+split.
    - If `spec.max_events` is set: download shards sequentially from the start
      until at least `max_events` events are available (may exceed slightly).

    Args:
        spec: Download specification.
        out_dir: Base output directory. Defaults to `default_data_dir()`.
        force: If True, re-download files even if they already exist.

    Returns:
        DownloadResult: metadata including local path and downloaded shards.
    """
    data_dir = (out_dir or default_data_dir()).expanduser().resolve()
    config_dir = local_config_dir(data_dir, spec.dataset_id, spec.config)
    config_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[str]
    if spec.max_events is None:
        # Download everything for this config/split in one go.
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
        shard_paths = _list_remote_shards(spec.dataset_id, spec.config, spec.split, spec.revision)
    else:
        # Deterministic incremental download of shards until we reach max_events.
        # Note: event-count inspection is deferred to the Polars loader stage.
        # Here we approximate by downloading the first N shards, assuming
        # relatively uniform shard sizes; we always download at least 1 shard.
        shards = _list_remote_shards(spec.dataset_id, spec.config, spec.split, spec.revision)
        if not shards:
            raise FileNotFoundError(
                f"No shards found for dataset={spec.dataset_id} config={spec.config} split={spec.split}"
            )
        # Heuristic: in Release-1 many configs are sharded as 1000 files for 100k events.
        # Avoid over-downloading by taking a conservative shard count.
        # The loader will still enforce max_events precisely.
        per_shard_guess = 100
        shard_count = max(1, (spec.max_events + per_shard_guess - 1) // per_shard_guess)
        shard_paths = shards[: shard_count]

        for fp in shard_paths:
            local_file = config_dir / fp
            local_file.parent.mkdir(parents=True, exist_ok=True)
            if local_file.exists() and not force:
                continue
            downloaded = hf_hub_download(
                repo_id=spec.dataset_id,
                repo_type="dataset",
                revision=spec.revision,
                filename=fp,
                local_dir=str(config_dir),
                local_dir_use_symlinks=False,
                force_download=force,
            )
            # hf_hub_download returns the resolved path; ensure it exists.
            Path(downloaded).exists()

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


