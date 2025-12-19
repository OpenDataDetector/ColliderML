"""Polars-based functional loader for locally downloaded ColliderML Parquet data."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Sequence, Union

import polars as pl

from colliderml.core.data.loader_config import LoaderConfig, ObjectType, load_config
from colliderml.core.hf_download import local_config_dir
from colliderml.core.tables import PolarsTable, select_events, table_names


def _parquet_glob(config_dir: Path, config: str, split: str) -> list[Path]:
    """Return local Parquet shard paths for a config+split.

    Expected on-disk layout (created by `colliderml download`):
      {config_dir}/data/{config}/{split}-*.parquet
    """
    pattern = str(config_dir / "data" / config / f"{split}-*.parquet")
    return [Path(p) for p in sorted(glob.glob(pattern))]


def _read_parquet(paths: Sequence[Path], lazy: bool) -> PolarsTable:
    """Read or scan parquet shards with Polars."""
    if len(paths) == 0:
        raise FileNotFoundError("No parquet shards found to load.")
    if lazy:
        return pl.scan_parquet([str(p) for p in paths])
    return pl.read_parquet([str(p) for p in paths])


def load_tables(cfg: Union[LoaderConfig, dict, str, Path]) -> Dict[str, PolarsTable]:
    """Load selected object tables as Polars tables.

    Args:
        cfg: LoaderConfig, dict, or YAML file path.

    Returns:
        Dict[str, PolarsTable]: mapping of object key -> Polars DataFrame/LazyFrame.

    Raises:
        FileNotFoundError: if expected local parquet shards are missing.
    """
    config = load_config(cfg)
    data_dir = config.resolved_data_dir()

    tables: Dict[str, PolarsTable] = {}
    channels = config.normalized_channels()
    for channel in channels:
        for obj in config.objects:
            name = f"{channel}_{config.pileup}_{obj}"
            cfg_dir = local_config_dir(data_dir, config.dataset_id, name)
            shards = _parquet_glob(cfg_dir, name, config.split)
            if not shards:
                raise FileNotFoundError(
                    f"Missing local shards for config '{name}'. "
                    f"Expected under: {cfg_dir}/data/{name}/{config.split}-*.parquet. "
                    f"Run: colliderml download --config {name}"
                )
            key = obj if len(channels) == 1 else f"{channel}.{obj}"
            tables[key] = _read_parquet(shards, config.lazy)

    if config.max_events is not None:
        tables = apply_max_events(tables, config.max_events)

    return tables


def apply_max_events(tables: Dict[str, PolarsTable], max_events: int, ref: str = "particles") -> Dict[str, PolarsTable]:
    """Select first N events from a reference table and apply to all tables.

    Args:
        tables: Mapping of table name -> Polars table.
        max_events: Number of events to keep (first N by reference table order).
        ref: Preferred reference table name. If missing, uses the first available.

    Returns:
        Dict[str, PolarsTable]: filtered tables.
    """
    ref_name = ref if ref in tables else table_names(tables)[0]
    ref_table = tables[ref_name]
    if isinstance(ref_table, pl.LazyFrame):
        event_ids = (
            ref_table.select(pl.col("event_id")).limit(max_events).collect().get_column("event_id")
        )
    else:
        event_ids = ref_table.select("event_id").head(max_events).get_column("event_id")
    return select_events(tables, event_ids)


