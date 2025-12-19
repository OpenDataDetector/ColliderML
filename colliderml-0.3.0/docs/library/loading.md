# Loading: Config-driven Local Parquet

The ColliderML loader reads **local Parquet shards** (downloaded via the CLI) using Polars.

## Load tables

Use `colliderml.core.load_tables` to read a set of object tables from the local cache.

```python
from colliderml.core import load_tables, collect_tables

cfg = {
    "dataset_id": "CERN/ColliderML-Release-1",
    "channels": "ttbar",
    "pileup": "pu0",
    "objects": ["particles", "tracker_hits", "calo_hits", "tracks"],
    "split": "train",
    "lazy": True,
    "max_events": 200,
}

tables = load_tables(cfg)      # dict[str, pl.DataFrame | pl.LazyFrame]
frames = collect_tables(tables)  # dict[str, pl.DataFrame]
```

## Key config fields

- `channels`: string or list (or `"all"`)
- `pileup`: `"pu0"`, `"pu200"`, etc.
- `objects`: `["particles", "tracker_hits", "calo_hits", "tracks"]`
- `split`: typically `"train"`
- `lazy`: `True` to `scan_parquet`, `False` to `read_parquet`
- `max_events`: exact event limit enforced locally across all tables
- `data_dir`: optional override for cache directory (otherwise uses defaults / env var)

## Event selection (`max_events`)

The loader enforces a consistent event selection across objects:

- Select `event_id`s from a reference table (by default `particles`)
- Filter all other tables to those `event_id`s

This means you can download full shards but still work with a deterministic, small event slice.


