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

## Events with no objects of a type

Each table has one row per event that has at least one object of that type. An event with no tracks, no
tracker hits or no calorimeter hits has **no row** in that table; it is not stored as an empty row.
`particles` has a row for every event, so it is the natural reference table.

Join on `event_id` with `particles` on the left (or an outer join) and treat a missing row as zero objects.
An inner join drops those events:

```python
events = frames["particles"].select("event_id").join(frames["tracks"], on="event_id", how="left")
```

How often this happens in Release 1:

| Config | Events checked | No row in `tracks` | No row in `tracker_hits` | No row in `calo_hits` |
|---|---|---|---|---|
| `ttbar_pu0` | 1,000,000 | 2 | 0 | 0 |
| `pileup_only_pu0` (first 10,000 events) | 10,000 | 3,891 | 2,677 | 2,627 |

In `pileup_only_pu0`, 38.9% of these events have no row in at least one of the three tables.

Some Release 1 `calo_hits` shards also lack events that do have calorimeter hits, or contain damaged hits;
see [ColliderML-Production#53](https://github.com/OpenDataDetector/ColliderML-Production/issues/53).


