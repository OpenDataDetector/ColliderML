# Library Overview

ColliderML ships a small Python library (`colliderml`) to make working with the dataset easier and more reproducible.

This section documents the **library features** (CLI, local loading, and utilities). If you only need a quick download, the HuggingFace `datasets` approach in the main docs still works — but the library workflow is often more convenient for analysis.

## Recommended workflow

Most analysis workflows follow this pattern:

- **Load** Parquet shards with the one-liner `colliderml.load(...)` — downloads on first call, then caches.
- **Explode** event-as-row tables into flat (object-as-row) tables.
- **Apply utilities** (pileup subsampling, decay traversal labels, calibration).
- *(optional)* **Simulate** new events yourself when you need more data or a different channel.
- *(optional)* **Score** your model against a built-in benchmark task.

```python
import colliderml
from colliderml.polars import explode_particles

# Download-and-load in one call; frames is dict[str, pl.DataFrame]
frames = colliderml.load("ttbar_pu0", tables=["particles"], max_events=200)
particles_flat = explode_particles(frames["particles"])   # particle-as-row
```

Power users can still reach the low-level config-driven loader directly:

```python
from colliderml.core import load_tables, collect_tables

tables = load_tables(
    {
        "channels": "ttbar",
        "pileup": "pu0",
        "objects": ["particles"],
        "split": "train",
        "lazy": True,
        "max_events": 200,
    }
)
frames = collect_tables(tables)
```

And when you need more events than the released dataset provides,
generate your own:

```python
result = colliderml.simulate(preset="ttbar-quick")
frames = colliderml.load("ttbar_pu0")   # still works — or read result.run_dir directly
```

## What you'll find here

- **CLI**: unified `colliderml` command (download, simulate, balance, …)
- **Loading**: one-line `colliderml.load()` + the config-driven `load_tables` under the hood
- **Exploding**: event-table → flat tables (`explode_*`)
- **Physics utilities**: pileup subsampling, decay traversal labels, calibration
- **Simulate**: run the full simulation pipeline (local or remote)
- **Remote (SaaS)**: HTTP client for the ColliderML backend
- **Tasks (benchmarks)**: built-in benchmark task registry + reference baselines
- **Benchmarks (timing)**: CI warn-only download-speed harness


