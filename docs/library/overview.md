# Library Overview

ColliderML ships a small Python library (`colliderml`) to make working with the dataset easier and more reproducible.

This section documents the **library features** (CLI, local loading, and utilities). If you only need a quick download, the HuggingFace `datasets` approach in the main docs still works — but the library workflow is often more convenient for analysis.

## Recommended workflow

Most analysis workflows follow this pattern:

- **Download** Parquet shards once to a local cache using the CLI.
- **Load** Parquet from disk (no network calls) with Polars-backed helpers.
- **Explode** event-as-row tables into flat (object-as-row) tables.
- **Apply utilities** (pileup subsampling, decay traversal labels, calibration).

```bash
colliderml download --channels ttbar --pileup pu0 --objects particles,tracker_hits,calo_hits,tracks --max-events 200
```

```python
from colliderml.core import load_tables, collect_tables
from colliderml.polars import explode_particles

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
frames = collect_tables(tables)  # dict[str, pl.DataFrame]
particles_evt = frames["particles"]          # event-as-row
particles_flat = explode_particles(particles_evt)  # particle-as-row
```

## What you’ll find here

- **CLI**: download + cache layout (`colliderml download`)
- **Loading**: config-driven local loading (`load_tables`)
- **Exploding**: event-table → flat tables (`explode_*`)
- **Physics utilities**: pileup subsampling, decay traversal labels, calibration
- **Benchmarks**: download-speed benchmarking (CI warn-only)


