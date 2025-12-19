# Exploding: Event Tables → Flat Tables

ColliderML stores each object collection as an **event table**:

- one row per `event_id`
- many columns are lists (one entry per particle/hit/cell in that event)

The `colliderml.polars` helpers “explode” these list columns into a flat, object-as-row table.

## Recommended pattern

Filter to the event(s) you care about **before** exploding:

```python
import polars as pl
from colliderml.polars import explode_particles

evt = frames["particles"].filter(pl.col("event_id") == event_id)
particles_flat = explode_particles(evt)
```

## Particles

```python
from colliderml.polars import explode_particles

particles_flat = explode_particles(frames["particles"])
```

Output includes:

- `event_id`
- `particle_index` (per-event row index)
- particle columns (scalar per row, previously list elements)

## Tracker hits

```python
from colliderml.polars import explode_tracker_hits

hits_flat = explode_tracker_hits(frames["tracker_hits"])
```

## Calorimeter hits (cells + contributions)

Calorimeter hits have nested contribution structure. Use:

```python
from colliderml.polars import explode_calo_cells_and_contribs

cells, contribs = explode_calo_cells_and_contribs(frames["calo_hits"])
```

- `cells`: one row per calorimeter cell hit
- `contribs`: one row per contribution (cell position repeated per contrib)



