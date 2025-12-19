"""Pileup subsampling utilities.

Implements a physically consistent pileup reduction at the event level by
removing high-index vertices and all detector activity attributable to particles
from those vertices.
"""

from __future__ import annotations

from typing import Dict, List, Mapping

import polars as pl

from colliderml.core.tables import PolarsTable


def subsample_pileup(
    tables: Mapping[str, PolarsTable],
    *,
    target_vertices: int,
    particles_key: str = "particles",
    tracker_hits_key: str = "tracker_hits",
    calo_hits_key: str = "calo_hits",
) -> Dict[str, PolarsTable]:
    """Subsample pileup by removing vertices with `vertex_primary > target_vertices`.

    This assumes the ColliderML schema:
    - `particles` table has list columns including:
      - `particle_id` (list[uint64])
      - `vertex_primary` (list[int]) with values 1..N
    - `tracker_hits` has list columns including `particle_id` aligned with hit fields.
    - `calo_hits` has:
      - cell-level list columns (e.g. detector, x,y,z,total_energy)
      - contribution list-of-lists: `contrib_particle_ids`, `contrib_energies`, `contrib_times`

    Args:
        tables: Mapping of object name -> Polars DataFrame/LazyFrame (one row per event).
        target_vertices: Target number of vertices to keep. Keeps vertices 1..K.
        particles_key: Key in `tables` for particle truth.
        tracker_hits_key: Key in `tables` for tracker hits (optional).
        calo_hits_key: Key in `tables` for calo hits (optional).

    Returns:
        Dict[str, PolarsTable]: new mapping with subsampled tables for keys present.
    """
    if target_vertices <= 0:
        raise ValueError("target_vertices must be positive")
    if particles_key not in tables:
        raise KeyError(f"Missing required particles table '{particles_key}'")

    particles = tables[particles_key]
    particles_filtered = _filter_particles_by_vertex_primary(particles, target_vertices=target_vertices)
    kept_ids_table = _kept_particle_ids_table(particles_filtered)

    out: Dict[str, PolarsTable] = dict(tables)
    out[particles_key] = particles_filtered

    if tracker_hits_key in tables:
        out[tracker_hits_key] = _filter_tracker_hits_by_particle_ids(
            tables[tracker_hits_key], kept_ids_table=kept_ids_table
        )

    if calo_hits_key in tables:
        out[calo_hits_key] = _filter_calo_hits_by_particle_ids(
            tables[calo_hits_key], kept_ids_table=kept_ids_table
        )

    return out


def _kept_particle_ids_table(particles_filtered: PolarsTable) -> PolarsTable:
    """Build a small table of (event_id, _kept_particle_ids) for joining."""
    if isinstance(particles_filtered, pl.LazyFrame):
        return particles_filtered.select(
            pl.col("event_id"), pl.col("particle_id").alias("_kept_particle_ids")
        )
    return particles_filtered.select(
        "event_id", pl.col("particle_id").alias("_kept_particle_ids")
    )


def _membership_mask(list_col: str, kept_list_col: str) -> pl.Expr:
    """Build a list[bool] mask indicating whether each element is in kept ids.

    Args:
        list_col: Name of list column containing ids to test.
        kept_list_col: Name of list column containing the kept ids set.

    Returns:
        pl.Expr: list[bool] mask aligned with `list_col`.
    """
    # Note: Polars forbids referencing named columns inside `list.eval`.
    # We use a struct UDF here to keep semantics correct and deterministic.
    return pl.struct([list_col, kept_list_col]).map_elements(
        lambda s: [x in set(s[kept_list_col] or []) for x in (s[list_col] or [])],
        return_dtype=pl.List(pl.Boolean),
    )


def _filter_list_by_mask(list_col: str, mask_col: str, return_dtype: pl.DataType) -> pl.Expr:
    """Filter a list column by an aligned list[bool] mask."""
    return pl.struct([list_col, mask_col]).map_elements(
        lambda s: [x for x, keep in zip((s[list_col] or []), (s[mask_col] or [])) if keep],
        return_dtype=return_dtype,
    )


def _filter_particles_by_vertex_primary(particles: PolarsTable, *, target_vertices: int) -> PolarsTable:
    """Filter all particle list columns to keep only vertex_primary <= K."""
    # Build a per-particle boolean mask list based on vertex_primary.
    # NOTE: Polars `list.eval` has constraints that can error in some versions;
    # we use a small expression UDF for correctness and stability.
    keep_mask_expr = pl.col("vertex_primary").map_elements(
        lambda xs: [v <= target_vertices for v in xs],
        return_dtype=pl.List(pl.Boolean),
    )

    # Identify list columns dynamically (excluding event_id).
    # For LazyFrame, infer schema from a zero-row collect.
    if isinstance(particles, pl.LazyFrame):
        schema = particles.collect_schema()
        list_cols = [
            name
            for name, dtype in schema.items()
            if name != "event_id" and isinstance(dtype, pl.List)
        ]
        # Add mask, then filter each list col using a struct UDF (stable across Polars versions).
        lf = particles.with_columns(_keep_particle=keep_mask_expr)
        filtered_exprs = []
        for c in list_cols:
            dtype = schema[c]
            filtered_exprs.append(_filter_list_by_mask(c, "_keep_particle", dtype).alias(c))
        return lf.with_columns(filtered_exprs).drop("_keep_particle")

    df = particles
    list_cols = [
        name
        for name, dtype in df.schema.items()
        if name != "event_id" and isinstance(dtype, pl.List)
    ]
    df2 = df.with_columns(_keep_particle=keep_mask_expr)
    filtered_exprs = []
    for c in list_cols:
        dtype = df.schema[c]
        filtered_exprs.append(_filter_list_by_mask(c, "_keep_particle", dtype).alias(c))
    return df2.with_columns(filtered_exprs).drop("_keep_particle")


def _filter_tracker_hits_by_particle_ids(
    tracker_hits: PolarsTable, *, kept_ids_table: PolarsTable
) -> PolarsTable:
    """Filter tracker hit list columns by whether hit particle_id is in kept ids."""
    # Hit mask is a list[bool] aligned with hit list columns.
    hit_mask = _membership_mask("particle_id", "_kept_particle_ids")

    if isinstance(tracker_hits, pl.LazyFrame):
        schema = tracker_hits.collect_schema()
        # Only filter original hit list columns; never include helper columns.
        list_cols = [
            name
            for name, dtype in schema.items()
            if name != "event_id" and not name.startswith("_") and isinstance(dtype, pl.List)
        ]
        lf = tracker_hits.join(kept_ids_table, on="event_id", how="left")
        lf = lf.with_columns(_keep_hit=hit_mask)
        filtered_exprs = []
        for c in list_cols:
            dtype = schema[c]
            filtered_exprs.append(_filter_list_by_mask(c, "_keep_hit", dtype).alias(c))
        return (
            lf.with_columns(filtered_exprs)
            .drop(["_kept_particle_ids", "_keep_hit"])
        )

    df = tracker_hits.join(kept_ids_table, on="event_id", how="left")
    if "_keep_hit" in df.columns:
        df = df.drop("_keep_hit")
    df = df.with_columns(_keep_hit=hit_mask)
    # Only filter original hit list columns (exclude helper columns).
    list_cols = [
        name
        for name, dtype in tracker_hits.schema.items()
        if name != "event_id" and not name.startswith("_") and isinstance(dtype, pl.List)
    ]
    filtered_exprs = []
    for c in list_cols:
        dtype = tracker_hits.schema[c]
        filtered_exprs.append(_filter_list_by_mask(c, "_keep_hit", dtype).alias(c))
    return df.with_columns(filtered_exprs).drop(["_kept_particle_ids", "_keep_hit"])


def _filter_calo_hits_by_particle_ids(
    calo_hits: PolarsTable, *, kept_ids_table: PolarsTable
) -> PolarsTable:
    """Filter calo cell contributions and drop cells with no remaining contributions."""
    required = {"contrib_particle_ids", "contrib_energies", "contrib_times"}
    if isinstance(calo_hits, pl.LazyFrame):
        cols = set(calo_hits.collect_schema().names())
    else:
        cols = set(calo_hits.columns)
    missing = required - cols
    if missing:
        raise KeyError(f"calo_hits missing required columns: {sorted(missing)}")

    # Identify cell-level list columns (List[Scalar]) to explode lock-step.
    # Contribution columns are handled separately and should remain List[Scalar] after explode.
    contrib_cols = ["contrib_particle_ids", "contrib_energies", "contrib_times"]
    cell_cols: list[str]
    if isinstance(calo_hits, pl.LazyFrame):
        schema = calo_hits.limit(0).collect().schema
        cell_cols = [
            name
            for name, dtype in schema.items()
            # Never treat total_energy as a regular cell list column; we recompute it.
            if name != "event_id"
            and name not in contrib_cols
            and name != "total_energy"
            and isinstance(dtype, pl.List)
        ]
    else:
        cell_cols = [
            name
            for name, dtype in calo_hits.schema.items()
            if name != "event_id"
            and name not in contrib_cols
            and name != "total_energy"
            and isinstance(dtype, pl.List)
        ]

    # Explode one row per cell, keeping contrib cols as lists-of-scalars.
    base = calo_hits.join(kept_ids_table, on="event_id", how="left")
    # Rename raw total_energy if present to avoid column name collisions downstream.
    if isinstance(base, pl.LazyFrame):
        if "total_energy" in base.collect_schema().names():
            base = base.rename({"total_energy": "_total_energy_raw"})
    else:
        if "total_energy" in base.columns:
            base = base.rename({"total_energy": "_total_energy_raw"})

    exploded = base.explode(cell_cols + contrib_cols)

    # Filter inner contribution lists in lock-step based on kept particle ids.
    contrib_mask = _membership_mask("contrib_particle_ids", "_kept_particle_ids")
    exploded = exploded.with_columns(_keep_contrib=contrib_mask)

    # Resolve contribution list dtypes post-explode so we preserve original types.
    if isinstance(exploded, pl.LazyFrame):
        ex_schema = exploded.collect_schema()
    else:
        ex_schema = exploded.schema

    filtered = (
        exploded.with_columns(
            _filter_list_by_mask("contrib_particle_ids", "_keep_contrib", ex_schema["contrib_particle_ids"]).alias(
                "contrib_particle_ids"
            ),
            _filter_list_by_mask("contrib_energies", "_keep_contrib", ex_schema["contrib_energies"]).alias(
                "contrib_energies"
            ),
            _filter_list_by_mask("contrib_times", "_keep_contrib", ex_schema["contrib_times"]).alias("contrib_times"),
        )
        .with_columns(total_energy=pl.col("contrib_energies").list.sum())
        .filter(pl.col("contrib_particle_ids").list.len() > 0)
        .drop(["_kept_particle_ids", "_keep_contrib"])
    )

    # Re-nest to one row per event by collecting exploded cells back into lists.
    # Group-by preserves row order as observed in the exploded frame.
    agg_exprs = [pl.col(c).implode() for c in cell_cols + contrib_cols + ["total_energy"]]
    # Preserve `_total_energy_raw` if present (handy for debugging); optional downstream.
    if "_total_energy_raw" in (filtered.collect_schema().names() if isinstance(filtered, pl.LazyFrame) else filtered.columns):
        agg_exprs.append(pl.col("_total_energy_raw").implode())
    return filtered.group_by("event_id", maintain_order=True).agg(agg_exprs)


