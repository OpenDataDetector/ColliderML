"""Polars helpers to explode nested ColliderML event tables into flat tables."""

from __future__ import annotations

from typing import List, Tuple

import polars as pl

from colliderml.core.tables import PolarsTable


def _to_pandas(table: PolarsTable):
    """Collect (if lazy) and convert to pandas.DataFrame."""
    import pandas as pd  # local import to keep pandas optional at import-time

    if isinstance(table, pl.LazyFrame):
        table = table.collect()
    out = table.to_pandas()
    # Type hint for callers/tests
    assert isinstance(out, pd.DataFrame)
    return out


def explode_event_table(
    table: PolarsTable,
    *,
    list_cols: List[str],
    index_name: str,
) -> PolarsTable:
    """Explode an event-table into a row-per-object table.

    Args:
        table: Event table with `event_id` and list columns.
        list_cols: List columns to explode in lock-step (same lengths per row).
        index_name: Name of generated per-event row index after exploding.

    Returns:
        PolarsTable: exploded table with `(event_id, index_name, ...)`.
    """
    # Explode list columns in lock-step, then create a per-event index.
    # `with_row_index` would create a global index; we want 0..N-1 within each event.
    # Polars cum_count() is 1-based, so subtract 1 to get 0-based.
    return table.explode(list_cols).with_columns(
        (pl.col("event_id").cum_count().over("event_id") - 1).alias(index_name)
    )


def explode_particles(particles: PolarsTable, *, index_name: str = "particle_index"):
    """Explode particles event-table into one row per particle (returns pandas)."""
    # Explode all list columns except event_id.
    if isinstance(particles, pl.LazyFrame):
        schema = particles.collect_schema()
        list_cols = [c for c, dt in schema.items() if c != "event_id" and isinstance(dt, pl.List)]
    else:
        list_cols = [c for c, dt in particles.schema.items() if c != "event_id" and isinstance(dt, pl.List)]
    return _to_pandas(explode_event_table(particles, list_cols=list_cols, index_name=index_name))


def explode_tracker_hits(tracker_hits: PolarsTable, *, index_name: str = "hit_index"):
    """Explode tracker hits event-table into one row per hit (returns pandas)."""
    if isinstance(tracker_hits, pl.LazyFrame):
        schema = tracker_hits.collect_schema()
        list_cols = [c for c, dt in schema.items() if c != "event_id" and isinstance(dt, pl.List)]
    else:
        list_cols = [c for c, dt in tracker_hits.schema.items() if c != "event_id" and isinstance(dt, pl.List)]
    return _to_pandas(explode_event_table(tracker_hits, list_cols=list_cols, index_name=index_name))


def explode_calo_cells_and_contribs(
    calo_hits: PolarsTable,
    *,
    cell_index_name: str = "cell_index",
    contrib_index_name: str = "contrib_index",
) -> Tuple["pandas.DataFrame", "pandas.DataFrame"]:
    """Explode calo hits into (cells_df, contribs_df) (returns pandas).

    Expected calo schema:
    - event_id
    - cell list columns: detector, x, y, z, total_energy, ...
    - contribution list-of-lists: contrib_particle_ids, contrib_energies, contrib_times

    Returns:
        (cells_df, contribs_df)
    """
    contrib_cols = ["contrib_particle_ids", "contrib_energies", "contrib_times"]
    if isinstance(calo_hits, pl.LazyFrame):
        schema = calo_hits.collect_schema()
        all_cols = schema.names()
        cell_cols = [
            c
            for c in all_cols
            if c != "event_id"
            and c not in contrib_cols
            and c != "total_energy"
            and isinstance(schema[c], pl.List)
        ]
    else:
        all_cols = calo_hits.columns
        cell_cols = [
            c
            for c, dt in calo_hits.schema.items()
            if c != "event_id" and c not in contrib_cols and c != "total_energy" and isinstance(dt, pl.List)
        ]

    explode_cols = cell_cols + (["total_energy"] if "total_energy" in all_cols else []) + contrib_cols
    lf = calo_hits.explode(explode_cols).with_columns(
        (pl.col("event_id").cum_count().over("event_id") - 1).alias(cell_index_name)
    )

    cells = lf.select(["event_id", cell_index_name] + cell_cols + (["total_energy"] if "total_energy" in all_cols else []))

    # Contribution-level: explode inner lists in lock-step, then unnest to scalar cols.
    contribs = (
        lf.select(["event_id", cell_index_name] + contrib_cols)
        .explode(contrib_cols)
        .with_columns(
            (pl.col(cell_index_name).cum_count().over(["event_id", cell_index_name]) - 1).alias(contrib_index_name)
        )
        .select(
            ["event_id", cell_index_name, contrib_index_name]
            + [
                pl.col("contrib_particle_ids").alias("particle_id"),
                pl.col("contrib_energies").alias("energy"),
                pl.col("contrib_times").alias("time"),
            ]
        )
    )

    return _to_pandas(cells), _to_pandas(contribs)


