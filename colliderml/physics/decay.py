"""Monte Carlo decay-graph utilities.

Implements per-event traversal of parent pointers to assign a primary ancestor ID
to every particle, handling edge cases like missing parents and loops.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Union

import polars as pl

from colliderml.core.tables import PolarsTable


def assign_primary_ancestor(
    particles: PolarsTable,
    *,
    particle_id_col: str = "particle_id",
    parent_id_col: str = "parent_id",
    output_col: str = "primary_ancestor_id",
    missing_parent_strategy: str = "self",
) -> PolarsTable:
    """Assign a primary ancestor particle id to all particles in each event.

    This expects a particles event-table with list columns for particle data:
    - `particle_id`: list[int]
    - `parent_id`: list[int] (same length; parent pointer per particle)

    Algorithm (per event):
    - Build mapping id -> parent_id for all particles in event.
    - For each particle:
      - Follow parent pointers until:
        - parent is null/-1: choose self (default) or null depending on strategy
        - parent is not present in this event's `particle_id` list (common after pruning): choose self (default) or null
        - loop detected: choose self and mark as loop-broken (not yet stored)

    Args:
        particles: Polars DataFrame/LazyFrame with event_id and particle list columns.
        particle_id_col: Name of particle id list column.
        parent_id_col: Name of parent id list column.
        output_col: Name of output list column to add.
        missing_parent_strategy: What to do when a parent is missing:
            - "self": assign self as ancestor
            - "null": assign None as ancestor

    Returns:
        PolarsTable: same type as input, with added `output_col` list column.
    """
    if missing_parent_strategy not in ("self", "null"):
        raise ValueError("missing_parent_strategy must be 'self' or 'null'")

    # Use a struct UDF to operate row-wise over lists (works for eager and lazy).
    def _per_event(row: dict) -> list[Optional[int]]:
        pids: List[int] = row.get(particle_id_col) or []
        parents: List[Optional[int]] = row.get(parent_id_col) or []

        # Build id->index mapping; parent pointers may refer by particle id.
        idx_by_id: Dict[int, int] = {pid: i for i, pid in enumerate(pids)}

        # Normalize parents list length.
        if len(parents) < len(pids):
            parents = parents + [None] * (len(pids) - len(parents))
        else:
            parents = parents[: len(pids)]

        out: List[Optional[int]] = []
        for i, pid in enumerate(pids):
            seen: set[int] = set()
            cur_id: int = pid
            while True:
                if cur_id in seen:
                    # loop detected - assign self as ancestor for this particle
                    out.append(pid if missing_parent_strategy == "self" else None)
                    break
                seen.add(cur_id)

                cur_idx = idx_by_id.get(cur_id)
                if cur_idx is None:
                    # cur_id not in table (shouldn't happen for first iter, but safety)
                    out.append(pid if missing_parent_strategy == "self" else None)
                    break

                parent = parents[cur_idx]
                if parent is None or parent == -1:
                    # cur_id is the root (no parent) -> assign cur_id as ancestor
                    out.append(cur_id if missing_parent_strategy == "self" else None)
                    break

                # If the parent isn't present in this event's stored particle list,
                # cur_id is effectively a root (common after pruning unstable parents).
                if int(parent) not in idx_by_id:
                    out.append(cur_id if missing_parent_strategy == "self" else None)
                    break

                cur_id = int(parent)

        return out

    cols = [particle_id_col, parent_id_col]

    return particles.with_columns(
        pl.struct(cols).map_elements(_per_event, return_dtype=pl.List(pl.Int64)).alias(output_col)
    )


