"""Functional helpers for Polars-backed ColliderML tables.

This module is intentionally functional: it operates on dictionaries mapping
table-name -> Polars DataFrame/LazyFrame.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Union

import polars as pl


PolarsTable = Union[pl.DataFrame, pl.LazyFrame]
Tables = Dict[str, PolarsTable]


def table_names(tables: Mapping[str, PolarsTable]) -> list[str]:
    """Return available table names."""
    return sorted(tables.keys())


def select_events(tables: Mapping[str, PolarsTable], event_ids: Union[pl.Series, List[int]]) -> Dict[str, PolarsTable]:
    """Filter all tables to a consistent set of event_ids.

    Args:
        tables: Mapping of table name -> Polars table. Each must have `event_id`.
        event_ids: Polars Series or Python list of event_ids to retain.

    Returns:
        Dict[str, PolarsTable]: filtered tables.
    """
    ids = event_ids.to_list() if isinstance(event_ids, pl.Series) else list(event_ids)
    out: Dict[str, PolarsTable] = {}
    for name, t in tables.items():
        out[name] = t.filter(pl.col("event_id").is_in(ids))
    return out


def collect_tables(tables: Mapping[str, PolarsTable], streaming: bool = False) -> Dict[str, pl.DataFrame]:
    """Collect all lazy tables into eager DataFrames.

    Args:
        tables: Mapping of table name -> Polars table.
        streaming: Whether to enable Polars streaming collection.

    Returns:
        Dict[str, pl.DataFrame]: eager tables.
    """
    out: Dict[str, pl.DataFrame] = {}
    for name, t in tables.items():
        out[name] = t.collect(streaming=streaming) if isinstance(t, pl.LazyFrame) else t
    return out


