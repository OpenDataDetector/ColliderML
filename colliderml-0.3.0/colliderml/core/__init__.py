"""Core functionality for ColliderML."""

from . import data
from .loader import load_tables, apply_max_events
from .tables import select_events, collect_tables, table_names

__all__ = [
    "data",
    "load_tables",
    "apply_max_events",
    "select_events",
    "collect_tables",
    "table_names",
]