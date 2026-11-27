"""collect_tables must work with the installed Polars (2.0 removed collect(streaming=...))."""

import polars as pl
import pytest

from colliderml.core import collect_tables


@pytest.mark.parametrize("streaming", [False, True])
def test_collect_tables_lazy(streaming):
    tables = {
        "particles": pl.LazyFrame({"event_id": [0, 1], "px": [[1.0], [2.0, 3.0]]}),
        "tracks": pl.DataFrame({"event_id": [0, 1]}),
    }
    out = collect_tables(tables, streaming=streaming)
    assert isinstance(out["particles"], pl.DataFrame)
    assert out["particles"]["event_id"].to_list() == [0, 1]
    assert out["tracks"] is tables["tracks"]
