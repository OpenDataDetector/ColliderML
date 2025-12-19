"""Unit tests for decay-chain traversal utilities (no network)."""

from __future__ import annotations

import polars as pl


def test_assign_primary_ancestor_with_primary_flags() -> None:
    from colliderml.physics.decay import assign_primary_ancestor

    # Two primaries (10, 20) and two descendants (11->10, 21->20).
    df = pl.DataFrame(
        {
            "event_id": [1],
            "particle_id": [[10, 11, 20, 21]],
            "parent_id": [[-1, 10, -1, 20]],
            "primary": [[True, False, True, False]],
        }
    )
    out = assign_primary_ancestor(df)
    # `primary` is ignored; roots inferred from parent links only.
    assert out["primary_ancestor_id"].to_list() == [[10, 10, 20, 20]]


def test_assign_primary_ancestor_missing_parent_self() -> None:
    from colliderml.physics.decay import assign_primary_ancestor

    df = pl.DataFrame(
        {
            "event_id": [1],
            "particle_id": [[10, 11]],
            "parent_id": [[-1, 999]],  # broken link
        }
    )
    out = assign_primary_ancestor(df, missing_parent_strategy="self")
    assert out["primary_ancestor_id"].to_list() == [[10, 11]]


def test_assign_primary_ancestor_loop_breaks() -> None:
    from colliderml.physics.decay import assign_primary_ancestor

    # 10 -> 11, 11 -> 10 forms a loop
    df = pl.DataFrame(
        {"event_id": [1], "particle_id": [[10, 11]], "parent_id": [[11, 10]]}
    )
    out = assign_primary_ancestor(df, missing_parent_strategy="self")
    assert out["primary_ancestor_id"].to_list() == [[10, 11]]


def test_assign_primary_ancestor_lazy() -> None:
    from colliderml.physics.decay import assign_primary_ancestor

    lf = pl.DataFrame(
        {
            "event_id": [1],
            "particle_id": [[10, 11]],
            "parent_id": [[-1, 10]],
            "primary": [[True, False]],
        }
    ).lazy()
    out = assign_primary_ancestor(lf)
    assert isinstance(out, pl.LazyFrame)
    collected = out.collect()
    # `primary` is ignored; roots inferred from parent links only.
    assert collected["primary_ancestor_id"].to_list() == [[10, 10]]


def test_assign_primary_ancestor_unique_count_equals_roots() -> None:
    """Unique ancestor count must equal number of root particles (parent not in particle_id set)."""
    from colliderml.physics.decay import assign_primary_ancestor

    # 3 roots: 10 (parent=-1), 20 (parent=-1), 30 (parent=999 broken)
    # 3 descendants: 11->10, 21->20, 31->30
    df = pl.DataFrame(
        {
            "event_id": [1],
            "particle_id": [[10, 11, 20, 21, 30, 31]],
            "parent_id": [[-1, 10, -1, 20, 999, 30]],  # 999 not in particle_id
        }
    )
    out = assign_primary_ancestor(df)
    ancestors = out["primary_ancestor_id"].to_list()[0]
    n_unique_ancestors = len(set(ancestors))

    # Count roots: parent_id == -1 or parent_id not in particle_id
    pids = set(df["particle_id"].to_list()[0])
    parents = df["parent_id"].to_list()[0]
    n_roots = sum(1 for p in parents if p == -1 or p not in pids)

    assert n_unique_ancestors == n_roots, f"Expected {n_roots} unique ancestors, got {n_unique_ancestors}"
    assert ancestors == [10, 10, 20, 20, 30, 30]


