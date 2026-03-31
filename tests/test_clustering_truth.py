"""Unit tests for clustering truth cluster construction (no network)."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest


def _make_simple_event():
    """Two ancestors (10, 20), three cells, known contributions.

    Cell 0: particle 11 (ancestor 10) deposits 3.0, particle 21 (ancestor 20) deposits 1.0
    Cell 1: particle 11 (ancestor 10) deposits 2.0
    Cell 2: particle 21 (ancestor 20) deposits 5.0
    """
    particles = pl.DataFrame(
        {
            "event_id": [1],
            "particle_id": [[10, 11, 20, 21]],
            "parent_id": [[-1, 10, -1, 20]],
            "energy": [[50.0, 30.0, 40.0, 20.0]],
            "px": [[1.0, 1.0, 1.0, 1.0]],
            "py": [[0.0, 0.0, 0.0, 0.0]],
            "pz": [[0.0, 0.0, 0.0, 0.0]],
        }
    )
    calo_hits = pl.DataFrame(
        {
            "event_id": [1],
            "detector": [[10, 10, 13]],
            "x": [[0.0, 1.0, 2.0]],
            "y": [[0.0, 0.0, 0.0]],
            "z": [[0.0, 0.0, 0.0]],
            "total_energy": [[4.0, 2.0, 5.0]],
            "contrib_particle_ids": [[[11, 21], [11], [21]]],
            "contrib_energies": [[[3.0, 1.0], [2.0], [5.0]]],
            "contrib_times": [[[0.1, 0.2], [0.1], [0.2]]],
        }
    )
    return particles, calo_hits


def test_basic_truth_clusters() -> None:
    from colliderml.clustering.truth import build_truth_clusters

    particles, calo_hits = _make_simple_event()
    tc = build_truth_clusters(calo_hits, particles, apply_calibration=False)

    assert isinstance(tc, pd.DataFrame)
    # Cell 0 has 2 ancestors, cells 1 and 2 have 1 each → 4 rows
    assert len(tc) == 4

    # Check ancestor 10 in cell 0 has energy 3.0
    row = tc[(tc["cell_index"] == 0) & (tc["primary_ancestor_id"] == 10)]
    assert len(row) == 1
    assert abs(row.iloc[0]["energy"] - 3.0) < 1e-6

    # Check ancestor 20 in cell 0 has energy 1.0
    row = tc[(tc["cell_index"] == 0) & (tc["primary_ancestor_id"] == 20)]
    assert len(row) == 1
    assert abs(row.iloc[0]["energy"] - 1.0) < 1e-6


def test_shared_cell_appears_multiple_times() -> None:
    """Cell receiving energy from 2 ancestors should appear twice."""
    from colliderml.clustering.truth import build_truth_clusters

    particles, calo_hits = _make_simple_event()
    tc = build_truth_clusters(calo_hits, particles, apply_calibration=False)

    cell0 = tc[tc["cell_index"] == 0]
    assert len(cell0) == 2
    assert set(cell0["primary_ancestor_id"]) == {10, 20}


def test_single_ancestor_cell() -> None:
    """Cells with a single ancestor should appear once."""
    from colliderml.clustering.truth import build_truth_clusters

    particles, calo_hits = _make_simple_event()
    tc = build_truth_clusters(calo_hits, particles, apply_calibration=False)

    cell1 = tc[tc["cell_index"] == 1]
    assert len(cell1) == 1
    assert cell1.iloc[0]["primary_ancestor_id"] == 10
    assert abs(cell1.iloc[0]["energy"] - 2.0) < 1e-6


def test_position_columns_present() -> None:
    from colliderml.clustering.truth import build_truth_clusters

    particles, calo_hits = _make_simple_event()
    tc = build_truth_clusters(calo_hits, particles, apply_calibration=False)

    for col in ["x", "y", "z", "detector", "total_energy"]:
        assert col in tc.columns, f"Missing column: {col}"


def test_multi_event() -> None:
    """Two events processed together."""
    from colliderml.clustering.truth import build_truth_clusters

    particles = pl.DataFrame(
        {
            "event_id": [1, 2],
            "particle_id": [[10, 11], [30, 31]],
            "parent_id": [[-1, 10], [-1, 30]],
            "energy": [[50.0, 30.0], [40.0, 20.0]],
            "px": [[1.0, 1.0], [1.0, 1.0]],
            "py": [[0.0, 0.0], [0.0, 0.0]],
            "pz": [[0.0, 0.0], [0.0, 0.0]],
        }
    )
    calo_hits = pl.DataFrame(
        {
            "event_id": [1, 2],
            "detector": [[10], [13]],
            "x": [[0.0], [1.0]],
            "y": [[0.0], [0.0]],
            "z": [[0.0], [0.0]],
            "total_energy": [[5.0], [3.0]],
            "contrib_particle_ids": [[[11]], [[31]]],
            "contrib_energies": [[[5.0]], [[3.0]]],
            "contrib_times": [[[0.1]], [[0.2]]],
        }
    )
    tc = build_truth_clusters(calo_hits, particles, apply_calibration=False)
    assert set(tc["event_id"]) == {1, 2}
    assert len(tc) == 2
