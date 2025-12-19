"""Unit tests for Polars flatten/explode helpers (no network)."""

from __future__ import annotations

import pandas as pd
import polars as pl


def test_explode_particles() -> None:
    from colliderml.polars.flatten import explode_particles

    particles = pl.DataFrame(
        {"event_id": [1], "particle_id": [[10, 11]], "px": [[1.0, 2.0]]}
    )
    out = explode_particles(particles)
    assert isinstance(out, pd.DataFrame)
    assert out["event_id"].tolist() == [1, 1]
    assert out["particle_index"].tolist() == [0, 1]
    assert out["particle_id"].tolist() == [10, 11]


def test_explode_calo_cells_and_contribs() -> None:
    from colliderml.polars.flatten import explode_calo_cells_and_contribs

    calo = pl.DataFrame(
        {
            "event_id": [1],
            "detector": [[0, 1]],
            "x": [[0.0, 1.0]],
            "total_energy": [[5.0, 6.0]],
            "contrib_particle_ids": [[[10, 11], [12]]],
            "contrib_energies": [[[1.0, 2.0], [3.0]]],
            "contrib_times": [[[0.1, 0.2], [0.3]]],
        }
    )
    cells, contribs = explode_calo_cells_and_contribs(calo)
    assert isinstance(cells, pd.DataFrame)
    assert isinstance(contribs, pd.DataFrame)
    assert cells.shape[0] == 2
    assert contribs.shape[0] == 3
    assert set(contribs.columns) == {"event_id", "cell_index", "contrib_index", "particle_id", "energy", "time"}
    # contrib_index should be per-cell (0..n-1)
    # first cell has 2 contribs -> 0,1; second cell has 1 contrib -> 0
    contribs_sorted = contribs.sort_values(["cell_index", "contrib_index"])
    assert contribs_sorted["cell_index"].tolist() == [0, 0, 1]
    assert contribs_sorted["contrib_index"].tolist() == [0, 1, 0]


