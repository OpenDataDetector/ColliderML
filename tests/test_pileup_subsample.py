"""Unit tests for pileup subsampling (no network)."""

from __future__ import annotations

import polars as pl


def test_subsample_pileup_filters_particles_hits_and_calo() -> None:
    from colliderml.physics.pileup import subsample_pileup

    # One event with 4 particles from vertices 1..4; target_vertices=2 keeps only first two.
    particles = pl.DataFrame(
        {
            "event_id": [1],
            "particle_id": [[10, 11, 12, 13]],
            "vertex_primary": [[1, 2, 3, 4]],
            "pdg_id": [[1, 2, 3, 4]],
        }
    )

    tracker_hits = pl.DataFrame(
        {
            "event_id": [1],
            "particle_id": [[10, 12, 13, 11]],
            "x": [[0.0, 1.0, 2.0, 3.0]],
        }
    )

    # Two calo cells with contributions; after filtering, contribs from 12/13 should be removed.
    calo_hits = pl.DataFrame(
        {
            "event_id": [1],
            "detector": [[0, 0]],
            "x": [[0.0, 1.0]],
            "y": [[0.0, 1.0]],
            "z": [[0.0, 1.0]],
            "total_energy": [[100.0, 200.0]],
            "contrib_particle_ids": [[[10, 12], [13]]],
            "contrib_energies": [[[5.0, 7.0], [9.0]]],
            "contrib_times": [[[1.0, 2.0], [3.0]]],
        }
    )

    out = subsample_pileup(
        {"particles": particles, "tracker_hits": tracker_hits, "calo_hits": calo_hits}, target_vertices=2
    )

    p = out["particles"]
    assert isinstance(p, pl.DataFrame)
    assert p["particle_id"].to_list() == [[10, 11]]
    assert p["vertex_primary"].to_list() == [[1, 2]]

    th = out["tracker_hits"]
    assert th["particle_id"].to_list() == [[10, 11]]
    assert th["x"].to_list() == [[0.0, 3.0]]

    ch = out["calo_hits"]
    # Second cell had only contrib from particle 13, so should be dropped.
    assert ch["detector"].to_list() == [[0]]
    assert ch["contrib_particle_ids"].to_list() == [[[10]]]
    assert ch["contrib_energies"].to_list() == [[[5.0]]]
    assert ch["total_energy"].to_list() == [[5.0]]


def test_subsample_pileup_works_lazy() -> None:
    from colliderml.physics.pileup import subsample_pileup

    particles = pl.DataFrame(
        {
            "event_id": [1],
            "particle_id": [[10, 11]],
            "vertex_primary": [[1, 2]],
        }
    ).lazy()
    tracker_hits = pl.DataFrame(
        {"event_id": [1], "particle_id": [[10, 11]], "x": [[0.0, 1.0]]}
    ).lazy()
    calo_hits = pl.DataFrame(
        {
            "event_id": [1],
            "detector": [[0]],
            "x": [[0.0]],
            "y": [[0.0]],
            "z": [[0.0]],
            "total_energy": [[5.0]],
            "contrib_particle_ids": [[[10]]],
            "contrib_energies": [[[5.0]]],
            "contrib_times": [[[1.0]]],
        }
    ).lazy()

    out = subsample_pileup(
        {"particles": particles, "tracker_hits": tracker_hits, "calo_hits": calo_hits}, target_vertices=1
    )
    assert isinstance(out["calo_hits"], pl.LazyFrame)
    collected = {k: v.collect() for k, v in out.items()}
    assert collected["particles"]["particle_id"].to_list() == [[10]]


