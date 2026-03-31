"""Build soft truth clusters from MC decay chains and calorimeter contributions.

A *truth cluster* groups all calorimeter energy deposits that originate from
the same primary ancestor particle (the earliest traceable particle in the
stored decay chain).  Each cell may belong to multiple truth clusters with
different energy fractions — showers overlap.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd
import polars as pl

from colliderml.core.tables import PolarsTable
from colliderml.physics.calibration import apply_calo_calibration, odd_default_calo_calibration
from colliderml.physics.decay import assign_primary_ancestor
from colliderml.polars.flatten import explode_calo_cells_and_contribs, explode_particles


def build_truth_clusters(
    calo_hits: PolarsTable,
    particles: PolarsTable,
    *,
    apply_calibration: bool = True,
) -> pd.DataFrame:
    """Build soft truth cluster assignments from decay chains and calo contributions.

    For every (event, cell, primary_ancestor) triple where the ancestor's decay
    products deposited energy in that cell, one row is produced.  A cell that
    received energy from two distinct ancestors therefore appears twice.

    Args:
        calo_hits: Event-table with calorimeter hit list columns
            (detector, x, y, z, total_energy, contrib_particle_ids,
            contrib_energies, contrib_times).
        particles: Event-table with particle list columns
            (particle_id, parent_id, energy, px, py, pz, …).
        apply_calibration: If True (default), apply the ODD default calo
            calibration before computing truth clusters.

    Returns:
        pandas DataFrame with columns:
            event_id          – event identifier
            cell_index        – per-event cell index (0-based)
            primary_ancestor_id – ID of the root ancestor for this contribution
            energy            – calibrated energy deposited by this ancestor's
                                decay chain in this cell
            x, y, z           – cell position
            detector          – detector enum id
            total_energy      – total (calibrated) cell energy across all ancestors
    """
    # ------------------------------------------------------------------
    # 1. Optionally calibrate calo energies
    # ------------------------------------------------------------------
    if apply_calibration:
        calo_hits = apply_calo_calibration(
            calo_hits, odd_default_calo_calibration(apply_to_contrib=True)
        )

    # ------------------------------------------------------------------
    # 2. Assign primary ancestor to every particle
    # ------------------------------------------------------------------
    particles_labeled = assign_primary_ancestor(particles)

    # ------------------------------------------------------------------
    # 3. Explode calo hits into cells + contributions
    # ------------------------------------------------------------------
    cells_df, contribs_df = explode_calo_cells_and_contribs(calo_hits)

    # ------------------------------------------------------------------
    # 4. Map each contributing particle_id → primary_ancestor_id
    # ------------------------------------------------------------------
    particles_flat = explode_particles(particles_labeled)
    pid_to_ancestor = particles_flat[["event_id", "particle_id", "primary_ancestor_id"]].copy()
    pid_to_ancestor["particle_id"] = pd.to_numeric(pid_to_ancestor["particle_id"], errors="coerce")
    pid_to_ancestor["primary_ancestor_id"] = pd.to_numeric(
        pid_to_ancestor["primary_ancestor_id"], errors="coerce"
    )

    contribs_df["particle_id"] = pd.to_numeric(contribs_df["particle_id"], errors="coerce")

    contribs_with_ancestor = contribs_df.merge(
        pid_to_ancestor, on=["event_id", "particle_id"], how="left"
    )
    # Drop contributions whose particle has no ancestor (e.g. not in particle table)
    contribs_with_ancestor = contribs_with_ancestor.dropna(subset=["primary_ancestor_id"])
    contribs_with_ancestor["primary_ancestor_id"] = contribs_with_ancestor[
        "primary_ancestor_id"
    ].astype("int64")

    # ------------------------------------------------------------------
    # 5. Aggregate energy per (event, cell, ancestor)
    # ------------------------------------------------------------------
    ancestor_energy = (
        contribs_with_ancestor.groupby(
            ["event_id", "cell_index", "primary_ancestor_id"], as_index=False
        )["energy"]
        .sum()
    )

    # ------------------------------------------------------------------
    # 6. Merge cell-level info (position, detector, total_energy)
    # ------------------------------------------------------------------
    pos_cols = [c for c in ["x", "y", "z", "detector", "total_energy"] if c in cells_df.columns]
    truth_clusters = ancestor_energy.merge(
        cells_df[["event_id", "cell_index"] + pos_cols],
        on=["event_id", "cell_index"],
        how="left",
    )

    return truth_clusters
