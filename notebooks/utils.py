"""Notebook helpers: calo/particle data prep and energy/ratio/response plots.

Used by colliderml_loader_exploration.ipynb to keep cells short.
Supports both single-event and all-events (no event filter) flows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from colliderml.physics import assign_primary_ancestor, CALO_DETECTOR_CODES
from colliderml.physics.calibration import apply_calo_calibration, odd_default_calo_calibration
from colliderml.polars import explode_calo_cells_and_contribs, explode_particles
from colliderml.viz import eta_from_xyz, eta_from_pxpypz, plot_binned_sums_with_xerr


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_calo_and_particles(
    frames: Dict[str, pl.DataFrame],
    event_ids: Optional[List[int]] = None,
    *,
    apply_calibration: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pl.DataFrame]:
    """Build contribs (calo), particles_flat (pandas), and particles_evt (Polars) for plotting.

    Args:
        frames: Dict with "calo_hits" and "particles" event-tables.
        event_ids: If provided, restrict to these event_id values; if None, use all events.
        apply_calibration: Whether to apply ODD default calo calibration.

    Returns:
        (contribs, particles_flat, particles_evt)
        - contribs: pandas, one row per (event_id, cell_index, particle_id) with energy, x, y, z, detector.
        - particles_flat: pandas, one row per particle.
        - particles_evt: Polars event-table (filtered to event_ids if provided).
    """
    calo_evt = frames["calo_hits"]
    particles_evt = frames["particles"]
    if event_ids is not None:
        calo_evt = calo_evt.filter(pl.col("event_id").is_in(event_ids))
        particles_evt = particles_evt.filter(pl.col("event_id").is_in(event_ids))
    if not calo_evt.height:
        raise RuntimeError("No calo hits for selected event(s)")
    if apply_calibration:
        calo_evt = apply_calo_calibration(calo_evt, odd_default_calo_calibration(apply_to_contrib=True))
    cells, contribs = explode_calo_cells_and_contribs(calo_evt)
    pos_cols = [c for c in ["x", "y", "z", "detector"] if c in cells.columns]
    contribs = contribs.merge(
        cells[["event_id", "cell_index"] + pos_cols],
        on=["event_id", "cell_index"],
        how="left",
    )
    particles_flat = explode_particles(particles_evt)
    return contribs, particles_flat, particles_evt


def _root_mask_per_event(particles_flat: pd.DataFrame) -> np.ndarray:
    """Root = no parent, or parent not in this event's particle set (particle IDs are per-event)."""
    parent = pd.to_numeric(particles_flat["parent_id"], errors="coerce")
    # (event_id, particle_id) pairs that exist in this table
    valid = particles_flat[["event_id", "particle_id"]].copy()
    valid["particle_id"] = pd.to_numeric(valid["particle_id"], errors="coerce")
    valid = valid.dropna(subset=["particle_id"])
    valid = valid.astype({"particle_id": "int64"})
    valid = valid.drop_duplicates()
    valid = valid.rename(columns={"particle_id": "parent_id"})
    # Merge: does (event_id, parent) exist as a particle in that event?
    pp = particles_flat[["event_id"]].copy()
    pp["_parent_num"] = parent
    merged = pp.merge(
        valid,
        left_on=["event_id", "_parent_num"],
        right_on=["event_id", "parent_id"],
        how="left",
        indicator=True,
    )
    parent_in_event = merged["_merge"] == "both"
    mask_root = (parent.isna() | (parent == -1) | ~parent_in_event).to_numpy(dtype=bool)
    return mask_root


def _ecal_hcal_masks(det: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mask_ecal, mask_hcal) from detector id array."""
    _ecal_ids = [
        CALO_DETECTOR_CODES["ecal_neg_endcap"],
        CALO_DETECTOR_CODES["ecal_barrel"],
        CALO_DETECTOR_CODES["ecal_pos_endcap"],
    ]
    _hcal_ids = [
        CALO_DETECTOR_CODES["hcal_neg_endcap"],
        CALO_DETECTOR_CODES["hcal_barrel"],
        CALO_DETECTOR_CODES["hcal_pos_endcap"],
    ]
    mask_ecal = np.isin(det, _ecal_ids)
    mask_hcal = np.isin(det, _hcal_ids)
    return mask_ecal, mask_hcal


def energy_plot_arrays(
    contribs: pd.DataFrame,
    particles_flat: pd.DataFrame,
    eta_max: float = 3.0,
) -> Dict[str, Any]:
    """Build arrays needed for energy vs eta and ratio vs eta plots.

    Args:
        contribs: From prepare_calo_and_particles (has energy, x, y, z, detector).
        particles_flat: From prepare_calo_and_particles (has energy, px, py, pz, parent_id, particle_id).
        eta_max: Restrict to |eta| < eta_max.

    Returns:
        Dict with: eta_calo, energy_dep, mask_ecal, mask_hcal, mask_total_calo, eta_mask_calo,
        eta_part, truth_energy, mask_root, eta_mask_part.
    """
    eta_calo = eta_from_xyz(
        contribs["x"].to_numpy().astype(float),
        contribs["y"].to_numpy().astype(float),
        contribs["z"].to_numpy().astype(float),
    )
    energy_dep = contribs["energy"].to_numpy().astype(float)
    det = contribs["detector"].to_numpy().astype(int)
    mask_ecal, mask_hcal = _ecal_hcal_masks(det)
    eta_mask_calo = np.abs(eta_calo) < eta_max
    mask_total_calo = (mask_ecal | mask_hcal) & eta_mask_calo

    eta_part = eta_from_pxpypz(
        particles_flat["px"].to_numpy().astype(float),
        particles_flat["py"].to_numpy().astype(float),
        particles_flat["pz"].to_numpy().astype(float),
    )
    truth_energy = particles_flat["energy"].to_numpy().astype(float)
    mask_root = _root_mask_per_event(particles_flat)
    eta_mask_part = np.abs(eta_part) < eta_max

    return {
        "eta_calo": eta_calo,
        "energy_dep": energy_dep,
        "mask_ecal": mask_ecal,
        "mask_hcal": mask_hcal,
        "mask_total_calo": mask_total_calo,
        "eta_mask_calo": eta_mask_calo,
        "eta_part": eta_part,
        "truth_energy": truth_energy,
        "mask_root": mask_root,
        "eta_mask_part": eta_mask_part,
    }


def build_per_root_response(
    contribs: pd.DataFrame,
    particles_flat: pd.DataFrame,
    particles_evt: pl.DataFrame,
    eta_max: float = 3.0,
) -> pd.DataFrame:
    """Per-root-particle response: (sum calibrated deposits from descendants) / (particle energy).

    Only includes roots that have at least one descendant deposit and are in |eta| < eta_max.

    Args:
        contribs: From prepare_calo_and_particles.
        particles_flat: From prepare_calo_and_particles.
        particles_evt: Polars particles event-table (same as passed to prepare_calo_and_particles).
        eta_max: Restrict to roots with |eta| < eta_max.

    Returns:
        DataFrame with columns: event_id, primary_ancestor_id, deposited_energy, truth_energy, response.
    """
    particles_evt_labeled = assign_primary_ancestor(particles_evt)
    particles_flat_labeled = explode_particles(particles_evt_labeled)
    pid_to_anc = particles_flat_labeled[["particle_id", "primary_ancestor_id"]].copy()
    pid_to_anc["particle_id"] = pd.to_numeric(pid_to_anc["particle_id"], errors="coerce")
    pid_to_anc["primary_ancestor_id"] = pd.to_numeric(pid_to_anc["primary_ancestor_id"], errors="coerce")
    contribs_with_anc = contribs.merge(pid_to_anc, on="particle_id", how="left")
    contribs_with_anc = contribs_with_anc.dropna(subset=["primary_ancestor_id"])
    deposited_per_root = (
        contribs_with_anc.groupby(["event_id", "primary_ancestor_id"], as_index=False)["energy"]
        .sum()
        .rename(columns={"energy": "deposited_energy"})
    )
    eta_part = eta_from_pxpypz(
        particles_flat["px"].to_numpy().astype(float),
        particles_flat["py"].to_numpy().astype(float),
        particles_flat["pz"].to_numpy().astype(float),
    )
    mask_root = _root_mask_per_event(particles_flat)
    eta_mask_part = np.abs(eta_part) < eta_max
    roots = particles_flat.loc[mask_root & eta_mask_part][["event_id", "particle_id", "energy"]].copy()
    roots = roots.rename(columns={"particle_id": "primary_ancestor_id", "energy": "truth_energy"})
    roots["primary_ancestor_id"] = pd.to_numeric(roots["primary_ancestor_id"], errors="coerce")
    merged = deposited_per_root.merge(
        roots,
        on=["event_id", "primary_ancestor_id"],
        how="inner",
    )
    merged["response"] = merged["deposited_energy"] / (merged["truth_energy"] + 1e-12)
    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def default_eta_bins(eta_max: float = 3.0, n_bins: int = 13) -> np.ndarray:
    """Default eta bin edges symmetric around 0."""
    return np.linspace(-eta_max, eta_max, n_bins)


def plot_energy_and_ratio_two_panel(
    arrays: Dict[str, Any],
    eta_bins: Optional[np.ndarray] = None,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (10, 8),
    **kwargs: Any,
) -> None:
    """Two-panel ATLAS-style plot: top = energy vs eta, bottom = ratio (measured/truth) vs eta.

    Args:
        arrays: From energy_plot_arrays().
        eta_bins: Bin edges; if None, use default_eta_bins().
        title_suffix: Appended to panel titles (e.g. "Event 0" or "All events (200)").
        figsize: (width, height).
        **kwargs: Passed to matplotlib (e.g. for tests).
    """
    import matplotlib.pyplot as plt

    if eta_bins is None:
        eta_bins = default_eta_bins()
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, **kwargs)
    ax = axes[0]

    eta_calo = arrays["eta_calo"]
    energy_dep = arrays["energy_dep"]
    mask_ecal = arrays["mask_ecal"]
    mask_hcal = arrays["mask_hcal"]
    mask_total_calo = arrays["mask_total_calo"]
    eta_mask_calo = arrays["eta_mask_calo"]
    eta_part = arrays["eta_part"]
    truth_energy = arrays["truth_energy"]
    mask_root = arrays["mask_root"]
    eta_mask_part = arrays["eta_mask_part"]

    plot_binned_sums_with_xerr(
        ax,
        eta_calo[mask_ecal & eta_mask_calo],
        energy_dep[mask_ecal & eta_mask_calo],
        bins=eta_bins,
        label="ECAL deposited (calibrated)",
        fmt="o-",
        lw=1.5,
        ms=3,
        capsize=0,
    )
    plot_binned_sums_with_xerr(
        ax,
        eta_calo[mask_hcal & eta_mask_calo],
        energy_dep[mask_hcal & eta_mask_calo],
        bins=eta_bins,
        label="HCAL deposited (calibrated)",
        fmt="o-",
        lw=1.5,
        ms=3,
        capsize=0,
    )
    plot_binned_sums_with_xerr(
        ax,
        eta_calo[mask_total_calo],
        energy_dep[mask_total_calo],
        bins=eta_bins,
        label="Total Calo deposited (calibrated)",
        fmt="o--",
        lw=1.5,
        ms=4,
        capsize=0,
        color="k",
    )
    plot_binned_sums_with_xerr(
        ax,
        eta_part[mask_root & eta_mask_part],
        truth_energy[mask_root & eta_mask_part],
        bins=eta_bins,
        label="Root-particle truth energy",
        fmt="o-",
        lw=1.5,
        ms=3,
        capsize=0,
    )
    ax.set_title(f"Energy vs eta (binned), {title_suffix}")
    ax.set_ylabel("Energy (sum in bin) [GeV]")
    ax.set_yscale("log")
    ax.legend()

    sum_dep, _ = np.histogram(
        eta_calo[mask_total_calo & eta_mask_calo],
        bins=eta_bins,
        weights=energy_dep[mask_total_calo & eta_mask_calo],
    )
    sum_truth, _ = np.histogram(
        eta_part[mask_root & eta_mask_part],
        bins=eta_bins,
        weights=truth_energy[mask_root & eta_mask_part],
    )
    centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    ratio = sum_dep / (sum_truth + 1e-12)
    ax2 = axes[1]
    ax2.plot(centers, ratio, "o-", label="Calibrated deposit / root truth")
    ax2.axhline(1.0, color="gray", linestyle="--", label="Perfect match")
    ax2.set_xlabel("eta")
    ax2.set_ylabel("Ratio (measured / truth)")
    ax2.set_ylim(0, 2)
    ax2.legend()
    plt.tight_layout()
    plt.show()


def plot_per_root_response_histogram(
    response_df: pd.DataFrame,
    title_suffix: str = "",
    bins: int = 50,
    range_xy: Tuple[float, float] = (0, 2),
    figsize: Tuple[float, float] = (8, 5),
    **kwargs: Any,
) -> None:
    """Histogram of per-root-particle response (calibrated deposit / truth).

    Args:
        response_df: From build_per_root_response().
        title_suffix: For title (e.g. "Event 0" or "All events (200)").
        bins: Number of histogram bins.
        range_xy: (xmin, xmax) for histogram.
        **kwargs: Passed to matplotlib.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
    ax.hist(
        response_df["response"],
        bins=bins,
        range=range_xy,
        edgecolor="black",
        alpha=0.7,
        label=f"Root particles, $|\\eta| < 3$, {title_suffix}",
    )
    ax.axvline(1.0, color="gray", linestyle="--", label="Perfect match")
    ax.set_xlabel("Response (calibrated deposit / truth energy)")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-root-particle response, {title_suffix}")
    ax.legend()
    plt.tight_layout()
    plt.show()
