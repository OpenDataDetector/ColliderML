"""Unit tests for calorimeter calibration (no network)."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def test_apply_calo_calibration_scales_total_and_contrib(tmp_path: Path) -> None:
    from colliderml.physics.calibration import apply_calo_calibration

    calo = pl.DataFrame(
        {
            "event_id": [1],
            "detector": [[0, 1]],
            "total_energy": [[10.0, 20.0]],
            "contrib_energies": [[[1.0, 2.0], [3.0]]],
        }
    )

    calib_yaml = tmp_path / "calib.yaml"
    calib_yaml.write_text(
        "detector_scale:\n  0: 2.0\n  1: 0.5\napply_to_contrib: true\n",
        encoding="utf-8",
    )

    out = apply_calo_calibration(calo, calib_yaml)
    assert out["total_energy"].to_list() == [[20.0, 10.0]]
    assert out["contrib_energies"].to_list() == [[[2.0, 4.0], [1.5]]]


def test_apply_calo_calibration_total_only(tmp_path: Path) -> None:
    from colliderml.physics.calibration import apply_calo_calibration

    calo = pl.DataFrame(
        {"event_id": [1], "detector": [[0]], "total_energy": [[10.0]]}
    )
    calib_yaml = tmp_path / "calib.yaml"
    calib_yaml.write_text("detector_scale:\n  0: 3.0\napply_to_contrib: false\n", encoding="utf-8")

    out = apply_calo_calibration(calo, calib_yaml)
    assert out["total_energy"].to_list() == [[30.0]]


def test_apply_calo_calibration_lazy(tmp_path: Path) -> None:
    from colliderml.physics.calibration import apply_calo_calibration

    calo = pl.DataFrame(
        {"event_id": [1], "detector": [[0]], "total_energy": [[10.0]]}
    ).lazy()
    calib_yaml = tmp_path / "calib.yaml"
    calib_yaml.write_text("detector_scale:\n  0: 2.0\n", encoding="utf-8")

    out = apply_calo_calibration(calo, calib_yaml)
    assert isinstance(out, pl.LazyFrame)
    collected = out.collect()
    assert collected["total_energy"].to_list() == [[20.0]]


def test_region_based_calo_calibration_table_values(tmp_path: Path) -> None:
    """Ensure ODD region scaling factors are applied to the correct detector IDs."""
    from colliderml.physics.calibration import apply_calo_calibration

    # Detector ids from production mapping:
    # ecal_neg_endcap=9, ecal_barrel=10, ecal_pos_endcap=11,
    # hcal_neg_endcap=12, hcal_barrel=13, hcal_pos_endcap=14
    calo = pl.DataFrame(
        {
            "event_id": [1],
            "detector": [[9, 10, 11, 12, 13, 14]],
            "total_energy": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        }
    )

    calib_yaml = tmp_path / "calib.yaml"
    calib_yaml.write_text(
        "detector_scale:\n"
        "  ecal_barrel: 37.5\n"
        "  ecal_endcap: 38.7\n"
        "  hcal_barrel: 45.0\n"
        "  hcal_endcap: 46.9\n",
        encoding="utf-8",
    )

    out = apply_calo_calibration(calo, calib_yaml)
    assert out["total_energy"].to_list() == [[38.7, 37.5, 38.7, 46.9, 45.0, 46.9]]


