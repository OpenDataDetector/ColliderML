"""Calorimeter calibration utilities.

Currently ColliderML only exposes calo hits. Calibration is applied directly
to the per-cell energies (and optionally contribution energies).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import polars as pl
import yaml

from colliderml.core.tables import PolarsTable
from colliderml.physics.detector_enums import CALO_DETECTOR_CODES, calo_region_to_ids
from colliderml.physics.constants import ODD_CALO_SCALING_V0


@dataclass(frozen=True)
class CaloCalibration:
    """Calibration parameters for calo hits.

    Attributes:
        detector_scale: Mapping from detector id -> multiplicative scale.
        default_scale: Scale applied when detector id not present in mapping.
        apply_to_contrib: Whether to also scale `contrib_energies`.
    """

    detector_scale: Dict[int, float]
    default_scale: float = 1.0
    apply_to_contrib: bool = True


def _as_list(x: Any) -> list:
    """Convert Polars/Python list-like values to a plain Python list."""
    if x is None:
        return []
    # In Polars map_elements, list values may arrive as a Series.
    if isinstance(x, pl.Series):
        return x.to_list()
    return list(x) if not isinstance(x, (str, bytes)) else [x]


def load_calo_calibration(path: Union[str, Path]) -> CaloCalibration:
    """Load calorimeter calibration from a YAML file.

    Expected YAML:
    - detector_scale:
        - keys may be detector IDs (ints/strings), e.g. {10: 37.5}
        - or region strings, e.g. {ecal_barrel: 37.5, ecal_endcap: 38.7}
    - default_scale: 1.0
    - apply_to_contrib: true
    """
    p = Path(path).expanduser().resolve()
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Calibration YAML must parse to a dict")
    det_map = payload.get("detector_scale") or {}
    if not isinstance(det_map, dict):
        raise ValueError("detector_scale must be a dict")
    detector_scale: Dict[int, float] = {}
    for k, v in det_map.items():
        if isinstance(k, int):
            detector_scale[int(k)] = float(v)
            continue
        ks = str(k).strip()
        # If numeric-as-string, treat as explicit detector id.
        if ks.isdigit() or (ks.startswith("-") and ks[1:].isdigit()):
            detector_scale[int(ks)] = float(v)
            continue
        # Otherwise treat as region string and expand to ids.
        for det_id in calo_region_to_ids(ks):
            detector_scale[int(det_id)] = float(v)
    return CaloCalibration(
        detector_scale=detector_scale,
        default_scale=float(payload.get("default_scale", 1.0)),
        apply_to_contrib=bool(payload.get("apply_to_contrib", True)),
    )


def odd_default_calo_calibration(*, apply_to_contrib: bool = True) -> CaloCalibration:
    """Default ODD calorimeter calibration based on region scaling factors.

    Returns:
        CaloCalibration: mapping applied to detector enum ids.
    """
    det_scale: Dict[int, float] = {}
    for region, scale in ODD_CALO_SCALING_V0.items():
        for det_id in calo_region_to_ids(region):
            det_scale[int(det_id)] = float(scale)
    return CaloCalibration(detector_scale=det_scale, default_scale=1.0, apply_to_contrib=apply_to_contrib)


def apply_calo_calibration(
    calo_hits: PolarsTable,
    calibration: Union[CaloCalibration, str, Path],
    *,
    detector_col: str = "detector",
    total_energy_col: str = "total_energy",
    contrib_energy_col: str = "contrib_energies",
) -> PolarsTable:
    """Apply calibration to calo hits.

    Args:
        calo_hits: Event-table with list columns including detector and energies.
        calibration: CaloCalibration or YAML path.
        detector_col: Detector id list column.
        total_energy_col: Total energy list column (per cell).
        contrib_energy_col: Contribution energies list-of-lists column.

    Returns:
        PolarsTable: calibrated calo hits, same type as input.
    """
    cal = load_calo_calibration(calibration) if not isinstance(calibration, CaloCalibration) else calibration

    # Build a vectorized mapping expression:
    # scale = detector.map_dict(detector_scale).fill_null(default)
    scale_expr = (
        pl.col(detector_col)
        .map_elements(
            lambda dets: [cal.detector_scale.get(int(d), cal.default_scale) for d in _as_list(dets)],
            return_dtype=pl.List(pl.Float64),
        )
        .alias("_calo_scale")
    )

    out = calo_hits.with_columns(scale_expr)

    # Scale total_energy per cell (list * list elementwise using struct UDF for stability).
    out = out.with_columns(
        pl.struct([total_energy_col, "_calo_scale"]).map_elements(
            lambda s: [
                float(e) * float(sc)
                for e, sc in zip(_as_list(s[total_energy_col]), _as_list(s["_calo_scale"]))
            ],
            return_dtype=pl.List(pl.Float64),
        ).alias(total_energy_col)
    )

    if cal.apply_to_contrib and contrib_energy_col in (out.collect_schema().names() if isinstance(out, pl.LazyFrame) else out.columns):
        # contrib_energies is list-of-lists aligned with cells; scale each inner list by the cell scale.
        out = out.with_columns(
            pl.struct([contrib_energy_col, "_calo_scale"]).map_elements(
                lambda s: [
                    [float(e) * float(sc) for e in _as_list(inner)]
                    for inner, sc in zip(_as_list(s[contrib_energy_col]), _as_list(s["_calo_scale"]))
                ],
                return_dtype=pl.List(pl.List(pl.Float64)),
            ).alias(contrib_energy_col)
        )

    return out.drop("_calo_scale")


