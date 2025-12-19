"""Detector enum codes used in ColliderML parquet outputs.

This is derived from the production pipeline enum definitions and kept here so:
- analysis code can interpret integer detector IDs
- calibration can be configured in terms of detector regions

Only the calo codes are required for current ColliderML release usage, but we
include the tracker mapping as well for completeness.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Mapping


# Tracker detector enum codes (uint8)
TRACKER_DETECTOR_CODES: Dict[str, int] = {
    # Pixel
    "pixel_neg_endcap": 0,
    "pixel_barrel": 1,
    "pixel_pos_endcap": 2,
    # Short strips
    "short_neg_endcap": 3,
    "short_barrel": 4,
    "short_pos_endcap": 5,
    # Long strips
    "long_neg_endcap": 6,
    "long_barrel": 7,
    "long_pos_endcap": 8,
}

# Calorimeter detector enum codes (uint8)
CALO_DETECTOR_CODES: Dict[str, int] = {
    # Electromagnetic calorimeter
    "ecal_neg_endcap": 9,
    "ecal_barrel": 10,
    "ecal_pos_endcap": 11,
    # Hadronic calorimeter
    "hcal_neg_endcap": 12,
    "hcal_barrel": 13,
    "hcal_pos_endcap": 14,
}


def calo_region_to_ids(region: str) -> List[int]:
    """Expand a calo region string to detector enum ids.

    Supported inputs (case-insensitive):
    - exact detector keys: e.g. 'ecal_barrel', 'hcal_neg_endcap'
    - grouped regions: 'ecal_endcap', 'hcal_endcap', 'ecal', 'hcal'

    Args:
        region: region string.

    Returns:
        List[int]: one or more detector IDs.
    """
    key = region.strip().lower().replace(" ", "_")

    # Direct mapping
    if key in CALO_DETECTOR_CODES:
        return [CALO_DETECTOR_CODES[key]]

    # Grouped regions
    if key == "ecal_endcap":
        return [CALO_DETECTOR_CODES["ecal_neg_endcap"], CALO_DETECTOR_CODES["ecal_pos_endcap"]]
    if key == "hcal_endcap":
        return [CALO_DETECTOR_CODES["hcal_neg_endcap"], CALO_DETECTOR_CODES["hcal_pos_endcap"]]
    if key == "ecal":
        return [
            CALO_DETECTOR_CODES["ecal_neg_endcap"],
            CALO_DETECTOR_CODES["ecal_barrel"],
            CALO_DETECTOR_CODES["ecal_pos_endcap"],
        ]
    if key == "hcal":
        return [
            CALO_DETECTOR_CODES["hcal_neg_endcap"],
            CALO_DETECTOR_CODES["hcal_barrel"],
            CALO_DETECTOR_CODES["hcal_pos_endcap"],
        ]

    raise ValueError(f"Unknown calo region '{region}'")


