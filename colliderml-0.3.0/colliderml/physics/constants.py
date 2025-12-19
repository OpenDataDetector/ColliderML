"""Physics and calibration constants for ColliderML.

Keep raw numerical constants here (no I/O, no Polars), so they can be reused
consistently across library code, notebooks, and docs.
"""

from __future__ import annotations

from typing import Dict

# Current estimate of the calorimeter scaling factors for Open Data Detector (ODD)
# regions, from the table you provided:
# - ECal Barrel: 37.5
# - ECal Endcap: 38.7
# - HCal Barrel: 45.0
# - HCal Endcap: 46.9
#
# Keys are region strings understood by `colliderml.physics.detector_enums.calo_region_to_ids`.
ODD_CALO_SCALING_V0: Dict[str, float] = {
    "ecal_barrel": 37.5,
    "ecal_endcap": 38.7,
    "hcal_barrel": 45.0,
    "hcal_endcap": 46.9,
}


