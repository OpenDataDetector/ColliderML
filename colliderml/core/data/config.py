"""Configuration and lightweight enums/constants for ColliderML.

This module provides base URLs and minimal legacy constants while we migrate
to a manifest-driven data selection model.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set


# Base URL configuration
BASE_URL: str = "https://portal.nersc.gov/cfs/m4958/ColliderML/"
MANIFEST_URL: str = f"{BASE_URL}manifest.json"

# Downloader base URLs list (kept for backward compatibility)
DEFAULT_URLS = [BASE_URL]


# Legacy-style enums (kept to avoid breaking existing imports/tests)
class PileupLevel(str, Enum):
    SINGLE = "single-particle"
    LOW = "pileup-10"
    HIGH = "pileup-200"


class DataType(str, Enum):
    RECO = "reco"
    TRUTH = "truth"
    MEASUREMENTS = "measurements"


# Minimal object configuration placeholder to satisfy existing imports
class _ObjectConfig:
    def __init__(self, data_type: DataType):
        self.data_type = data_type


# Representative objects; the authoritative list comes from the manifest
OBJECT_CONFIGS: Dict[str, _ObjectConfig] = {
    "tracks": _ObjectConfig(DataType.RECO),
    "particle_flow": _ObjectConfig(DataType.RECO),
    "particles": _ObjectConfig(DataType.TRUTH),
    "tracker_hits": _ObjectConfig(DataType.MEASUREMENTS),
}


# Representative physics processes; authoritative set comes from the manifest
VALID_PROCESSES: Set[str] = {"ttbar", "wjets", "zjets", "susy", "higgs", "qcd", "exotics"}


# Legacy constants used by older CLI/tests; superseded by manifest data
EVENTS_PER_FILE: int = 1000

# Optional dataset sizes for summary planning (will be superseded by manifest totals)
DATASET_SIZES: Dict[str, int] = {
    "ttbar": 100_000,
    "qcd": 100_000,
}


def get_object_path(
    pileup: PileupLevel,
    process: str,
    object_name: str,
    start_event: int,
    end_event: int,
) -> str:
    """Build a legacy-style relative path for an object file.

    This is maintained only for backward compatibility with existing tests and
    scripts. New code should select paths from the manifest instead.
    """

    # Very lightweight validation to match test expectations
    if process not in VALID_PROCESSES:
        raise ValueError("Invalid process")
    if object_name not in OBJECT_CONFIGS:
        raise ValueError("Invalid object type")

    # Use a fixed version segment to satisfy test path format
    version = "v1"
    data_type = OBJECT_CONFIGS[object_name].data_type.value
    pileup_str = pileup.value if isinstance(pileup, PileupLevel) else str(pileup)
    filename = f"{pileup_str}.{process}.{version}.{data_type}.{object_name}.events{start_event}-{end_event}.h5"
    rel_path = f"{pileup_str}/{process}/{version}/{data_type}/{object_name}/{filename}"
    return rel_path


__all__ = [
    "BASE_URL",
    "MANIFEST_URL",
    "DEFAULT_URLS",
    "PileupLevel",
    "DataType",
    "OBJECT_CONFIGS",
    "VALID_PROCESSES",
    "EVENTS_PER_FILE",
    "DATASET_SIZES",
    "get_object_path",
]


