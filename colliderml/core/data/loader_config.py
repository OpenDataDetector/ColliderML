"""Loader configuration for ColliderML: normalize and resolve load parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import yaml

from colliderml.core.hf_download import DEFAULT_DATASET_ID, DEFAULT_SPLIT, default_data_dir

from .config import VALID_PROCESSES


# Type alias for object type names (particles, tracks, etc.)
ObjectType = str


@dataclass(frozen=True)
class LoaderConfig:
    """Configuration for loading local Parquet tables via load_tables."""

    dataset_id: str
    channels: Union[str, List[str]]
    pileup: str
    objects: List[str]
    split: str
    lazy: bool
    max_events: Optional[int] = None
    data_dir: Optional[Union[str, Path]] = None

    def resolved_data_dir(self) -> Path:
        """Return the base data directory (override or default from env/cache)."""
        if self.data_dir is not None:
            return Path(self.data_dir).expanduser().resolve()
        return default_data_dir()

    def normalized_channels(self) -> List[str]:
        """Return channels as a list (single string -> [s], 'all' -> VALID_PROCESSES)."""
        if self.channels == "all":
            return sorted(VALID_PROCESSES)
        if isinstance(self.channels, str):
            return [self.channels]
        return list(self.channels)


@dataclass
class PileupSubsampleConfig:
    """Placeholder for pileup subsampling configuration. Not yet implemented."""

    pass


@dataclass
class CalibrationConfig:
    """Placeholder for calibration configuration. Not yet implemented."""

    pass


def load_config(cfg: Union[LoaderConfig, dict, str, Path]) -> LoaderConfig:
    """Normalize cfg into a LoaderConfig.

    Args:
        cfg: LoaderConfig (returned as-is), dict of loader options, or path to YAML file.

    Returns:
        LoaderConfig instance.

    Raises:
        TypeError: If cfg type is not supported.
        ValueError: If required keys are missing from dict/YAML.
    """
    if isinstance(cfg, LoaderConfig):
        return cfg
    if isinstance(cfg, (str, Path)):
        path = Path(cfg)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("YAML config must parse to a dict")
        cfg = data
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be LoaderConfig, dict, or path; got {type(cfg)}")

    dataset_id = cfg.get("dataset_id", DEFAULT_DATASET_ID)
    channels = cfg.get("channels")
    if channels is None:
        raise ValueError("config must include 'channels'")
    pileup = cfg.get("pileup")
    if pileup is None:
        raise ValueError("config must include 'pileup'")
    objects = cfg.get("objects")
    if objects is None:
        raise ValueError("config must include 'objects'")
    split = cfg.get("split", DEFAULT_SPLIT)
    lazy = cfg.get("lazy", False)
    max_events = cfg.get("max_events")
    data_dir = cfg.get("data_dir")

    return LoaderConfig(
        dataset_id=dataset_id,
        channels=channels,
        pileup=pileup,
        objects=list(objects),
        split=split,
        lazy=lazy,
        max_events=max_events,
        data_dir=data_dir,
    )


__all__ = [
    "LoaderConfig",
    "PileupSubsampleConfig",
    "CalibrationConfig",
    "ObjectType",
    "load_config",
]
