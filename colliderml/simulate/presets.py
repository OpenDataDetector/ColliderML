"""Simulation presets.

A *preset* is a named bundle of simulation parameters (channel, event count,
pileup, estimated cost) that ships as package data inside this library. It
gives users a friendly shorthand for common workloads::

    colliderml simulate --preset ttbar-quick

The preset catalogue lives in :file:`presets.yaml`, resolved via
:func:`importlib.resources` so it works the same whether the package is
installed from a wheel, an sdist, or ``pip install -e .``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Dict, Mapping, Optional

import yaml

#: Package-qualified path to the bundled presets file.
PRESETS_PACKAGE_RESOURCE = "colliderml.simulate"
_PRESETS_FILENAME = "presets.yaml"


@dataclass(frozen=True)
class Preset:
    """Resolved simulation preset.

    Attributes:
        name: Preset identifier (e.g. ``"ttbar-quick"``).
        channel: Physics channel (e.g. ``"ttbar"``, ``"higgs_portal"``).
        events: Number of events to simulate.
        pileup: Pileup level (``0`` means no pileup, ``200`` is HL-LHC).
        description: One-line human-readable summary (optional).
    """

    name: str
    channel: str
    events: int
    pileup: int
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dict view of the preset (for JSON/YAML dumps)."""
        return {
            "name": self.name,
            "channel": self.channel,
            "events": self.events,
            "pileup": self.pileup,
            "description": self.description,
        }


def _read_presets_file() -> Mapping[str, Mapping[str, Any]]:
    """Read the bundled :file:`presets.yaml` via :mod:`importlib.resources`.

    Returns:
        The ``presets`` mapping from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file is missing from the package.
        ValueError: If the YAML payload is malformed.
    """
    resource = resources.files(PRESETS_PACKAGE_RESOURCE).joinpath(_PRESETS_FILENAME)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Presets file {_PRESETS_FILENAME} not found in package "
            f"{PRESETS_PACKAGE_RESOURCE}. This usually means the package "
            "was built without package_data — reinstall from a wheel or "
            "run `pip install -e .` from a checkout."
        )
    with resources.as_file(resource) as path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{_PRESETS_FILENAME} did not parse to a mapping.")
    presets = data.get("presets")
    if not isinstance(presets, dict):
        raise ValueError(f"{_PRESETS_FILENAME} is missing a top-level 'presets' mapping.")
    return presets


def load_presets() -> Dict[str, Preset]:
    """Load the full preset catalogue.

    Returns:
        Mapping of preset name → :class:`Preset`.
    """
    raw = _read_presets_file()
    out: Dict[str, Preset] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Preset '{name}' must be a mapping, got {type(spec).__name__}.")
        channel = spec.get("channel")
        events = spec.get("events")
        pileup = spec.get("pileup", 0)
        description = spec.get("description", "")
        if channel is None:
            raise ValueError(f"Preset '{name}' is missing required field 'channel'.")
        if events is None:
            raise ValueError(f"Preset '{name}' is missing required field 'events'.")
        out[name] = Preset(
            name=name,
            channel=str(channel),
            events=int(events),
            pileup=int(pileup),
            description=str(description),
        )
    return out


def resolve_preset(name: str, presets: Optional[Mapping[str, Preset]] = None) -> Preset:
    """Look up a preset by name.

    Args:
        name: Preset identifier.
        presets: Optional pre-loaded catalogue (saves a YAML re-read).

    Returns:
        The resolved :class:`Preset`.

    Raises:
        ValueError: If the name is not in the catalogue. The error message
            lists all available preset names so users can self-correct.
    """
    catalogue = presets if presets is not None else load_presets()
    if name not in catalogue:
        available = ", ".join(sorted(catalogue.keys()))
        raise ValueError(
            f"Unknown simulation preset '{name}'. Available presets: {available}"
        )
    return catalogue[name]
