"""Simulation subsystem for ColliderML.

Exposes :func:`simulate` — the end-to-end orchestrator that runs the full
Pythia/MadGraph → Geant4 → digitization → parquet pipeline inside a Docker
or Podman container, or submits to the SaaS backend for remote execution.

The pipeline scripts and stage YAML templates live in the companion
``colliderml-production`` repository; this subpackage auto-clones that repo
into a local cache on first use (mirroring how :mod:`colliderml.simulate`
already handles OpenDataDetector and MG5aMC_PY8_interface). Users never have
to know about this — it is an implementation detail of :func:`simulate`.

Typical use::

    import colliderml

    result = colliderml.simulate(preset="ttbar-quick")
    print(result.run_dir)                 # e.g. /.../colliderml_output/.../runs/0

    # or supply parameters explicitly:
    result = colliderml.simulate(
        channel="higgs_portal",
        events=10,
        pileup=10,
    )

    # or submit to the backend (no Docker needed locally):
    result = colliderml.simulate(preset="higgs-portal-quick", remote=True)
"""

from __future__ import annotations

from colliderml.simulate.api import SimulationResult, simulate
from colliderml.simulate.presets import (
    PRESETS_PACKAGE_RESOURCE,
    Preset,
    load_presets,
    resolve_preset,
)

__all__ = [
    "simulate",
    "SimulationResult",
    "load_presets",
    "resolve_preset",
    "Preset",
    "PRESETS_PACKAGE_RESOURCE",
]
