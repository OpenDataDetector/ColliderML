"""Simulation subsystem for ColliderML.

Exposes :func:`simulate` — the end-to-end orchestrator that runs the
Pythia/MadGraph → Geant4 → tracking (→ particle flow → parquet) pipeline in
Docker or Podman containers, or submits to the SaaS backend for remote
execution.

The pipeline scripts and stage YAML templates live in the companion
``colliderml-production`` repository; this subpackage auto-clones that repo
into a local cache on first use. The repository ref and the container images
are pinned together in :file:`pipeline_config.yaml`
(:func:`load_pipeline_config`).

Typical use::

    import colliderml

    result = colliderml.simulate(preset="ttbar-quick")
    print(result.run_dir)                 # e.g. /.../colliderml_output/.../runs/0

    # full chain with particle flow; final tables under result.parquet_dir:
    result = colliderml.simulate(preset="ttbar-pflow-quick")
    print(result.tables()["pfos"])

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

import sys
import types

from colliderml.simulate.api import SimulationResult, simulate
from colliderml.simulate.config import PipelineConfig, load_pipeline_config
from colliderml.simulate.presets import (
    PRESETS_PACKAGE_RESOURCE,
    Preset,
    load_presets,
    resolve_preset,
)

class _CallableModule(types.ModuleType):
    """Lets ``colliderml.simulate(...)`` work whether the name resolves to the
    function (first attribute access on ``colliderml``) or to this subpackage
    (after anything imported ``colliderml.simulate``)."""

    def __call__(self, *args, **kwargs):
        return simulate(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = [
    "simulate",
    "SimulationResult",
    "PipelineConfig",
    "load_pipeline_config",
    "load_presets",
    "resolve_preset",
    "Preset",
    "PRESETS_PACKAGE_RESOURCE",
]
