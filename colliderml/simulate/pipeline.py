"""Pipeline stage definitions and config discovery.

This module knows the shape of each supported channel's pipeline (which
scripts to run in what order) and how to locate the YAML stage configs
inside the cloned :mod:`colliderml-production` repository.

Keeping ``CHANNEL_STAGES`` here — rather than reading it from a manifest
file in the production repo — lets us validate channels at import time and
keep type checking strict. When a new channel is added to the production
pipeline, add an entry here *and* ensure the corresponding config YAML files
exist under ``configs_development/docker_test/<channel>/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class Stage:
    """A single pipeline stage.

    Attributes:
        name: Human-readable name for CLI progress reporting.
        stage: Machine identifier used by the pipeline runner
            (e.g. ``"pythia_generation"``). Matches the hook in
            :mod:`~colliderml.simulate.docker` that copies MadGraph output
            across run directories.
        script: Path *relative to* ``scripts/`` in the production repo
            (e.g. ``"simulation/pythia_gen.py"``).
        config: YAML filename inside the channel's config directory
            (e.g. ``"pythia_config.yaml"``).
    """

    name: str
    stage: str
    script: str
    config: str


#: Pipeline definition for each supported channel.
#:
#: Ordering matters — stages run sequentially, and later stages depend on
#: artefacts from earlier ones.
CHANNEL_STAGES: Dict[str, Tuple[Stage, ...]] = {
    "higgs_portal": (
        Stage(
            name="Pythia Generation",
            stage="pythia_generation",
            script="simulation/pythia_gen.py",
            config="pythia_config.yaml",
        ),
        Stage(
            name="Detector Simulation",
            stage="simulation",
            script="simulation/ddsim_run.py",
            config="simulation_config.yaml",
        ),
        Stage(
            name="Digitization & Reconstruction",
            stage="digitization",
            script="simulation/digi_and_reco.py",
            config="digitization_config.yaml",
        ),
        Stage(
            name="Parquet Conversion",
            stage="convert_all",
            script="postprocessing/convert_all.py",
            config="convert_all.yaml",
        ),
    ),
    "ttbar": (
        Stage(
            name="MadGraph Init",
            stage="madgraph_init",
            script="simulation/madgraph_init.py",
            config="madgraph_init_config.yaml",
        ),
        Stage(
            name="MadGraph Generation",
            stage="madgraph_generation",
            script="simulation/madgraph_gen.py",
            config="madgraph_generation_config.yaml",
        ),
        Stage(
            name="Pythia Generation",
            stage="pythia_generation",
            script="simulation/pythia_gen.py",
            config="pythia_config.yaml",
        ),
        Stage(
            name="Detector Simulation",
            stage="simulation",
            script="simulation/ddsim_run.py",
            config="simulation_config.yaml",
        ),
        Stage(
            name="Digitization & Reconstruction",
            stage="digitization",
            script="simulation/digi_and_reco.py",
            config="digitization_config.yaml",
        ),
        Stage(
            name="Parquet Conversion",
            stage="convert_all",
            script="postprocessing/convert_all.py",
            config="convert_all.yaml",
        ),
    ),
    "single_muon": (
        Stage(
            name="Pythia Generation",
            stage="pythia_generation",
            script="simulation/pythia_gen.py",
            config="pythia_config.yaml",
        ),
        Stage(
            name="Detector Simulation",
            stage="simulation",
            script="simulation/ddsim_run.py",
            config="simulation_config.yaml",
        ),
        Stage(
            name="Digitization & Reconstruction",
            stage="digitization",
            script="simulation/digi_and_reco.py",
            config="digitization_config.yaml",
        ),
        Stage(
            name="Parquet Conversion",
            stage="convert_all",
            script="postprocessing/convert_all.py",
            config="convert_all.yaml",
        ),
    ),
}


def get_channel_stages(channel: str) -> Tuple[Stage, ...]:
    """Return the ordered pipeline stages for a channel.

    Args:
        channel: Physics channel name.

    Returns:
        Tuple of :class:`Stage` in execution order.

    Raises:
        ValueError: If ``channel`` is not a known channel.
    """
    if channel not in CHANNEL_STAGES:
        available = ", ".join(sorted(CHANNEL_STAGES.keys()))
        raise ValueError(f"Unknown simulation channel '{channel}'. Available: {available}")
    return CHANNEL_STAGES[channel]


def list_channels() -> List[str]:
    """Return all known simulation channels, sorted alphabetically."""
    return sorted(CHANNEL_STAGES.keys())


def find_config_dir(channel: str, prod_root: Path) -> Path:
    """Find the stage-config directory for ``channel`` inside a production clone.

    The search order mirrors the pattern used by the production repo's own
    scripts:

    1. ``<prod_root>/configs_development/docker_test/<channel>/`` — complete
       per-channel configs (preferred).
    2. ``<prod_root>/configs_production/templates/`` — shared templates
       (fallback for channels without dedicated dev configs).

    Args:
        channel: Physics channel name.
        prod_root: Path to a :mod:`colliderml-production` checkout (usually
            the cache directory populated by
            :func:`colliderml.simulate.docker.clone_colliderml_production`).

    Returns:
        Path to the config directory.

    Raises:
        FileNotFoundError: If neither candidate directory exists.
    """
    dev_dir = prod_root / "configs_development" / "docker_test" / channel
    if dev_dir.is_dir():
        return dev_dir
    templates_dir = prod_root / "configs_production" / "templates"
    if templates_dir.is_dir():
        return templates_dir
    raise FileNotFoundError(
        f"No config directory found for channel '{channel}'. Searched: "
        f"{dev_dir}, {templates_dir}. Is the colliderml-production clone at "
        f"{prod_root} complete? Try deleting it and letting simulate() re-clone."
    )


def generate_stage_manifest(
    channel: str,
    *,
    prod_root: Path,
) -> List[Dict[str, str]]:
    """Assemble the list of (script, config) pairs to execute for a channel.

    Only stages whose config YAML actually exists in the production checkout
    are included — this is how ``single_muon`` gets to skip MadGraph without
    any special casing.

    Args:
        channel: Physics channel name.
        prod_root: Root of the :mod:`colliderml-production` checkout.

    Returns:
        List of dicts with keys ``name``, ``stage``, ``script``,
        ``config_path`` (the last is *relative to* ``prod_root`` so it can
        be passed straight to a container that mounts ``prod_root`` as
        ``/workspace``).
    """
    stages = get_channel_stages(channel)
    config_dir = find_config_dir(channel, prod_root)

    manifest: List[Dict[str, str]] = []
    for stage in stages:
        config_file = config_dir / stage.config
        if not config_file.exists():
            continue
        manifest.append(
            {
                "name": stage.name,
                "stage": stage.stage,
                "script": stage.script,
                "config_path": str(config_file.relative_to(prod_root)),
            }
        )
    return manifest


def load_stage_template(config_dir: Path, stage: Stage) -> Dict[str, object]:
    """Load a stage's YAML config template as a plain dict.

    Used by tests and by any caller that wants to introspect or mutate a
    stage config before it's passed to the container.

    Args:
        config_dir: Directory returned by :func:`find_config_dir`.
        stage: Stage whose config should be loaded.

    Returns:
        Parsed YAML dict.

    Raises:
        FileNotFoundError: If the stage's config YAML does not exist.
    """
    path = config_dir / stage.config
    if not path.exists():
        raise FileNotFoundError(f"Stage config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Stage config {path} did not parse to a mapping.")
    return data
