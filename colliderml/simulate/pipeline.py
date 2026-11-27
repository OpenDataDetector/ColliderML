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
        image: Which container image runs the stage: ``"sim"`` or ``"reco"``.
        particle_flow_only: ``True`` for the stages that only run when
            particle flow is requested (everything after tracking).
    """

    name: str
    stage: str
    script: str
    config: str
    image: str = "sim"
    particle_flow_only: bool = False


_DETECTOR_STAGES: Tuple[Stage, ...] = (
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
)

#: Stages after tracking: truth tables, calorimeter digitisation and Pandora
#: particle flow (reco image), and the final parquet tables under
#: ``<output_dir>/parquet/``.
_PARTICLE_FLOW_STAGES: Tuple[Stage, ...] = (
    Stage(
        name="Truth Particle Tables",
        stage="particles_full",
        script="postprocessing/convert_particles_full.py",
        config="particles_full_config.yaml",
        particle_flow_only=True,
    ),
    Stage(
        name="Calorimeter Digitization & Particle Flow",
        stage="pandora_reco",
        script="simulation/pandora_reco.py",
        config="pandora_reco_config.yaml",
        image="reco",
        particle_flow_only=True,
    ),
    Stage(
        name="Calorimeter & Particle Flow Tables",
        stage="reco_tables",
        script="postprocessing/convert_reco_tables.py",
        config="reco_tables_config.yaml",
        image="reco",
        particle_flow_only=True,
    ),
    Stage(
        name="Parquet Packaging",
        stage="package_parquet",
        script="postprocessing/package_native_parquet.py",
        config="package_parquet_config.yaml",
        particle_flow_only=True,
    ),
)

#: Pipeline definition for each supported channel.
#:
#: Ordering matters: stages run sequentially, and later stages depend on
#: artefacts from earlier ones.
#:
#: Without particle flow the chain ends at Digitization & Reconstruction,
#: which writes the tracker_hits / particles / tracks parquet of the run
#: directly through the ACTS Arrow plugin. With particle flow the remaining
#: stages run as well and the final tables land in ``<output_dir>/parquet/``.
CHANNEL_STAGES: Dict[str, Tuple[Stage, ...]] = {
    "higgs_portal": (
        Stage(
            name="Pythia Generation",
            stage="pythia_generation",
            script="simulation/pythia_gen.py",
            config="pythia_config.yaml",
        ),
        *_DETECTOR_STAGES,
        *_PARTICLE_FLOW_STAGES,
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
        *_DETECTOR_STAGES,
        *_PARTICLE_FLOW_STAGES,
    ),
    # NOTE: `single_muon` is intentionally absent. Its preset was removed and it
    # was mis-routed through Pythia generation (it needs an ACTS particle-gun
    # stage, not a hard-process generator). Re-add it only with a correct
    # particle-gun pipeline.
}


def get_channel_stages(channel: str, *, particle_flow: bool = False) -> Tuple[Stage, ...]:
    """Return the ordered pipeline stages for a channel.

    Args:
        channel: Physics channel name.
        particle_flow: Include the stages after tracking
            (:attr:`Stage.particle_flow_only`). Off by default.

    Returns:
        Tuple of :class:`Stage` in execution order.

    Raises:
        ValueError: If ``channel`` is not a known channel.
    """
    if channel not in CHANNEL_STAGES:
        available = ", ".join(sorted(CHANNEL_STAGES.keys()))
        raise ValueError(f"Unknown simulation channel '{channel}'. Available: {available}")
    stages = CHANNEL_STAGES[channel]
    if particle_flow:
        return stages
    return tuple(stage for stage in stages if not stage.particle_flow_only)


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
    particle_flow: bool = False,
) -> List[Dict[str, str]]:
    """Assemble the list of (script, config) pairs to execute for a channel.

    A generation or tracking stage whose config YAML is absent from the
    production checkout is skipped. With ``particle_flow=True`` every
    particle-flow stage must have its config: a checkout without them
    cannot run the full chain, and stopping early would look like success.

    Args:
        channel: Physics channel name.
        prod_root: Root of the :mod:`colliderml-production` checkout.
        particle_flow: Include the stages after tracking (truth tables,
            Pandora, calo/PFO tables, parquet packaging).

    Returns:
        List of dicts with keys ``name``, ``stage``, ``script``, ``image``
        (``"sim"`` or ``"reco"``) and ``config_path`` (*relative to*
        ``prod_root`` so it can be passed straight to a container that
        mounts ``prod_root`` as ``/workspace``).

    Raises:
        FileNotFoundError: If ``particle_flow`` is set and the checkout
            lacks the config of a particle-flow stage.
    """
    stages = get_channel_stages(channel, particle_flow=particle_flow)
    config_dir = find_config_dir(channel, prod_root)

    manifest: List[Dict[str, str]] = []
    for stage in stages:
        config_file = config_dir / stage.config
        if not config_file.exists():
            if stage.particle_flow_only:
                raise FileNotFoundError(
                    f"{config_file} not found: the colliderml-production checkout at "
                    f"{prod_root} has no particle-flow configs for channel '{channel}'. "
                    "It is probably older than the pinned pipeline release; delete it "
                    "to let simulate() re-clone."
                )
            continue
        manifest.append(
            {
                "name": stage.name,
                "stage": stage.stage,
                "script": stage.script,
                "image": stage.image,
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
