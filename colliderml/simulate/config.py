"""Pinned pipeline versions for :func:`colliderml.simulate`.

The pipeline code (a git ref of the production repository) and the two container
images are pinned together in :file:`pipeline_config.yaml`, which ships as package
data. :func:`load_pipeline_config` reads it and applies the environment-variable
overrides, so a development branch or a locally built image can be tried without
editing the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from typing import Mapping, Optional

import yaml

_CONFIG_PACKAGE = "colliderml.simulate"
_CONFIG_FILENAME = "pipeline_config.yaml"

#: Environment variable that overrides each field of :class:`PipelineConfig`.
ENV_OVERRIDES = {
    "production_ref": "COLLIDERML_PRODUCTION_REF",
    "sim_image": "COLLIDERML_SIM_IMAGE",
    "reco_image": "COLLIDERML_RECO_IMAGE",
}


@dataclass(frozen=True)
class PipelineConfig:
    """Versions one :func:`~colliderml.simulate.simulate` call runs.

    Attributes:
        production_repo: Git URL of the repository with the pipeline scripts.
        production_ref: Tag, branch or commit of that repository.
        sim_image: Image for generation, Geant4, tracking and packaging.
        reco_image: Image for calorimeter digitisation and Pandora.
    """

    production_repo: str
    production_ref: str
    sim_image: str
    reco_image: str


def load_pipeline_config(environ: Optional[Mapping[str, str]] = None) -> PipelineConfig:
    """Read :file:`pipeline_config.yaml` and apply environment overrides.

    Args:
        environ: Environment to read overrides from. Defaults to ``os.environ``.

    Returns:
        The effective :class:`PipelineConfig`.

    Raises:
        ValueError: If the file lacks one of the four fields.
    """
    env = os.environ if environ is None else environ
    text = resources.files(_CONFIG_PACKAGE).joinpath(_CONFIG_FILENAME).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{_CONFIG_FILENAME} did not parse to a mapping.")
    values = {}
    for name in ("production_repo", "production_ref", "sim_image", "reco_image"):
        value = env.get(ENV_OVERRIDES.get(name, ""), "") or data.get(name)
        if not value:
            raise ValueError(f"{_CONFIG_FILENAME} is missing required field '{name}'.")
        values[name] = str(value)
    return PipelineConfig(**values)


__all__ = ["ENV_OVERRIDES", "PipelineConfig", "load_pipeline_config"]
