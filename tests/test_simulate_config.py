"""Tests for :mod:`colliderml.simulate.config` (pinned pipeline versions)."""

from __future__ import annotations

import pytest

from colliderml.simulate import config as config_mod
from colliderml.simulate import docker as docker_mod
from colliderml.simulate.config import ENV_OVERRIDES, PipelineConfig, load_pipeline_config


def test_bundled_config_has_all_fields() -> None:
    cfg = load_pipeline_config(environ={})
    assert isinstance(cfg, PipelineConfig)
    assert cfg.production_repo.startswith("https://")
    assert cfg.production_ref
    assert cfg.sim_image and cfg.reco_image
    assert cfg.sim_image != cfg.reco_image


def test_docker_module_defaults_come_from_the_bundled_config() -> None:
    cfg = load_pipeline_config(environ={})
    assert docker_mod.DEFAULT_IMAGE == cfg.sim_image
    assert docker_mod.DEFAULT_RECO_IMAGE == cfg.reco_image
    assert docker_mod.COLLIDERML_PRODUCTION_REF == cfg.production_ref
    assert docker_mod.COLLIDERML_PRODUCTION_REPO == cfg.production_repo


@pytest.mark.parametrize("field", sorted(ENV_OVERRIDES))
def test_environment_overrides_one_field(field: str) -> None:
    base = load_pipeline_config(environ={})
    cfg = load_pipeline_config(environ={ENV_OVERRIDES[field]: "override"})
    assert getattr(cfg, field) == "override"
    for other in ("production_repo", "production_ref", "sim_image", "reco_image"):
        if other != field:
            assert getattr(cfg, other) == getattr(base, other)


def test_empty_environment_value_is_ignored() -> None:
    base = load_pipeline_config(environ={})
    assert load_pipeline_config(environ={"COLLIDERML_SIM_IMAGE": ""}) == base


def test_production_ref_env_override_reaches_the_clone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLLIDERML_PRODUCTION_REF", "my-branch")
    assert docker_mod._resolve_production_ref() == "my-branch"
    monkeypatch.delenv("COLLIDERML_PRODUCTION_REF")
    assert docker_mod._resolve_production_ref() == docker_mod.COLLIDERML_PRODUCTION_REF


def test_missing_field_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resource:
        def joinpath(self, name):
            return self

        def read_text(self, encoding="utf-8"):
            return "production_repo: x\nproduction_ref: y\nsim_image: z\n"

    monkeypatch.setattr(config_mod.resources, "files", lambda package: _Resource())
    with pytest.raises(ValueError, match="reco_image"):
        load_pipeline_config(environ={})
