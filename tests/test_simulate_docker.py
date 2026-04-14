"""Integration tests for :mod:`colliderml.simulate.docker`.

These tests hit the real container runtime and/or the real Git clone
endpoint, so they are marked ``integration`` and excluded from
``make test-fast``. Run with::

    pytest -m integration tests/test_simulate_docker.py

Or just the runtime smoke test (no network needed)::

    pytest tests/test_simulate_docker.py::test_get_container_runtime_detects_docker_or_podman
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from colliderml.simulate import docker as docker_mod
from colliderml.simulate.docker import (
    ContainerRuntimeError,
    clone_colliderml_production,
    default_cache_root,
    get_container_runtime,
)


def test_get_container_runtime_detects_docker_or_podman() -> None:
    """Lightweight sanity check: detects whichever runtime is installed.

    Not marked integration because it only looks at ``$PATH`` — no daemon,
    no network. Skipped if neither runtime is installed (common on CI).
    """
    if not (shutil.which("docker") or shutil.which("podman")):
        pytest.skip("neither docker nor podman on PATH")
    rt = get_container_runtime()
    assert rt in {"docker", "podman"}


def test_get_container_runtime_raises_when_neither_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_mod.shutil, "which", lambda name: None)
    with pytest.raises(ContainerRuntimeError):
        get_container_runtime()


def test_default_cache_root_honours_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("COLLIDERML_CACHE", str(tmp_path))
    root = default_cache_root()
    assert root == (tmp_path / "simulate").resolve()


def test_default_cache_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLLIDERML_CACHE", raising=False)
    root = default_cache_root()
    assert root.name == "simulate"
    assert root.parent.name == "colliderml"


@pytest.mark.integration
def test_clone_colliderml_production_fetches_repo(tmp_path: Path) -> None:
    """Really clone the production repo into a scratch dir.

    Marked integration because it needs GitHub network access (~30 s for a
    shallow clone).
    """
    clone = clone_colliderml_production(cache_root=tmp_path)
    assert clone.is_dir()
    assert (clone / ".git").exists()
    # The production repo must contain the canonical pipeline scripts;
    # if these are missing our auto-clone target has drifted.
    assert (clone / "scripts" / "cli" / "setup_container_env.sh").exists()
    assert (clone / "configs_development" / "docker_test").is_dir()
