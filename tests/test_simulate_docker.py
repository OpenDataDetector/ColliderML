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
import subprocess
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


def test_get_container_runtime_falls_back_to_podman_hpc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        docker_mod.shutil, "which", lambda name: "/usr/bin/podman-hpc" if name == "podman-hpc" else None
    )
    assert get_container_runtime() == "podman-hpc"


@pytest.mark.parametrize(
    "runtime,query", [("docker", "inspect"), ("podman", "exists"), ("podman-hpc", "exists")]
)
def test_check_image_available_uses_the_runtime_query(
    monkeypatch: pytest.MonkeyPatch, runtime: str, query: str
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    assert docker_mod.check_image_available("img:tag", runtime=runtime) is True
    assert calls == [[runtime, "image", query, "img:tag"]]


def _prod_checkout(tmp_path: Path) -> Path:
    prod = tmp_path / "prod"
    (prod / "scripts" / "cli").mkdir(parents=True)
    (prod / docker_mod.STAGE_ENV_SCRIPT).write_text("")
    return prod


def test_run_stage_builds_the_container_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    prod = _prod_checkout(tmp_path)
    docker_mod.run_stage(
        prod_root=prod,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        script="simulation/pandora_reco.py",
        config_path="configs/pandora.yaml",
        stage="pandora_reco",
        image_kind="reco",
        seed=7,
        run_id="3",
        image="reco:test",
        runtime="podman",
    )
    cmd, kwargs = calls[0]
    assert cmd[:3] == ["podman", "run", "--rm"]
    assert f"{prod}:/workspace" in cmd
    assert cmd[cmd.index("--entrypoint") + 1] == "/bin/bash"
    assert cmd[-3] == "reco:test"
    inner = cmd[-1]
    assert "source /workspace/scripts/cli/docker_stage_env.sh reco && " in inner
    assert "--record /output/pipeline_timing.csv --stage pandora_reco -- python3 pandora_reco.py" in inner
    assert "--config /workspace/configs/pandora.yaml" in inner
    assert "--output-subdir 3 --seed 7" in inner
    assert kwargs["timeout"] is None


def test_run_stage_rejects_a_checkout_without_the_stage_env_script(tmp_path: Path) -> None:
    (tmp_path / "prod").mkdir()
    with pytest.raises(ContainerRuntimeError, match="docker_stage_env.sh"):
        docker_mod.run_stage(
            prod_root=tmp_path / "prod",
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            script="simulation/pythia_gen.py",
            config_path="c.yaml",
            runtime="docker",
        )


def test_run_pipeline_routes_stages_to_their_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    used = []

    def fake_run_stage(**kwargs):
        used.append((kwargs["stage"], kwargs["image_kind"], kwargs["image"]))
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(docker_mod, "run_stage", fake_run_stage)
    manifest = [
        {"name": "Sim", "stage": "simulation", "script": "a.py", "config_path": "a.yaml", "image": "sim"},
        {"name": "Pandora", "stage": "pandora_reco", "script": "b.py", "config_path": "b.yaml", "image": "reco"},
        {"name": "Old entry", "stage": "digitization", "script": "c.py", "config_path": "c.yaml"},
    ]
    results = docker_mod.run_pipeline(
        channel="ttbar",
        output_dir=tmp_path / "out",
        stage_manifest=manifest,
        image="sim:test",
        reco_image="reco:test",
        prod_root=_prod_checkout(tmp_path),
        runtime="docker",
    )
    assert used == [
        ("simulation", "sim", "sim:test"),
        ("pandora_reco", "reco", "reco:test"),
        ("digitization", "sim", "sim:test"),
    ]
    assert [r.returncode for r in results] == [0, 0, 0]
    assert (tmp_path / "prod" / ".cache").is_dir()


def test_madgraph_output_is_copied_into_the_run_directory(tmp_path: Path) -> None:
    staged = tmp_path / "runs" / "all" / "0"
    staged.mkdir(parents=True)
    (staged / "events.hepmc.gz").write_bytes(b"hepmc")
    docker_mod._copy_madgraph_output(tmp_path, "0")
    assert (tmp_path / "runs" / "0" / "events.hepmc.gz").read_bytes() == b"hepmc"


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
    assert (clone / docker_mod.STAGE_ENV_SCRIPT).exists()
    assert (clone / "configs_development" / "docker_test").is_dir()
