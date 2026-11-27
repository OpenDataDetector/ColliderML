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
    assert rt in docker_mod.SUPPORTED_RUNTIMES


def test_get_container_runtime_prefers_podman_hpc_over_podman(monkeypatch: pytest.MonkeyPatch) -> None:
    # where podman-hpc is installed, plain podman is on PATH as well
    monkeypatch.setattr(
        docker_mod.shutil, "which", lambda name: None if name == "docker" else f"/usr/bin/{name}"
    )
    assert get_container_runtime() == "podman-hpc"
    monkeypatch.setattr(
        docker_mod.shutil, "which", lambda name: "/usr/bin/podman" if name == "podman" else None
    )
    assert get_container_runtime() == "podman"


def test_image_defaults_follow_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        docker_mod.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0)
    )
    monkeypatch.setenv("COLLIDERML_SIM_IMAGE", "sim:env")
    monkeypatch.setenv("COLLIDERML_RECO_IMAGE", "reco:env")
    assert docker_mod.pull_image(runtime="podman", interactive=False) is True
    assert calls[-1] == ["podman", "image", "exists", "sim:env"]
    prod = _prod_checkout(tmp_path)
    for kind, expected in (("sim", "sim:env"), ("reco", "reco:env")):
        docker_mod.run_stage(prod_root=prod, output_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                             script="a/b.py", config_path="c.yaml", image_kind=kind, runtime="podman")
        assert expected in calls[-1]


def test_pull_image_pulls_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1 if cmd[1:3] == ["image", "inspect"] else 0)

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    assert docker_mod.pull_image("img:1", runtime="docker", interactive=False) is True
    assert calls == [["docker", "image", "inspect", "img:1"], ["docker", "pull", "img:1"]]


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
    image_at = cmd.index("reco:test")
    assert cmd[image_at + 1] == "-c"
    # every setting is a positional parameter of the fixed in-container command
    assert cmd[image_at + 3:] == [
        "stage",
        "scripts/cli/docker_stage_env.sh",
        "reco",
        "scripts/cli/stage_timer.py",
        "simulation/pandora_reco.py",
        "pandora_reco",
        "/workspace/configs/pandora.yaml",
        "3",
        "7",
    ]
    inner = cmd[image_at + 2]
    assert 'source "/workspace/$1" "$2" || exit 1' in inner
    assert '--stage "$5" -- python3 "$(basename "$4")" --config "$6"' in inner
    assert "pandora_reco" not in inner and "configs/" not in inner
    assert kwargs["timeout"] is None


def test_run_stage_takes_container_config_paths_and_host_proxies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        docker_mod.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0)
    )
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("https_proxy", "http://proxy:3128")
    docker_mod.run_stage(
        prod_root=_prod_checkout(tmp_path),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        script="simulation/pythia_gen.py",
        config_path="/output/configs/0/pythia_generation__pythia_config.yaml",
        runtime="docker",
    )
    cmd = calls[0]
    assert "/output/configs/0/pythia_generation__pythia_config.yaml" in cmd
    assert "https_proxy=http://proxy:3128" in cmd
    assert not any(arg.startswith("HTTP_PROXY=") for arg in cmd)
    assert "pythia_gen" in cmd  # stage label defaults to the script name


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
    monkeypatch.setenv("COLLIDERML_CACHE", str(tmp_path / "cache"))
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
    # the Geant4 cache sits next to the production checkout, not inside it
    assert (tmp_path / "cache" / "simulate" / "container_cache").is_dir()
    assert not (tmp_path / "prod" / ".cache").exists()


def _fake_git(monkeypatch: pytest.MonkeyPatch, outcomes: dict) -> list:
    """Replace subprocess.run in the docker module; outcomes maps a git subcommand to a return code."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, outcomes.get(cmd[3], 0), stdout="", stderr="no such ref")

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)
    return calls


def test_cached_clone_is_reused_offline_when_the_ref_is_already_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "colliderml-production" / ".git").mkdir(parents=True)
    calls = _fake_git(monkeypatch, {"fetch": 1, "checkout": 0})
    clone = clone_colliderml_production(cache_root=tmp_path, ref="pipeline-v9")
    assert clone == tmp_path / "colliderml-production"
    assert [c[3] for c in calls] == ["fetch", "checkout"]
    assert calls[1][-1] == "pipeline-v9"


def test_cached_clone_that_cannot_reach_the_ref_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "colliderml-production" / ".git").mkdir(parents=True)
    _fake_git(monkeypatch, {"fetch": 1, "checkout": 1})
    with pytest.raises(ContainerRuntimeError, match="Could not check out 'pipeline-v9'"):
        clone_colliderml_production(cache_root=tmp_path, ref="pipeline-v9")
