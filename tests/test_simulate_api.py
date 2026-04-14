"""Unit tests for :mod:`colliderml.simulate.api`.

All Docker and network calls are mocked via ``monkeypatch`` so these tests
run in the fast gate. Integration coverage for real container runs lives in
:mod:`tests.test_simulate_docker` behind the ``integration`` marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from colliderml.simulate import api as api_mod
from colliderml.simulate.api import SimulationResult, simulate
from colliderml.simulate.docker import StageRun
from colliderml.simulate.pipeline import get_channel_stages


def test_simulate_requires_preset_or_channel() -> None:
    with pytest.raises(ValueError, match="preset or an explicit"):
        simulate()


def test_simulate_requires_events_when_no_preset() -> None:
    with pytest.raises(ValueError, match="requires `events`"):
        simulate(channel="ttbar")


def test_simulate_rejects_unknown_channel() -> None:
    with pytest.raises(ValueError, match="Unknown simulation channel"):
        simulate(channel="not_a_channel", events=10)


def test_get_channel_stages_includes_known_channels() -> None:
    assert get_channel_stages("ttbar")
    assert get_channel_stages("higgs_portal")
    assert get_channel_stages("single_muon")


def _install_simulate_stubs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Dict[str, Any]:
    """Replace the heavy-weight deps of simulate() with lightweight fakes.

    Returns a dict of captured call arguments so tests can assert against
    what the orchestrator asked for without actually running Docker.
    """
    captured: Dict[str, Any] = {"pull_image": [], "run_pipeline": []}

    def fake_check_runtime_available(runtime: Any = None) -> str:
        return "docker"

    def fake_pull_image(image: str, *, runtime: Any = None, interactive: bool = True) -> bool:
        captured["pull_image"].append(image)
        return True

    def fake_clone() -> Path:
        clone = tmp_path / "prod"
        clone.mkdir(parents=True, exist_ok=True)
        return clone

    def fake_generate_stage_manifest(channel: str, *, prod_root: Path) -> List[Dict[str, str]]:
        captured["manifest_channel"] = channel
        captured["manifest_prod_root"] = prod_root
        return [
            {
                "name": "Pythia Generation",
                "stage": "pythia_generation",
                "script": "simulation/pythia_gen.py",
                "config_path": f"configs_development/docker_test/{channel}/pythia_config.yaml",
            },
            {
                "name": "Detector Simulation",
                "stage": "simulation",
                "script": "simulation/ddsim_run.py",
                "config_path": f"configs_development/docker_test/{channel}/simulation_config.yaml",
            },
        ]

    def fake_run_pipeline(**kwargs: Any) -> List[StageRun]:
        captured["run_pipeline"].append(kwargs)
        run_dir = Path(kwargs["output_dir"]) / "runs" / kwargs["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "merged_events.hepmc3").write_bytes(b"stub")
        return [
            StageRun(name=s["name"], stage=s["stage"], returncode=0)
            for s in kwargs["stage_manifest"]
        ]

    monkeypatch.setattr(api_mod, "check_runtime_available", fake_check_runtime_available)
    monkeypatch.setattr(api_mod, "pull_image", fake_pull_image)
    monkeypatch.setattr(api_mod, "clone_colliderml_production", fake_clone)
    monkeypatch.setattr(api_mod, "generate_stage_manifest", fake_generate_stage_manifest)
    monkeypatch.setattr(api_mod, "run_pipeline", fake_run_pipeline)
    return captured


def test_simulate_local_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_simulate_stubs(monkeypatch, tmp_path)

    result = simulate(
        channel="higgs_portal",
        events=5,
        pileup=0,
        output_dir=tmp_path / "out",
        quiet=True,
    )

    assert isinstance(result, SimulationResult)
    assert result.channel == "higgs_portal"
    assert result.events == 5
    assert result.pileup == 0
    assert result.output_dir == (tmp_path / "out").resolve()
    assert result.run_dir == (tmp_path / "out").resolve() / "runs" / "0"
    assert result.remote_request_id is None
    assert [s.returncode for s in result.stages] == [0, 0]
    assert captured["manifest_channel"] == "higgs_portal"
    assert captured["pull_image"], "pull_image should have been called"
    assert captured["run_pipeline"], "run_pipeline should have been called"
    assert captured["run_pipeline"][0]["run_id"] == "0"


def test_simulate_with_preset_applies_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_simulate_stubs(monkeypatch, tmp_path)
    result = simulate(preset="ttbar-quick", output_dir=tmp_path / "out", quiet=True)
    assert result.channel == "ttbar"
    assert result.events == 10
    assert result.pileup == 0


def test_simulate_explicit_args_override_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_simulate_stubs(monkeypatch, tmp_path)
    result = simulate(
        preset="ttbar-quick",
        events=25,
        pileup=40,
        output_dir=tmp_path / "out",
        quiet=True,
    )
    # channel still comes from the preset; events/pileup from the explicit args.
    assert result.channel == "ttbar"
    assert result.events == 25
    assert result.pileup == 40


def test_simulate_empty_manifest_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_simulate_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        api_mod, "generate_stage_manifest", lambda channel, *, prod_root: []
    )
    with pytest.raises(RuntimeError, match="No stages were generated"):
        simulate(
            channel="higgs_portal",
            events=1,
            pileup=0,
            output_dir=tmp_path / "out",
            quiet=True,
        )


def test_simulation_result_list_files_includes_stage_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_simulate_stubs(monkeypatch, tmp_path)
    result = simulate(
        channel="higgs_portal",
        events=1,
        pileup=0,
        output_dir=tmp_path / "out",
        quiet=True,
    )
    files = result.list_files()
    assert any(fp.name == "merged_events.hepmc3" for fp in files)
