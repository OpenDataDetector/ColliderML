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
        # Materialise stub YAML files alongside the manifest entries so the
        # runtime-overrides overlay step (which reads each stage's YAML to
        # apply events/pileup/seed kwargs) has something to load.
        import yaml as _yaml
        cfg_dir = prod_root / f"configs_development/docker_test/{channel}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "pythia_config.yaml").write_text(
            _yaml.safe_dump({"events": 10, "pileup": 10, "seed": 42})
        )
        (cfg_dir / "simulation_config.yaml").write_text(
            _yaml.safe_dump({"events": 10, "pileup": 10, "seed": 42})
        )
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


# ---------------------------------------------------------------------------
# Runtime override overlays (events / pileup / seed kwargs actually take effect)
# ---------------------------------------------------------------------------
class TestRuntimeOverrides:
    """``simulate(events=N, pileup=M)`` must propagate into the per-stage
    YAML configs that the container reads. Without the overlay step the
    static YAMLs win and the caller's kwargs get silently dropped after
    the user-facing 'Running … (N events, pileup=M)' message."""

    def test_overlay_overrides_events_pileup_seed(self, tmp_path: Path) -> None:
        import yaml as _yaml
        from colliderml.simulate.api import _apply_runtime_overrides

        # Fake prod_root with one stage config containing all three keys.
        cfg_dir = tmp_path / "configs_development/docker_test/higgs_portal"
        cfg_dir.mkdir(parents=True)
        cfg_path = cfg_dir / "pythia_config.yaml"
        cfg_path.write_text(_yaml.safe_dump({
            "events": 10, "pileup": 10, "seed": 42, "channel": "higgs_portal",
        }))

        manifest = [{
            "name": "Pythia Generation",
            "stage": "pythia_generation",
            "script": "simulation/pythia_gen.py",
            "config_path": "configs_development/docker_test/higgs_portal/pythia_config.yaml",
        }]

        new_manifest = _apply_runtime_overrides(
            manifest, prod_root=tmp_path, run_id="0",
            events=2, pileup=5, seed=99,
        )
        # config_path was rewritten to the overlay
        assert new_manifest[0]["config_path"].startswith(".overlays/0/")
        overlay = tmp_path / new_manifest[0]["config_path"]
        assert overlay.exists()

        # Loaded values reflect the kwargs
        loaded = _yaml.safe_load(overlay.read_text())
        assert loaded["events"] == 2
        assert loaded["pileup"] == 5
        assert loaded["seed"] == 99

    def test_overlay_preserves_unrelated_keys(self, tmp_path: Path) -> None:
        import yaml as _yaml
        from colliderml.simulate.api import _apply_runtime_overrides

        cfg_dir = tmp_path / "configs_development/docker_test/higgs_portal"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "pythia_config.yaml").write_text(_yaml.safe_dump({
            "events": 10,
            "pythia_settings": ["Higgs:useBSM = on", "35:m0 = 125.0"],
            "vertex_sigma_xy": 0.0125,
        }))

        manifest = [{"name": "P", "stage": "pythia_generation",
                     "script": "s", "config_path": "configs_development/docker_test/higgs_portal/pythia_config.yaml"}]
        new_manifest = _apply_runtime_overrides(
            manifest, prod_root=tmp_path, run_id="0",
            events=2, pileup=5, seed=42,
        )
        loaded = _yaml.safe_load((tmp_path / new_manifest[0]["config_path"]).read_text())
        assert loaded["events"] == 2
        # Hand-crafted Pythia knobs survive
        assert loaded["pythia_settings"] == ["Higgs:useBSM = on", "35:m0 = 125.0"]
        assert loaded["vertex_sigma_xy"] == 0.0125

    def test_overlay_skips_missing_keys(self, tmp_path: Path) -> None:
        """A stage's YAML without an ``events`` key shouldn't gain one —
        e.g. madgraph_init_config doesn't take an event count, we shouldn't
        forge it into the config."""
        import yaml as _yaml
        from colliderml.simulate.api import _apply_runtime_overrides

        cfg_dir = tmp_path / "configs_development/docker_test/ttbar"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "madgraph_init_config.yaml").write_text(_yaml.safe_dump({
            "process": "p p > t t~",
            "order": "NLO",
            # no `events`, no `pileup`
        }))
        manifest = [{"name": "MG init", "stage": "madgraph_init",
                     "script": "s", "config_path": "configs_development/docker_test/ttbar/madgraph_init_config.yaml"}]
        new_manifest = _apply_runtime_overrides(
            manifest, prod_root=tmp_path, run_id="0",
            events=2, pileup=5, seed=42,
        )
        loaded = _yaml.safe_load((tmp_path / new_manifest[0]["config_path"]).read_text())
        assert "events" not in loaded
        assert "pileup" not in loaded
        # Stage-specific keys preserved
        assert loaded["process"] == "p p > t t~"

    def test_overlay_per_run_id_namespacing(self, tmp_path: Path) -> None:
        import yaml as _yaml
        from colliderml.simulate.api import _apply_runtime_overrides

        cfg = tmp_path / "configs_development/docker_test/higgs_portal/pythia_config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(_yaml.safe_dump({"events": 10, "pileup": 10}))
        manifest = [{"name": "P", "stage": "pythia_generation",
                     "script": "s", "config_path": "configs_development/docker_test/higgs_portal/pythia_config.yaml"}]

        _apply_runtime_overrides(manifest, prod_root=tmp_path, run_id="alpha",
                                 events=2, pileup=5, seed=42)
        _apply_runtime_overrides(manifest, prod_root=tmp_path, run_id="beta",
                                 events=99, pileup=200, seed=42)

        alpha = _yaml.safe_load((tmp_path / ".overlays/alpha/pythia_generation__pythia_config.yaml").read_text())
        beta = _yaml.safe_load((tmp_path / ".overlays/beta/pythia_generation__pythia_config.yaml").read_text())
        assert alpha["events"] == 2 and alpha["pileup"] == 5
        assert beta["events"] == 99 and beta["pileup"] == 200
