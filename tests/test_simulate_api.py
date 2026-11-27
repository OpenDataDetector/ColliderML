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
from colliderml.simulate.pipeline import (
    CHANNEL_STAGES,
    generate_stage_manifest,
    get_channel_stages,
)

PARTICLE_FLOW_STAGE_IDS = ["particles_full", "pandora_reco", "reco_tables", "package_parquet"]


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


@pytest.mark.parametrize("channel", sorted(CHANNEL_STAGES))
def test_particle_flow_stages_are_opt_in(channel: str) -> None:
    tracking = [s.stage for s in get_channel_stages(channel)]
    full = [s.stage for s in get_channel_stages(channel, particle_flow=True)]
    assert tracking[-1] == "digitization"
    assert full == tracking + PARTICLE_FLOW_STAGE_IDS
    assert full == [s.stage for s in CHANNEL_STAGES[channel]]


def test_only_the_pandora_stages_use_the_reco_image() -> None:
    images = {s.stage: s.image for s in get_channel_stages("ttbar", particle_flow=True)}
    assert {stage for stage, image in images.items() if image == "reco"} == {
        "pandora_reco",
        "reco_tables",
    }
    assert set(images.values()) == {"sim", "reco"}


def _write_stage_configs(prod_root: Path, channel: str, stages: List[str]) -> None:
    cfg_dir = prod_root / "configs_development" / "docker_test" / channel
    cfg_dir.mkdir(parents=True)
    by_id = {s.stage: s for s in CHANNEL_STAGES[channel]}
    for stage in stages:
        (cfg_dir / by_id[stage].config).write_text("events: 10\n")
        script = prod_root / "scripts" / by_id[stage].script
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("")


def test_manifest_carries_image_and_stops_after_tracking_by_default(tmp_path: Path) -> None:
    all_ids = [s.stage for s in CHANNEL_STAGES["higgs_portal"]]
    _write_stage_configs(tmp_path, "higgs_portal", all_ids)

    tracking = generate_stage_manifest("higgs_portal", prod_root=tmp_path)
    assert [m["stage"] for m in tracking] == ["pythia_generation", "simulation", "digitization"]

    full = generate_stage_manifest("higgs_portal", prod_root=tmp_path, particle_flow=True)
    assert [m["stage"] for m in full] == all_ids
    assert [m["image"] for m in full if m["stage"] in ("pandora_reco", "reco_tables")] == ["reco", "reco"]
    assert all(m["image"] == "sim" for m in tracking)


def test_manifest_refuses_a_checkout_without_a_stage_config(tmp_path: Path) -> None:
    _write_stage_configs(tmp_path, "higgs_portal", ["pythia_generation", "simulation", "digitization"])
    assert len(generate_stage_manifest("higgs_portal", prod_root=tmp_path)) == 3
    with pytest.raises(FileNotFoundError, match="stage 'particles_full'"):
        generate_stage_manifest("higgs_portal", prod_root=tmp_path, particle_flow=True)

    (tmp_path / "scripts/simulation/digi_and_reco.py").unlink()
    with pytest.raises(FileNotFoundError, match="scripts/simulation/digi_and_reco.py not found"):
        generate_stage_manifest("higgs_portal", prod_root=tmp_path)

    (tmp_path / "configs_development/docker_test/higgs_portal/simulation_config.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="stage 'simulation'"):
        generate_stage_manifest("higgs_portal", prod_root=tmp_path)


def test_a_channel_without_docker_configs_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "configs_production" / "templates").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No config directory for channel 'ttbar'"):
        generate_stage_manifest("ttbar", prod_root=tmp_path)


def _install_simulate_stubs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Dict[str, Any]:
    """Replace the heavy-weight deps of simulate() with lightweight fakes.

    Returns a dict of captured call arguments so tests can assert against
    what the orchestrator asked for without actually running Docker.
    """
    captured: Dict[str, Any] = {"pull_image": [], "run_pipeline": []}

    def fake_check_runtime_available(runtime: Any = None) -> str:
        return runtime or "docker"

    def fake_pull_image(image: str, *, runtime: Any = None, interactive: bool = True) -> bool:
        captured["pull_image"].append(image)
        return True

    def fake_clone() -> Path:
        clone = tmp_path / "prod"
        clone.mkdir(parents=True, exist_ok=True)
        return clone

    def fake_generate_stage_manifest(
        channel: str, *, prod_root: Path, particle_flow: bool = False
    ) -> List[Dict[str, str]]:
        captured["manifest_channel"] = channel
        captured["manifest_prod_root"] = prod_root
        captured["manifest_particle_flow"] = particle_flow
        # Write a stub YAML per stage, with the keys the real configs have, so the
        # per-run config step (events/pileup/seed kwargs) has something to load.
        import yaml as _yaml
        cfg_dir = prod_root / f"configs_development/docker_test/{channel}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        all_events = {"particles_full", "pandora_reco", "reco_tables"}  # read their whole input
        manifest = []
        for stage in get_channel_stages(channel, particle_flow=particle_flow):
            cfg = {"events": -1 if stage.stage in all_events else 10, "pileup": 10, "seed": 42}
            if stage.stage == "package_parquet":
                cfg = {"run_size": 10, "chunk_size": 10}
            (cfg_dir / stage.config).write_text(_yaml.safe_dump(cfg))
            manifest.append(
                {
                    "name": stage.name,
                    "stage": stage.stage,
                    "script": stage.script,
                    "image": stage.image,
                    "config_path": f"configs_development/docker_test/{channel}/{stage.config}",
                }
            )
        return manifest

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
    assert [s.stage for s in result.stages] == ["pythia_generation", "simulation", "digitization"]
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


def test_simulate_particle_flow_preset_runs_full_chain_with_both_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_simulate_stubs(monkeypatch, tmp_path)
    result = simulate(
        preset="ttbar-pflow-quick",
        output_dir=tmp_path / "out",
        image="sim:test",
        reco_image="reco:test",
        quiet=True,
    )
    assert result.particle_flow is True
    assert captured["manifest_particle_flow"] is True
    assert captured["pull_image"] == ["sim:test", "reco:test"]
    call = captured["run_pipeline"][0]
    assert call["image"] == "sim:test"
    assert call["reco_image"] == "reco:test"
    assert result.parquet_dir == (tmp_path / "out").resolve() / "parquet"


def test_particle_flow_run_keeps_all_events_in_the_stages_after_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    captured = _install_simulate_stubs(monkeypatch, tmp_path)
    simulate(
        preset="ttbar-pflow-quick", events=3, pandora_processes=2, runtime="podman-hpc",
        output_dir=tmp_path / "out", quiet=True,
    )
    call = captured["run_pipeline"][0]
    assert call["runtime"] == "podman-hpc"
    assert len(call["stage_manifest"]) == 9
    events = {
        m["stage"]: yaml.safe_load((tmp_path / "out" / m["config_path"][len("/output/"):]).read_text()).get("events")
        for m in call["stage_manifest"]
    }
    # an event cap in reco_tables is refused when Pandora ran in slices
    assert events["particles_full"] == events["pandora_reco"] == events["reco_tables"] == -1
    assert events["madgraph_generation"] == events["simulation"] == events["digitization"] == 3
    packaging = yaml.safe_load((tmp_path / "out/configs/0/package_parquet__package_parquet_config.yaml").read_text())
    assert packaging == {"run_size": 3, "chunk_size": 3}


def test_simulate_without_particle_flow_pulls_only_the_sim_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_simulate_stubs(monkeypatch, tmp_path)
    result = simulate(preset="ttbar-quick", output_dir=tmp_path / "out", quiet=True)
    assert result.particle_flow is False
    assert captured["manifest_particle_flow"] is False
    assert len(captured["pull_image"]) == 1


def test_simulate_particle_flow_argument_overrides_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_simulate_stubs(monkeypatch, tmp_path)
    simulate(preset="ttbar-pflow-quick", particle_flow=False, output_dir=tmp_path / "a", quiet=True)
    assert captured["manifest_particle_flow"] is False
    simulate(preset="ttbar-quick", particle_flow=True, output_dir=tmp_path / "b", quiet=True)
    assert captured["manifest_particle_flow"] is True


def test_simulate_default_images_follow_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_simulate_stubs(monkeypatch, tmp_path)
    monkeypatch.setenv("COLLIDERML_SIM_IMAGE", "sim:from-env")
    monkeypatch.setenv("COLLIDERML_RECO_IMAGE", "reco:from-env")
    simulate(preset="ttbar-pflow-quick", output_dir=tmp_path / "out", quiet=True)
    assert captured["pull_image"] == ["sim:from-env", "reco:from-env"]


def test_simulate_remote_rejects_particle_flow() -> None:
    with pytest.raises(ValueError, match="not available with remote=True"):
        simulate(preset="ttbar-pflow-quick", remote=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"events": 0},
        {"events": 5, "pileup": -1},
        {"events": 5, "threads": 1},
        {"events": 5, "pandora_processes": 0},
    ],
)
def test_simulate_rejects_out_of_range_counts(kwargs: Dict[str, int]) -> None:
    with pytest.raises(ValueError):
        simulate(channel="ttbar", **kwargs)


def test_simulation_result_tables_groups_files_by_table(tmp_path: Path) -> None:
    result = SimulationResult(
        channel="ttbar", events=2, pileup=0, output_dir=tmp_path, run_dir=tmp_path / "runs" / "0",
        particle_flow=True,
    )
    assert result.tables() == {}
    for group, table in (("reco", "pfos"), ("reco", "tracks"), ("truth", "particles")):
        folder = result.parquet_dir / group / table
        folder.mkdir(parents=True)
        (folder / f"x.{group}.{table}.events0-1.parquet").write_bytes(b"")
    tables = result.tables()
    assert sorted(tables) == ["particles", "pfos", "tracks"]
    assert tables["pfos"][0].name == "x.reco.pfos.events0-1.parquet"

    tracking_only = SimulationResult(
        channel="ttbar", events=2, pileup=0, output_dir=tmp_path, run_dir=tmp_path / "runs" / "0"
    )
    assert tracking_only.tables() == {}


def test_simulate_refuses_an_output_directory_with_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_simulate_stubs(monkeypatch, tmp_path)
    simulate(preset="ttbar-pflow-quick", output_dir=tmp_path / "out", quiet=True)
    with pytest.raises(FileExistsError, match="already holds a run"):
        simulate(preset="ttbar-pflow-quick", seed=7, output_dir=tmp_path / "out", quiet=True)
    assert len(captured["run_pipeline"]) == 1


def test_particle_flow_needs_the_output_directory_to_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_simulate_stubs(monkeypatch, tmp_path)
    simulate(preset="ttbar-quick", output_dir=tmp_path / "out", run_id="1", quiet=True)
    with pytest.raises(FileExistsError, match="already holds a run"):
        simulate(preset="ttbar-pflow-quick", output_dir=tmp_path / "out", quiet=True)
    assert len(captured["run_pipeline"]) == 1


def test_tracking_runs_can_share_an_output_directory_with_different_run_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_simulate_stubs(monkeypatch, tmp_path)
    simulate(preset="ttbar-quick", output_dir=tmp_path / "out", run_id="a", quiet=True)
    simulate(preset="ttbar-quick", output_dir=tmp_path / "out", run_id="b", quiet=True)
    with pytest.raises(FileExistsError):
        simulate(preset="ttbar-quick", output_dir=tmp_path / "out", run_id="a", quiet=True)
    assert [c["run_id"] for c in captured["run_pipeline"]] == ["a", "b"]


@pytest.mark.parametrize("run_id", ["../x", "/abs", "a b", "", "a/b", ".hidden"])
def test_simulate_rejects_run_ids_that_are_not_plain_names(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        simulate(channel="ttbar", events=2, run_id=run_id)


def test_particle_flow_requires_the_default_run_id() -> None:
    with pytest.raises(ValueError, match="run_id='0'"):
        simulate(preset="ttbar-pflow-quick", run_id="3")


def test_colliderml_simulate_is_callable_in_any_import_order() -> None:
    import importlib

    import colliderml

    with pytest.raises(ValueError, match="requires either a preset"):
        colliderml.simulate()
    importlib.import_module("colliderml.simulate")  # binds the subpackage on the parent
    with pytest.raises(ValueError, match="requires either a preset"):
        colliderml.simulate()
    with pytest.raises(ValueError, match="requires either a preset"):
        colliderml.simulate()


def test_two_output_directories_do_not_share_stage_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    captured = _install_simulate_stubs(monkeypatch, tmp_path)
    simulate(channel="higgs_portal", events=3, pileup=0, output_dir=tmp_path / "a", quiet=True)
    simulate(channel="higgs_portal", events=7, pileup=0, output_dir=tmp_path / "b", quiet=True)
    for name, events in (("a", 3), ("b", 7)):
        configs = sorted((tmp_path / name / "configs" / "0").glob("*.yaml"))
        assert len(configs) == 3
        assert all(yaml.safe_load(c.read_text())["events"] == events for c in configs)
    first = captured["run_pipeline"][0]["stage_manifest"][0]["config_path"]
    assert first == "/output/configs/0/pythia_generation__pythia_config.yaml"


def test_simulate_empty_manifest_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_simulate_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        api_mod, "generate_stage_manifest", lambda channel, *, prod_root, particle_flow=False: []
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
def _host_path(output_dir: Path, container_path: str) -> Path:
    """Host file behind a container path under /output."""
    assert container_path.startswith("/output/")
    return output_dir / container_path[len("/output/"):]


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
            manifest, prod_root=tmp_path, output_dir=tmp_path / "out", run_id="0",
            events=2, pileup=5, seed=99,
        )
        # config_path was rewritten to the container path of the per-run config
        assert new_manifest[0]["config_path"].startswith("/output/configs/0/")
        overlay = _host_path(tmp_path / "out", new_manifest[0]["config_path"])
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
            manifest, prod_root=tmp_path, output_dir=tmp_path / "out", run_id="0",
            events=2, pileup=5, seed=42,
        )
        loaded = _yaml.safe_load(_host_path(tmp_path / "out", new_manifest[0]["config_path"]).read_text())
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
            manifest, prod_root=tmp_path, output_dir=tmp_path / "out", run_id="0",
            events=2, pileup=5, seed=42,
        )
        loaded = _yaml.safe_load(_host_path(tmp_path / "out", new_manifest[0]["config_path"]).read_text())
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

        out = tmp_path / "out"
        _apply_runtime_overrides(manifest, prod_root=tmp_path, output_dir=out, run_id="alpha",
                                 events=2, pileup=5, seed=42)
        _apply_runtime_overrides(manifest, prod_root=tmp_path, output_dir=out, run_id="beta",
                                 events=99, pileup=200, seed=42)

        alpha = _yaml.safe_load((out / "configs/alpha/pythia_generation__pythia_config.yaml").read_text())
        beta = _yaml.safe_load((out / "configs/beta/pythia_generation__pythia_config.yaml").read_text())
        assert alpha["events"] == 2 and alpha["pileup"] == 5
        assert beta["events"] == 99 and beta["pileup"] == 200

    def test_overlay_sets_threads_processes_and_packaging_sizes(self, tmp_path: Path) -> None:
        import yaml

        from colliderml.simulate.api import _apply_runtime_overrides

        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        configs = {
            "simulation": {"events": 10, "threads": 1},
            "pandora_reco": {"events": -1, "processes": 1},
            "package_parquet": {"run_size": 10, "chunk_size": 10, "row_group_size": 100000},
        }
        manifest = []
        for stage, cfg in configs.items():
            (cfg_dir / f"{stage}.yaml").write_text(yaml.safe_dump(cfg))
            manifest.append({"stage": stage, "config_path": f"configs/{stage}.yaml"})

        new = _apply_runtime_overrides(
            manifest, prod_root=tmp_path, output_dir=tmp_path / "out", run_id="0", events=3, pileup=0, seed=1,
            threads=4, pandora_processes=2,
        )
        out = {m["stage"]: yaml.safe_load(_host_path(tmp_path / "out", m["config_path"]).read_text())
               for m in new}
        assert out["simulation"] == {"events": 3, "threads": 4}
        assert out["pandora_reco"] == {"events": -1, "processes": 2}
        assert out["package_parquet"] == {"run_size": 3, "chunk_size": 3, "row_group_size": 100000}

    def test_overlay_leaves_threads_alone_when_not_requested(self, tmp_path: Path) -> None:
        import yaml

        from colliderml.simulate.api import _apply_runtime_overrides

        (tmp_path / "sim.yaml").write_text(yaml.safe_dump({"events": 10, "threads": 8, "processes": 5}))
        new = _apply_runtime_overrides(
            [{"stage": "simulation", "config_path": "sim.yaml"}],
            prod_root=tmp_path, output_dir=tmp_path / "out", run_id="0", events=3, pileup=0, seed=1,
        )
        assert yaml.safe_load(_host_path(tmp_path / "out", new[0]["config_path"]).read_text()) == {
            "events": 3, "threads": 8, "processes": 5,
        }

    @pytest.mark.parametrize("run_id,expected", [
        ("0", "/output/runs/all/0/events.hepmc.gz"),
        ("7", "/output/runs/all/7/events.hepmc.gz"),
        ("scan-a", "/output/runs/scan-a/events.hepmc.gz"),
    ])
    def test_overlay_points_pythia_at_the_madgraph_output(
        self, tmp_path: Path, run_id: str, expected: str
    ) -> None:
        import yaml

        from colliderml.simulate.api import _apply_runtime_overrides

        (tmp_path / "pythia.yaml").write_text(
            yaml.safe_dump({"events": 10, "hard_scatter_file": "/output/runs/all/0/events.hepmc.gz"})
        )
        new = _apply_runtime_overrides(
            [{"stage": "pythia_generation", "config_path": "pythia.yaml"}],
            prod_root=tmp_path, output_dir=tmp_path / "out", run_id=run_id, events=3, pileup=0, seed=1,
        )
        cfg = yaml.safe_load(_host_path(tmp_path / "out", new[0]["config_path"]).read_text())
        assert cfg["hard_scatter_file"] == expected
