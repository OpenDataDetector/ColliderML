"""Top-level :func:`simulate` orchestration function.

This is the user-facing entry point. It understands the four ways a caller
can drive a run:

1. By preset: ``simulate(preset="ttbar-quick")``
2. By explicit parameters: ``simulate(channel="higgs_portal", events=10, pileup=10)``
3. Locally (via Docker/Podman): the default
4. Remotely (via the backend SaaS): ``simulate(..., remote=True)``

All container lifecycle, production-repo cloning, and config generation is
delegated to :mod:`~colliderml.simulate.docker` and
:mod:`~colliderml.simulate.pipeline`. This module is deliberately thin — it
coordinates, prints progress, and wraps results in a user-friendly object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from colliderml.simulate.config import load_pipeline_config
from colliderml.simulate.docker import (
    StageRun,
    check_runtime_available,
    clone_colliderml_production,
    pull_image,
    run_pipeline,
)
from colliderml.simulate.pipeline import generate_stage_manifest, get_channel_stages
from colliderml.simulate.presets import Preset, resolve_preset


@dataclass
class SimulationResult:
    """The outcome of a successful :func:`simulate` call.

    Wraps enough metadata to inspect the run and lazily load its tabular
    outputs via :func:`colliderml.load`.

    Attributes:
        channel: Physics channel that was simulated.
        events: Requested number of events.
        pileup: Pileup level.
        output_dir: Host directory passed as ``--output`` (one level above
            ``runs/<run_id>/``).
        run_dir: The per-run output directory containing the HepMC, EDM4hep,
            and parquet outputs (e.g. ``<output_dir>/runs/0/``).
        stages: Per-stage execution results.
        remote_request_id: Set only when ``remote=True`` — the backend's
            request identifier for later polling.
        particle_flow: Whether the run included particle flow and the final
            parquet tables.
    """

    channel: str
    events: int
    pileup: int
    output_dir: Path
    run_dir: Path
    stages: List[StageRun] = field(default_factory=list)
    remote_request_id: Optional[str] = None
    particle_flow: bool = False

    @property
    def parquet_dir(self) -> Path:
        """Directory of the final tables of a particle-flow run.

        Laid out as ``<parquet_dir>/{truth,reco}/<table>/*.parquet``. Runs
        without particle flow do not create it; their per-run tables are
        under :attr:`run_dir`.
        """
        return self.output_dir / "parquet"

    def list_files(self) -> List[Path]:
        """List every file produced under :attr:`run_dir`, sorted."""
        if not self.run_dir.exists():
            return []
        return sorted(p for p in self.run_dir.rglob("*") if p.is_file())

    def tables(self) -> Dict[str, List[Path]]:
        """Final parquet files of a particle-flow run, by table name.

        Returns:
            Mapping such as ``{"pfos": [...], "tracks": [...]}``; empty if
            :attr:`parquet_dir` does not exist.
        """
        found: Dict[str, List[Path]] = {}
        for path in sorted(self.parquet_dir.glob("*/*/*.parquet")):
            found.setdefault(path.parent.name, []).append(path)
        return found

    def __repr__(self) -> str:  # pragma: no cover - trivial
        n_files = len(self.list_files())
        return (
            f"SimulationResult(channel={self.channel!r}, events={self.events}, "
            f"pileup={self.pileup}, files={n_files}, run_dir={str(self.run_dir)!r})"
        )


def _apply_runtime_overrides(
    manifest: list,
    *,
    prod_root: Path,
    run_id: str,
    events: int,
    pileup: int,
    seed: int,
    threads: Optional[int] = None,
    pandora_processes: Optional[int] = None,
) -> list:
    """Write per-run YAML overlays so the caller's kwargs actually take effect.

    Each stage in the manifest points at a static YAML config under
    ``prod_root/configs_development/docker_test/<channel>/`` with hardcoded
    defaults (e.g. ``events: 10``). This helper:

      1. Reads each stage's original YAML.
      2. Overrides the keys below with the caller's values, only where the
         key already exists in the YAML (we do not add fields a stage script
         does not know about) and the value is not ``None``.
      3. Writes the modified YAML to
         ``prod_root/.overlays/<run_id>/<stage>__<original_basename>``.
      4. Returns a manifest with each entry's ``config_path`` rewritten
         to point at the overlay.

    Keys: ``events``, ``pileup``, ``seed``, ``threads``, ``processes`` (the
    Pandora stage's process count) and, for the packaging stage, ``run_size``
    and ``chunk_size``, both set to ``events`` because the one run holds all
    events and they go into one file per table.

    The overlay path stays under ``prod_root`` so the existing
    ``--config /workspace/<config_path>`` in ``run_stage`` keeps working
    unchanged. ``run_id`` namespacing lets concurrent runs coexist.
    """
    import yaml

    overlay_root = prod_root / ".overlays" / run_id
    overlay_root.mkdir(parents=True, exist_ok=True)

    overrides = {
        "events": events,
        "pileup": pileup,
        "seed": seed,
        "threads": threads,
        "processes": pandora_processes,
        "run_size": events,
        "chunk_size": events,
    }

    new_manifest = []
    for stage in manifest:
        orig_rel = stage["config_path"]
        orig_path = prod_root / orig_rel
        with open(orig_path) as f:
            cfg = yaml.safe_load(f) or {}

        for key, value in overrides.items():
            if key in cfg and value is not None:
                cfg[key] = value

        orig_basename = Path(orig_rel).name
        overlay_rel = Path(".overlays") / run_id / f"{stage['stage']}__{orig_basename}"
        overlay_path = prod_root / overlay_rel
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        with open(overlay_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        new_stage = dict(stage)
        new_stage["config_path"] = str(overlay_rel)
        new_manifest.append(new_stage)

    return new_manifest


def _default_output_dir(channel: str, pileup: int, events: int) -> Path:
    """Compute a sensible default output directory if the caller didn't give one."""
    return (Path.cwd() / "colliderml_output" / f"{channel}_pu{pileup}_{events}evt").resolve()


def _coerce_preset(
    preset: Optional[Union[str, Preset]],
    *,
    channel: Optional[str],
    events: Optional[int],
    pileup: Optional[int],
    particle_flow: Optional[bool] = None,
) -> tuple[str, int, int, bool]:
    """Merge a preset (if given) with explicit arguments.

    Explicit arguments win over preset values — this lets users take a
    preset as a starting point and bump, say, the pileup without writing
    their own preset.

    Returns:
        ``(channel, events, pileup, particle_flow)``.
    """
    if preset is not None:
        resolved = resolve_preset(preset) if isinstance(preset, str) else preset
        final_channel = channel or resolved.channel
        final_events = events if events is not None else resolved.events
        final_pileup = pileup if pileup is not None else resolved.pileup
        final_particle_flow = resolved.particle_flow if particle_flow is None else particle_flow
    else:
        if channel is None:
            raise ValueError(
                "simulate() requires either a preset or an explicit `channel`. "
                "Try colliderml.simulate(preset='ttbar-quick')."
            )
        if events is None:
            raise ValueError(
                "simulate() requires `events` when no preset is given. "
                "For example: colliderml.simulate(channel='ttbar', events=10)."
            )
        final_channel = channel
        final_events = int(events)
        final_pileup = int(pileup) if pileup is not None else 0
        final_particle_flow = bool(particle_flow)

    # Channel sanity: catches typos at call time instead of container runtime.
    get_channel_stages(final_channel)
    if int(final_events) < 1:
        raise ValueError(f"`events` must be at least 1, got {final_events}.")
    if int(final_pileup) < 0:
        raise ValueError(f"`pileup` must not be negative, got {final_pileup}.")
    return final_channel, int(final_events), int(final_pileup), bool(final_particle_flow)


def _simulate_remote(
    *, channel: str, events: int, pileup: int, seed: int
) -> SimulationResult:
    """Delegate to the remote SaaS client (see :mod:`colliderml.remote`).

    The heavy-weight import is deferred so users who only do local sim do
    not pay for ``requests`` at import time.
    """
    try:
        from colliderml.remote import submit  # noqa: WPS433 (deferred import by design)
    except ImportError as exc:  # pragma: no cover - exercised in commit B2
        raise ImportError(
            "remote=True requires the remote extra: pip install 'colliderml[remote]'"
        ) from exc

    submission = submit(channel=channel, events=events, pileup=pileup, seed=seed)
    return SimulationResult(
        channel=channel,
        events=events,
        pileup=pileup,
        output_dir=Path.cwd() / "colliderml_output",
        run_dir=Path.cwd() / "colliderml_output" / "remote" / str(submission.request_id),
        stages=[],
        remote_request_id=str(submission.request_id),
    )


def simulate(
    *,
    preset: Optional[Union[str, Preset]] = None,
    channel: Optional[str] = None,
    events: Optional[int] = None,
    pileup: Optional[int] = None,
    seed: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
    image: Optional[str] = None,
    reco_image: Optional[str] = None,
    particle_flow: Optional[bool] = None,
    threads: Optional[int] = None,
    pandora_processes: Optional[int] = None,
    remote: bool = False,
    run_id: str = "0",
    prod_root: Optional[Union[str, Path]] = None,
    runtime: Optional[str] = None,
    quiet: bool = False,
) -> SimulationResult:
    """Run the ColliderML simulation pipeline.

    The pipeline is: event generation (Pythia, or MadGraph + Pythia for
    NLO channels) → detector simulation (Geant4 via DDSim) → tracker
    digitization, ACTS tracking and vertexing. With ``particle_flow=True``
    it continues with calorimeter digitization and Pandora particle flow,
    and writes the final parquet tables (tracks, tracker_hits, calo_cells,
    calo_clusters, pfos, vertices, particles, generator_particles) to
    ``<output_dir>/parquet/``. Every stage runs inside a container.

    Args:
        preset: Named preset (e.g. ``"ttbar-pflow-quick"``). Sets
            ``channel``, ``events``, ``pileup`` and ``particle_flow`` if
            they are not given explicitly. See
            :func:`~colliderml.simulate.load_presets`.
        channel: Physics channel (e.g. ``"ttbar"``, ``"higgs_portal"``).
        events: Number of events to simulate.
        pileup: Pileup level (``0`` for no pileup, ``200`` for HL-LHC).
        seed: Random seed forwarded to each stage.
        output_dir: Host directory for outputs. Defaults to
            ``./colliderml_output/<channel>_pu<pileup>_<events>evt/``.
        image: Sim image reference (generation, Geant4, tracking,
            packaging). Defaults to the pinned image, see
            :func:`~colliderml.simulate.load_pipeline_config`.
        reco_image: Reco image reference (calorimeter digitization and
            Pandora). Defaults to the pinned image. Only used with
            ``particle_flow=True``.
        particle_flow: Run the stages after tracking. Defaults to the
            preset's setting, or ``False`` without a preset.
        threads: Threads for Geant4 and ACTS. Defaults to the stage
            configs' value (1).
        pandora_processes: Concurrent Pandora processes. Defaults to the
            stage config's value (1). One process needs about 2 GB without
            pileup and about 11 GB per event at pileup 200.
        remote: If ``True``, submit the job to the SaaS backend instead of
            running locally. Requires ``pip install 'colliderml[remote]'``.
            The backend does not run particle flow yet.
        run_id: Subdirectory under ``output_dir/runs/`` for this run.
        prod_root: Host path to an existing :mod:`colliderml-production`
            checkout. Leave ``None`` to auto-clone on first use.
        runtime: Override container runtime (``"docker"``, ``"podman"`` or
            ``"podman-hpc"``). Auto-detected if ``None``.
        quiet: Suppress per-stage progress messages.

    Returns:
        :class:`SimulationResult` with paths to the outputs and per-stage
        results.

    Raises:
        ValueError: If neither a preset nor explicit channel/events are
            supplied, if the requested channel is unknown, or if
            ``remote=True`` is combined with particle flow.
        ContainerRuntimeError: If ``remote=False`` and no container runtime
            is available locally.
        RuntimeError: If any pipeline stage exits non-zero.
    """
    resolved_channel, resolved_events, resolved_pileup, use_particle_flow = _coerce_preset(
        preset, channel=channel, events=events, pileup=pileup, particle_flow=particle_flow
    )

    if remote:
        if use_particle_flow:
            raise ValueError(
                "Particle flow is not available with remote=True yet. Run it locally, "
                "or pass particle_flow=False to stop after tracking."
            )
        return _simulate_remote(
            channel=resolved_channel,
            events=resolved_events,
            pileup=resolved_pileup,
            seed=seed,
        )

    # Local container path.
    pinned = load_pipeline_config()
    sim_image = image or pinned.sim_image
    pflow_image = reco_image or pinned.reco_image

    rt = check_runtime_available(runtime)
    needed_images = [sim_image, pflow_image] if use_particle_flow else [sim_image]
    for needed in needed_images:
        if not pull_image(needed, runtime=rt):
            raise RuntimeError(f"Container image {needed} not available. Cannot proceed.")

    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _default_output_dir(resolved_channel, resolved_pileup, resolved_events)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prod = (
        Path(prod_root).expanduser().resolve()
        if prod_root is not None
        else clone_colliderml_production()
    )

    manifest = generate_stage_manifest(
        resolved_channel, prod_root=prod, particle_flow=use_particle_flow
    )
    if not manifest:
        raise RuntimeError(
            f"No stages were generated for channel '{resolved_channel}'. "
            f"Is the colliderml-production clone at {prod} up to date?"
        )

    # Apply the caller's kwargs by writing per-run overlay YAMLs under
    # prod_root/.overlays/<run_id>/. Each stage script reads its config
    # from disk, so without the overlays the static configs would win.
    manifest = _apply_runtime_overrides(
        manifest,
        prod_root=prod,
        run_id=run_id,
        events=resolved_events,
        pileup=resolved_pileup,
        seed=seed,
        threads=threads,
        pandora_processes=pandora_processes,
    )

    def _on_start(i: int, total: int, name: str) -> None:
        if not quiet:
            print(f"  [{i + 1}/{total}] {name} ...", flush=True)

    def _on_end(i: int, total: int, name: str, rc: int) -> None:
        if not quiet:
            status = "done" if rc == 0 else f"FAILED (exit {rc})"
            print(f"  [{i + 1}/{total}] {name} — {status}", flush=True)

    if not quiet:
        print(
            f"Running {resolved_channel} pipeline "
            f"({resolved_events} events, pileup={resolved_pileup}"
            f"{', particle flow' if use_particle_flow else ''}) ..."
        )

    stage_results = run_pipeline(
        channel=resolved_channel,
        output_dir=out_dir,
        stage_manifest=manifest,
        seed=seed,
        run_id=run_id,
        image=sim_image,
        reco_image=pflow_image,
        prod_root=prod,
        runtime=rt,
        on_stage_start=_on_start,
        on_stage_end=_on_end,
    )

    run_dir = out_dir / "runs" / run_id
    result = SimulationResult(
        channel=resolved_channel,
        events=resolved_events,
        pileup=resolved_pileup,
        output_dir=out_dir,
        run_dir=run_dir,
        stages=list(stage_results),
        particle_flow=use_particle_flow,
    )
    if not quiet:
        print(f"Done. Output: {run_dir}")
        if run_dir.exists():
            for fp in sorted(run_dir.glob("*")):
                if fp.is_file():
                    size_mb = fp.stat().st_size / (1024 * 1024)
                    print(f"  {fp.name} ({size_mb:.1f} MB)")
        if use_particle_flow:
            print(f"Tables: {result.parquet_dir}")
            for name in result.tables():
                print(f"  {name}")

    return result


__all__ = ["simulate", "SimulationResult"]
