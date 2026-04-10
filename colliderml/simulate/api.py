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
from typing import List, Optional, Union

from colliderml.simulate.docker import (
    DEFAULT_IMAGE,
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
    """

    channel: str
    events: int
    pileup: int
    output_dir: Path
    run_dir: Path
    stages: List[StageRun] = field(default_factory=list)
    remote_request_id: Optional[str] = None

    def list_files(self) -> List[Path]:
        """List every file produced under :attr:`run_dir`, sorted."""
        if not self.run_dir.exists():
            return []
        return sorted(p for p in self.run_dir.rglob("*") if p.is_file())

    def __repr__(self) -> str:  # pragma: no cover - trivial
        n_files = len(self.list_files())
        return (
            f"SimulationResult(channel={self.channel!r}, events={self.events}, "
            f"pileup={self.pileup}, files={n_files}, run_dir={str(self.run_dir)!r})"
        )


def _default_output_dir(channel: str, pileup: int, events: int) -> Path:
    """Compute a sensible default output directory if the caller didn't give one."""
    return (Path.cwd() / "colliderml_output" / f"{channel}_pu{pileup}_{events}evt").resolve()


def _coerce_preset(
    preset: Optional[Union[str, Preset]],
    *,
    channel: Optional[str],
    events: Optional[int],
    pileup: Optional[int],
) -> tuple[str, int, int]:
    """Merge a preset (if given) with explicit arguments.

    Explicit arguments win over preset values — this lets users take a
    preset as a starting point and bump, say, the pileup without writing
    their own preset.
    """
    if preset is not None:
        resolved = resolve_preset(preset) if isinstance(preset, str) else preset
        final_channel = channel or resolved.channel
        final_events = events if events is not None else resolved.events
        final_pileup = pileup if pileup is not None else resolved.pileup
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

    # Channel sanity: catches typos at call time instead of container runtime.
    get_channel_stages(final_channel)
    return final_channel, int(final_events), int(final_pileup)


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
    image: str = DEFAULT_IMAGE,
    remote: bool = False,
    run_id: str = "0",
    prod_root: Optional[Union[str, Path]] = None,
    runtime: Optional[str] = None,
    quiet: bool = False,
) -> SimulationResult:
    """Run the ColliderML simulation pipeline.

    The pipeline is: event generation (Pythia, or MadGraph + Pythia for
    NLO channels) → detector simulation (Geant4 via DDSim) → digitization
    and reconstruction (ACTS) → parquet conversion. Every stage runs inside
    the OpenDataDetector software container.

    Args:
        preset: Named preset (e.g. ``"ttbar-quick"``). Sets ``channel``,
            ``events`` and ``pileup`` if they are not given explicitly. See
            :func:`~colliderml.simulate.load_presets`.
        channel: Physics channel (e.g. ``"ttbar"``, ``"higgs_portal"``).
        events: Number of events to simulate.
        pileup: Pileup level (``0`` for no pileup, ``200`` for HL-LHC).
        seed: Random seed forwarded to each stage.
        output_dir: Host directory for outputs. Defaults to
            ``./colliderml_output/<channel>_pu<pileup>_<events>evt/``.
        image: Container image reference. Defaults to the pinned OpenData
            Detector image.
        remote: If ``True``, submit the job to the SaaS backend instead of
            running locally. Requires ``pip install 'colliderml[remote]'``.
        run_id: Subdirectory under ``output_dir/runs/`` for this run.
        prod_root: Host path to an existing :mod:`colliderml-production`
            checkout. Leave ``None`` to auto-clone on first use.
        runtime: Override container runtime (``"docker"`` or ``"podman"``).
            Auto-detected if ``None``.
        quiet: Suppress per-stage progress messages.

    Returns:
        :class:`SimulationResult` with paths to the outputs and per-stage
        results.

    Raises:
        ValueError: If neither a preset nor explicit channel/events are
            supplied, or if the requested channel is unknown.
        ContainerRuntimeError: If ``remote=False`` and no container runtime
            is available locally.
        RuntimeError: If any pipeline stage exits non-zero.
    """
    resolved_channel, resolved_events, resolved_pileup = _coerce_preset(
        preset, channel=channel, events=events, pileup=pileup
    )

    if remote:
        return _simulate_remote(
            channel=resolved_channel,
            events=resolved_events,
            pileup=resolved_pileup,
            seed=seed,
        )

    # Local container path.
    rt = check_runtime_available(runtime)
    if not pull_image(image, runtime=rt):
        raise RuntimeError("Container image not available. Cannot proceed.")

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

    manifest = generate_stage_manifest(resolved_channel, prod_root=prod)
    if not manifest:
        raise RuntimeError(
            f"No stages were generated for channel '{resolved_channel}'. "
            f"Is the colliderml-production clone at {prod} up to date?"
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
            f"({resolved_events} events, pileup={resolved_pileup}) ..."
        )

    stage_results = run_pipeline(
        channel=resolved_channel,
        output_dir=out_dir,
        stage_manifest=manifest,
        seed=seed,
        run_id=run_id,
        image=image,
        prod_root=prod,
        runtime=rt,
        on_stage_start=_on_start,
        on_stage_end=_on_end,
    )

    run_dir = out_dir / "runs" / run_id
    if not quiet:
        print(f"Done. Output: {run_dir}")
        if run_dir.exists():
            for fp in sorted(run_dir.glob("*")):
                if fp.is_file():
                    size_mb = fp.stat().st_size / (1024 * 1024)
                    print(f"  {fp.name} ({size_mb:.1f} MB)")

    return SimulationResult(
        channel=resolved_channel,
        events=resolved_events,
        pileup=resolved_pileup,
        output_dir=out_dir,
        run_dir=run_dir,
        stages=list(stage_results),
    )


__all__ = ["simulate", "SimulationResult"]
