"""Container runtime management for local simulation.

This module wraps ``docker``, ``podman`` or ``podman-hpc`` (whichever is
installed) and runs each pipeline stage inside a container. Two images are
involved: the *sim* image (generation, Geant4, ACTS tracking, packaging) and the
*reco* image (calorimeter digitisation and Pandora particle flow).

The pipeline scripts, stage configs and the in-container environment setup all
live in the companion :mod:`colliderml-production` repository, not in the
public library. :func:`clone_colliderml_production` fetches that repo at the
pinned git ref into a user-visible cache directory the first time it is
needed; subsequent runs reuse the clone. The clone is then mounted inside the
container as ``/workspace``.

The pinned ref and the image names come from
:mod:`colliderml.simulate.config` (:file:`pipeline_config.yaml`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from colliderml.simulate.config import ENV_OVERRIDES, load_pipeline_config

_PINNED = load_pipeline_config(environ={})

#: Default sim image (generation, Geant4, ACTS tracking, packaging); override
#: with ``image=...`` on :func:`simulate` or the ``--image`` CLI flag.
DEFAULT_IMAGE = _PINNED.sim_image

#: Default reco image (calorimeter digitisation and Pandora); override with
#: ``reco_image=...`` or ``--reco-image``.
DEFAULT_RECO_IMAGE = _PINNED.reco_image

#: Git repository that houses the simulation pipeline scripts and configs.
COLLIDERML_PRODUCTION_REPO = _PINNED.production_repo

#: Pinned ref of that repository, bumped together with the images in
#: :file:`pipeline_config.yaml` when a new pipeline release is cut.
COLLIDERML_PRODUCTION_REF = _PINNED.production_ref

#: Environment variable override for the pinned ref (useful for testing
#: against an in-flight branch of the production repo).
COLLIDERML_PRODUCTION_REF_ENV = ENV_OVERRIDES["production_ref"]

#: Runtimes probed by :func:`get_container_runtime`, in order of preference.
SUPPORTED_RUNTIMES = ("docker", "podman", "podman-hpc")

#: Script in the production repo that sets up a stage's environment in the
#: container; takes the stage's image kind (``sim`` or ``reco``).
STAGE_ENV_SCRIPT = "scripts/cli/docker_stage_env.sh"

#: Script in the production repo that records wall time and peak memory of a
#: stage in ``<output>/pipeline_timing.csv``.
STAGE_TIMER_SCRIPT = "scripts/cli/stage_timer.py"


class ContainerRuntimeError(RuntimeError):
    """Raised when neither docker nor podman is available/usable."""


@dataclass(frozen=True)
class StageRun:
    """Result of a single pipeline stage invocation.

    Attributes:
        name: Human-readable stage name (for reporting).
        stage: Machine identifier.
        returncode: Exit code from the container command.
    """

    name: str
    stage: str
    returncode: int


def default_cache_root() -> Path:
    """Return the default cache directory for simulation artefacts.

    Honors ``$COLLIDERML_CACHE`` if set, otherwise falls back to
    ``~/.cache/colliderml/simulate``.
    """
    env = os.environ.get("COLLIDERML_CACHE")
    if env:
        return Path(env).expanduser().resolve() / "simulate"
    return (Path.home() / ".cache" / "colliderml" / "simulate").resolve()


def get_container_runtime() -> str:
    """Return the container runtime binary to use.

    Takes the first of :data:`SUPPORTED_RUNTIMES` on ``$PATH``: Docker,
    then Podman, then ``podman-hpc`` (the Podman wrapper on HPC systems
    such as NERSC Perlmutter).

    Returns:
        ``"docker"``, ``"podman"`` or ``"podman-hpc"``.

    Raises:
        ContainerRuntimeError: If none is installed.
    """
    for runtime in SUPPORTED_RUNTIMES:
        if shutil.which(runtime):
            return runtime
    raise ContainerRuntimeError(
        "No container runtime on PATH (looked for docker, podman, podman-hpc).\n"
        "  - Docker: https://docs.docker.com/get-docker/\n"
        "  - Podman: https://podman.io/getting-started/installation\n"
        "Or use remote simulation: colliderml.simulate(..., remote=True)"
    )


def check_runtime_available(runtime: Optional[str] = None) -> str:
    """Confirm the runtime daemon is actually responsive.

    Args:
        runtime: Runtime binary to probe. Auto-detected if ``None``.

    Returns:
        The runtime name.

    Raises:
        ContainerRuntimeError: If the runtime is installed but the daemon
            is not running or is unreachable.
    """
    rt = runtime or get_container_runtime()
    try:
        result = subprocess.run(
            [rt, "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContainerRuntimeError(f"{rt} daemon timed out after 10s.") from exc
    if result.returncode != 0:
        raise ContainerRuntimeError(
            f"{rt} is installed but not responding.\n"
            f"stderr: {result.stderr.strip()}"
        )
    return rt


def check_image_available(image: str = DEFAULT_IMAGE, *, runtime: Optional[str] = None) -> bool:
    """Return ``True`` iff the container image is already present locally."""
    rt = runtime or get_container_runtime()
    # `image exists` is the Podman query; it also sees podman-hpc's migrated
    # images, which `image inspect` misses on compute nodes.
    query = "inspect" if Path(rt).name == "docker" else "exists"
    result = subprocess.run(
        [rt, "image", query, image],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def pull_image(
    image: str = DEFAULT_IMAGE,
    *,
    interactive: bool = True,
    runtime: Optional[str] = None,
) -> bool:
    """Ensure the container image is available locally, pulling if needed.

    The images are large (about 6 GB for the sim image, 27 GB for the reco
    image), so we ask before starting a fresh pull. Pass
    ``interactive=False`` for CI or scripted callers.

    Args:
        image: Image reference (``<registry>/<repo>:<tag>``).
        interactive: If ``True`` and ``stdin`` is a tty, prompt before
            downloading.
        runtime: Runtime to use; auto-detected if ``None``.

    Returns:
        ``True`` if the image is now available, ``False`` if the user
        declined the interactive prompt.
    """
    rt = runtime or get_container_runtime()
    if check_image_available(image, runtime=rt):
        return True
    if interactive and sys.stdin.isatty():
        print(f"\nContainer image {image} is not present locally.")
        print("This is a large download (about 6 GB for the sim image, 27 GB for the reco image).")
        try:
            response = input("Continue? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return False
        if response in ("n", "no"):
            print("Aborted. Use remote=True to run on the backend instead.")
            return False
    print(f"Pulling {image} via {rt}...")
    result = subprocess.run([rt, "pull", image], timeout=3600)
    return result.returncode == 0


def _resolve_production_ref() -> str:
    """Return the effective colliderml-production git ref to clone."""
    return os.environ.get(COLLIDERML_PRODUCTION_REF_ENV) or COLLIDERML_PRODUCTION_REF


def clone_colliderml_production(
    cache_root: Optional[Path] = None,
    *,
    ref: Optional[str] = None,
    force_refresh: bool = False,
) -> Path:
    """Clone or refresh the :mod:`colliderml-production` repository into the cache.

    This is the mechanism that makes the public library a thin orchestrator:
    all pipeline scripts, stage YAML templates, and the container bootstrap
    script live in the production repo, and we fetch them on demand.

    The repo is cloned into ``<cache_root>/colliderml-production/``. If the
    clone already exists we fetch and fast-forward to the requested ref
    (so bumping :data:`COLLIDERML_PRODUCTION_REF` picks up automatically).

    Args:
        cache_root: Cache directory (default: :func:`default_cache_root`).
        ref: Git ref to check out. Defaults to
            :data:`COLLIDERML_PRODUCTION_REF` (or the
            ``COLLIDERML_PRODUCTION_REF`` env var override).
        force_refresh: If ``True``, delete any existing clone and re-clone
            from scratch. Use when the cache is suspected to be corrupt.

    Returns:
        Path to the cloned production repo, ready to mount as ``/workspace``.
    """
    root = (cache_root or default_cache_root()).expanduser().resolve()
    clone_dir = root / "colliderml-production"
    target_ref = ref or _resolve_production_ref()

    if force_refresh and clone_dir.exists():
        shutil.rmtree(clone_dir)

    if clone_dir.exists() and (clone_dir / ".git").exists():
        # Fetch + checkout to the desired ref.
        subprocess.run(
            ["git", "-C", str(clone_dir), "fetch", "--quiet", "origin", target_ref],
            check=False,
            timeout=300,
        )
        checkout = subprocess.run(
            ["git", "-C", str(clone_dir), "checkout", "--quiet", "FETCH_HEAD"],
            check=False,
            timeout=60,
        )
        if checkout.returncode != 0:
            # Fallback: try as a branch/tag name.
            subprocess.run(
                ["git", "-C", str(clone_dir), "checkout", "--quiet", target_ref],
                check=False,
                timeout=60,
            )
        return clone_dir

    # Fresh clone.
    root.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {COLLIDERML_PRODUCTION_REPO} @ {target_ref} into {clone_dir} ...")
    result = subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            target_ref,
            COLLIDERML_PRODUCTION_REPO,
            str(clone_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        # Retry without --branch (works for commit SHAs that --branch rejects).
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        result = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                COLLIDERML_PRODUCTION_REPO,
                str(clone_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise ContainerRuntimeError(
                f"Failed to clone {COLLIDERML_PRODUCTION_REPO}: {result.stderr.strip()}"
            )
        subprocess.run(
            ["git", "-C", str(clone_dir), "checkout", "--quiet", target_ref],
            check=True,
            timeout=60,
        )
    return clone_dir


def _cache_dir(prod_root: Path) -> Path:
    """Return (creating it if needed) the cache directory mounted as ``/cache``.

    It lives at ``<prod_root>/.cache`` and holds what the in-container setup
    downloads on first use: the Geant4 datasets (about 2 GB) and a few Python
    packages. Everything else the stages need is in the images.
    """
    cache_dir = prod_root / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def run_stage(
    *,
    prod_root: Path,
    output_dir: Path,
    cache_dir: Path,
    script: str,
    config_path: str,
    stage: Optional[str] = None,
    image_kind: str = "sim",
    seed: int = 42,
    run_id: str = "0",
    image: str = DEFAULT_IMAGE,
    runtime: Optional[str] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a single pipeline stage inside the container.

    Args:
        prod_root: Host path to the cloned production repo (mounted as
            ``/workspace`` inside the container).
        output_dir: Host directory for outputs (mounted as ``/output``).
        cache_dir: Host directory for external caches (mounted as
            ``/cache``).
        script: Stage script, relative to ``scripts/`` in the production
            repo (e.g. ``"simulation/pythia_gen.py"``).
        config_path: Path to the stage's YAML config, relative to
            ``prod_root``.
        stage: Stage identifier written to ``pipeline_timing.csv``.
            Defaults to the script's file name.
        image_kind: ``"sim"`` or ``"reco"``: which environment the
            production repo's setup script prepares. Must match ``image``.
        seed: Random seed forwarded to the stage script.
        run_id: Run subdirectory under ``output_dir/runs/``.
        image: Container image reference.
        runtime: Runtime to use; auto-detected if ``None``.
        timeout: Seconds after which the stage is abandoned. ``None``
            (the default) waits indefinitely: a pile-up 200 Geant4 stage
            can take hours.

    Returns:
        :class:`subprocess.CompletedProcess` for the container run.

    Raises:
        ContainerRuntimeError: If the production checkout predates the
            two-image pipeline (no stage environment script).
    """
    rt = runtime or get_container_runtime()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (prod_root / STAGE_ENV_SCRIPT).is_file():
        raise ContainerRuntimeError(
            f"{prod_root} has no {STAGE_ENV_SCRIPT}: this colliderml-production "
            f"checkout is older than {COLLIDERML_PRODUCTION_REF}, which this version "
            "of colliderml needs. Delete the checkout to let simulate() re-clone it."
        )

    script_dir = os.path.dirname(script)
    script_name = os.path.basename(script)
    stage_label = stage or Path(script_name).stem

    container_cmd = (
        f"source /workspace/{STAGE_ENV_SCRIPT} {image_kind} && "
        f"cd /workspace/scripts/{script_dir} && "
        f"python3 /workspace/{STAGE_TIMER_SCRIPT} "
        f"--record /output/pipeline_timing.csv --stage {stage_label} -- "
        f"python3 {script_name} "
        f"--config /workspace/{config_path} "
        f"--output /output/runs "
        f"--output-subdir {run_id} "
        f"--seed {seed}"
    )

    cmd: List[str] = [
        rt,
        "run",
        "--rm",
        "-v",
        f"{prod_root}:/workspace",
        "-v",
        f"{output_dir}:/output",
        "-v",
        f"{cache_dir}:/cache",
        "-e",
        "COLLIDERML_CACHE=/cache",
        "--entrypoint",
        "/bin/bash",
        image,
        "-c",
        container_cmd,
    ]
    return subprocess.run(cmd, timeout=timeout)


def _copy_madgraph_output(output_dir: Path, run_id: str) -> None:
    """Bridge MadGraph output to where the Pythia stage expects it.

    MadGraph writes ``runs/all/<run_id>/events.hepmc.gz`` but the downstream
    Pythia stage reads ``runs/<run_id>/events.hepmc.gz``. This keeps that
    stitch out of the stage scripts themselves.
    """
    src = output_dir / "runs" / "all" / run_id / "events.hepmc.gz"
    dst_dir = output_dir / "runs" / run_id
    dst = dst_dir / "events.hepmc.gz"
    if src.exists() and not dst.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))


def run_pipeline(
    *,
    channel: str,
    output_dir: Path,
    stage_manifest: List[dict],
    seed: int = 42,
    run_id: str = "0",
    image: str = DEFAULT_IMAGE,
    reco_image: str = DEFAULT_RECO_IMAGE,
    prod_root: Optional[Path] = None,
    runtime: Optional[str] = None,
    on_stage_start: Optional[Callable[[int, int, str], None]] = None,
    on_stage_end: Optional[Callable[[int, int, str, int], None]] = None,
) -> List[StageRun]:
    """Run every stage in a manifest, stopping on the first non-zero exit.

    Args:
        channel: Physics channel (used for post-stage hooks like the
            MadGraph → Pythia bridge).
        output_dir: Host directory for outputs (mounted as ``/output``).
        stage_manifest: List of dicts produced by
            :func:`colliderml.simulate.pipeline.generate_stage_manifest`.
        seed: Random seed forwarded to each stage.
        run_id: Run subdirectory under ``output_dir/runs/``.
        image: Sim image reference.
        reco_image: Reco image reference, used by the stages whose manifest
            entry has ``image == "reco"``.
        prod_root: Host path to the production-repo clone. If ``None``,
            :func:`clone_colliderml_production` is called.
        runtime: Runtime to use; auto-detected if ``None``.
        on_stage_start: Optional callback ``(index, total, stage_name)``
            invoked before each stage.
        on_stage_end: Optional callback ``(index, total, stage_name,
            returncode)`` invoked after each stage.

    Returns:
        List of :class:`StageRun` results, one per executed stage.

    Raises:
        RuntimeError: If any stage exits non-zero.
    """
    rt = runtime or check_runtime_available()
    if prod_root is None:
        prod_root = clone_colliderml_production()
    cache_dir = _cache_dir(prod_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(stage_manifest)
    results: List[StageRun] = []

    for i, stage in enumerate(stage_manifest):
        if on_stage_start is not None:
            on_stage_start(i, total, stage["name"])

        image_kind = stage.get("image", "sim")
        completed = run_stage(
            prod_root=prod_root,
            output_dir=output_dir,
            cache_dir=cache_dir,
            script=stage["script"],
            config_path=stage["config_path"],
            stage=stage["stage"],
            image_kind=image_kind,
            seed=seed,
            run_id=run_id,
            image=reco_image if image_kind == "reco" else image,
            runtime=rt,
        )
        stage_run = StageRun(
            name=stage["name"], stage=stage["stage"], returncode=completed.returncode
        )
        results.append(stage_run)

        if on_stage_end is not None:
            on_stage_end(i, total, stage["name"], completed.returncode)

        if completed.returncode != 0:
            raise RuntimeError(
                f"Stage '{stage['name']}' failed with exit code {completed.returncode}. "
                f"Check the container output above for details."
            )

        if stage["stage"] == "madgraph_generation":
            _copy_madgraph_output(output_dir, run_id)

    return results
