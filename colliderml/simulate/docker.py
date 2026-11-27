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

#: Pinned sim image (generation, Geant4, ACTS tracking, packaging), as shipped
#: in the package. The functions below resolve their default at call time, so
#: ``COLLIDERML_SIM_IMAGE`` applies; override per call with ``image=...``.
DEFAULT_IMAGE = _PINNED.sim_image

#: Pinned reco image (calorimeter digitisation and Pandora), as shipped;
#: ``COLLIDERML_RECO_IMAGE`` or ``reco_image=...`` override it.
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
#: ``podman-hpc`` comes before ``podman``: where the wrapper is installed,
#: plain ``podman`` is on ``$PATH`` too but does not see the migrated images.
SUPPORTED_RUNTIMES = ("docker", "podman-hpc", "podman")

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
    then ``podman-hpc`` (the Podman wrapper on HPC systems such as NERSC
    Perlmutter), then Podman.

    Returns:
        ``"docker"``, ``"podman"`` or ``"podman-hpc"``.

    Raises:
        ContainerRuntimeError: If none is installed.
    """
    for runtime in SUPPORTED_RUNTIMES:
        if shutil.which(runtime):
            return runtime
    raise ContainerRuntimeError(
        "No container runtime on PATH (looked for docker, podman-hpc, podman).\n"
        "  - Docker: https://docs.docker.com/get-docker/\n"
        "  - Podman: https://podman.io/getting-started/installation\n"
        "Runs without particle flow can also go to the backend: simulate(..., remote=True)"
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


def check_image_available(image: Optional[str] = None, *, runtime: Optional[str] = None) -> bool:
    """Return ``True`` iff the container image is already present locally.

    ``image`` defaults to the pinned sim image."""
    image = image or load_pipeline_config().sim_image
    rt = runtime or get_container_runtime()
    name = Path(rt).name
    if name == "podman-hpc":
        # Images migrated to the shared read-only store run and are listed, but
        # `image exists` and `image inspect` do not find them.
        listing = subprocess.run(
            [rt, "images", "--format", "{{.Repository}}:{{.Tag}} {{.Repository}}@{{.Digest}}"],
            capture_output=True,
            text=True,
        )
        if listing.returncode == 0 and image in listing.stdout.split():
            return True
    # `image exists` is the Podman query, `image inspect` the Docker one.
    query = "inspect" if name == "docker" else "exists"
    result = subprocess.run(
        [rt, "image", query, image],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def pull_image(
    image: Optional[str] = None,
    *,
    interactive: bool = True,
    runtime: Optional[str] = None,
) -> bool:
    """Ensure the container image is available locally, pulling if needed.

    The images are large (about 6 GB for the sim image and 27 GB for the
    reco image once unpacked), so we ask before starting a fresh pull. Pass
    ``interactive=False`` for CI or scripted callers.

    Args:
        image: Image reference (``<registry>/<repo>:<tag>``). Defaults to
            the pinned sim image.
        interactive: If ``True`` and ``stdin`` is a tty, prompt before
            downloading.
        runtime: Runtime to use; auto-detected if ``None``.

    Returns:
        ``True`` if the image is now available, ``False`` if the user
        declined the interactive prompt.
    """
    image = image or load_pipeline_config().sim_image
    rt = runtime or get_container_runtime()
    if check_image_available(image, runtime=rt):
        return True
    if interactive and sys.stdin.isatty():
        print(f"\nContainer image {image} is not present locally.")
        print("This is a large download (unpacked: about 6 GB for the sim image, 27 GB for the reco image).")
        try:
            response = input("Continue? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return False
        if response in ("n", "no"):
            print("Aborted.")
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
        # Fetch the ref and check it out. Without network the fetch fails; a ref
        # that is already in the clone (a tag from an earlier run) still works.
        fetched = subprocess.run(
            ["git", "-C", str(clone_dir), "fetch", "--quiet", "origin", target_ref],
            capture_output=True,
            timeout=300,
        )
        for candidate in (["FETCH_HEAD"] if fetched.returncode == 0 else []) + [target_ref]:
            checkout = subprocess.run(
                ["git", "-C", str(clone_dir), "checkout", "--quiet", "--detach", candidate],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if checkout.returncode == 0:
                return clone_dir
        raise ContainerRuntimeError(
            f"Could not check out '{target_ref}' in {clone_dir}: {checkout.stderr.strip()}\n"
            "Check the network connection, or delete that directory to clone it again."
        )

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


def container_cache_dir() -> Path:
    """Return (creating it if needed) the cache directory mounted as ``/cache``.

    It is ``<default_cache_root()>/container_cache``, next to the production
    checkout and not inside it, so re-cloning the checkout keeps it. It holds
    what the in-container setup downloads on first use: the Geant4 datasets
    (about 2 GB). Everything else the stages need is in the images.
    """
    cache_dir = default_cache_root() / "container_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
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
    image: Optional[str] = None,
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
        config_path: The stage's YAML config: a path relative to
            ``prod_root``, or an absolute path inside the container
            (``/output/...``).
        stage: Stage identifier written to ``pipeline_timing.csv``.
            Defaults to the script's file name.
        image_kind: ``"sim"`` or ``"reco"``: which environment the
            production repo's setup script prepares. Must match ``image``.
        seed: Random seed forwarded to the stage script.
        run_id: Run subdirectory under ``output_dir/runs/``.
        image: Container image reference. Defaults to the pinned image
            of ``image_kind``.
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
    if image is None:
        pinned = load_pipeline_config()
        image = pinned.reco_image if image_kind == "reco" else pinned.sim_image
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (prod_root / STAGE_ENV_SCRIPT).is_file():
        raise ContainerRuntimeError(
            f"{prod_root} has no {STAGE_ENV_SCRIPT}: this colliderml-production "
            f"checkout is older than {COLLIDERML_PRODUCTION_REF}, which this version "
            "of colliderml needs. Delete the checkout to let simulate() re-clone it."
        )

    stage_label = stage or Path(script).stem
    # Overlay configs live in the output directory and come as container paths.
    container_config = config_path if config_path.startswith("/") else f"/workspace/{config_path}"

    # The stage's settings are positional parameters of a fixed command, so no
    # value is re-read as shell source.
    container_cmd = (
        'source "/workspace/$1" "$2" || exit 1; '
        'cd "/workspace/scripts/$(dirname "$4")" || exit 1; '
        'exec python3 "/workspace/$3" --record /output/pipeline_timing.csv --stage "$5" -- '
        'python3 "$(basename "$4")" --config "$6" --output /output/runs '
        '--output-subdir "$7" --seed "$8"'
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
    ]
    # Proxy settings of the host, for the first-run download of the Geant4 data.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        if os.environ.get(name):
            cmd += ["-e", f"{name}={os.environ[name]}"]
    cmd += [
        "--entrypoint",
        "/bin/bash",
        image,
        "-c",
        container_cmd,
        "stage",
        STAGE_ENV_SCRIPT,
        image_kind,
        STAGE_TIMER_SCRIPT,
        script,
        stage_label,
        container_config,
        run_id,
        str(seed),
    ]
    return subprocess.run(cmd, timeout=timeout)


def run_pipeline(
    *,
    channel: str,
    output_dir: Path,
    stage_manifest: List[dict],
    seed: int = 42,
    run_id: str = "0",
    image: Optional[str] = None,
    reco_image: Optional[str] = None,
    prod_root: Optional[Path] = None,
    runtime: Optional[str] = None,
    on_stage_start: Optional[Callable[[int, int, str], None]] = None,
    on_stage_end: Optional[Callable[[int, int, str, int], None]] = None,
) -> List[StageRun]:
    """Run every stage in a manifest, stopping on the first non-zero exit.

    Args:
        channel: Physics channel (not used; kept for callers that pass it).
        output_dir: Host directory for outputs (mounted as ``/output``).
        stage_manifest: List of dicts produced by
            :func:`colliderml.simulate.pipeline.generate_stage_manifest`.
        seed: Random seed forwarded to each stage.
        run_id: Run subdirectory under ``output_dir/runs/``.
        image: Sim image reference. Defaults to the pinned image.
        reco_image: Reco image reference, used by the stages whose manifest
            entry has ``image == "reco"``. Defaults to the pinned image.
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
    cache_dir = container_cache_dir()

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

    return results
