# `colliderml.simulate`

API reference for the simulation subsystem. For the conceptual overview
(container runtimes, caching, the pipeline stages) see
[Local Simulation](../guide/simulation.md); for the SaaS variant see
[Remote Simulation](../guide/remote-simulation.md).

## Top-level entry point

### `simulate(...)` → `SimulationResult`

```python
colliderml.simulate(
    *,
    preset: str | Preset | None = None,
    channel: str | None = None,
    events: int | None = None,
    pileup: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
    image: str = DEFAULT_IMAGE,
    remote: bool = False,
    run_id: str = "0",
    prod_root: str | Path | None = None,
    runtime: str | None = None,
    quiet: bool = False,
) -> SimulationResult
```

Runs the full pipeline for one (channel, events, pileup) configuration.

**Parameter semantics**

- `preset` — name of a bundled preset (see
  [`list_presets()`](#list_presets)), or a [`Preset`](#preset-dataclass)
  instance. Supplies defaults for `channel`, `events`, and `pileup`.
- Explicit `channel` / `events` / `pileup` **always win** over the preset's
  values. Pass just one (e.g. `pileup=40`) to tweak a preset without
  defining a new one.
- `output_dir` defaults to
  `./colliderml_output/<channel>_pu<pileup>_<events>evt/`.
- `seed` is forwarded to every stage so the full run is reproducible.
- `image` pins the OpenDataDetector software image. Leave at the default
  unless you are rebuilding the image yourself.
- `remote=True` submits to the SaaS backend instead of running locally —
  requires `pip install "colliderml[remote]"` and a valid HuggingFace
  token. See [`colliderml.remote`](./remote.md) for the SaaS client.
- `run_id` names the subdirectory under `output_dir/runs/`. The default
  `"0"` is fine for single-run workflows; bump it for parameter sweeps.
- `prod_root` lets advanced callers point at a pre-existing
  `colliderml-production` checkout. Leave `None` for the default
  auto-clone behaviour.
- `runtime` forces a container runtime (`"docker"` or `"podman"`);
  auto-detected otherwise.
- `quiet=True` suppresses per-stage progress messages.

**Errors**

- `ValueError` — neither preset nor explicit channel is supplied, or the
  channel is unknown.
- `ContainerRuntimeError` — no container runtime is available and
  `remote=False`.
- `RuntimeError` — a pipeline stage exited non-zero.

**Example**

```python
import colliderml

result = colliderml.simulate(
    preset="higgs-portal-quick",
    output_dir="runs/hp10",
    seed=1234,
)
print(result.run_dir)
for fp in result.list_files():
    print(" ", fp.name)
```

## `SimulationResult` dataclass

```python
@dataclass
class SimulationResult:
    channel: str
    events: int
    pileup: int
    output_dir: Path
    run_dir: Path
    stages: list[StageRun]
    remote_request_id: str | None = None
```

| Field | Meaning |
| :-- | :-- |
| `channel` | Physics channel that was simulated. |
| `events` | Requested event count (actual may differ slightly for some channels). |
| `pileup` | Pileup level. |
| `output_dir` | The directory you passed (or the computed default). |
| `run_dir` | `<output_dir>/runs/<run_id>/` — the per-run artefact directory. |
| `stages` | One `StageRun(name, stage, returncode)` per executed stage. |
| `remote_request_id` | Only set when `remote=True`; the backend's request ID. |

Methods:

- `list_files()` → sorted `list[Path]` of every file under `run_dir`.

## `Preset` dataclass

```python
@dataclass(frozen=True)
class Preset:
    name: str
    channel: str
    events: int
    pileup: int
    description: str = ""
```

Helper methods:

- `as_dict()` — plain dict view for JSON/YAML serialisation.

## `load_presets() -> dict[str, Preset]` { #list_presets }

Loads the bundled preset catalogue from the package's `presets.yaml`. The
file ships as package data, so it works in editable installs, wheels, and
sdists alike.

```python
from colliderml.simulate import load_presets
catalogue = load_presets()
for name, preset in sorted(catalogue.items()):
    print(f"{name:25s} {preset.channel:15s} events={preset.events:>6}  pileup={preset.pileup}")
```

## `resolve_preset(name, presets=None) -> Preset`

Looks up a single preset by name. Raises `ValueError` with the full list
of available names on a miss — so typos surface immediately.

## Auto-cloning the production repo

The pipeline scripts, YAML stage configs, and container bootstrap script
live in [`OpenDataDetector/colliderml-production`](https://github.com/OpenDataDetector/colliderml-production).
The public library clones that repository on first use into a cache
directory and mounts it inside the container as `/workspace`.

```python
from colliderml.simulate.docker import (
    clone_colliderml_production,
    default_cache_root,
    COLLIDERML_PRODUCTION_REF,
)

print("cache root:", default_cache_root())
print("pinned ref:", COLLIDERML_PRODUCTION_REF)

# Idempotent; just fetches if the clone already exists.
prod = clone_colliderml_production()
print("cloned to:", prod)
```

**Overriding the pinned ref** — set the `COLLIDERML_PRODUCTION_REF`
environment variable before calling `simulate()`. This is the supported
way for pipeline developers to test an in-flight branch without editing
the library.

**Forcing a re-clone** — pass `force_refresh=True` to
`clone_colliderml_production`, or delete the cache directory and let
`simulate()` rebuild it.

## Container runtime helpers

All exported from `colliderml.simulate.docker`:

| Function | Purpose |
| :-- | :-- |
| `get_container_runtime()` | Return `"docker"` or `"podman"`; raise `ContainerRuntimeError` if neither is installed. |
| `check_runtime_available(runtime=None)` | Verify the runtime's daemon is actually reachable (`runtime info`). |
| `check_image_available(image, *, runtime=None)` | Return `True` iff the image is already pulled locally. |
| `pull_image(image, *, interactive=True, runtime=None)` | Pull the image, prompting before the ~10 GB download when running on a tty. |
| `default_cache_root()` | `$COLLIDERML_CACHE`/`simulate` or `~/.cache/colliderml/simulate` if unset. |
| `clone_colliderml_production(cache_root=None, *, ref=None, force_refresh=False)` | Clone or refresh the production repo at the pinned ref. |
| `run_pipeline(...)` | Low-level multi-stage runner used by `simulate()`. |

You rarely need to call these directly — they are documented here so
that users debugging CI failures or writing their own orchestrators
can see the primitives.

## Pipeline introspection

`colliderml.simulate.pipeline` exposes the channel→stages mapping that
`simulate()` uses internally. Useful when you want to reason about what
would run *before* spinning up a container:

```python
from colliderml.simulate.pipeline import (
    CHANNEL_STAGES,
    get_channel_stages,
    list_channels,
)

print(list_channels())
for stage in get_channel_stages("ttbar"):
    print(f"  {stage.name:35s} ({stage.script})")
```

And to preview the manifest that will be handed to the container runner:

```python
from pathlib import Path
from colliderml.simulate.docker import clone_colliderml_production
from colliderml.simulate.pipeline import generate_stage_manifest

prod = clone_colliderml_production()
for step in generate_stage_manifest("higgs_portal", prod_root=prod):
    print(step["name"], "→", step["config_path"])
```

## See also

- [Local Simulation guide](../guide/simulation.md) — conceptual overview
- [Remote Simulation guide](../guide/remote-simulation.md) — SaaS variant
- [`colliderml.load()`](./loading.md) — load simulation output
- [Benchmark tasks](./tasks.md) — score simulated events against a task
