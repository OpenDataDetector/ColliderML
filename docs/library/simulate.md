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
    image: str | None = None,
    reco_image: str | None = None,
    particle_flow: bool | None = None,
    threads: int | None = None,
    pandora_processes: int | None = None,
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
  [`load_presets()`](#list_presets)), or a [`Preset`](#preset-dataclass)
  instance. Supplies defaults for `channel`, `events`, `pileup` and
  `particle_flow`.
- Explicit `channel` / `events` / `pileup` **always win** over the preset's
  values. Pass just one (e.g. `pileup=40`) to tweak a preset without
  defining a new one.
- `output_dir` defaults to
  `./colliderml_output/<channel>_pu<pileup>_<events>evt/`.
- `seed` is forwarded to every stage. Whether two runs with the same seed give identical events has not been tested with the two-thread default.
- `image` is the sim image (generation, Geant4, ACTS tracking, packaging)
  and `reco_image` the reco image (calorimeter digitisation and Pandora).
  Both default to the [pinned versions](#pinned-pipeline-versions).
- `particle_flow=True` continues after tracking: truth particle tables,
  calorimeter digitisation and Pandora particle flow, and the final
  parquet tables under `<output_dir>/parquet/`. The default is the
  preset's setting, or `False` without a preset. See
  [Local Simulation](../guide/simulation.md#run-the-full-chain).
- `threads` sets the Geant4 and ACTS threads, `pandora_processes` the
  number of concurrent Pandora processes. `threads` defaults to 2, the
  minimum (with one thread the Geant4 step of the pinned sim image writes
  every event twice), `pandora_processes` to 1. One Pandora
  process needs about 2 GB without pile-up and about 6 GB at
  pile-up 200.
- `remote=True` submits to the SaaS backend instead of running locally —
  requires `pip install "colliderml[remote]"` and a valid HuggingFace
  token. See [`colliderml.remote`](./remote.md) for the SaaS client.
- `run_id` names the subdirectory under `output_dir/runs/`. Runs without
  particle flow can share an `output_dir` under different `run_id`s. A
  particle-flow run must keep the default `"0"` and needs an `output_dir`
  of its own. `simulate()` raises `FileExistsError` if the run is already
  there.
- `prod_root` lets advanced callers point at a pre-existing
  `colliderml-production` checkout. Leave `None` for the default
  auto-clone behaviour.
- `runtime` forces a container runtime (`"docker"`, `"podman"` or
  `"podman-hpc"`); auto-detected otherwise.
- `quiet=True` suppresses per-stage progress messages.

**Errors**

- `ValueError` - neither preset nor explicit channel is supplied, the
  channel is unknown, or `remote=True` is combined with particle flow.
- `ContainerRuntimeError` — no container runtime is available and
  `remote=False`.
- `RuntimeError` — a pipeline stage exited non-zero.
- `FileExistsError` - the output directory already holds the run.
- `FileNotFoundError` - the production checkout lacks the script or the
  config of a stage.

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
    particle_flow: bool = False
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
| `particle_flow` | Whether the run included particle flow and the final tables. |

Properties and methods:

- `parquet_dir` → `<output_dir>/parquet/`, the final tables of a
  particle-flow run, laid out as `{truth,reco}/<table>/*.parquet`.
- `list_files()` → sorted `list[Path]` of every file under `run_dir`.
- `tables()` → `dict[str, list[Path]]` of the final parquet files by table
  name (`"pfos"`, `"tracks"`, ...); empty for a run without particle flow.

## `Preset` dataclass

```python
@dataclass(frozen=True)
class Preset:
    name: str
    channel: str
    events: int
    pileup: int
    description: str = ""
    particle_flow: bool = False
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

## Pinned pipeline versions

The production-repo ref and the two container images are pinned together
in `colliderml/simulate/pipeline_config.yaml`, which ships with the
package:

```python
from colliderml.simulate import load_pipeline_config

cfg = load_pipeline_config()
print(cfg.production_ref, cfg.sim_image, cfg.reco_image)
```

| Field | Environment override | Per-call override |
| :-- | :-- | :-- |
| `production_ref` | `COLLIDERML_PRODUCTION_REF` | `prod_root=` (your own checkout) |
| `sim_image` | `COLLIDERML_SIM_IMAGE` | `image=` / `--image` |
| `reco_image` | `COLLIDERML_RECO_IMAGE` | `reco_image=` / `--reco-image` |

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
| `get_container_runtime()` | Return `"docker"`, `"podman-hpc"` or `"podman"`, whichever is found first in that order; raise `ContainerRuntimeError` if none is installed. |
| `check_runtime_available(runtime=None)` | Verify the runtime's daemon is actually reachable (`runtime info`). |
| `check_image_available(image, *, runtime=None)` | Return `True` iff the image is already pulled locally. |
| `pull_image(image=None, *, interactive=True, runtime=None)` | Pull the image (default: the pinned sim image), prompting before the download when running on a tty. Unpacked, the sim image is 6 GB and the reco image 27 GB. |
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
for stage in get_channel_stages("ttbar", particle_flow=True):
    print(f"  {stage.name:42s} {stage.image:5s} ({stage.script})")
```

`get_channel_stages(channel)` returns the stages up to tracking;
`particle_flow=True` adds the four stages after it. Each `Stage` names the
image it runs in (`"sim"` or `"reco"`). `generate_stage_manifest` raises
`FileNotFoundError` if the production checkout lacks the config of a stage.

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

- [Local Simulation guide](../guide/simulation.md): running the full chain, start to finish
- [Remote Simulation guide](../guide/remote-simulation.md): SaaS variant
- [`colliderml.load()`](./loading.md): load the released dataset
- [Benchmark tasks](./tasks.md): score simulated events against a task
