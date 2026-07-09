---
search: false
---

# Running production stages (`run_stage.py`)

How to run any ColliderML production stage on Perlmutter with the **podman-hpc
container** run mechanism that replaced the old shifter + cvmfs + conda patchwork.
Scripts live in [`colliderml-production`](https://github.com/OpenDataDetector/colliderml-production) under `scripts/cli/`.

::: tip Which entry point?
- **`run_stage.py`** (this page) — NERSC production: interactive smoke runs *and* SLURM
  at scale, driven entirely by a config YAML. Launches podman-hpc for you.
- **`run_docker.sh` / `run_pipeline_docker.sh`** — local laptop / dev, or the public
  `colliderml.simulate` library. Plain `docker run` of the single `sw` image. Not covered here.
:::

## TL;DR

A stage is **one config YAML**. You never pass the stage name or output dir on the
command line — they are fields *in* the config. To run the digitization stage of a
dataset interactively, on a Perlmutter login (or exclusive/interactive compute) node:

```bash
python scripts/cli/run_stage.py \
    configs_production/drift_beamspot/single_muon_10GeV/digitization_config.yaml
```

`run_stage.py` merges the env defaults, git-commits the config for provenance, then
runs the stage script *inside* the stage's container via:

```
podman-hpc run --rm -v /global/cfs/cdirs/m4958:/global/cfs/cdirs/m4958 \
    -v $PSCRATCH:$PSCRATCH --entrypoint /bin/bash <image> \
    -c "source setup_container_env.sh && python simulation/digi_and_reco.py --config … --output … --seed …"
```

CFS and `$PSCRATCH` are bind-mounted at their **same absolute paths**, so no path
translation is ever needed inside the container.

## Mental model

- **Config-driven.** `run_stage.py` takes config file(s), not flags describing the work.
  The `stage:` field selects the stage; everything else (events, container, execution
  mode, output) is config too.
- **One config = one stage.** A dataset is produced by running several configs in order
  (gen → sim → digi → …), each its own file.
- **`env_setup.yaml` supplies defaults.** The stage config is merged over
  `scripts/cli/env_setup.yaml` (the container variant), which provides per-stage
  in-container env setup and the default image mapping.
- **The host launches the container.** You run `run_stage.py` on the login node; it emits
  `podman-hpc run …`. You are *not* inside a container when you call it.
- **Every run is git-committed.** `run_stage.py` refuses to run on `master`/`main` (use a
  config branch) and writes an `expanded_config.yaml` snapshot + `.git_commit_success`
  marker into the output dir for reproducibility.

## The container fields

Set these in the stage config (usually under `common:`), or inherit from `env_setup.yaml`:

| Field | Meaning |
| :-- | :-- |
| `common.container` | Image the stage runs in, e.g. `ghcr.io/opendatadetector/sw:pr-8` (single-image) or `localhost/colliderml/acts-arrow:…` (sim half of the two-container model). |
| `common.container_tarball` | CFS path to the image tarball, `podman-hpc load -i`'d once per node in SLURM mode (login nodes have no registry pull). |
| `common.stage_containers` | *Optional* per-stage image override — e.g. route `calo_digitization`/`pandora_reco` to the reco image while everything else uses the sim image. Falls back to `common.container`. |

::: info Single- vs two-container
Newer campaigns point `common.container` at one self-contained image (`sw:pr-8`) and run
everything there. The two-container model splits reco out because the sw image's ROOT
lacks the Eve component KalTest/Pandora needs — see `docs/TWO_CONTAINER_PIPELINE.md` in
the production repo. Both are driven the same way through `run_stage.py`; only the
container fields differ.
:::

::: warning Pin images by digest before scaling
`pr-4`/`pr-8` are *mutable* PR tags. For a large final campaign, pin the image by digest
so a re-tag can't silently change the stack.
:::

## Execution modes

Set `job_config.execution_mode` in the config, or override with `--execution-mode`.
Precedence: CLI flag > config > default (`distributed_slurm`).

| Mode | What it does |
| :-- | :-- |
| `interactive` | Runs the stage as a subprocess **now**, in the container, on the current node. Use for smoke tests / small runs / debugging. |
| `distributed_slurm` | One SLURM job; runs fan out across nodes (`n_runs`, `runs_per_node`). The default. |
| `monolithic_slurm` | One SLURM job, all runs on one allocation. |
| `multi_node_slurm` | Multi-node job; also forced when you pass **several** config files at once. |

::: tip `debug_output_dir`
On an `interactive` run, setting `debug_output_dir` makes the stage write to that one
directory **and skips the validation/guardian phase**. This is the standard way to drive
a single pilot run into a known dataset path. For a real production sample, use a SLURM
mode instead so validation runs.
:::

## CLI reference

```
python scripts/cli/run_stage.py <config.yaml> [<config2.yaml> …] [options]
```

| Argument | Meaning |
| :-- | :-- |
| `configs` (positional, 1+) | Stage config path(s). Multiple → one combined `multi_node_slurm` job. |
| `--execution-mode {interactive,monolithic_slurm,distributed_slurm,multi_node_slurm}` | Override the config's mode. |
| `--dry-run` | SLURM modes: write the batch scripts to a dry-run dir instead of `sbatch`. |
| `--run-range START END` | Subset of run IDs (distributed modes). |
| `--run-list ID [ID …]` | Explicit run IDs. |
| `--force-commit` | Git-commit even if nothing changed / the marker exists. |
| `--allow-master` | Permit running on `master`/`main` (normally blocked). |

There is intentionally **no** `--stage`, `--config`, `--output`, or `--debug-output-dir`
flag — those are config fields.

## Anatomy of a stage config

A live production config (`configs_production/drift_beamspot/single_muon_10GeV/digitization_config.yaml`):

```yaml
campaign: "drift_beamspot"
dataset:  "single_muon_10GeV"
version:  "v1"
stage:    "digitization"          # -> simulation/digi_and_reco.py

common:
  container:         "ghcr.io/opendatadetector/sw:pr-8"
  container_tarball: "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/stress_mu200/sw_pr8_image.tar"

job_config:
  execution_mode: "interactive"   # interactive | distributed_slurm | monolithic_slurm | multi_node_slurm
  n_runs: 1
  runs_per_node: 1
  time_limit: "01:00:00"
  qos: "interactive"

events: 10000
threads: 64

# stage-specific knobs (read by digi_and_reco.py):
odd_geo_dir: /opt/odd
digi_config: /opt/odd/config/odd-digi-smearing-config.json
output_parquet_arrow: True        # native ACTS parquet: tracker_simhits + tracker_hits + particles + tracks
digi: True
reco: True

# interactive single-run into a known dataset dir; skips validation/guardian:
debug_output_dir: "/global/cfs/cdirs/m4958/data/ColliderML/simulation/drift_beamspot/single_muon_10GeV/v1/runs/0"
```

## Stages and their scripts

`cli_utils.STAGE_SCRIPT_MAP` maps `stage:` → script. The container column is the
two-container model (single-image campaigns run all of these in one image):

| `stage:` | script | container |
| :-- | :-- | :-- |
| `madgraph_init` | `simulation/madgraph_init.py` | sim |
| `madgraph_generation` | `simulation/madgraph_gen.py` | sim |
| `pythia_generation` | `simulation/pythia_gen.py` | sim |
| `particlegun_generation` | `simulation/particlegun_gen.py` | sim |
| `merge_smear` | `simulation/merge_and_smear.py` | sim |
| `simulation` | `simulation/ddsim_run.py` | sim |
| `digitization` | `simulation/digi_and_reco.py` | sim |
| `calo_digitization` | `simulation/calo_digitization.py` | **reco** |
| `pandora_reco` | `simulation/pandora_reco.py` | **reco** |
| `build_tracks` · `build_tracker_hits` · `build_particles` · `build_manifest` · `convert_all` | `postprocessing/*.py` | sim |

Config layout: `configs_production/<campaign>/<dataset>/<stage>_config.yaml`.
(`configs_development/docker_test/*` are for the Docker path, not `run_stage.py`.)

## Worked examples

**Interactive pilot** (one run into a dataset dir) — set `execution_mode: interactive`
and `debug_output_dir`, then run each stage in order:

```bash
python scripts/cli/run_stage.py .../single_muon_10GeV/simulation_config.yaml
python scripts/cli/run_stage.py .../single_muon_10GeV/digitization_config.yaml
```

**SLURM at scale** — set `execution_mode: distributed_slurm`, `n_runs`, `runs_per_node`,
`time_limit`, `qos`, then:

```bash
python scripts/cli/run_stage.py .../ttbar/digitization_config.yaml            # submit
python scripts/cli/run_stage.py .../ttbar/digitization_config.yaml --dry-run  # inspect scripts only
python scripts/cli/run_stage.py .../ttbar/digitization_config.yaml --run-range 0 100
```

## What happens under the hood

- **interactive:** `run_interactive()` wraps the stage in `build_podman_run_prefix()`
  (`podman-hpc run … bash -c "source setup_container_env.sh && python <script> --config …
  --output <run_dir> --output-subdir <ID> --seed <dataset>_<version>_run<ID>"`) and runs
  it now.
- **SLURM:** `job_submission.py` writes a batch script whose preamble does
  `podman-hpc image exists <c> || podman-hpc load -i <tarball>` once per node, then each
  `srun` task runs `podman-hpc run …`; a validation/guardian job is appended (unless
  `debug_output_dir` is set).

The internal flag is still named `use_shifter` for legacy reasons, but the emitted
command is `podman-hpc run` — verified free of `shifter`/`cvmfs`/`conda`.

## Gotchas

- **Run from a config branch, not `master`** (git-guard); or pass `--allow-master`.
- **`podman-hpc` needs a login node or an *exclusive*/interactive/debug compute node** —
  it fails **silently** on plain `shared`-qos nodes.
- **Pin images by digest** before a large campaign; `pr-*` tags are mutable.
- **`debug_output_dir` disables validation** — great for pilots; use SLURM modes for real
  samples so the validation/guardian phase runs.
- **Reco stages are CPU-only** (`-C cpu`); GPU nodes are only for GNN training.

::: warning Status (2026-07)
The container run mechanism is **not yet merged to `master`** in `colliderml-production`
— `master` is still shifter/cvmfs/conda. It lives in two stacked open PRs: **#42
"Container approach"** (the `run_stage`/`cli_utils`/`job_submission`/`env_setup.yaml`
changes described here) and **#43 "Pandora reco stages"**. The top-level `README.md` and
`scripts/cli/README.md` in that repo still describe the old conda/cvmfs flow and should be
pointed here.
:::
