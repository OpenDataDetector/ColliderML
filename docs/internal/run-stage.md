---
search: false
---

# Running production stages (`run_stage.py`)

The complete operator's guide to ColliderML production on Perlmutter: how
`run_stage.py` works, **both run mechanisms** (the original shifter/cvmfs one and
the new podman-hpc container one), how configs are structured, how parquet gets
written (legacy `convert_all` vs the native Arrow writer, with regression
evidence), and every gotcha we know about. Scripts live in
[`colliderml-production`](https://github.com/OpenDataDetector/colliderml-production)
under `scripts/cli/`.

::: tip Which entry point?
- **`run_stage.py`** (this page) — NERSC production: interactive smoke runs *and*
  SLURM at scale, driven entirely by a config YAML.
- **`run_docker.sh` / `run_pipeline_docker.sh`** — local laptop / dev, or the public
  `colliderml.simulate` library. Plain `docker run`. Not covered here.
:::

## The mental model (unchanged across both mechanisms)

`run_stage.py` itself is **identical** in the old and new worlds — the run
mechanism lives entirely in its helpers (`cli_utils.py`, `env_setup.yaml`,
`job_submission.py`). What never changes:

- **Config-driven.** You pass config file(s), not flags describing the work. The
  `stage:` field selects the stage; everything else (events, environment,
  execution mode, output) is config too. There is intentionally **no** `--stage`,
  `--config`, or `--output` flag.
- **One config = one stage.** A dataset is produced by running several configs in
  order (gen → sim → digi → convert), each its own file, under
  `configs_production/<campaign>/<dataset>/`.
- **`env_setup.yaml` supplies environment defaults.** The stage config is merged
  over it; per-stage blocks list the shell lines sourced before the stage script
  runs.
- **Every run is git-committed.** `run_stage.py` refuses to run on `master`/`main`
  (use a config branch, or `--allow-master`) and writes an
  `expanded_config.yaml` snapshot + `.git_commit_success` marker into the dataset
  dir for provenance.
- **The stage script invocation is always** `python <script>.py --config <yaml>
  --output <run_dir> --output-subdir <ID> --seed <dataset>_<version>_run<ID>`.

## Two run mechanisms — old and new

| | **Old (shifter / cvmfs / conda)** | **New (podman-hpc container)** |
| :-- | :-- | :-- |
| Where it lives | GitHub `master` (what all pre-2026-06 datasets used) | PRs [#42](https://github.com/OpenDataDetector/colliderml-production/pull/42) / [#43](https://github.com/OpenDataDetector/colliderml-production/pull/43) + `feat/two-container-pipeline`; used by `drift_beamspot` onward |
| Environment | mixed: shifter-ATLAS image + cvmfs LCG views + bare-metal ACTS/dd4hep + conda | one self-contained image (spack HEP stack), no cvmfs, no conda |
| Container tech | `shifter --image=… --module=cvmfs` | `podman-hpc run` (+ `podman-hpc load -i <tarball>` per node) |
| Which stages containerised | only `SHIFTER_STAGES` (pythia/gun gen, merge_smear, sim, digi, calo_digi, pandora); **MadGraph ran on the host** | **every stage** runs in the image (MadGraph included — the MG5-PY8 interface is baked in) |
| Default image | `registry.cern.ch/atlasadc/atlas-grid-almalinux9` | `ghcr.io/opendatadetector/sw:pr-8` (or the two-container sim/reco pair) |
| Datasets produced | `hard_scatter/*` (zprime, tth, …), `full_pileup/*`, pilots | `drift_beamspot/*`, everything new |

### How the old mechanism ran a stage

For a stage in `SHIFTER_STAGES`, `cli_utils` emitted
`shifter --image=registry.cern.ch/atlasadc/atlas-grid-almalinux9 --module=cvmfs …`
and the per-stage `env_setup.yaml` block sourced the environment inside it — e.g.
`simulation` did the ATLAS local setup
(`source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh`), `madgraph` sourced a
cvmfs LCG view (`source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh <view>`) plus a
local python env. `madgraph_init`/`madgraph_generation` ran **directly on the
host** against a host-built MG5. This is the deeply-proven path — every released
dataset up to mid-2026 came from it — but it depends on cvmfs availability, host
toolchains, and ATLAS infrastructure that ColliderML doesn't control.

### How the new mechanism runs a stage

You run `run_stage.py` on a login node (you are *not* inside a container); it
launches the container for you:

```
podman-hpc run --rm -v /global/cfs/cdirs/m4958:/global/cfs/cdirs/m4958 \
    -v $PSCRATCH:$PSCRATCH --entrypoint /bin/bash <image> \
    -c "source setup_container_env.sh && python simulation/digi_and_reco.py --config … --output …"
```

CFS and `$PSCRATCH` are bind-mounted at their **same absolute paths** — no path
translation anywhere. In SLURM mode, `job_submission.py`'s preamble does
`podman-hpc image exists <c> || podman-hpc load -i <tarball>` once per node
(login/compute nodes can't pull registries), then each `srun` task wraps the stage
in `podman-hpc run`. The internal flag is still named `use_shifter` for legacy
reasons; the emitted command is podman-hpc, verified free of
`shifter`/`cvmfs`/`conda`.

Container selection is config:

| Field | Meaning |
| :-- | :-- |
| `common.container` | Image the stage runs in (e.g. `ghcr.io/opendatadetector/sw:pr-8`) |
| `common.container_tarball` | CFS tarball `podman-hpc load`ed per node in SLURM mode |
| `common.stage_containers` | *Optional* per-stage override — routes `calo_digitization`/`pandora_reco` to the key4hep reco image in the two-container model (see `docs/TWO_CONTAINER_PIPELINE.md`); everything else falls back to `common.container` |

## Anatomy of a stage config (annotated)

A live production config (`configs_production/drift_beamspot/single_muon_10GeV/digitization_config.yaml`):

```yaml
campaign: "drift_beamspot"        # -> output under <output_base_dir>/<campaign>/<dataset>/<version>/
dataset:  "single_muon_10GeV"
version:  "v1"
stage:    "digitization"          # -> STAGE_SCRIPT_MAP -> simulation/digi_and_reco.py

common:                           # NEW-mechanism fields (absent in old-mechanism configs)
  container:         "ghcr.io/opendatadetector/sw:pr-8"
  container_tarball: "/global/cfs/cdirs/m4958/.../sw_pr8_image.tar"

job_config:
  execution_mode: "interactive"   # interactive | distributed_slurm | monolithic_slurm | multi_node_slurm
  n_runs: 1                       # SLURM modes: number of runs; nodes = n_runs / runs_per_node
  runs_per_node: 1
  time_limit: "01:00:00"
  qos: "interactive"

events: 10000
threads: 64

# --- stage-specific knobs, read by the stage script ---
odd_geo_dir: /opt/odd             # geometry the image ships
digi_config: /opt/odd/config/odd-digi-smearing-config.json
digi: True
reco: True
output_parquet_arrow: True        # native ACTS parquet (see writer section below)

# interactive single-run into a known dataset dir; DISABLES validation/guardian:
debug_output_dir: ".../drift_beamspot/single_muon_10GeV/v1/runs/0"
```

## Execution modes & CLI

Mode precedence: **CLI `--execution-mode` > `job_config.execution_mode` > default
(`distributed_slurm`)**.

| Mode | What it does |
| :-- | :-- |
| `interactive` | Runs the stage as a subprocess **now**, on the current node. Smoke tests, pilots, debugging. |
| `distributed_slurm` | One SLURM job; runs fan out across nodes. The production default. |
| `monolithic_slurm` | One SLURM job, everything in one allocation (e.g. `madgraph_init`, which can't parallelise). |
| `multi_node_slurm` | Multi-node job; **forced** when you pass several configs at once. |

```
python scripts/cli/run_stage.py <config.yaml> [<config2.yaml> …] [options]
  --execution-mode {interactive,monolithic_slurm,distributed_slurm,multi_node_slurm}
  --dry-run              # SLURM modes: write batch scripts, don't sbatch — ALWAYS do this first
  --run-range START END  # subset of runs (START inclusive, END exclusive)
  --run-list ID [ID …]   # explicit run IDs
  --force-commit         # git-commit even if unchanged
  --allow-master         # permit running on master/main
```

`debug_output_dir` (config field, interactive mode): the stage writes to that one
directory and **the validation/guardian phase is skipped** (the validator expects
numbered `runs/N` subdirs). Standard for pilots; never for real samples.

## Stages

| `stage:` | script | notes |
| :-- | :-- | :-- |
| `madgraph_init` | `simulation/madgraph_init.py` | one-time process compile; monolithic; **old: host / new: in-image** |
| `madgraph_generation` | `simulation/madgraph_gen.py` | parallel event gen off the process tarball |
| `pythia_generation` | `simulation/pythia_gen.py` | |
| `particlegun_generation` | `simulation/particlegun_gen.py` | |
| `merge_smear` | `simulation/merge_and_smear.py` | pileup merge and/or vertex smear |
| `simulation` | `simulation/ddsim_run.py` | Geant4 via ddsim |
| `digitization` | `simulation/digi_and_reco.py` | ACTS digi → seeding → CKF → ambiguity |
| `calo_digitization` | `simulation/calo_digitization.py` | reco image (two-container) |
| `pandora_reco` | `simulation/pandora_reco.py` | reco image (two-container) |
| `convert_all` | `postprocessing/convert_all.py` | ROOT/edm4hep → parquet (legacy writer) |
| `build_*`, `build_manifest` | `postprocessing/*.py` | per-object converters / manifest |

## Parquet: `convert_all` vs the native Arrow writer

Two ways to get parquet, and this is where trust matters:

**Legacy `convert_all`** (the proven path): digi writes ROOT
(`measurements.root`, `particles.root`, `tracksummary_ambi.root`), then the
`convert_all` stage reads ROOT + edm4hep via `pyedm4hep` and writes
`parquet/{truth,reco}/…`. Every released dataset used this. It **still works in
the new container** — all its deps (`pyarrow`, `pyedm4hep`, `polars`, …) are
verified importable in `sw:pr-8`, and the env-setup pip guard now checks all of
them (not just pyarrow) before deciding the install is complete.

**Native Arrow writer** (`output_parquet_arrow: True` on the digi config): the
ACTS Arrow plugin writes `particles` / `tracker_hits` / `tracker_simhits` /
`tracks` parquet **directly during digitization** — no convert_all needed for
those tables. Used by `drift_beamspot` onward.

### Regression evidence (2026-07)

A same-seed digitization of the same edm4hep input was run in `sw:pr-8` emitting
**both** outputs, then compared
(`tests/regression/test_actsnative_vs_v1.py`):

- **Fitted track parameters are bit-identical** — max |native − convert_all| for
  `d0`, `z0`, `phi`, `theta`, `qOverP` = **0.0** across all matched tracks.
- Particles: identical per-event counts and PDG multisets. Tracker hits:
  consistent under the schema mapping (native rows are measurements /
  Release-2 layout; v1 rows per-measurement).
- **Known, intended differences:** (1) `convert_all` **drops events with zero
  tracks**; the native writer keeps them as empty event-rows — downstream code
  must not assume identical event_id density. (2) Table schemas differ (flat v1
  vs Release-2 nested layout).

### Calo: convert_all only (for now)

**No currently-available image can write native calo parquet.** The
tracker-hits-v2 ACTS rebase dropped the calo converter machinery (PR #5441 was
not re-applied), verified empirically: `EDM4hepCaloHitInputConverter` is absent
in both `sw:pr-8` and the older arrow-dev image. `digi_and_reco.py`
feature-detects this and logs a warning. Until an image restores it, **calo
parquet comes exclusively from `convert_all`** (or `convert_calo_digi.py` on
Pandora reco output) — i.e. the old trusted path is the only path for calo.

## Gotchas (all of them)

- **Run from a config branch, not `master`** — the git-guard blocks it
  (`--allow-master` to override).
- **`podman-hpc` fails *silently* on shared-qos nodes.** Use a login node or an
  exclusive/interactive/debug allocation. (Reco/Pandora stages are CPU-only:
  `-C cpu`.)
- **The `/cache` ODD-v4 shadow trap** (manual container runs only): sourcing
  `setup_container_env.sh` by hand builds an ODD **v4** factory lib into
  `/cache` and prepends it to `LD_LIBRARY_PATH`, shadowing the image's
  `/opt/odd` v6 factories → dd4hep "no factory" crash in any geometry stage.
  Production via `run_stage` avoids this (it leaves `COLLIDERML_CACHE` pointing
  at the image's baked cache). If you source the setup manually, strip it:
  `export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/cache' | paste -sd:)`
- **Pin images by digest before scaling.** `pr-4`/`pr-8` are *mutable* PR tags; a
  re-tag can silently change the stack under a campaign.
- **A host-built `madgraph_process.tgz` is NOT portable into the container**
  (FastJet ABI + CutTools mismatch). Regenerate the process in-image with
  `madgraph_init` when switching mechanisms. Do not assume the tarball transfers.
- **`debug_output_dir` disables validation** — pilots only.
- **File-exists ≠ stage-done.** ROOT writers create files at start and finalize
  at sequencer end; a killed job leaves readable-but-partial files. Gate on the
  job log's "Processed N events", which is what the dataset loaders do.
- **Never merge ROOT histogram files** across runs (corrupts widths);
  row-concatenate TTrees / parquet instead.
- **Stale failure reports:** the guardian's `failure_report_*_FINAL.txt` can be
  false alarms from earlier attempts (e.g. globbing `events.hepmc3` when the
  config writes `events.hepmc`). The authoritative record is
  `validation_reports/*.json` + what's on disk.

## Worked examples

```bash
# interactive pilot into a known dir (debug_output_dir set in the config):
python scripts/cli/run_stage.py configs_production/drift_beamspot/single_muon_10GeV/digitization_config.yaml

# production at scale — ALWAYS dry-run first, read the emitted batch script:
python scripts/cli/run_stage.py .../ttbar/digitization_config.yaml --dry-run
python scripts/cli/run_stage.py .../ttbar/digitization_config.yaml
python scripts/cli/run_stage.py .../ttbar/digitization_config.yaml --run-range 0 100   # subset

# one-time NLO process build (new mechanism: runs in-image):
python scripts/cli/run_stage.py .../ttbar/madgraph_init_config.yaml --execution-mode monolithic_slurm
```

## Status / provenance (2026-07)

- GitHub `master` = **old mechanism** (shifter/cvmfs/conda). The new container
  mechanism is in open PRs **#42** (podman-hpc run mechanism) and **#43**
  (Pandora reco stages), with the drift_beamspot campaign on top (#50). The
  running copy on Perlmutter is `colliderml-prod-container` branch
  `feat/two-container-pipeline`.
- Back-compat regression suite: `tests/regression/` (same-seed dual-output
  driver + pytest comparison). Calo coverage blocked on the #5441 machinery.
- The old `README.md` / `scripts/cli/README.md` in the production repo predate
  the container mechanism; this page supersedes them.
