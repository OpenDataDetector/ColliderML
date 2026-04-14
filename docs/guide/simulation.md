# Local Simulation

ColliderML can generate new events on your own machine using the full
physics pipeline: Pythia (or MadGraph + Pythia for NLO processes) →
Geant4 detector simulation via DDSim → digitisation and track
reconstruction via ACTS → parquet conversion. Everything runs inside the
official OpenDataDetector software container, so you do not need to
install any HEP tooling yourself — just Docker or Podman.

This page covers the **local** story: running the pipeline on your own
workstation or a shared node. If you would rather hand the work off to a
shared cluster, see [Remote Simulation](./remote-simulation.md).

## When to run locally

| Use local simulation when | Use remote simulation when |
| :-- | :-- |
| You want tight iteration on a few events | You need >100 events with pileup |
| You have Docker / Podman + ~20 GB free disk | You do not have a container runtime |
| You want offline reproducibility | You want someone else to pay the compute bill |
| You are debugging a pipeline change | You are running a benchmark |

A typical iteration loop — `simulate → inspect → tweak → simulate again`
— fits comfortably within 10–15 minutes for the `*-quick` presets.

## Prerequisites

1. **Python 3.10 or newer** — see [Installation](./installation.md).
2. **Either Docker or Podman** on your `$PATH`. The library auto-detects
   whichever is installed; Docker is preferred if both are present.
3. **Disk headroom**:
   - ~10 GB for the OpenDataDetector software image (pulled once)
   - ~2 GB for the Geant4 physics data tables (downloaded on first run)
   - 50 MB to a few hundred MB per run, depending on event count
4. The simulate extras installed:
   ```bash
   pip install "colliderml[sim]"
   ```

## The first run, annotated

```python
import colliderml

result = colliderml.simulate(preset="ttbar-quick")
print(result.run_dir)
print([p.name for p in result.list_files()])
```

On first invocation, `simulate()` will:

1. **Detect the container runtime** (`docker` or `podman`). If neither is
   available you get a clear error pointing at
   [Remote Simulation](./remote-simulation.md) instead.
2. **Prompt before pulling** the ~10 GB container image (skipped in
   non-interactive environments — pass `image="..."` to override the
   pinned tag).
3. **Clone `colliderml-production`** into
   `~/.cache/colliderml/simulate/colliderml-production/`. This repo holds
   the stage scripts, YAML config templates, and container bootstrap
   script. It is pinned to a known-good git ref for reproducibility.
4. **Clone OpenDataDetector v4.0.4** (CERN GitLab) and
   **MG5aMC_PY8_interface** (GitHub) into the same cache directory. These
   provide the detector geometry and MadGraph/Pythia bridge that the
   container expects at `/cache/odd-v4` and `/cache/MG5aMC_PY8_interface`.
5. **Run each pipeline stage** in order. Progress is printed to stdout;
   the first stage typically takes a few seconds, detector sim is the
   long pole (~10 minutes for 10 events).
6. **Return a `SimulationResult`** with paths to the outputs.

Everything cached in step 2–4 persists between runs, so the *second*
invocation skips straight to step 5 and starts producing events
immediately.

## Presets versus explicit parameters

Presets are named shorthand for common workloads:

```python
# Preset: bundled (channel, events, pileup) triple.
colliderml.simulate(preset="ttbar-quick")

# Explicit parameters.
colliderml.simulate(channel="higgs_portal", events=10, pileup=10)

# Preset as a starting point, but bump pileup:
colliderml.simulate(preset="ttbar-quick", pileup=40)
```

Explicit arguments always win over the preset's defaults. List the full
preset catalogue on the CLI:

```bash
colliderml list-presets
```

## What comes out

`SimulationResult.run_dir` points at a per-run directory (e.g.
`colliderml_output/ttbar_pu0_10evt/runs/0/`) containing:

| File | Stage | Format |
| :-- | :-- | :-- |
| `events.hepmc.gz` | MadGraph (ttbar only) | HepMC2 |
| `merged_events.hepmc3` | Pythia + pileup | HepMC3 |
| `edm4hep.root` | DDSim / Geant4 | EDM4hep ROOT |
| `measurements.root` | Digitisation | ACTS ROOT |
| `particles.root` | Truth particles | ACTS ROOT |
| `tracksummary_ambi.root` | Reconstruction | ACTS ROOT |
| `*.parquet` | Parquet conversion | Parquet |

The parquet outputs can be loaded with the same
[`colliderml.load()`](../library/loading.md) helper used for the released
dataset, so downstream analysis code does not care whether the events
came from HuggingFace or from your own machine.

## Cache layout and disk hygiene

Everything is rooted at `$COLLIDERML_CACHE` (defaults to
`~/.cache/colliderml/`):

```
~/.cache/colliderml/simulate/
├── colliderml-production/          # pinned production-repo clone
│   └── .cache/
│       ├── odd-v4/                 # OpenDataDetector geometry
│       ├── g4data/                 # Geant4 physics tables (~2 GB)
│       └── MG5aMC_PY8_interface/   # MG5 → Pythia bridge
```

Wipe the whole tree at any time — `simulate()` will re-populate it on
the next run. If a stage complains that a cache is corrupt, call
`colliderml.simulate.docker.clone_colliderml_production(force_refresh=True)`
to re-fetch just the production-repo clone.

## Overriding the pinned pipeline ref

The pinned `colliderml-production` ref is a module-level constant in
[`colliderml.simulate.docker`](../library/simulate.md#auto-cloning-the-production-repo).
To point at an in-flight branch of the production repo without editing
the library:

```bash
export COLLIDERML_PRODUCTION_REF=my-feature-branch
python -c "import colliderml; colliderml.simulate(preset='ttbar-quick')"
```

This is the supported escape hatch for pipeline developers working on
both repos at once; end users should stick with the pinned default.

## Troubleshooting

**"Neither docker nor podman is installed."** — install one of them, or
run remotely with `simulate(..., remote=True)`.

**"daemon is not responding"** — Docker Desktop needs to be running (on
macOS/Windows) or the `dockerd` service needs to be up (on Linux). Try
`docker info` from a shell to confirm.

**First run gets stuck on "Pulling …"** — the OpenDataDetector image is
~10 GB. Check your connection and let it finish; subsequent runs reuse
the cached image.

**Stage fails with `libOpenDataDetector.so not found`** — the production
checkout in the cache is corrupt. Re-clone with
`force_refresh=True`, or delete `~/.cache/colliderml/simulate/` and let
`simulate()` start fresh.

**"No stages were generated"** — the pinned production-repo ref is
missing config YAML files for the requested channel. Bump
`COLLIDERML_PRODUCTION_REF` to a newer ref, or file an issue.

## Next steps

- Run the CLI equivalent: [`colliderml simulate`](../library/cli.md#simulate)
- Look up the `simulate()` signature: [Simulate API](../library/simulate.md)
- Pipe the outputs into a benchmark: [Benchmark Tasks](./tasks.md)
- Offload heavy runs: [Remote Simulation](./remote-simulation.md)
