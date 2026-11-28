# Local Simulation

ColliderML can generate new events on your own machine with the same chain
that produced the released dataset:

1. MadGraph and Pythia8 generate the events (Pythia8 alone for the Pythia-only channels).
2. Geant4 simulates the Open Data Detector (ODD), through DDSim.
3. ACTS digitises the tracker, finds and fits tracks, and reconstructs primary vertices.
4. The calorimeter hits are digitised and Pandora builds clusters and particle-flow objects.
5. Everything is written as parquet tables with the layout of the released dataset.

Every stage runs in a container: steps 1 to 3 in the OpenDataDetector
software image, step 4 in the ColliderML reconstruction image. You need
Python and Docker or Podman, and no physics software of your own.

You can stop after step 3 (tracking only), which skips the reconstruction
image and is quicker to set up.

This page covers running the chain on your own workstation or on a node of
a shared cluster. To hand the work to the ColliderML backend instead, see
[Remote Simulation](./remote-simulation.md).

## When to run locally

| Use local simulation when | Use remote simulation when |
| :-- | :-- |
| You want tight iteration on a few events | You need >100 events with pile-up |
| You have Docker or Podman and the disk space below | You do not have a container runtime |
| You want offline reproducibility | You want someone else to pay the compute bill |
| You need particle flow (remote runs stop after tracking) | You only need tracks |

## What you need

| | |
| :-- | :-- |
| Software | Python 3.10 or newer, `git`, and Docker, Podman or `podman-hpc` (see [HPC systems](#on-an-hpc-system-with-podman-hpc)) |
| CPU | x86-64 with AVX2. ARM machines, Apple Silicon included, are not supported |
| Disk | About 40 GB with particle flow: the two images take 6 GB and 27 GB once unpacked, the Geant4 data 2 GB, and ten events without pile-up 0.4 GB. Tracking only: about 9 GB |
| Memory | In ten events without pile-up the largest process used 2.2 GB. Pile-up needs much more, see [below](#more-events-more-pile-up) |
| Network | For the first run, to download the images, the pipeline code and the Geant4 data. Later runs reuse the local copies |

The timings and memory numbers on this page come from one machine (a node of
NERSC Perlmutter, AMD EPYC 7763, with `podman-hpc`). They show what to
expect; they are not tested minimum requirements.

## Install

```bash
pip install "colliderml[sim]"
```

Particle flow needs `colliderml` 0.5.0 or later: `colliderml list-presets`
must show `ttbar-pflow-quick`.

## Run the full chain

```bash
colliderml simulate --preset ttbar-pflow-quick --output ./ttbar_pflow
```

or from Python:

```python
from colliderml.simulate import simulate

result = simulate(preset="ttbar-pflow-quick", output_dir="./ttbar_pflow")
print(result.parquet_dir)
```

This runs ten ttbar events without pile-up. The stages are printed as they
run, each followed by its own log (not shown here):

```
Running ttbar pipeline (10 events, pileup=0, particle flow) ...
  [1/9] MadGraph Init ...
  [1/9] MadGraph Init — done
  [2/9] MadGraph Generation ...
  [2/9] MadGraph Generation — done
  ...
  [9/9] Parquet Packaging ...
  [9/9] Parquet Packaging — done
Done. Output: /path/to/ttbar_pflow/runs/0
  digi_provenance.json (0.0 MB)
  edm4hep.root (41.9 MB)
  ...
Tables: /path/to/ttbar_pflow/parquet
  truth_tracks
  ...
```

The stages in between are Pythia Generation, Detector Simulation,
Digitization & Reconstruction, Truth Particle Tables, Calorimeter
Digitization & Particle Flow, and Calorimeter & Particle Flow Tables. The
logs of Detector Simulation and Digitization & Reconstruction contain
`Error in cling::AutoLoadingVisitor::InsertIntoAutoLoadingState` lines from
ROOT; in our runs they did not affect the output.

The output directory must be new: `simulate` stops if it already holds a run.

For tracking only, leave out particle flow:

```bash
colliderml simulate --preset ttbar-quick --output ./ttbar_tracks
```

This stops after stage 5 and does not need the reconstruction image.

## What happens on the first run

1. The container runtime is detected: `docker`, then `podman-hpc`, then
   `podman`. `--runtime` (or `runtime=`) chooses one.
2. You are asked before each image is pulled. In a non-interactive session
   the pull goes ahead.
3. The pipeline code (`colliderml-production`) is cloned into
   `~/.cache/colliderml/simulate/`, at the version this release of the
   library is pinned to.
4. The Geant4 physics data (about 2 GB) is downloaded into the same cache.
5. The stages run, and a `SimulationResult` with the output paths is returned.

Steps 2 to 4 happen once; later runs start with the stages.

## How long it takes

`ttbar-pflow-quick` uses two threads for Geant4 and ACTS and one Pandora
process. We ran it with the images already pulled and an empty cache, so the
whole-run time includes the clone of the pipeline code and the Geant4 data
download:

| Stage | Wall time | Largest process |
| :-- | --: | --: |
| MadGraph Init | 4 min 50 s | 1.2 GB |
| MadGraph Generation | 14 s | 0.4 GB |
| Pythia Generation | under 1 s | 0.3 GB |
| Detector Simulation | 2 min 18 s | 2.2 GB |
| Digitization & Reconstruction | 13 s | 1.7 GB |
| Truth Particle Tables | 2 s | 0.4 GB |
| Calorimeter Digitization & Particle Flow | 13 s | 2.0 GB |
| Calorimeter & Particle Flow Tables | 4 s | 0.4 GB |
| Parquet Packaging | 2 s | 0.7 GB |
| Whole run, first run in a new cache | 10 min 52 s | |

MadGraph Init compiles and integrates the NLO process. Its cost does not
depend on the number of events, so it is close to half of a ten-event run and a small
part of a large one. It uses every core it finds (128 on the test machine),
so expect it to take longer on a laptop.

Wall time and peak memory of every stage of your own run are written to
`<output>/pipeline_timing.csv`.

## What comes out

```
ttbar_pflow/
├── parquet/
│   ├── truth/
│   │   ├── particles/            # every particle of the event (generator and Geant4)
│   │   ├── generator_particles/  # generator history of the hard scatter
│   │   └── tracker_simhits/      # Geant4 tracker hits
│   └── reco/
│       ├── tracker_hits/         # digitised tracker clusters
│       ├── tracks/               # ACTS tracks (CKF, ambiguity-resolved)
│       ├── truth_tracks/         # tracks fitted on truth-assigned hits
│       ├── vertices/             # primary vertices (AMVF)
│       ├── calo_cells/           # digitised calorimeter cells
│       ├── calo_clusters/        # Pandora clusters
│       └── pfos/                 # Pandora particle-flow objects
├── runs/0/                       # intermediate files of every stage (HepMC, EDM4hep, ROOT)
├── configs/0/                    # the stage configs this run used
├── docker_test/                  # the compiled MadGraph process
└── pipeline_timing.csv
```

A tracking-only run has `particles`, `tracker_simhits`, `tracker_hits`,
`tracks` and `truth_tracks`, in `runs/0/`.

For ten events without pile-up the tables take 17 MB, the intermediate
files 173 MB and the MadGraph process 230 MB. Only `parquet/` is needed for
analysis.

Each table has one row per event. Every column except `event_id` is a list
with one entry per object. The tables are linked by ids within an event:

| From | Column | Points to |
| :-- | :-- | :-- |
| `tracks` | `hit_ids` | position in the event's `tracker_hits` lists (0-based) |
| `tracks` | `majority_particle_id` | `particles.particle_id` |
| `tracker_hits` | `particle_ids` | `particles.particle_id` |
| `pfos` | `track_ids` | `tracks.track_id` |
| `pfos` | `cluster_ids` | `calo_clusters.cluster_id` |
| `calo_clusters` | `cell_ids` | `calo_cells.cell_id` |
| `calo_cells` | `contrib_particle_ids` | `particles.particle_id` |

When you join:

- A track without a majority particle has `majority_particle_id = 2**64 - 1`.
- `generator_particles` is a subset of `particles` with the same
  `particle_id`: the particles of the hard scatter that the generator
  produced before the final state (the tops, the W bosons, the showers).
- `color_a` and `color_b` are 0 for every particle: the simulation output
  carries no colour-flow information.

`colliderml.load()` reads only the released dataset. Read your own tables with `pyarrow` or `pandas`. Here we count
the particle-flow objects and tracks per event:

```python
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

parquet = Path("ttbar_pflow/parquet")
pfos = pq.read_table(next((parquet / "reco/pfos").glob("*.parquet"))).to_pandas()
tracks = pd.read_parquet(next((parquet / "reco/tracks").glob("*.parquet")), columns=["event_id", "track_id"])

pfos["n_pfos"] = pfos["pfo_id"].str.len()
pfos["n_charged"] = pfos["charge"].apply(lambda q: int(np.count_nonzero(q)))
pfos["pt_sum"] = [np.hypot(px, py).sum() for px, py in zip(pfos["px"], pfos["py"])]
summary = pfos[["event_id", "n_pfos", "n_charged", "pt_sum"]].merge(
    tracks.assign(n_tracks=tracks["track_id"].str.len())[["event_id", "n_tracks"]], on="event_id")
print(summary)
```

Our test run printed (momenta in GeV):

```
   event_id  n_pfos  n_charged      pt_sum  n_tracks
0         0     199         28  286.083862        29
1         1     232         56  345.497101        56
2         2     236         46  314.021729        46
3         3     120         28  419.020050        28
4         4     204         47  350.671509        48
5         5     204         50  328.044983        51
6         6     276         66  531.719971        66
7         7     283         64  474.373596        64
8         8     206         38  389.224915        38
9         9     187         32  378.065125        33
```

A second run with the same seed can print different numbers: the Geant4
step with two threads is not always reproducible (see `seed` in the
[Simulate API](../library/simulate.md)).

`SimulationResult.tables()` returns the same files by table name.

## More events, more pile-up

```bash
colliderml simulate --channel ttbar --particle-flow \
    --events 100 --pileup 0 --threads 8 --pandora-processes 8 \
    --output ./ttbar_100
```

`--threads` sets the Geant4 and ACTS threads (2 or more). `--pandora-processes`
splits the Pandora stage into that many processes, each on a slice of the
events.

Pile-up is the mean number of extra proton-proton interactions overlaid on
each event. The released ColliderML pile-up samples use 200:

```bash
colliderml simulate --channel ttbar --particle-flow \
    --events 2 --pileup 200 --threads 2 --pandora-processes 2 \
    --output ./ttbar_pu200
```

We measured this command on the same machine:

| Stage | Wall time | Largest process |
| :-- | --: | --: |
| MadGraph Init, MadGraph Generation | 5 min 7 s | 1.0 GB |
| Pythia Generation (with the pile-up interactions) | 8 s | 0.3 GB |
| Detector Simulation | 11 min 49 s | 11.3 GB |
| Digitization & Reconstruction | 58 s | 7.5 GB |
| Truth Particle Tables | 6 s | 0.9 GB |
| Calorimeter Digitization & Particle Flow | 28 s | 5.4 GB per process |
| Calorimeter & Particle Flow Tables | 10 s | 1.1 GB |
| Parquet Packaging | 4 s | 0.9 GB |
| Whole run | 20 min 29 s | |

The run wrote 1.7 GB, of which the tables are 160 MB. These two events have
1,500 tracks and 5,800 particle-flow objects each. A Pandora process needed
up to 6.3 GiB for one pile-up 200 event in our tests of the reconstruction
image, so keep `--pandora-processes` times that below your free memory.

## Presets versus explicit parameters

Presets are named shorthand for common workloads:

```python
from colliderml.simulate import simulate

# Preset: channel, events, pile-up and particle flow in one name.
simulate(preset="ttbar-pflow-quick")

# Explicit parameters.
simulate(channel="higgs_portal", events=10, pileup=10, particle_flow=True)

# A preset as a starting point, with more pile-up:
simulate(preset="ttbar-pflow-quick", pileup=40, output_dir="./ttbar_pu40")
```

Explicit arguments win over the preset. `colliderml list-presets` lists them.

## On an HPC system with podman-hpc

Where the runtime is `podman-hpc` (NERSC Perlmutter, for example):

1. Pull the images on a login node before you start. `podman-hpc pull` also
   makes an image visible to the compute nodes; for an image you built or
   loaded yourself, run `podman-hpc migrate <image>`. The image names are
   the `sim_image` and `reco_image` values printed by
   `python -c "from colliderml.simulate import load_pipeline_config as c; print(c())"`.
2. Run the simulation on a compute node, one simulation per job step. Two
   job steps that started containers of the same image at the same time
   failed in our test.
3. Put the cache on a file system that handles many small files well. The
   Geant4 data is about 43,000 files, and copying it to a parallel file
   system took 28 minutes in our test. Set `COLLIDERML_CACHE` to a
   node-local or scratch directory.

## Cache layout

Everything is under `$COLLIDERML_CACHE` (default `~/.cache/colliderml/`):

```
~/.cache/colliderml/simulate/
├── colliderml-production/          # pinned clone of the pipeline code
└── container_cache/
    └── g4data/                     # Geant4 physics data (~2 GB)
```

You can delete the tree at any time; the next run fills it again. To fetch
only the pipeline code again, call
`colliderml.simulate.docker.clone_colliderml_production(force_refresh=True)`.

## Overriding the pinned pipeline versions

The `colliderml-production` version and the two container images are pinned
together in
[`colliderml/simulate/pipeline_config.yaml`](../library/simulate.md#pinned-pipeline-versions).
Each has an environment variable override: `COLLIDERML_PRODUCTION_REF`,
`COLLIDERML_SIM_IMAGE` and `COLLIDERML_RECO_IMAGE`. To run a branch of the
pipeline code without editing the library:

```bash
export COLLIDERML_PRODUCTION_REF=my-feature-branch
colliderml simulate --preset ttbar-quick --output ./test
```

This is meant for pipeline developers; for anything you want to compare with
the released data, keep the pinned versions.

## Known limitations

| Limitation | What it means for you |
| :-- | :-- |
| No vertex-to-track links with the public sim image | `reco/vertices` has the vertex positions, covariances and track counts. The columns `track_ids`, `track_weights`, `truth_interaction` and `truth_interaction_dz` need an ACTS writer that the public image does not have yet |
| Simpler ttbar process than the released samples | This pipeline generates `p p > t t~` at NLO without extra jets; the released samples use FxFx-merged multi-jet matrix elements. Our ten events have fewer tracks and particle-flow objects per event than the released-format sample (46 and 213 against 62 and 268) |
| x86-64 only | There are no ARM builds of either image |
| One run per output directory, one file per table | For more than a few hundred events with pile-up, make several calls with different `seed` and `output_dir` |
| Two threads at least | With one thread the Geant4 step of the current sim image writes every event twice, so `--threads 1` is refused |
| `truth/particles` at pile-up | Every particle of the event is in the table, with 30 columns, at any pile-up. The released pile-up 200 sample has the 18-column Release 1 particle table there |
| No remote particle flow | `remote=True` stops after tracking |
| Docker writes files as root | With Docker (not Podman) the output belongs to root: `sudo chown -R $USER ./ttbar_pflow` |

## Troubleshooting

| Message | What to do |
| :-- | :-- |
| `No container runtime on PATH` | Install Docker or Podman, or run remotely with `simulate(..., remote=True)` |
| `daemon is not responding` | Start Docker Desktop on macOS and Windows, or the `dockerd` service on Linux; `docker info` shows whether it runs |
| The first run stays at `Pulling ...` | The images are 6 GB and 27 GB. Let the pull finish; later runs reuse them |
| `... has no scripts/cli/docker_stage_env.sh`, or `... not found: the colliderml-production checkout at ...` | The cached pipeline code does not match this version of the library. Delete `~/.cache/colliderml/simulate/colliderml-production/` and run again |
| `... already holds a run` | `simulate()` does not write into an output directory that has results in it. Choose a new `output_dir` |

If a stage fails, send us the stage name, its error output and
`<output>/pipeline_timing.csv` if it exists.

## Next steps

- All `simulate()` arguments: [Simulate API](../library/simulate.md)
- The CLI equivalent: [`colliderml simulate`](../library/cli.md#simulate-run-the-pipeline)
- Use the outputs in a benchmark: [Benchmark Tasks](./tasks.md)
- Run on the ColliderML backend: [Remote Simulation](./remote-simulation.md)
