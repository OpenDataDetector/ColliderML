# Simulate and reconstruct ttbar with particle flow

This tutorial runs the full ColliderML chain on your own machine, for ten
ttbar events without pile-up:

1. MadGraph and Pythia8 generate the events.
2. Geant4 simulates the Open Data Detector (ODD).
3. ACTS digitises the tracker, finds tracks and fits vertices.
4. The calorimeter hits are digitised and Pandora builds particle-flow objects.
5. Everything is written as parquet tables with the layout of the released dataset.

Every stage runs in a container, so there is no physics software to install.

## What you need

| | |
| :-- | :-- |
| Software | Python 3.10 or newer, `git`, and Docker or Podman (`podman-hpc` on HPC systems, see [below](#on-an-hpc-system-with-podman-hpc)) |
| CPU | x86-64 with AVX2. ARM machines, Apple Silicon included, are not supported |
| Disk | About 40 GB: the two images are 6 GB and 27 GB once unpacked, the Geant4 data 2 GB, and ten events without pile-up 0.4 GB |
| Memory | In the ten-event run the processes together used at most 2.3 GB (resident memory, sampled every 2 s). Pile-up needs much more, see [below](#more-events-more-pile-up) |
| Network | For the first run: the images, the pipeline code and the Geant4 data are downloaded. Later runs still contact GitHub to check the pipeline version, and use the local copy when offline |

The numbers on this page come from one test on a single machine (a node of
NERSC Perlmutter, AMD EPYC 7763, with `podman-hpc`), run through the
pipeline's own script with the settings the commands below use. They show
what to expect and are not tested minimum requirements. The chain has not yet
been run with Docker or plain Podman; please report what you find.

## Install

```bash
pip install "colliderml[sim]"
```

Particle flow needs `colliderml` 0.5.0 or later (`colliderml list-presets`
must show `ttbar-pflow-quick`).

## Run

```bash
colliderml simulate --preset ttbar-pflow-quick --output ./ttbar_pflow
```

or from Python:

```python
from colliderml.simulate import simulate

result = simulate(preset="ttbar-pflow-quick", output_dir="./ttbar_pflow")
print(result.parquet_dir)
```

The first run asks before it pulls the two container images. It then clones
the pipeline code into `~/.cache/colliderml/simulate/` and downloads the
Geant4 datasets there. Later runs reuse all of it.

The output directory must be new: `simulate` stops if `./ttbar_pflow` already
holds a run.

The stages are printed as they run:

```
Running ttbar pipeline (10 events, pileup=0, particle flow) ...
  [1/9] MadGraph Init ...
  [2/9] MadGraph Generation ...
  [3/9] Pythia Generation ...
  [4/9] Detector Simulation ...
  [5/9] Digitization & Reconstruction ...
  [6/9] Truth Particle Tables ...
  [7/9] Calorimeter Digitization & Particle Flow ...
  [8/9] Calorimeter & Particle Flow Tables ...
  [9/9] Parquet Packaging ...
Done.
```

## How long it takes

The preset runs ten events without pile-up, with two threads for Geant4 and
ACTS and one Pandora process. With the images and the Geant4 data already in
place:

| Stage | Wall time | Largest process |
| :-- | --: | --: |
| MadGraph Init | 6 min 36 s | 1.0 GB |
| MadGraph Generation | 15 s | 0.4 GB |
| Pythia Generation | 1 s | 0.3 GB |
| Detector Simulation | 2 min 23 s | 2.2 GB |
| Digitization & Reconstruction | 17 s | 1.7 GB |
| Truth Particle Tables | 3 s | 0.5 GB |
| Calorimeter Digitization & Particle Flow | 38 s | 1.9 GB |
| Calorimeter & Particle Flow Tables | 7 s | 0.5 GB |
| Parquet Packaging | 3 s | 0.8 GB |
| Whole run, container start-up included | 12 min 24 s | |

MadGraph Init compiles the NLO matrix elements and integrates the process.
Its cost doesn't depend on the number of events, so it is half of a
ten-event run and a small part of a large one. MadGraph uses every core it
finds (128 on the test machine), so expect this stage to take longer on a
laptop.

On the first run, add the time to pull the two images, which take 33 GB on
disk once unpacked (the transfer is smaller). The Geant4 data is another
2 GB, in about 43,000 files.

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
│       ├── vertices/             # AMVF primary vertices
│       ├── calo_cells/           # digitised calorimeter cells
│       ├── calo_clusters/        # Pandora clusters
│       └── pfos/                 # Pandora particle-flow objects
├── runs/0/                       # intermediate files of every stage (HepMC, EDM4hep, ROOT)
├── configs/0/                    # the stage configs this run used
├── docker_test/                  # the compiled MadGraph process
└── pipeline_timing.csv
```

For these ten events the tables take 17 MB, the intermediate files in
`runs/0/` 173 MB and the MadGraph process 230 MB. Only `parquet/` is needed
for analysis.

Each table is one parquet file with one row per event. Every column except
`event_id` is a list with one entry per object. The tables are linked by ids
within an event:

| From | Column | Points to |
| :-- | :-- | :-- |
| `tracks` | `hit_ids` | position in the event's `tracker_hits` lists (0-based) |
| `tracks` | `majority_particle_id` | `particles.particle_id` |
| `tracker_hits` | `particle_ids` | `particles.particle_id` |
| `pfos` | `track_ids` | `tracks.track_id` |
| `pfos` | `cluster_ids` | `calo_clusters.cluster_id` |
| `calo_clusters` | `cell_ids` | `calo_cells.cell_id` |
| `calo_cells` | `contrib_particle_ids` | `particles.particle_id` |

Two things to know when you join:

- A track without a majority particle has `majority_particle_id = 2**64 - 1`.
- `generator_particles` is a subset of `particles` with the same
  `particle_id`: the particles of the hard scatter that the generator
  produced before the final state (the tops, the W bosons, the showers).
- `color_a` and `color_b` are 0 for every particle: the simulation output
  of this pipeline carries no colour-flow information.

Here we read the particle-flow objects and the tracks, and count the charged
objects per event:

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

Expect your numbers to differ in detail. We have not tested whether a
second run with the same seed gives the same events.

`SimulationResult.tables()` returns the same files by table name, if you
would rather not build the paths by hand.

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

Pile-up 200 costs far more per event. We measured this command (two events,
two threads, two Pandora processes) on the same machine:

| Stage | Wall time | Largest process |
| :-- | --: | --: |
| MadGraph Init, MadGraph Generation | 8 min 10 s | 1.0 GB |
| Pythia Generation (with the pile-up interactions) | 19 s | 0.3 GB |
| Detector Simulation | 12 min 48 s | 11.3 GB |
| Digitization & Reconstruction | 1 min 37 s | 7.4 GB |
| Truth Particle Tables | 9 s | 1.0 GB |
| Calorimeter Digitization & Particle Flow | 12 min 16 s | 5.6 GB per process |
| Calorimeter & Particle Flow Tables | 13 s | 1.3 GB |
| Parquet Packaging | 7 s | 1.0 GB |
| Whole run | 37 min 49 s | |

The processes of this run together used at most 11.5 GB, during the
detector simulation. The run wrote 1.8 GB, of which the tables are 160 MB.
These two events have 1,500 tracks and 5,800 particle-flow objects each.

The larger of the two Pandora processes used 5.6 GB, for one event. Keep
`--pandora-processes` times that below your free memory.

## On an HPC system with podman-hpc

Where the runtime is `podman-hpc` (NERSC Perlmutter, for example), a few
things differ from a laptop:

1. Pull the images on a login node before you start. `podman-hpc pull` also
   makes an image visible to the compute nodes; for an image you built or
   loaded yourself, run `podman-hpc migrate <image>`. The image names are
   the `sim_image` and `reco_image` values printed by
   `python -c "from colliderml.simulate import load_pipeline_config as c; print(c())"`.
2. The runtime is found automatically (`docker`, then `podman-hpc`, then
   `podman`). Add `--runtime podman-hpc` to force it.
3. Put the cache on a file system that handles many small files well. The
   Geant4 data is about 43,000 files, and copying it to a parallel file
   system took 28 minutes in our test. Set `COLLIDERML_CACHE` to a
   node-local or scratch directory.
4. Run one simulation per job step. Two job steps that start containers of
   the same image at the same time failed in our test.

Run the simulation on a compute node.

## Known limitations

| Limitation | What it means for you |
| :-- | :-- |
| No vertex-to-track links with the public sim image | `reco/vertices` has the vertex positions, covariances and track counts. The columns `track_ids`, `track_weights`, `truth_interaction` and `truth_interaction_dz` need an ACTS writer that the public image does not have yet. With an image that has it (`--image`), the pipeline writes them |
| Simpler ttbar process than the released samples | This pipeline generates `p p > t t~` at NLO without extra jets. The released samples use FxFx-merged multi-jet matrix elements. Our ten events have fewer tracks and particle-flow objects per event than the released-format pilot sample (46 and 215 against 62 and 268). We have not shown that the process alone explains the difference |
| x86-64 only | There are no ARM builds of either image |
| One run per output directory, one file per table | `simulate()` writes all events into a single run. For more than a few hundred events with pile-up, make several calls with different `seed` and `output_dir` |
| Two threads at least | With one thread the Geant4 step of the current sim image writes every event twice, so `--threads 1` is refused |
| `truth/particles` at pile-up | Every particle of the event is in the table, with 30 columns, at any pile-up. The released pile-up 200 pilot sample has the 18-column Release 1 particle table there |
| No remote particle flow | `remote=True` stops after tracking |
| Docker writes files as root | With Docker (Podman is not affected) the output belongs to root. Change the owner afterwards: `sudo chown -R $USER ./ttbar_pflow` |

## Next steps

- All `simulate()` arguments: [Simulate API](../library/simulate.md)
- Stages, caching and the pinned pipeline versions: [Local Simulation](./simulation.md)
