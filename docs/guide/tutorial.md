# Platform tutorial

A from-zero walkthrough of the ColliderML platform from a
researcher's perspective. By the end you'll have **downloaded a
tracker-hits sample from HuggingFace**, **simulated events locally
with the official container**, **submitted a simulation request to
the SaaS backend**, **scored a tracking submission against the
leaderboard**, and **published your "model" as an HF repo that the
model-zoo Space indexes**.

There's an executable version of this tutorial in
[`notebooks/tutorial.ipynb`](https://github.com/OpenDataDetector/ColliderML/blob/main/notebooks/tutorial.ipynb).
Every code block below comes straight out of that notebook, so you
can read along here and run any piece you like without leaving the
docs.

Each chapter ends with a **"what just happened"** section that
unpacks the architecture under the hood, so the tutorial also
doubles as a conceptual map.

## Prerequisites

| | |
|---|---|
| **Python** | 3.10 or newer. |
| **Install** | `pip install --pre 'colliderml[all]'`  (pre-release until `0.4.0` final; drop `--pre` once it ships). |
| **Disk headroom** | ~2 GB for the Chapter 1 dataset slice. |
| **Chapter 2** | Docker or Podman on `$PATH`, plus ~12 GB for the pipeline container and Geant4 datasets (one-time download). |
| **Chapters 3–4** | A running backend. Easiest: clone the `colliderml-production` repo and run `scripts/setup_tutorial_env.sh path/to/colliderml-production/backend`. |
| **Chapters 3–5** | A HuggingFace account and a token (`huggingface-cli login`). |

If you're on a machine with a tight `$HOME` quota (e.g. NERSC
login nodes), redirect caches before anything else:

```bash
export COLLIDERML_DATA_DIR=/scratch/$USER/colliderml-cache
export HF_HOME=/scratch/$USER/hf-cache
```

## How the pieces fit together

```
┌──────────────┐   HTTP    ┌────────────────┐   SFAPI   ┌───────────┐
│ your laptop  │──────────▶│ backend (FastAPI│──────────▶│ Perlmutter│
│ colliderml.* │           │  +  Postgres)  │           │  (Slurm)  │
└──────┬───────┘           └────────┬───────┘           └─────┬─────┘
       │                            │                         │
       │  HuggingFace Hub           │   uploads artefacts     │
       ▼                            ▼                         ▼
┌──────────────┐            ┌──────────────────┐      ┌───────────┐
│ datasets,    │            │ HF dataset repos │◀─────│ pipeline  │
│ model zoo,   │            │  (simulated      │      │  output   │
│ Spaces       │            │   events)        │      └───────────┘
└──────────────┘            └──────────────────┘
```

Chapters 1 and 5 use only the HuggingFace side (leftmost column).
Chapter 2 is your laptop alone. Chapters 3 and 4 exercise the full
stack.

---

## Chapter 1 — Loading data from HuggingFace

The canonical ColliderML datasets live at
[`CERN/ColliderML-Release-1`](https://huggingface.co/datasets/CERN/ColliderML-Release-1).
Each config is named `<channel>_<pileup>`, e.g. `ttbar_pu0` or
`higgs_portal_pu200`. The library's `load()` handles cache-aware
download and Polars loading in one call.

```python
from colliderml.core.hf_download import discover_remote_configs
import colliderml

configs = discover_remote_configs("CERN/ColliderML-Release-1")
print(f"{len(configs)} configs available; first six:", configs[:6])

frames = colliderml.load(
    "ttbar_pu0",
    tables=["tracker_hits", "particles"],
    max_events=200,
)
row = frames["tracker_hits"].row(0, named=True)
print("event_id:", row["event_id"], "number of hits:", len(row["hit_id"]))
```

### What just happened

- `discover_remote_configs` hit the HF dataset repo and extracted unique
  `data/<config>/...` prefixes — the config list is derived from
  what's on disk in the repo, not a hardcoded manifest.
- `colliderml.load(...)` resolved `"ttbar_pu0"` into the set of
  parquet shards for the requested tables, downloaded any missing
  shards into `$COLLIDERML_DATA_DIR` (default `~/.cache/colliderml`),
  and loaded them with the Polars-backed loader.
- `max_events=200` is an **in-memory slice**, not a download limit —
  the full shard has to be fetched once, then we take the first 200
  rows. For a tiny sample this is wasteful; for real workflows the
  cache pays for itself from the second call onward.

The nested schema (one row per event, columns as lists across hits)
is central to the library's performance story — Polars can lazily
scan these shards without exploding them into flat tables.

---

## Chapter 2 — Run the pipeline yourself: local simulation

ColliderML ships a container
(`ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0`)
that bundles MadGraph, Pythia, Geant4 + ddsim, and ACTS. The
`colliderml.simulate` subpackage drives the full pipeline —
hard-scatter generation → parton shower → detector simulation →
track reconstruction — in a single call.

For this chapter we use the `single-muon` preset, the fastest (~5
minutes on a modern laptop).

```python
from colliderml.simulate import load_presets
import colliderml

for name, cfg in load_presets().items():
    print(f"  {name:24}  channel={cfg['channel']:14} events={cfg['events']:>4} pileup={cfg['pileup']}")

result = colliderml.simulate(preset="single-muon", quiet=True)
print("run directory:", result.run_dir)
print("stages run:   ", [s.name for s in result.stages])
```

### What just happened

On first call, `colliderml.simulate` did three things before running
anything:

1. **Detected the runtime** — `docker` preferred, `podman` fallback.
2. **Cloned `colliderml-production@pipeline-v0.1.0`** into `.cache/`
   for the pipeline scripts. This auto-clone pattern mirrors the one
   already used for the ODD geometry and the MG5 ↔ Pythia8
   interface.
3. **Built the ODD detector geometry and downloaded Geant4
   datasets** into the cache (both one-time, ~10 minutes combined).

Then for each stage (Pythia → DDSim → Digi + Reco → Parquet) it
started a container with the pipeline script for that stage, mounted
the run directory, and ran to completion. The `SimulationResult` you
get back records which stages ran, how long each took, and where the
outputs are.

The **same** parquet output format that Chapter 1 downloaded from HF
is what this pipeline writes locally, so you can round-trip a
simulated sample through `colliderml.load()` without caring about
the source.

---

## Chapter 3 — Scale up: Simulation as a Service

Single-muon runs in minutes on your laptop. A full `ttbar + PU=200`
run takes a day of CPU time and tens of GB of output — not
something to babysit locally. The SaaS backend accepts a JSON
request, queues a Slurm job on NERSC Perlmutter via the SFAPI,
polls it to completion, and uploads the artefacts to an HF dataset
repo under your account.

From the user's side it's three calls: `submit`, `wait_for`, then
download the resulting HF repo like any other dataset.

```python
from colliderml import remote

print(f"You have {remote.balance():.0f} credits available.")

sub = remote.submit(channel="ttbar", events=10, pileup=10)
print(f"request_id: {sub.request_id}, state: {sub.state}, credits: {sub.credits_charged}")

final = remote.wait_for(sub.request_id, poll_interval=1, timeout=120)
print("final state:", final.state, "output repo:", final.output_hf_repo)
```

An identical second `submit` hits the backend's dedup cache:

```python
dup = remote.submit(channel="ttbar", events=10, pileup=10)
assert dup.request_id == sub.request_id  # same request
assert dup.credits_charged == 0          # no double-charge
```

### What just happened

Inside the backend, `POST /v1/simulate` did the following:

1. **Hashed the request config** and checked the dedup table — if a
   completed request with the same hash exists within the last 7
   days, returns it verbatim (`credits_charged=0`).
2. **Ran credit and abuse checks.** The kill switch
   (`POST /admin/freeze`) short-circuits here with a 503 if an
   operator has tripped it.
3. **Inserted a row** into `simulation_requests` with state `queued`.
4. **Handed off to `SFAPIRunner.submit`:**
   - **With real NERSC credentials** (`SFAPI_CLIENT_ID`,
     `SFAPI_CLIENT_SECRET` set on the backend): renders
     `app/sbatch_template.sh.j2`, uploads it to `/pscratch/sd/.../colliderml/`,
     POSTs to the SFAPI `/compute/jobs` endpoint, spawns a polling
     task that hits `/compute/jobs/{id}` every 60 s.
   - **In mock mode** (default for local dev): spawns a task that
     marks the request completed after 2 seconds.
5. **Returned** the `(request_id, state=queued, credits_charged, …)`
   tuple.

`remote.wait_for` polls `GET /v1/requests/{id}` every
`poll_interval` seconds until the state is terminal (`completed` /
`failed` / `cancelled`). On success you get back an `output_hf_repo`
URL — a fresh HF dataset repo under your account containing the
parquet shards, loadable via the same `colliderml.load()` you used
in Chapter 1.

::: tip Mock vs real SFAPI
The backend runs in mock mode unless `SFAPI_CLIENT_ID` and
`SFAPI_CLIENT_SECRET` are set. Mock mode reproduces the full
state-machine (queued → running → completed) in about 2 seconds
so the API shape is exact — you just don't get real pipeline
output. See the internal [SFAPI runner docs](../internal/sfapi.md)
for the production setup.
:::

---

## Chapter 4 — Benchmark a tracking algorithm

The `tracking` task asks you to reconstruct particle tracks from
detector hits in `ttbar_pu200` events. The dataset ships *candidate*
tracker hits; your job is to group them into track assignments. A
submission is a parquet file with three columns: `event_id`,
`hit_id`, `track_id`.

### The tracking efficiency definition

The primary metric is **TrackML weighted efficiency**, scored by the
double-majority rule on a held-out eval split (events 90 000–99 999):

> A reconstructed track is **correct** if
> **one truth particle owns ≥50% of the track's hits**
> AND
> **that particle contributes ≥50% of its own hits to the track**.
>
> Efficiency = sum of hit weights in correct tracks ÷ total hit weight.

Two companion metrics unpack the failure modes:

- **Fake rate** — fraction of tracks failing the first half of the
  rule (no single particle dominates). Usually means "this track is
  made of hits from multiple particles."
- **Duplicate rate** — fraction of truth particles matched to more
  than one reconstructed track. Usually means "this particle got
  split across fragments."
- **Physics efficiency (pT > 1 GeV)** — fraction of high-pT primary
  particles with *any* reconstructed track. A physics-centric
  complement to the TrackML metric.

All four are scored **server-side** on truth that's never shipped to
clients. You can reproduce the local numbers with the helpers in
`colliderml.tasks.tracking.metrics`, but the leaderboard number is
authoritative.

### A pedagogical submission

The library ships two oracle baselines under
`colliderml.tasks.tracking.baselines.oracle`:

- `perfect_oracle_predictions(truth_hits)` — assigns each hit to its
  truth particle. Scores ~1.0 on everything by construction. The
  metric's upper bound.
- `noised_oracle_predictions(truth_hits, *, split_fraction, merge_fraction, seed)` —
  starts from the perfect oracle and deliberately degrades it.
  `split_fraction` breaks tracks into two asymmetric pieces (attacks
  the *"track owns majority of particle's hits"* half of the rule);
  `merge_fraction` unifies track pairs within an event (attacks the
  *"one particle owns majority of track's hits"* half).

```python
import colliderml
from colliderml.tasks.tracking.baselines import (
    noised_oracle_predictions,
    perfect_oracle_predictions,
)
from colliderml.tasks.tracking.metrics import trackml_weighted_efficiency

tracking = colliderml.tasks.get("tracking")
truth = tracking.load(tables=["tracker_hits"], event_range=(90_000, 90_050))["tracker_hits"]

perfect = perfect_oracle_predictions(truth)
noised = noised_oracle_predictions(truth, split_fraction=0.1, merge_fraction=0.1, seed=42)

print("perfect oracle eff:", trackml_weighted_efficiency(perfect, truth))
print("noised oracle eff: ", trackml_weighted_efficiency(noised, truth))
```

Submit for real scoring on the leaderboard:

```python
result = colliderml.tasks.submit("tracking", noised)
print("server scores: ", result["scores"])
print("credits earned:", result["credits_earned"])
```

### What just happened

The backend's `POST /v1/benchmark/tracking/submit` took the parquet
bytes, loaded them with pyarrow, and ran the same
`trackml_weighted_efficiency` / `fake_rate` / `duplicate_rate` /
`physics_eff_pt1` code you just ran locally — but on the **full
10 000-event eval split** backed by a truth table that never leaves
the server. If the result beats the current best on any metric with
`higher_is_better=True`, credits are written to the user's ledger.
The leaderboard Space (`spaces/leaderboard/`) polls
`/v1/leaderboard/tracking` to render the public table.

**Why an oracle baseline is a useful submission.** It's not a
tracking algorithm — it's a calibration point. Perfect oracle scores
≈1 by construction, so it tells you the metric's true ceiling.
Noised oracle with small fractions shows how quickly the TrackML
score punishes each failure mode:

- `split_fraction`: each split turns one correct track into (one
  correct + one fake). The efficiency drop is roughly linear in
  `split_fraction` because the remaining ~70% of the original track
  still passes the double-majority rule.
- `merge_fraction`: harsher — a single merge sacrifices **both**
  contributing tracks, so the drop is roughly quadratic near
  zero.

When you write a real tracker (DBSCAN on (η, φ, r), a GNN, a CKF),
this is the yardstick.

---

## Chapter 5 — Share your model: the model zoo

The ColliderML model zoo is a thin HuggingFace Hub filter: any model
on the Hub tagged `colliderml` shows up in
[`spaces/model-zoo`](https://huggingface.co/spaces/OpenDataDetector/colliderml-model-zoo).
There's no backend registry, no approval process — the `tag` is the
contract. Publishing is a standard `huggingface_hub.create_repo` +
`upload_folder`.

```python
import pathlib, textwrap
from huggingface_hub import HfApi, create_repo, upload_folder

hf_user = HfApi().whoami()["name"]
model_dir = pathlib.Path("/tmp/colliderml-tutorial-model")
model_dir.mkdir(exist_ok=True)

(model_dir / "config.json").write_text(
    '{"kind": "noised-oracle", "split_fraction": 0.1, "merge_fraction": 0.1}\n'
)
(model_dir / "README.md").write_text(textwrap.dedent("""
    ---
    tags:
      - colliderml
      - tracking
    license: mit
    ---
    # Noised-oracle tracker (tutorial)

    Pedagogical baseline for the ColliderML tracking task. Not a real
    tracker — serves as a calibration point for evaluating real
    submissions.
"""))

repo_id = f"{hf_user}/colliderml-tutorial-tracker"
create_repo(repo_id, exist_ok=True)
upload_folder(folder_path=str(model_dir), repo_id=repo_id)
print("published:", f"https://huggingface.co/{repo_id}")
```

Refresh the model-zoo Space — your repo now appears.

### What just happened

You created an HF model repo under your namespace with a README that
has `tags: [colliderml, tracking]` in its YAML frontmatter. The
model-zoo Space runs

```python
huggingface_hub.list_models(filter="colliderml")
```

on every page load and renders whatever comes back. Your repo is now
one of the results.

**Tradeoff.** The simplicity is deliberate: no backend registry
means no moderation, no provenance linking, no guaranteed
compatibility with evaluation tooling. The contract is just the
tag. A future iteration could add a `/v1/models` backend endpoint
that stitches together a model card with its leaderboard score(s),
or runs the model's inference against the eval set to produce a
deterministic score — but that's a later conversation.

---

## Chapter 6 — Recap and next steps

You just drove every public surface of the ColliderML platform:

| Chapter | Surface |
|---|---|
| 1 | `colliderml.load()`, `colliderml.core.hf_download` |
| 2 | `colliderml.simulate()`, preset catalogue, container auto-clone |
| 3 | `colliderml.remote.{submit, wait_for, balance}`, dedup cache |
| 4 | `colliderml.tasks.{get, evaluate, submit}`, tracking metrics, oracle baselines |
| 5 | `huggingface_hub.create_repo`, the `tags: [colliderml]` contract |

### Reference card

```python
import colliderml
from colliderml import remote
from colliderml.tasks.tracking.baselines import noised_oracle_predictions
from huggingface_hub import create_repo, upload_folder

# 1. Load data
frames = colliderml.load("ttbar_pu0", tables=["tracker_hits"], max_events=200)

# 2. Run the pipeline locally
result = colliderml.simulate(preset="single-muon")

# 3. Submit a remote simulation
sub = remote.submit(channel="ttbar", events=10_000, pileup=200)
final = remote.wait_for(sub.request_id)

# 4. Score a tracking submission
preds = noised_oracle_predictions(truth_hits, split_fraction=0.05)
result = colliderml.tasks.submit("tracking", preds)

# 5. Publish a model (standard HF)
create_repo(f"{user}/my-tracker", exist_ok=True)
upload_folder(folder_path="...", repo_id=f"{user}/my-tracker")
```

### What's not covered

- **Other tasks.** `jets`, `anomaly`, `tracking_latency`,
  `tracking_small`, `data_loading` all ship with reference baselines
  and the same `colliderml.tasks.{get, evaluate, submit}` surface.
- **Writing your own task.** Subclass
  `colliderml.tasks.BenchmarkTask`, register with `@register`, ship
  reference baselines. An operator adds the new task to the
  backend's task list.
- **Running the platform yourself.** The
  [operator docs](../internal/overview.md) (unlisted, reachable by
  URL) cover backend deployment, SFAPI credentials, admin Space
  operations, and the container image build checklist.
- **Webhooks and multi-node.** The backend supports posting a
  webhook when a request completes and has multi-node sbatch
  templates for the `*-benchmark` presets — useful for CI/CD-style
  workflows.

Issues, questions, ideas:
[github.com/OpenDataDetector/ColliderML/issues](https://github.com/OpenDataDetector/ColliderML/issues).
