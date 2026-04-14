# Benchmark Tasks

ColliderML ships a small catalogue of benchmark tasks so that every
model can be compared on an equal footing. Each task defines:

- a **dataset** to draw events from (e.g. ``ttbar_pu200``)
- a **held-out eval range** (event IDs) that participants must not train on
- the **inputs** a model is allowed to see
- a set of **metrics** with directions (higher-is-better or lower-is-better)

This page is the conceptual tour. For the complete API, see
[`colliderml.tasks`](../library/tasks.md).

## The catalogue at a glance

| Task name | Dataset | Inputs | Primary metric | What it measures |
| :-- | :-- | :-- | :-- | :-- |
| `tracking` | `ttbar_pu200` | `tracker_hits` | `trackml_eff` | Track reconstruction quality |
| `jets` | `ttbar_pu0` | `tracks`, `calo_hits` | `btag_auc` | Jet flavour tagging (b / c / light) |
| `anomaly` | mixed SM + BSM | `tracks`, `calo_hits` | `auroc` | BSM event detection against SM background |
| `tracking_latency` | `ttbar_pu200` | `tracker_hits` | `events_per_sec` | Inference throughput for tracking |
| `tracking_small` | `ttbar_pu200` | `tracker_hits` | `trackml_eff` per tier | Tracking under a parameter budget |
| `data_loading` | `ttbar_pu200` | `tracker_hits` | `events_per_sec_local` | Raw data loading throughput |

Get the list programmatically at any time:

```python
import colliderml.tasks
print(colliderml.tasks.list_tasks())
```

## Installing the task runner

The task registry is in the base install, but the reference baselines
(BDT for jets, IsolationForest for anomaly detection) use scikit-learn.
Install the optional extra to get them:

```bash
pip install "colliderml[tasks]"
```

You can evaluate your own predictions against a task without the
`tasks` extra — you only need it if you want to run the shipped
baselines verbatim.

## The scoring contract

Every task exposes three methods on its instance:

1. **`load_eval_inputs()`** — returns the held-out eval data as a dict
   of pyarrow Tables. Users call this to build their inference
   pipeline; the backend calls it to run baselines on demand.
2. **`validate_predictions(preds)`** — raises `ValueError` if your
   submission has the wrong columns, sums, coverage, etc. Run this
   locally before uploading anything.
3. **`score(preds)`** — returns ``{metric_name: value}``.

The top-level helper `colliderml.tasks.evaluate` chains the last two:

```python
import colliderml.tasks

scores = colliderml.tasks.evaluate("tracking", "my_tracks.parquet")
print(scores)
# {'trackml_eff': 0.87, 'fake_rate': 0.03, 'dup_rate': 0.04, 'physics_eff_pt1': 0.92}
```

## Eval splits — "what can I train on?"

Each task withholds a range of event IDs (``eval_event_range``) for
scoring. A valid submission covers *at least half* of that range —
partial submissions are allowed so users can iterate quickly, but full
coverage is required to win a credit award on the leaderboard.

Use the task's own `load()` helper to pull just the eval events:

```python
task = colliderml.tasks.get("tracking")
hits = task.load(tables=["tracker_hits"], event_range=task.eval_event_range)
# hits["tracker_hits"] is a pyarrow Table with events 90_000..100_000
```

For training, load the complement — events *outside* the eval range.
The loader is explicit enough that "this data is safe to train on" is
easy to enforce in your own pipeline.

## Submitting to the leaderboard

Local evaluation gives you a preview. To earn credits and appear on
the public leaderboard, submit to the backend:

```python
import colliderml.tasks

result = colliderml.tasks.submit("tracking", "my_tracks.parquet")
print(result["scores"])         # backend's canonical scores
print(result["credits_earned"]) # non-zero if you beat a previous best
```

Submission requires the `remote` extra (for `requests`) and a
HuggingFace token — see
[Remote Simulation](./remote-simulation.md#prerequisites) for auth
setup. The backend re-scores your submission against its held-out
truth set (which is *not* bundled with the library, so you can't
overfit on it) and awards credits on new bests.

## Running the reference baselines

Each task ships a simple reference baseline you can run as a module:

```bash
# CKF baseline for the tracking task — converts a pipeline run's
# tracksummary_ambi.root into the task's expected parquet schema.
python -m colliderml.tasks.tracking.baselines.ckf \
    --run-dir ./colliderml_output/ttbar_pu200_100evt/runs/0 \
    --output ckf_preds.parquet

# GBDT baseline for jet classification
python -m colliderml.tasks.jets.baselines.bdt \
    --channel ttbar_pu0 --max-events 1000 --output bdt_preds.parquet

# IsolationForest baseline for anomaly detection
python -m colliderml.tasks.anomaly.baselines.isoforest \
    --output iso_preds.parquet
```

The BDT and IsolationForest baselines use scikit-learn; the CKF
baseline only needs a completed pipeline run, which you can produce
with `colliderml simulate` (see [Local Simulation](./simulation.md)).

## Systems tasks vs. physics tasks

The first three tasks (`tracking`, `jets`, `anomaly`) are **physics
tasks** — they measure the quality of a model's output against truth.

The last three (`tracking_latency`, `tracking_small`, `data_loading`)
are **systems tasks** — they measure operational properties (wall-clock
time, parameter count, I/O throughput). Systems tasks exist because
the best scientific model is not always the most useful: a 10 GB
transformer that wins the tracking leaderboard is worse than a 10 kB
network that hits 95% of its score if you actually have to deploy it
at collider trigger latencies.

## What's next

- Browse the [API reference](../library/tasks.md) for every function signature
- Read about [remote simulation](./remote-simulation.md) to generate training data
- Check the live [Leaderboard](https://huggingface.co/spaces/ColliderML/leaderboard)
  (deployed from this repo's `spaces/leaderboard/`)
- Open a PR to add a new task or baseline — contributors earn credits!
