# `colliderml.tasks`

API reference for the benchmark task registry. For a conceptual tour
(what each task measures, eval splits, leaderboard workflow), see
[Benchmark Tasks](../guide/tasks.md).

Install the optional extra if you want the reference baselines:

```bash
pip install "colliderml[tasks]"
```

The registry itself and local scoring work with the base install —
only the BDT and IsolationForest baselines require scikit-learn.

## Registry

### `list_tasks() -> list[str]`

Returns every registered task name, sorted alphabetically.

```python
>>> import colliderml.tasks
>>> colliderml.tasks.list_tasks()
['anomaly', 'data_loading', 'jets', 'tracking', 'tracking_latency', 'tracking_small']
```

### `get(name: str) -> BenchmarkTask`

Returns a fresh instance of the task registered under ``name``.
Raises `ValueError` with the full list of available names on a miss.

### `register(cls: type[BenchmarkTask]) -> type[BenchmarkTask]`

Class decorator that registers a subclass of
[`BenchmarkTask`](#benchmarktask-abc). Used internally by each task
module; external code can also use it to register custom tasks in a
plugin. Rejects duplicate `name` values so accidental shadowing
surfaces as an `ValueError` at import time.

```python
from colliderml.tasks import BenchmarkTask, register

@register
class MyTask(BenchmarkTask):
    name = "my_task"
    dataset = "ttbar_pu0"
    eval_event_range = (95_000, 100_000)
    inputs = ["tracks"]
    metrics = ["my_metric"]
    higher_is_better = {"my_metric": True}

    def load_eval_inputs(self):
        return self.load(tables=self.inputs, event_range=self.eval_event_range)

    def validate_predictions(self, preds):
        ...

    def score(self, preds):
        return {"my_metric": 0.42}
```

## Evaluation

### `evaluate(task_name, predictions) -> dict[str, float]`

Runs `validate_predictions()` and `score()` on the registered task.

```python
scores = colliderml.tasks.evaluate("tracking", "my_tracks.parquet")
```

`predictions` accepts:

- a path (`str` or `pathlib.Path`) to a parquet file
- an existing `pyarrow.Table`
- a `pandas.DataFrame`

Anything else raises `TypeError`.

### `submit(task_name, predictions, *, backend_url=None) -> dict`

Uploads predictions to the backend leaderboard and returns the
backend's response (typically `{'scores': ..., 'credits_earned': ...}`).

Requires the `remote` extra (`pip install 'colliderml[remote]'`) and a
HuggingFace token. See [`colliderml.remote`](./remote.md) for auth
details. On a non-2xx response the function raises `RuntimeError` with
the backend's error message verbatim.

The submission flow:

1. Coerce `predictions` to a pyarrow Table.
2. Validate locally (raises `ValueError` on bad submissions so you
   don't waste bandwidth).
3. Score locally and include the preview in the multipart body.
4. POST the parquet bytes to
   `POST /v1/benchmark/<task_name>/submit` with the HF bearer token.
5. Return the backend's JSON body.

## `BenchmarkTask` ABC { #benchmarktask-abc }

Base class for every task. Subclasses set the following class
attributes:

| Attribute | Type | Meaning |
| :-- | :-- | :-- |
| `name` | `str` | Identifier used in URLs, the CLI, the leaderboard, and the registry. |
| `dataset` | `str` | `<channel>_<pileup>` key like `"ttbar_pu200"`. |
| `eval_event_range` | `tuple[int, int]` | Half-open range of event IDs withheld for scoring. |
| `inputs` | `list[str]` | Table names the task exposes to the model. |
| `metrics` | `list[str]` | Metric names in display order (first is primary). |
| `higher_is_better` | `dict[str, bool]` | Per-metric direction. Defaults to `True`. |

Subclasses must implement:

- `load_eval_inputs() -> dict[str, pa.Table]` — return the held-out
  eval data as pyarrow Tables.
- `validate_predictions(preds: pa.Table) -> None` — raise `ValueError`
  if the submission is malformed.
- `score(preds: pa.Table) -> dict[str, float]` — compute every metric.

The base class provides a `load()` helper that subclasses use inside
`load_eval_inputs()` and `score()`:

```python
def load(
    self,
    *,
    tables: list[str],
    dataset: str | None = None,
    max_events: int | None = None,
    event_range: tuple[int, int] | None = None,
) -> dict[str, pa.Table]: ...
```

`dataset` defaults to `self.dataset` — override it only for tasks
like anomaly detection that pull from multiple datasets. The helper
auto-downloads missing parquets via
[`colliderml.core.hf_download`](./loading.md) before delegating to
[`colliderml.core.loader.load_tables`](./loading.md).

`is_better(metric, new, current) -> bool` on the base class honours
`higher_is_better` so the leaderboard can decide whether a submission
beats a prior best.

## Per-task details

### `tracking` (`TrackingTask`)

- **Dataset**: `ttbar_pu200`
- **Eval range**: `(90_000, 100_000)`
- **Inputs**: `tracker_hits`
- **Metrics**: `trackml_eff`, `fake_rate`, `dup_rate`, `physics_eff_pt1`
- **Higher is better**: `trackml_eff`, `physics_eff_pt1`
- **Schema required on preds**: `event_id`, `hit_id`, `track_id`

The primary metric is TrackML weighted efficiency: the sum of hit
weights across all correctly reconstructed tracks over the total hit
weight. A track is "correct" when (a) a majority of its hits come
from one particle, and (b) that particle contributes ≥ 50% of its own
hits to the track.

### `jets` (`JetClassificationTask`)

- **Dataset**: `ttbar_pu0`
- **Eval range**: `(90_000, 100_000)`
- **Inputs**: `tracks`, `calo_hits`
- **Metrics**: `btag_auc`, `light_rej_70`, `c_rej_70`
- **Schema required on preds**: `event_id`, `jet_id`, `prob_b`,
  `prob_c`, `prob_light` (probabilities must sum to 1 per row, ±0.02)

The local `score()` currently synthesises a deterministic truth label
for preview purposes. The backend scorer overrides this with the
canonical truth file when you submit via `submit()`.

### `anomaly` (`AnomalyDetectionTask`)

- **Dataset**: virtual `mixed_sm_bsm` (draws from 3 SM + 4 BSM channels)
- **Eval range**: `(0, 10_000)` per channel
- **Inputs**: `tracks`, `calo_hits`
- **Metrics**: `auroc`, `sig_eff_1fpr`
- **Schema required on preds**: `event_id`, `channel`, `anomaly_score`

SM channels: `ttbar_pu0`, `zmumu_pu0`, `zee_pu0`.
BSM channels: `higgs_portal_pu0`, `susy_gmsb_pu0`, `hidden_valley_pu0`,
`zprime_pu0`.

`load_eval_inputs()` silently skips channels that aren't cached
locally — on a developer laptop that has only downloaded some of the
BSM datasets, the task still works with whatever is present.

### `tracking_latency` (`TrackingLatencyTask`)

- **Dataset**: `ttbar_pu200`
- **Eval range**: `(99_000, 100_000)` (1000 events)
- **Metrics**: `wallclock_s`, `events_per_sec`
- **Schema required**: 1-row table with `wallclock_s`, `n_events`

Users report the time they measured; the backend re-runs the code in
a standardised container for canonical numbers.

### `tracking_small` (`TrackingSmallModelTask`)

- **Dataset**: `ttbar_pu200`
- **Eval range**: `(90_000, 100_000)`
- **Metrics**: `trackml_eff`, `n_params`, `tier`
- **Schema required**: tracking predictions + a constant `n_params` column

Parameter tiers: tier 1 < 10k, tier 2 < 100k, tier 3 < 1M, tier 4 is
"does not qualify". The leaderboard sorts by tier first (smaller
tiers first), then by TrackML efficiency within a tier.

### `data_loading` (`DataLoadingTask`)

- **Dataset**: `ttbar_pu200`
- **Eval range**: `(0, 10_000)`
- **Metrics**: `events_per_sec_local`, `events_per_sec_streaming`
- **Schema required**: 1-row table with `local_seconds`,
  `streaming_seconds`, `n_events`

## Reference baselines

Three reference baselines ship with the library, runnable as Python
modules:

| Baseline | Module | Dependency | Purpose |
| :-- | :-- | :-- | :-- |
| CKF (tracking) | `colliderml.tasks.tracking.baselines.ckf` | — | Converts a pipeline run's tracksummary into the task's schema |
| GBDT (jets) | `colliderml.tasks.jets.baselines.bdt` | scikit-learn | Shallow gradient-boosted trees on per-event kinematics |
| IsoForest (anomaly) | `colliderml.tasks.anomaly.baselines.isoforest` | scikit-learn | Unsupervised anomaly detection on per-event features |

All three expose a `main()` entry point and a `--help` flag. The
scikit-learn dependency is gated inside the baseline scripts — the
main `tasks` package stays importable even if scikit-learn is not
installed.

## Internal helpers (rarely needed)

- `colliderml.tasks.parse_dataset_name(name)` — split
  `"ttbar_pu200"` into `("ttbar", "pu200")`.
- `colliderml.tasks.load_task_data(dataset, *, tables, max_events=None, event_range=None)`
  — the free-function that `BenchmarkTask.load()` calls under the hood.

## See also

- [Benchmark Tasks guide](../guide/tasks.md) — conceptual overview
- [`colliderml.load`](./loading.md) — the low-level loader used by tasks
- [`colliderml.remote`](./remote.md) — the SaaS client used by `submit()`
- [Leaderboard Space](https://huggingface.co/spaces/ColliderML/leaderboard) — live rankings
