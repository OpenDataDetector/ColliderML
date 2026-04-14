# Model Zoo

**Live:** [huggingface.co/spaces/CERN/colliderml-model-zoo](https://huggingface.co/spaces/CERN/colliderml-model-zoo)
**Source:** [`spaces/model-zoo/`](https://github.com/OpenDataDetector/ColliderML/tree/main/spaces/model-zoo)

A browser for HuggingFace models tagged `colliderml`. Discovery is
entirely tag-based: the Space calls
`huggingface_hub.list_models(filter="colliderml", limit=500, full=True)`
and lays the results out in a sortable table.

## Getting your model listed

Add the `colliderml` tag to your model card and (optionally) an
appropriate task tag:

```yaml
---
tags:
- colliderml
- tracking        # or jets, anomaly, or omit for "general"
task_category: object-detection
---
```

Re-push the model — the Space picks up the new tag on its next refresh
(tag queries go through HuggingFace's live API; no re-deploy needed).

## Columns

| Column | Source |
| :-- | :-- |
| `model` | `m.modelId` (repo name) |
| `task` | First of `tracking` / `jets` / `anomaly` in the tag list, else `general` |
| `downloads` | HuggingFace download count |
| `likes` | HuggingFace like count |
| `pipeline` | `m.pipeline_tag` |
| `url` | Direct link to the model card |

The default sort is by download count, descending. The **Filter by
task** dropdown lets you narrow to one of the three physics tasks or
to `general` (everything else).

## Authentication

Public models only — no HuggingFace sign-in needed.

## Running it locally

```bash
cd spaces/model-zoo
pip install -r requirements.txt
python app.py
```

## Limitations

- The Space caps the query at 500 models per refresh. If you have more
  than 500 models tagged `colliderml` (a nice problem to have), bump
  the `limit=` argument in `fetch_models()`.
- Task detection is naïve — it looks for a literal tag match. If you
  use different tags (e.g. `track-reco` instead of `tracking`) your
  model shows up under `general`.

## See also

- [Benchmark Tasks guide](../guide/tasks.md) — the tasks users train
  against when they publish a model
- [Leaderboard](./leaderboard.md) — where shipped models go to battle
