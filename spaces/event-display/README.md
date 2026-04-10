---
title: ColliderML Event Display
emoji: 🔭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
---

# ColliderML Event Display

Interactive 3D visualisation of single events from the
[ColliderML datasets](https://huggingface.co/datasets/CERN/ColliderML-Release-1).

Pick a physics process and an event ID to see the tracker hits, reconstructed
tracks, and truth particles inside the OpenDataDetector geometry.

## Local development

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

## Caching

The app pre-caches a small set of events per dataset on first load via
`colliderml.load()`. Subsequent loads are instant. To populate the on-disk
cache ahead of deploy, run:

```bash
python cache_events.py
```

Cached parquets land under `_cached_events/<dataset>/*.parquet` and are
picked up automatically at runtime.

## Deployment

Synced automatically from `spaces/event-display/` on the
[OpenDataDetector/ColliderML](https://github.com/OpenDataDetector/ColliderML)
repo via the `sync-spaces.yml` workflow (see commit B7 of the v0.4.0
migration).
