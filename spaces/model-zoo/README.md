---
title: ColliderML Model Zoo
emoji: 🦒
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
---

# ColliderML Model Zoo

Browse models tagged `colliderml` on HuggingFace. Upload your own by
adding the `colliderml` tag to your model card.

## Curation

Models are discovered automatically via
`huggingface_hub.list_models(filter="colliderml")`. If you want your model
shown here, add the following to your model card:

```yaml
---
tags:
- colliderml
task_category: object-detection   # or whatever applies
---
```

The zoo sorts by download count by default. You can also filter by task
(`tracking`, `jets`, `anomaly`).

## Deployment

Synced automatically from `spaces/model-zoo/` on the
[OpenDataDetector/ColliderML](https://github.com/OpenDataDetector/ColliderML)
repo via the `sync-spaces.yml` workflow.
