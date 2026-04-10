---
title: ColliderML Leaderboard
emoji: 🏆
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: apache-2.0
---

# ColliderML Benchmark Leaderboard

Six tasks. Beat any metric, earn credits. Reproduce someone else's result,
earn credits. Contribute a baseline, earn credits.

## Tasks

### Physics

1. **tracking** — Track reconstruction on `ttbar_pu200` (TrackML weighted efficiency, fake / dup rates)
2. **jets** — Jet classification on `ttbar_pu0` (b-tag AUC, flavour rejections)
3. **anomaly** — BSM anomaly detection on mixed SM/BSM events (AUROC, signal efficiency @ 1% FPR)

### Systems

4. **tracking_latency** — Wall-clock time to process the eval split
5. **tracking_small** — Best efficiency under a parameter budget (three tiers)
6. **data_loading** — Throughput of `colliderml.load()` and downstream iteration

Task metadata is loaded from
[`colliderml.tasks`](https://opendatadetector.github.io/ColliderML/library/tasks)
at startup so adding a new task in the library automatically surfaces it
here without re-deploying the Space.

## How to submit

1. Sign in with HuggingFace.
2. Choose a task tab.
3. Upload your predictions as a Parquet file.
4. The server re-scores with the held-out truth and awards credits for any
   metric that beats the current best.

## Reproduction

Every leaderboard row has a **Reproduce** button. Run the original submitter's
code yourself, upload your own predictions, and if you land within 2% on
every metric you earn 20 credits. This is the self-policing mechanism that
keeps the leaderboard honest.

## Deployment

Synced automatically from `spaces/leaderboard/` on the
[OpenDataDetector/ColliderML](https://github.com/OpenDataDetector/ColliderML)
repo via the `sync-spaces.yml` workflow.
