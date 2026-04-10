---
title: ColliderML Simulation
emoji: ⚛️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: true
license: apache-2.0
---

# ColliderML Simulation Service

Submit custom ColliderML simulation requests without installing anything
locally. This Space is a thin frontend for the ColliderML backend — it
sends the same payloads that the `colliderml` pip package sends and
returns the same results.

## Usage

1. Click **Sign in with HuggingFace** (you'll get 10 free credits on your first sign-in).
2. Choose a physics channel, number of events, and pileup.
3. Click **Submit**. You'll be given a request ID and an email when the job
   completes.
4. Use the **Chat** tab if you'd rather describe what you need in plain English —
   the Claude-powered agent estimates the compute cost, checks your balance,
   and asks for confirmation before submitting.

## Environment variables

| Variable              | Purpose                                                                                    |
|-----------------------|--------------------------------------------------------------------------------------------|
| `COLLIDERML_BACKEND`  | Backend URL (default: `https://api.colliderml.com`)                                        |
| `ANTHROPIC_API_KEY`   | Required for the Chat tab. Without it the chat tab shows a "not configured" message.       |

## See also

- [Event display](https://huggingface.co/spaces/CERN/colliderml-event-display) — visualise existing datasets
- [Leaderboard](https://huggingface.co/spaces/CERN/colliderml-leaderboard) — benchmarks and credits
- [`colliderml` pip package](https://pypi.org/project/colliderml/) — same capabilities from the CLI
- [Remote simulation guide](https://opendatadetector.github.io/ColliderML/guide/remote-simulation)

## Deployment

Synced automatically from `spaces/simulation-form/` on the
[OpenDataDetector/ColliderML](https://github.com/OpenDataDetector/ColliderML)
repo via the `sync-spaces.yml` workflow.
