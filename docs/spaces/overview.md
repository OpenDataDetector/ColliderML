# HuggingFace Spaces

ColliderML ships four public HuggingFace Spaces, each a lightweight
Gradio frontend for a different slice of the platform. The code lives
under [`spaces/`](https://github.com/OpenDataDetector/ColliderML/tree/main/spaces)
in this repository and is auto-synced to HuggingFace on every push to
`main` (staging previews sync on every push to `staging`) via the
`sync-spaces.yml` workflow.

## The four Spaces

| Space | What it does | Needs login? |
| :-- | :-- | :-- |
| [Event Display](./event-display.md) | Interactive 3D viewer for single events | No |
| [Simulation Form](./simulation-form.md) | Submit custom simulations via form or chat agent | HuggingFace OAuth |
| [Leaderboard](./leaderboard.md) | Browse + submit to benchmark task leaderboards | HuggingFace OAuth for submission |
| [Model Zoo](./model-zoo.md) | Browse HF models tagged `colliderml` | No |

A fifth Space — `colliderml-admin` — is **private** and lives in the
[`colliderml-production`](https://github.com/OpenDataDetector/colliderml-production)
repository instead. It exposes operator controls (credit grants, user
bans, kill switch) and is gated behind a separate admin token.

## How the Spaces fit the platform

```
┌──────────────────────┐         ┌──────────────────────┐
│  colliderml pip pkg  │         │  HuggingFace Spaces  │
│  (this repo)         │         │  (this repo, synced) │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                │
           │                                │  HF OAuth → bearer token
           │                                │
           ▼                                ▼
       ┌──────────────────────────────────────┐
       │   ColliderML Backend (FastAPI)       │
       │   (colliderml-production repo)       │
       │                                      │
       │   POST /v1/simulate                  │
       │   GET  /v1/me, /v1/requests/{id}     │
       │   POST /v1/benchmark/{task}/submit   │
       │   GET  /v1/leaderboard/{task}        │
       └──────────────────┬───────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │  NERSC SFAPI    │
                │  Perlmutter     │
                └─────────────────┘
```

The Spaces **never talk to NERSC directly** — every button in every Space
is a thin wrapper around a backend HTTP endpoint. The backend does all
the authentication, quota enforcement, credit accounting, and actual
SFAPI dispatch.

This means the same features are available whether you reach for the
pip package or a Space: `colliderml simulate --remote` and the
Simulation Form Space both POST to `/v1/simulate`; `colliderml.tasks.submit`
and the Leaderboard Space both POST to `/v1/benchmark/<task>/submit`.

## Embedding a Space in your own site

HuggingFace Spaces have a built-in iframe embed:

```html
<iframe
    src="https://huggingface.co/spaces/CERN/colliderml-event-display"
    frameborder="0"
    width="850"
    height="450">
</iframe>
```

The four public Spaces carry `hf_oauth: true` where needed and an
Apache-2.0 license, so they're safe to embed in papers, blog posts, or
lab pages. Please keep the attribution link visible.

## Running a Space locally

Each Space is a self-contained Gradio app:

```bash
cd spaces/event-display
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

For Spaces that need backend auth (`simulation-form`, `leaderboard`), set
`COLLIDERML_BACKEND=http://localhost:8000` to point at a local backend
checkout, and supply a HuggingFace token via env var instead of OAuth.

## Deployment workflow

`.github/workflows/sync-spaces.yml` (landing in commit B7 of the v0.4.0
migration) matrix-builds every directory under `spaces/` and pushes it
to the corresponding HuggingFace Space via `huggingface_hub.upload_folder`.
On `staging` pushes the Spaces are synced to suffixed variants
(`-staging`) so you can preview changes without clobbering the
production Space.

See the individual Space pages for each one's specifics.
