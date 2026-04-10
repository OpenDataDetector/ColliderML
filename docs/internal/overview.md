---
search: false
---

# Internal Documentation

::: warning Audience
These pages document the **backend service, SFAPI runner, admin tooling, and deployment infrastructure** that live in the
[`colliderml-production`](https://github.com/OpenDataDetector/colliderml-production) repository.
They are written for platform operators and contributors — not for end users of the `colliderml` pip package.
:::

## What lives where

| Concern | Repository |
| :-- | :-- |
| pip package (`colliderml`), benchmark tasks, HF Spaces, docs site | `OpenDataDetector/ColliderML` (this repo) |
| FastAPI backend, SFAPI runner, pipeline scripts, container image workflow, admin dashboard | `OpenDataDetector/colliderml-production` |

## Pages in this section

- [Backend](./backend.md) — FastAPI service architecture, routes, database schema, auth flow, credit ledger
- [SFAPI Runner](./sfapi.md) — NERSC Superfacility API integration, sbatch templates, multi-node submission
- [Admin Dashboard](./admin.md) — Gradio admin Space, freeze/grant/ban/usage operations, kill switch
- [Deployment](./deployment.md) — Render deploy, Docker Compose local dev, environment variables, database migrations
- [Container Image](./container-image.md) — `OpenDataDetector/sw` image contents, build checklist, known issues

These pages are intentionally **excluded from the site navigation and search index**.
They are reachable only by direct URL.
