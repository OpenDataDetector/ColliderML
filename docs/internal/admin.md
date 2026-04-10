---
search: false
---

# Admin Dashboard

The **colliderml-admin** Space is a private Gradio app that exposes
operator controls for the ColliderML backend. It lives in
`colliderml-production/spaces/colliderml-admin/` and is deployed to
a HuggingFace Space gated behind a shared admin token.

## Authentication

The Space shows a login box on load. Enter the `ADMIN_TOKEN` value
(the same secret set as an env var on the backend). All subsequent API
calls include `X-Admin-Token: <token>` in the request header.

Backend URL is read from the `COLLIDERML_BACKEND` env var (default:
`https://api.colliderml.com`).

## Tabs

### Usage

- **Refresh** button fetches `GET /admin/usage?limit=50`
- Displays a bar chart of the top 50 users ranked by node-hours consumed
  this calendar month
- Summary line: total node-hours, total requests

### Analytics

Three visualisations from the `/admin/analytics/*` endpoints:

| Chart | Endpoint | Description |
| :-- | :-- | :-- |
| Requests per channel | `/admin/analytics/channels` | Bar chart of count + node-hours per physics channel |
| Daily node-hours | `/admin/analytics/daily?days=30` | Line chart over the last 30 days |
| Failure rate | `/admin/analytics/failures` | Single percentage for the current month |

### User management

| Control | Endpoint | Notes |
| :-- | :-- | :-- |
| **Grant** | `POST /admin/grant` | Fields: `hf_username`, `delta` (positive or negative), `reason` |
| **Ban** | `POST /admin/ban` | Sets `banned=true`; user gets 403 on next request |
| **Unban** | `POST /admin/ban` | Sets `banned=false` |

Grants are recorded in `credit_transactions` with `reason=admin_grant`.

### Kill switch

Two buttons:

| Button | Effect |
| :-- | :-- |
| **FREEZE** | `POST /admin/freeze?frozen=true` — all new simulation submissions return 503 |
| **UNFREEZE** | `POST /admin/freeze?frozen=false` — resume normal operation |

The freeze flag lives in `global_config.submissions_frozen`. It does
**not** cancel jobs already in flight — only prevents new ones.

## Kill switch procedure

When something goes wrong (runaway costs, NERSC outage, security
incident):

1. Click **FREEZE** in the admin Space (or `curl -X POST /admin/freeze?frozen=true -H "X-Admin-Token: ..."`)
2. Investigate the issue
3. If needed, **Ban** specific users
4. Fix the root cause
5. Click **UNFREEZE** to resume
6. Optionally **Grant** refunds to affected users

## Deployment

The admin Space is synced by `colliderml-production/.github/workflows/sync-spaces.yml`
(matrix entry: `colliderml-admin`). It is **not** part of the public
repo's `sync-spaces.yml` workflow.

Required HuggingFace Space secrets (set manually via Space Settings):

| Secret | Purpose |
| :-- | :-- |
| `COLLIDERML_BACKEND` | Backend URL |
| `ADMIN_TOKEN` | Shared admin secret |
