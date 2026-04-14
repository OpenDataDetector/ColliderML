---
search: false
---

# Backend Service

The ColliderML backend is a **FastAPI** application that accepts simulation requests, manages user credits, dispatches jobs to NERSC Perlmutter via the Superfacility API, and hosts benchmark leaderboards.

Source: `colliderml-production/backend/`

## Technology stack

| Layer | Choice |
| :-- | :-- |
| Framework | FastAPI 0.115+ |
| Database | Supabase (PostgreSQL 14+) via `asyncpg` |
| Authentication | HuggingFace OAuth bearer tokens |
| Job dispatch | NERSC SFAPI (`sfapi-client>=0.1.0`) |
| Container runtime | Shifter on Perlmutter |
| Output storage | HuggingFace Datasets |
| Templating | Jinja2 (sbatch scripts) |
| Email | aiosmtplib (optional) |

## Routes

### User routes (`/v1/*`)

All require `Authorization: Bearer <hf-token>`.

| Method | Path | Purpose |
| :-- | :-- | :-- |
| `POST` | `/v1/simulate` | Submit a simulation request |
| `GET` | `/v1/requests/{request_id}` | Status + output URI for one request |
| `GET` | `/v1/requests` | List current user's requests (paginated, limit=50) |
| `GET` | `/v1/me` | User profile: username, email, credits, timestamps |
| `GET` | `/v1/me/transactions` | Credit ledger entries (paginated, limit=100) |
| `GET` | `/v1/datasets` | Static catalogue of 21 dataset variants |

### Benchmark routes (`/v1/benchmark/*`)

| Method | Path | Purpose |
| :-- | :-- | :-- |
| `GET` | `/v1/benchmark/tasks` | List all benchmark tasks with metrics |
| `POST` | `/v1/benchmark/{task}/submit` | Upload predictions (Parquet), auto-score, award credits |
| `POST` | `/v1/benchmark/{task}/reproduce/{id}` | Reproduce a submission; earn 20 credits if within 2% |
| `GET` | `/v1/leaderboard/{task}` | Leaderboard for a task (limit=100) |

### Admin routes (`/admin/*`)

Require `X-Admin-Token: <shared-secret>` header.

| Method | Path | Purpose |
| :-- | :-- | :-- |
| `POST` | `/admin/freeze?frozen={bool}` | Kill switch — pause all new submissions |
| `POST` | `/admin/grant` | Grant/deduct credits: `{hf_username, delta, reason}` |
| `POST` | `/admin/ban` | Ban/unban a user: `{hf_username, banned}` |
| `GET` | `/admin/usage?limit=20` | Top users by node-hours this month |
| `GET` | `/admin/analytics/channels` | Request count + node-hours per channel |
| `GET` | `/admin/analytics/daily?days=30` | Daily node-hours time series |
| `GET` | `/admin/analytics/failures` | Failure rate this month |

### Webhook routes (`/webhooks/*`)

| Method | Path | Purpose |
| :-- | :-- | :-- |
| `POST` | `/webhooks/github` | GitHub PR-merged webhook (HMAC-SHA256 verified) |

### Health

| Method | Path | Purpose |
| :-- | :-- | :-- |
| `GET` | `/healthz` | Readiness probe |

## Database schema

Six core tables, one view:

### `users`

| Column | Type | Notes |
| :-- | :-- | :-- |
| `hf_username` | text PK | HuggingFace username |
| `email` | text | nullable |
| `credits` | numeric(10,3) | Current balance |
| `banned` | bool | default false |
| `created_at` | timestamptz | |
| `last_seen_at` | timestamptz | Updated on every auth |
| `notes` | text | Operator notes |

### `credit_transactions`

Append-only ledger. Every credit movement is recorded.

| Column | Type | Notes |
| :-- | :-- | :-- |
| `id` | bigserial PK | |
| `hf_username` | text FK | |
| `delta` | numeric(10,3) | Positive = credit, negative = debit |
| `reason` | text | `signup`, `spent_on_request`, `reconciliation_refund`, `refund_submit_failed`, `refund_job_failed`, `beat_<task>`, `github_pr_merged`, `admin_grant` |
| `metadata` | jsonb | Request hash, submission ID, PR URL, etc. |
| `created_at` | timestamptz | |

### `simulation_requests`

| Column | Type | Notes |
| :-- | :-- | :-- |
| `id` | uuid PK | |
| `hf_username` | text FK | |
| `channel` | text | Physics channel name |
| `events` | int | 1–100,000 |
| `pileup` | int | 0–200 |
| `seed` | int | default 42 |
| `config_hash` | text | SHA-256 of (channel, events, pileup, seed) |
| `state` | text | `queued` → `submitted` → `running` → `completed` \| `failed` \| `cancelled` |
| `nersc_jobid` | text | SLURM job ID |
| `estimated_node_hours` | numeric(10,3) | |
| `actual_node_hours` | numeric(10,3) | |
| `credits_charged` | numeric(10,3) | |
| `output_hf_repo` | text | `CERN/ColliderML-Service-<id>` |
| `error_message` | text | |
| `created_at`, `updated_at` | timestamptz | |

Unique constraint on `config_hash` where state in `(queued, submitted, running, completed)` — prevents duplicate in-flight requests.

### `global_config`

Single-row table (id=1).

| Column | Default | Purpose |
| :-- | :-- | :-- |
| `monthly_node_hours_cap` | 500 | Global monthly ceiling |
| `submissions_frozen` | false | Kill switch |
| `seed_credits` | 10 | Credits granted on first sign-in |

### `benchmark_submissions`, `benchmark_bests`, `benchmark_reproductions`

Leaderboard tables tracking per-task submissions, current bests per metric, and reproduction attempts.

### `gh_hf_mapping`

Maps GitHub usernames to HuggingFace usernames for automated credit grants on PR merges.

### `monthly_usage` (view)

Aggregates `node_hours` and `n_requests` per user for the current calendar month.

## Authentication flow

1. Extract `Authorization: Bearer <token>` header
2. Verify against `https://huggingface.co/api/whoami-v2`
3. First sign-in: create user row, grant `seed_credits`, record `signup` transaction
4. Check `banned` flag — reject with 403 if true
5. Update `last_seen_at`, return user dict

## Credit system

**Unit:** 1 credit = 1 node-hour on Perlmutter CPU queue.

**Cost formula:**
```
base_seconds = channel_base[channel]   # 5s–90s per event
pileup_factor = 1 + pileup / 50
overhead = madgraph_overhead[channel]   # 0 or 300s
node_hours = (overhead + base_seconds * events * pileup_factor) / 3600
```

**Pre-submission gates** (checked in order):
1. Banned? → 403
2. Frozen? → 503
3. Burst: ≥10 submissions in 5 min → 429
4. Duplicate: same config_hash completed in 7 days → 409
5. Monthly cap: `used + est > cap` → 503
6. Balance: `credits < est` → 402
7. Atomic deduction inside a database transaction

**Refunds:** Automatic on submission failure, job failure, or over-estimation (reconciliation).

**Earning credits:**
- Beat a leaderboard metric: 30–50 credits per metric
- Reproduce a submission (within 2%): 20 credits
- GitHub PR merged with recognized labels: 10–200 credits
- Admin grant: arbitrary
