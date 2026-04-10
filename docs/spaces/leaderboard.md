# Leaderboard

**Live:** [huggingface.co/spaces/CERN/colliderml-leaderboard](https://huggingface.co/spaces/CERN/colliderml-leaderboard)
**Source:** [`spaces/leaderboard/`](https://github.com/OpenDataDetector/ColliderML/tree/main/spaces/leaderboard)

Public scoreboard for the six benchmark tasks shipped in
[`colliderml.tasks`](../library/tasks.md). Browse rankings, submit your
own predictions, or reproduce someone else's to earn credits.

## Layout

One tab per task. Inside each tab you get:

1. A table of the top 100 submissions (sortable, read-only). Columns
   include submitter HF username, all task metrics, credits earned for
   that submission, and whether it's a shipped baseline.
2. A **Submit predictions** accordion — upload a parquet file, the
   backend re-scores it against held-out truth, and any metric that
   beats the current best earns credits.
3. A **Reproduce a submission** accordion — enter a submission ID and
   upload your own predictions. If every metric lands within 2% of the
   original you earn 20 credits. This is the self-policing mechanism
   that keeps the board honest.

## Task discovery

The task list is loaded from the installed
[`colliderml.tasks`](../library/tasks.md) package at startup:

```python
import colliderml.tasks as _tasks
TASKS = _tasks.list_tasks()
```

Adding a new task in the library is enough to surface it in the UI —
no Space redeploy beyond a pin-bump on the `colliderml` package in
`requirements.txt`.

The Space falls back to a hardcoded task list if the `colliderml`
import fails (e.g. during a first-boot partial install).

## Authentication

Submission and reproduction both require a HuggingFace OAuth sign-in.
The received token is forwarded as a bearer credential to the
`/v1/benchmark/<task>/submit` and `/v1/benchmark/<task>/reproduce/<id>`
backend endpoints. Browsing the rankings does **not** require a
sign-in.

## Credit economics

| Action | Reward |
| :-- | :-- |
| Submit a new best on any metric | 30–50 credits (task-dependent) |
| Add a new baseline that becomes the bar to beat | 100 credits |
| Reproduce a submission within 2% tolerance on every metric | 20 credits |
| Contribute a new task or channel | 200 credits |

Credit balances are shared across every entry point in the platform
(the pip package, the Simulation Form Space, and this Space all draw
from the same backend ledger).

## Required Space secrets

| Secret | Purpose |
| :-- | :-- |
| `COLLIDERML_BACKEND` | Backend URL; defaults to `https://api.colliderml.com` |

## Running it locally

```bash
cd spaces/leaderboard
pip install -r requirements.txt
export COLLIDERML_BACKEND=http://localhost:8000
python app.py
```

## See also

- [Benchmark Tasks guide](../guide/tasks.md) — the conceptual tour
- [`colliderml.tasks`](../library/tasks.md) — the registry that
  populates the task tabs
- [Simulation Form](./simulation-form.md) — for generating training data
