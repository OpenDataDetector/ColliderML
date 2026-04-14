# Remote Simulation

The ColliderML SaaS backend runs the same pipeline as
[Local Simulation](./simulation.md) — but on NERSC Perlmutter, dispatched
via a small FastAPI service. You submit a job, get a request ID, and
collect the results from a per-request HuggingFace dataset when the run
completes. No container runtime needed on your side.

This page covers the operator-facing story. For the API reference (every
function signature, error type, env var), see
[`colliderml.remote`](../library/remote.md).

## When to use remote simulation

| Use remote when | Use local instead when |
| :-- | :-- |
| You need more than a few hundred events | You want tight iteration cycles |
| You want realistic pileup (mu ≥ 10) | You have Docker/Podman already |
| You are on a laptop / CI runner without Docker | You need offline reproducibility |
| You want the job's output hosted on HuggingFace automatically | You are debugging the pipeline scripts themselves |
| You want to earn credits by submitting benchmark models | |

Remote runs are idempotent and *deduplicated*: if a request with the
same parameters has already succeeded, the backend reuses the existing
dataset and charges zero credits.

## Prerequisites

1. **Python 3.10 or newer**.
2. **The remote extra**:
   ```bash
   pip install "colliderml[remote]"
   ```
   This pulls in `requests`. Everything else (`huggingface_hub`,
   `pyyaml`) is already in the base install.
3. **A HuggingFace account and token** with *read* permissions:
   ```bash
   huggingface-cli login     # interactive
   # or
   export HF_TOKEN=hf_xxxxxxxx
   ```
   The token is used both to authenticate with the backend *and* to
   pull the resulting dataset from HuggingFace.

## Credits

The backend uses a simple credit model:

- **1 credit ≈ 1 node-hour ≈ 100 events at `pu0`** (roughly; the backend
  prices each request based on the pipeline stages it will run)
- **New users get a 10-credit seed grant** on first login
- **Benchmark wins and PRs to the model zoo earn additional credits** —
  see [Benchmark Tasks](./tasks.md)

Check your balance any time:

```python
import colliderml
print(colliderml.balance(), "credits")
```

Or on the CLI:

```bash
colliderml balance
```

## Submitting a job

There are two entry points:

### `colliderml.simulate(remote=True)` — integrated

The drop-in counterpart to [local simulation](./simulation.md). Returns a
`SimulationResult` whose `remote_request_id` is set so you can poll it
later.

```python
import colliderml

result = colliderml.simulate(
    preset="higgs-portal-quick",
    remote=True,
)
print(result.remote_request_id)
```

`simulate()` returns **immediately** after the backend acknowledges the
request. Use the request ID with `colliderml.remote.wait_for()` or the
`colliderml status` CLI to track progress.

### `colliderml.remote.submit(...)` — direct

Lower-level than `simulate()`. You get the full :class:`RemoteSubmission`
object back, with credits charged, estimated node-hours, and the initial
state. Use this when you want tight control over the submit → poll →
download loop.

```python
from colliderml.remote import submit, wait_for

submission = submit(channel="ttbar", events=1000, pileup=200)
print(submission.request_id, submission.credits_charged, "credits charged")

final = wait_for(
    submission.request_id,
    poll_interval=30,
    on_poll=lambda s: print(f"  state={s.state}"),
)
print("done:", final.output_hf_repo)
```

## Polling, status, cancellation

```python
from colliderml.remote import status, wait_for

# One-shot status check
snap = status("req-abc123")
print(snap.state, snap.output_hf_repo)

# Block until done, with progress callback
final = wait_for("req-abc123", poll_interval=60)
```

Terminal states are `completed`, `failed`, and `cancelled`. `wait_for`
raises `RuntimeError` on `failed` / `cancelled` and `TimeoutError` when
a supplied `timeout` elapses.

**Cancellation** — contact the backend admin (see the internal
[admin](../internal/admin.md) runbook) for now; a user-facing
cancellation endpoint is planned.

## Collecting outputs

Completed requests expose their output dataset at:

```
hf.co/datasets/ColliderML/req-<request_id>
```

You can load it with the same convenience you use for the canonical
release:

```python
import colliderml
from colliderml.core.hf_download import DownloadSpec, download_config
from colliderml.core.loader import load_tables
from colliderml.core.data.loader_config import LoaderConfig

# Pull outputs for the completed request into the local cache
download_config(DownloadSpec(
    dataset_id="ColliderML/req-abc123",
    config="ttbar_pu200_tracker_hits",
))

# Then load locally
tables = load_tables(LoaderConfig(
    dataset_id="ColliderML/req-abc123",
    channels="ttbar",
    pileup="pu200",
    objects=["tracker_hits"],
    split="train",
    lazy=False,
))
```

(After commit B4 lands, the one-liner `colliderml.load("ttbar_pu200",
dataset_id="ColliderML/req-abc123")` will subsume all of the above.)

## Configuration

| Env var | Meaning | Default |
| :-- | :-- | :-- |
| `COLLIDERML_BACKEND` | Backend base URL | `https://api.colliderml.com` |
| `HF_TOKEN` | Explicit HuggingFace token (overrides hub login) | — |
| `HUGGING_FACE_HUB_TOKEN` | Fallback env var | — |

Pass `backend_url=` to any client function to override per-call.

## Troubleshooting

**`RuntimeError: Authentication failed`** — the backend rejected your
token. Re-run `huggingface-cli login` and try again. Make sure your
token has at least *read* permissions.

**`RuntimeError: Insufficient credits`** — the backend priced your
request higher than your current balance. Reduce `events`, lower
`pileup`, or earn more credits by submitting a benchmark entry. Run
`colliderml balance` to see what you have.

**`RuntimeError: Duplicate request`** — someone (possibly you) already
ran exactly this request. The backend typically returns the existing
dataset URL in the error detail — use that instead of re-submitting.

**`Could not reach ColliderML backend`** — network, DNS, or the backend
is down. Check the status page linked from the docs home.

**`ImportError: The colliderml.remote client requires the 'remote' extra`**
— run `pip install 'colliderml[remote]'`.

## Next steps

- [Remote API reference](../library/remote.md)
- [Local simulation guide](./simulation.md) — when to stay on your own machine
- [Benchmark tasks](./tasks.md) — earn more credits
