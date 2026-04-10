# `colliderml.remote`

API reference for the ColliderML SaaS backend client. For the conceptual
overview — credits, auth, when to use remote vs. local — see
[Remote Simulation](../guide/remote-simulation.md).

This module lives behind the **`remote` extra**:

```bash
pip install "colliderml[remote]"
```

The only additional runtime dependency is `requests`; everything else is
already in the base install.

## Public surface

All of these are importable directly from `colliderml.remote`:

| Symbol | Kind | Purpose |
| :-- | :-- | :-- |
| [`submit`](#submit) | function | `POST /v1/simulate` |
| [`status`](#status) | function | `GET /v1/requests/{id}` |
| [`wait_for`](#wait_for) | function | submit-and-poll loop |
| [`get_me`](#get_me) | function | `GET /v1/me` |
| [`balance`](#balance) | function | convenience wrapper around `get_me()` |
| [`RemoteSubmission`](#remotesubmission-dataclass) | dataclass | snapshot of a request |
| `DEFAULT_BACKEND_URL` | constant | base URL (overridable) |
| `POLL_INTERVAL_SECONDS` | constant | default poll cadence for `wait_for` |

## `submit(...)` → `RemoteSubmission` { #submit }

```python
colliderml.remote.submit(
    *,
    channel: str,
    events: int,
    pileup: int,
    seed: int = 42,
    backend_url: str | None = None,
) -> RemoteSubmission
```

Submits a new simulation request. Returns immediately after the backend
acknowledges the submission — to block until the job is done, follow up
with [`wait_for`](#wait_for).

**Behaviour**

- `HF_TOKEN` (or the `huggingface-cli` saved token) is used as the
  bearer credential.
- If the backend recognises the request as an exact duplicate of a
  completed run, the returned `RemoteSubmission` is populated from the
  cached result and no new compute is scheduled. A reuse notice is
  printed to stderr.
- Non-2xx responses are translated into `RuntimeError` with an
  actionable message: 401 points users at `huggingface-cli login`, 402
  surfaces the backend's "insufficient credits" detail verbatim, 409 is
  reported as a duplicate, and everything else is a generic
  "Backend error {code}".

**Raises**

- `RuntimeError` — auth, credits, or backend error.
- `ImportError` — the `remote` extra is not installed.

## `status(request_id, *, backend_url=None)` → `RemoteSubmission` { #status }

Fetches the current state of an existing request. The returned object is
a **new snapshot**; it does not share identity with any previous
`RemoteSubmission` for the same request.

```python
from colliderml.remote import status

snap = status("req-abc123")
if snap.is_terminal:
    print(snap.output_hf_repo)
```

## `wait_for(...)` { #wait_for }

```python
colliderml.remote.wait_for(
    request_id: str,
    *,
    backend_url: str | None = None,
    timeout: float | None = None,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    on_poll: Callable[[RemoteSubmission], None] | None = None,
) -> RemoteSubmission
```

Polls `request_id` until the backend reports a terminal state
(`completed`, `failed`, or `cancelled`). Returns the final snapshot, or
raises:

- `TimeoutError` if `timeout` seconds elapse before completion.
- `RuntimeError` if the request ends in `failed` or `cancelled`.

**Progress reporting** — pass `on_poll=` to receive a callback with the
latest `RemoteSubmission` on every non-terminal poll. Useful for CLI
progress lines:

```python
wait_for(rid, on_poll=lambda s: print(f"  state={s.state}"))
```

## `get_me(*, backend_url=None)` → `dict` { #get_me }

Returns the authenticated user's profile. Shape (defined by the backend):

```json
{
  "hf_username": "alice",
  "credits": 42.5,
  "created_at": "2025-04-01T12:34:56Z",
  "total_requests": 7,
  "completed_requests": 5
}
```

## `balance(*, backend_url=None)` → `float` { #balance }

Thin wrapper around `get_me()["credits"]`. Used by the CLI
`colliderml balance` command and re-exported as `colliderml.balance`
(after commit B4).

```python
import colliderml
print(colliderml.balance(), "credits remaining")
```

## `RemoteSubmission` dataclass { #remotesubmission-dataclass }

```python
@dataclass
class RemoteSubmission:
    request_id: str
    state: str                       # submitted | queued | running | completed | failed | cancelled
    channel: str
    events: int
    pileup: int
    credits_charged: float = 0.0
    estimated_node_hours: float = 0.0
    estimated_completion_seconds: int = 0
    output_hf_repo: str | None = None
    backend_url: str = DEFAULT_BACKEND_URL
```

Properties and methods:

- `.is_terminal` → `bool`. `True` when `state` ∈ {`completed`,
  `failed`, `cancelled`}.
- `.refresh()` → mutates in-place with the latest `status()` and
  returns `self`.

The `raw` field (hidden from `repr`) stores the backend's full response
body for forward-compatibility — new fields the backend starts sending
show up there before they're promoted to dedicated attributes.

## Environment variables

| Name | Meaning | Default |
| :-- | :-- | :-- |
| `COLLIDERML_BACKEND` | Backend base URL (overridden by `backend_url=` keyword) | `https://api.colliderml.com` |
| `HF_TOKEN` | HuggingFace API token | — |
| `HUGGING_FACE_HUB_TOKEN` | Fallback HF token env var | — |

## Authentication

The client never asks for a password. Token resolution follows this
order, first hit wins:

1. `HF_TOKEN` env var
2. `HUGGING_FACE_HUB_TOKEN` env var
3. [`huggingface_hub.get_token()`](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/authentication#huggingface_hub.get_token)
   — the modern helper (hub ≥ 0.20)
4. [`HfFolder.get_token()`](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/authentication#huggingface_hub.HfFolder.get_token)
   — legacy fallback for older hub versions

No token → `RuntimeError` with a one-line recipe for fixing it.

## See also

- [Remote Simulation guide](../guide/remote-simulation.md)
- [`colliderml.simulate`](./simulate.md) — integrated entry point that
  drives `submit()` under the hood when `remote=True`.
- [`colliderml.load`](./loading.md) — load per-request HF datasets
  produced by completed runs.
