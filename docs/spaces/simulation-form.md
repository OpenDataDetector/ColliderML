# Simulation Form

**Live:** [huggingface.co/spaces/CERN/colliderml-simulation](https://huggingface.co/spaces/CERN/colliderml-simulation)
**Source:** [`spaces/simulation-form/`](https://github.com/OpenDataDetector/ColliderML/tree/main/spaces/simulation-form)

A graphical front door for the SaaS simulation backend. Two tabs:

1. **Simulate** — a form with dropdowns for channel, events, and pileup.
   Submit, get a request ID, then hit **Refresh status** to poll until
   the job completes.
2. **Chat** — a natural-language interface backed by Claude that can
   estimate compute costs, check your credit balance, and submit jobs
   on your behalf after you confirm.

Both tabs talk to the same backend that the
[`colliderml.simulate(remote=True)`](../library/simulate.md) Python API
uses. Whichever entry point you pick, you end up in the same queue.

## Authentication

Authentication is HuggingFace OAuth. Click **Sign in with HuggingFace**
and the Space receives a scoped token that it forwards as a bearer
credential to the backend. The backend verifies the token, checks
quotas, and charges your credit balance.

Every user starts with 10 credits on first sign-in. See
[Remote Simulation](../guide/remote-simulation.md#credits) for the
credit economics.

## The form tab

| Field | Meaning |
| :-- | :-- |
| Physics channel | `ttbar`, `higgs_portal`, `zmumu`, `zee`, `diphoton`, `jets`, `susy_gmsb`, `hidden_valley`, `zprime`, `single_muon` |
| Events | Number of events to simulate (1–100 000) |
| Pileup | Pileup level in steps of 10, up to 200 (HL-LHC) |
| Seed | Random seed forwarded to every stage (default 42) |

After submission the UI shows:

- the backend's assigned request ID,
- an estimated cost in credits,
- an estimated wall-clock completion time,
- a note if the request was deduplicated against an existing run
  (zero credits in that case).

## The chat tab

The chat agent is Claude (`claude-sonnet-4-6`) with three tools:

- `check_balance` — queries `/v1/me` and reports credits remaining.
- `estimate_compute` — dry-runs the backend's capacity estimate and
  returns credits + estimated minutes.
- `submit_simulation` — posts to `/v1/simulate`. The system prompt
  requires the agent to confirm cost + parameters with the user before
  calling this tool.

The chat tab only works if the Space has `ANTHROPIC_API_KEY` set as a
Space-level secret. Without it the chat tab shows a "not configured"
message but the rest of the Space still works.

## Required Space secrets

| Secret | Purpose |
| :-- | :-- |
| `COLLIDERML_BACKEND` | Backend URL; defaults to `https://api.colliderml.com` |
| `ANTHROPIC_API_KEY` | Needed for the Chat tab; without it the tab degrades gracefully |

See the
[sync-spaces workflow docs](../internal/deployment.md#space-secrets)
for how to set these after the first automated sync.

## Running it locally

```bash
cd spaces/simulation-form
pip install -r requirements.txt
export COLLIDERML_BACKEND=http://localhost:8000   # point at your local backend
export ANTHROPIC_API_KEY=sk-ant-...               # optional, only for the Chat tab
python app.py
```

Use a HuggingFace token instead of OAuth for local testing — see the
`fetch_me` call in `app.py` for where to inject one.

## See also

- [Remote Simulation guide](../guide/remote-simulation.md) — the CLI
  equivalent of this Space
- [`colliderml.simulate`](../library/simulate.md) — the Python API it
  mirrors
- [`colliderml.remote`](../library/remote.md) — the client library
  the backend speaks to
