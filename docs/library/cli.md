# CLI reference

The unified `colliderml` command groups every supported operation
under one entry point. Each subcommand has its own `--help`.

```bash
colliderml --help
colliderml <subcommand> --help
```

## `download` — fetch dataset shards

Download a subset (heuristic shard selection) for a process / pileup / object set:

```bash
colliderml download \
  --channels ttbar \
  --pileup pu0 \
  --objects particles,tracker_hits,calo_hits,tracks \
  --split train \
  --max-events 200
```

`--max-events` is a best-effort way to limit how many shards are
downloaded — the Polars loader then enforces exact event selection
locally. Data goes into `$COLLIDERML_DATA_DIR` (default
`~/.cache/colliderml`) or wherever you point `--out`.

## `list-configs` — inspect the cache or the remote repo

```bash
colliderml list-configs                       # locally cached configs
colliderml list-configs --remote              # query HuggingFace for what exists
```

## `simulate` — run the pipeline

New in v0.4.0. Runs the full Pythia / MadGraph / Geant4 / ACTS
pipeline, either locally inside a Docker or Podman container or via
the SaaS backend.

```bash
# Named preset, local container run (needs Docker/Podman):
colliderml simulate --preset ttbar-quick --local

# Explicit parameters, writes into ./runs/hp10/
colliderml simulate \
  --channel higgs_portal --events 10 --pileup 10 \
  --local --output ./runs/hp10

# Submit to the SaaS backend (needs an HF token):
colliderml simulate --preset ttbar-benchmark --remote
```

Options:

| Flag | Meaning |
| :-- | :-- |
| `--preset <name>` | Named preset; sets channel/events/pileup if not supplied. |
| `--channel <name>` | Physics channel (`ttbar`, `higgs_portal`, `single_muon`, …). |
| `--events <N>` | Number of events to generate. |
| `--pileup <N>` | Pileup level (0 for no pileup, 200 for HL-LHC). |
| `--seed <N>` | Random seed forwarded to every stage. Default: 42. |
| `--output <dir>` | Host output directory. Defaults to `./colliderml_output/<channel>_pu<pileup>_<events>evt`. |
| `--image <tag>` | Override the container image. Defaults to the pinned ODD sw image. |
| `--local` / `--remote` | Mutually exclusive; default is local if a runtime is available. |
| `--run-id <name>` | Subdirectory under `runs/`. Default: `0`. |
| `--quiet` | Suppress per-stage progress messages. |

See the [Local Simulation](../guide/simulation.md) and
[Remote Simulation](../guide/remote-simulation.md) guides for the full
story.

## `list-presets` — print the bundled preset catalogue

```bash
colliderml list-presets
```

Outputs one line per preset with its name, channel, event count,
pileup, and description. See
[`colliderml.simulate.load_presets`](./simulate.md#list_presets) for
the API equivalent.

## `balance` — show your credit balance

New in v0.4.0. Queries the SaaS backend's `/v1/me` endpoint and prints
your credit balance. Requires the `[remote]` extra and a HuggingFace
token.

```bash
colliderml balance
# user:    alice
# credits: 42.00

colliderml balance --json    # also print the raw /v1/me body
```

## `status` — check a remote request

New in v0.4.0. Fetches a remote simulation request's state from
`/v1/requests/<request-id>`.

```bash
colliderml status req-abc123
# request_id:  req-abc123
# state:       running
# channel:     ttbar
# events:      1000
# pileup:      200

colliderml status req-abc123 --json   # full backend response body
```

## Cache location

- By default, data is stored in `~/.cache/colliderml`.
- Override via `$COLLIDERML_DATA_DIR` or `--out /path/to/cache` on the
  `download` / `list-configs` subcommands.
- The `simulate` subsystem uses a separate subdirectory
  (`$COLLIDERML_CACHE/simulate/`, default
  `~/.cache/colliderml/simulate/`) for its container cache, the
  auto-cloned production repo, and the Geant4 physics tables.


