# CLI: Download to Local Cache

The CLI is the supported way to download Parquet shards to disk. The library loader then reads **only from the local cache**.

## Download

Download a small subset (heuristic shard selection) for a process/pileup/object set:

```bash
colliderml download \
  --channels ttbar \
  --pileup pu0 \
  --objects particles,tracker_hits,calo_hits,tracks \
  --split train \
  --max-events 200
```

### Notes on `--max-events`

HuggingFace hosts Parquet files in **shards**. `--max-events` is a best-effort way to limit how many shards are downloaded; the loader can enforce exact event selection locally.

## Cache location

- By default, data is stored in a local cache directory.
- You can override via:
  - environment variable: `COLLIDERML_DATA_DIR`
  - CLI flag: `--out /path/to/cache`

## Listing configs

Use:

```bash
colliderml list-configs
```

This can list locally-cached configs, and optionally query remote configs depending on CLI flags.


