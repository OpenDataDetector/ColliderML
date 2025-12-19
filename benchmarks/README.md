## Benchmarks

This directory is focused on **download speed benchmarks** for ColliderML datasets hosted on HuggingFace.

### What we measure

- **Initial download time**: time to fetch a specific Parquet shard from HuggingFace into the local cache.
- **Cached re-load time**: time to fetch the same file again with caching enabled (should be much faster).

### CI behavior

- CI runs a small, deterministic download benchmark:
  - Dataset: `CERN/ColliderML-Release-1`
  - Configs: `ttbar_pu0_particles`, `ttbar_pu0_tracker_hits`, `ttbar_pu0_calo_hits`, `ttbar_pu0_tracks`
  - Shard: the first shard (lexicographically) per config
- CI is **warn-only**: it will emit warnings if thresholds are exceeded, but will not fail the job.

### Local usage

Run the benchmark (writes a JSON result file under `benchmarks/results/`):

- `python benchmarks/download_benchmark.py --out ~/.cache/colliderml`


