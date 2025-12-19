# Benchmarks

Benchmarks live under the top-level `benchmarks/` directory and focus on **download speed**.

## Download benchmark runner

Run locally:

```bash
python benchmarks/download_benchmark.py --help
```

Typical usage benchmarks a small, deterministic subset (e.g. a fixed set of configs and one shard each) and writes machine-readable JSON output.

## CI integration

GitHub Actions runs a lightweight benchmark in **warn-only** mode to detect large regressions without flaking CI due to network variability.

See:

- `benchmarks/README.md`
- `.github/workflows/tests.yml`


