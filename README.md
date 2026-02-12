# ColliderML

[![Tests](https://github.com/OpenDataDetector/ColliderML/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/OpenDataDetector/ColliderML/actions/workflows/tests.yml)
![Coverage](./coverage.svg)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern machine learning library for high-energy physics data analysis.

## Installation

```bash
pip install colliderml
```

For development: `pip install -e ".[dev]"`

## Getting the data

**Option 1 — CLI (download to local cache, then load with the library):**

```bash
colliderml download --channels ttbar --pileup pu0 --objects particles,tracker_hits,calo_hits,tracks --max-events 200
```

Cache location: default `~/.cache/colliderml`, or set `COLLIDERML_DATA_DIR`. List downloaded configs: `colliderml list-configs`.

**Option 2 — HuggingFace only:**

```python
from datasets import load_dataset
dataset = load_dataset("CERN/ColliderML-Release-1", "ttbar_pu0_particles", split="train")
```

## Using the library

The notebook [notebooks/colliderml_loader_exploration.ipynb](notebooks/colliderml_loader_exploration.ipynb) shows how to use the convenience functions: loading local Parquet with `load_tables`, exploding event tables, pileup subsampling, calibration, and plotting.

Full docs: <https://opendatadetector.github.io/ColliderML>

## Development

```bash
pytest -v -m "not integration"
```

Docs are built with VitePress: `npm ci --prefix docs && npm run --prefix docs docs:build`.

## License

[MIT License](LICENSE)
