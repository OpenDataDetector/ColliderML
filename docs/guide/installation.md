# Installation

## Requirements

- Python 3.10 or 3.11
- pip or conda

## Install from PyPI

The easiest way to install ColliderML is via pip:

```bash
pip install colliderml
```

This will install ColliderML and its core dependencies:
- `datasets` - HuggingFace datasets library for data loading
- `numpy` - Numerical computing
- `h5py` - HDF5 file support (for local data inspection)

## Install from Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/OpenDataDetector/ColliderML.git
cd colliderml

# Install in development mode
pip install -e .
```

### Optional feature sets (`extras`)

ColliderML ships a handful of optional extras so that the base install
stays lean. Pick whichever you need:

```bash
# Run simulation pipelines locally (requires Docker or Podman separately)
pip install "colliderml[sim]"

# Submit simulation jobs to the SaaS backend (no Docker needed)
pip install "colliderml[remote]"

# Reference baselines for the benchmark task runner (brings scikit-learn)
pip install "colliderml[tasks]"

# Development tools (testing, formatting, linting, type-checking)
pip install "colliderml[dev]"
```

The `sim` extra pulls in only the Python dependencies required to drive
the container runtime — the actual container image and supporting data
are fetched on first use of `colliderml.simulate()`. See
[Local Simulation](./simulation.md) for the full story and disk-space
expectations.

The `remote` extra installs the `requests` library and gives you
`colliderml.remote.submit`, `status`, `balance`, and the integrated
`colliderml.simulate(remote=True)` path — see
[Remote Simulation](./remote-simulation.md). You will also need a
HuggingFace account and token, which is how the backend authenticates
you.

The `tasks` extra pulls in `scikit-learn` for the reference baselines
shipped with each benchmark task (GBDT for jets, IsolationForest for
anomaly detection). You only need it if you want to run the shipped
baselines verbatim — the task *registry* and local scoring work with
the base install. See [Benchmark Tasks](./tasks.md) for details.

The `dev` extra includes:
- `pytest` and `pytest-cov` for testing
- `black` for code formatting
- `ruff` for linting
- `mypy` for type checking

## Using Conda

If you prefer conda:

```bash
# Create a new environment
conda create -n colliderml python=3.11
conda activate colliderml

# Install ColliderML
pip install colliderml
```

## Verify Installation

Test your installation:

```python
import colliderml
print(f"ColliderML version: {colliderml.__version__}")

# Test loading a dataset
from datasets import load_dataset
dataset = load_dataset(
    "CERN/Colliderml-release-1",
    "ttbar_pu0_particles",
    split="train"
)
print(f"Successfully loaded {len(dataset)} events")
```

## Troubleshooting

### HuggingFace Hub Access

ColliderML datasets are public and don't require authentication. However, if you experience connection issues:

1. Check your internet connection
2. Try setting a HuggingFace cache directory:
   ```bash
   export HF_HOME=/path/to/cache
   ```
3. Consult the [HuggingFace datasets documentation](https://huggingface.co/docs/datasets)

### Python Version Issues

ColliderML requires Python 3.10 or 3.11. Check your version:

```bash
python --version
```

If you have an older version, consider using conda or pyenv to manage multiple Python versions.

## Next Steps

- Continue to the [Quickstart Guide](./quickstart.md) to start using ColliderML
- Learn about [Data Structure](./data-structure.md) and available datasets
