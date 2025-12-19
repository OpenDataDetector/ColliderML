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

### Development Installation

To install with development dependencies (testing, formatting, etc.):

```bash
pip install -e ".[dev]"
```

This includes:
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
