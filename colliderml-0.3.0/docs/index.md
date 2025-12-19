# ColliderML

<AboutData>

The ColliderML dataset is the largest-yet source of full-detail simulation in a virtual detector experiment.

**Why virtual?** The simulation choices are not tied to a construction timeline, there are no budget limitations, no politics. The only goals are to produce the most realistic physics on a detailed detector geometry, with significant computating challenges, in an ML-friendly structure.

The ColliderML dataset provides comprehensive simulation data for machine learning applications in high-energy physics, with detailed detector responses and physics object reconstructions.

</AboutData>

## Get the Data

The ColliderML dataset is hosted on [HuggingFace](https://huggingface.co/datasets/CERN/Colliderml-release-1) and can be accessed using the `datasets` library.

::: tip Prefer a more convenient workflow?
ColliderML also ships a Python library (`colliderml`) with a CLI download + local loading workflow and utilities (exploding tables, pileup subsampling, calibration, etc.). See [Library docs](/library/overview).
:::

### Quick Start

1. Install the datasets library:
```bash
pip install datasets
```

2. Load a dataset:
```python
from datasets import load_dataset

# Load particles data from ttbar events (no pileup)
dataset = load_dataset("CERN/Colliderml-release-1", "ttbar_pu0_particles")
```

### Interactive Configuration

Use the configurator below to customize your dataset selection and generate the corresponding code:

<DataConfig />

If there are errors or unexpected behavior, please [open an issue](https://github.com/OpenDataDetector/ColliderML/issues) on the GitHub repository.
<!-- CHANGELOG:DATASET:START -->
::: details Dataset Changelog (latest 5)
- (0.2.0 — 2025-11-07) - Datasets now hosted on HuggingFace Hub for easier access and distribution.
- (0.2.0 — 2025-11-07) - Support for standard HuggingFace `datasets` library for data loading.
- (0.2.0 — 2025-11-07) - Migrated from NERSC manifest-based distribution to HuggingFace datasets.
- (0.2.0 — 2025-11-07) - Data now stored in Parquet format with improved compression and accessibility.
- (0.1.0 — 2025-09-08) - Initial release of ColliderML dataset with ttbar and ggf physics processes.
See the full changelog: [Changelog](/changelog).
:::
<!-- CHANGELOG:DATASET:END -->
