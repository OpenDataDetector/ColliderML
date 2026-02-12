# ColliderML

<AboutData>

The ColliderML dataset is the largest-yet source of full-detail simulation in a virtual detector experiment.

**Why virtual?** The simulation choices are not tied to a construction timeline, there are no budget limitations, no politics. The only goals are to produce the most realistic physics on a detailed detector geometry, with significant computating challenges, in an ML-friendly structure.

The ColliderML dataset provides comprehensive simulation data for machine learning applications in high-energy physics, with detailed detector responses and physics object reconstructions.

</AboutData>

## Get the Data

Download data with the **colliderml** CLI, then load it in Python with Polars. The dataset is also on [HuggingFace](https://huggingface.co/datasets/CERN/Colliderml-release-1) if you prefer the `datasets` library.

### Quick Start

1. Install and download:

```bash
pip install colliderml
colliderml download --channels ttbar --pileup pu0 --objects particles,tracker_hits,calo_hits,tracks --max-events 200
```

2. Load in Python (same config as the download):

```python
from colliderml.core import load_tables, collect_tables

cfg = {
    "dataset_id": "CERN/ColliderML-Release-1",
    "channels": "ttbar",
    "pileup": "pu0",
    "objects": ["particles", "tracker_hits", "calo_hits", "tracks"],
    "split": "train",
    "lazy": False,
    "max_events": 200,
}
tables = load_tables(cfg)
frames = collect_tables(tables)  # dict[str, pl.DataFrame]
# e.g. frames["particles"], frames["tracker_hits"]
```

For more (exploding tables, pileup subsampling, calibration), see the [library docs](/library/overview) and the [exploration notebook](https://github.com/OpenDataDetector/ColliderML/blob/main/notebooks/colliderml_loader_exploration.ipynb).

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
