# Quickstart Guide

This guide will help you get started with ColliderML datasets from HuggingFace Hub.

## Installation

Install ColliderML using pip:

```bash
pip install colliderml
```

Or install from source:

```bash
git clone https://github.com/OpenDataDetector/ColliderML.git
cd ColliderML
pip install -e .
```

## Loading Your First Dataset

The ColliderML dataset is hosted on HuggingFace Hub and can be loaded using the standard `datasets` library.
Each configuration is large (`ttbar_pu0_particles` is 223 GB in 1000 shards), so start with one shard:

```python
from datasets import load_dataset

# Load one shard of the ttbar particles dataset (no pileup): 1000 events, 0.2 GB
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    data_files="data/ttbar_pu0_particles/train-00000-of-01000.parquet",
    split="train",
)

print(f"Loaded {len(dataset)} events")
```

This prints `Loaded 1000 events`. Passing the configuration name instead
(`load_dataset("CERN/ColliderML-Release-1", "ttbar_pu0_particles", split="train")`)
downloads all of its shards. The examples below use `dataset` from this block.

## Prefer a local CLI + loader workflow?

The HuggingFace `datasets` approach is great for quick access. For analysis workflows, ColliderML also provides a more convenient pattern:

- `colliderml.load("ttbar_pu0", max_events=200)`: one line that downloads on first call, then caches
- explode event tables into flat tables with `colliderml.polars.explode_*`

```python
import colliderml
from colliderml.polars import explode_particles

# Downloads on first call, reads from the local cache afterwards.
frames = colliderml.load("ttbar_pu0", tables=["particles"], max_events=200)
particles_flat = explode_particles(frames["particles"])
```

See the [Library overview](/library/overview), then:

- [CLI download](/library/cli)
- [Local loading](/library/loading)
- [Exploding tables](/library/exploding)

## Generate events yourself

Instead of downloading pre-generated data, you can run the full
ColliderML chain yourself (generation, Geant4, tracking and Pandora
particle flow, in containers with Docker or Podman), or submit a job to
the ColliderML backend:

```python
import colliderml

# Local: needs `pip install "colliderml[sim]"` plus Docker or Podman.
result = colliderml.simulate(preset="ttbar-pflow-quick", output_dir="./ttbar_pflow")
print(result.parquet_dir)                   # the parquet tables land here

# Remote: needs `pip install "colliderml[remote]"` and an HF token.
result = colliderml.simulate(preset="higgs-portal-quick", remote=True)
print(result.remote_request_id)
```

Full details in the [Local Simulation](./simulation.md) and
[Remote Simulation](./remote-simulation.md) guides.

## Score your model against a benchmark task

New in v0.4.0. Six built-in benchmark tasks (`tracking`, `jets`,
`anomaly`, and three systems tasks) let you compare any model on
equal footing:

```python
import colliderml.tasks

# What's available?
print(colliderml.tasks.list_tasks())

# Score local predictions
scores = colliderml.tasks.evaluate("tracking", "my_preds.parquet")
print(scores)

# Upload to the leaderboard (earns credits on new bests)
colliderml.tasks.submit("tracking", "my_preds.parquet")
```

See the [Benchmark Tasks guide](./tasks.md) for the full workflow.

## Understanding Dataset Structure

The ColliderML dataset is organized with configurations that combine:

- **Process**: The physics process being simulated (e.g., `ttbar`, `ggf`, `dihiggs`)
- **Pileup**: The pileup condition (e.g., `pu0` for no pileup, `pu200` for 200 pileup)
- **Object Type**: The detector data type or hierarchy level

### Available Configurations

Each configuration name follows the pattern `{process}_{pileup}_{object_type}`. For example:
- `ttbar_pu0_particles`
- `ggf_pu200_calo_hits`
- `dihiggs_pu0_tracks`

ColliderML provides multiple views of collision events:

- `particles`: Truth-level particle information (Monte Carlo truth)
- `tracker_hits`: Detector measurements in the tracking system
- `calo_hits`: Detector measurements in the calorimeter
- `tracks`: Reconstructed particle tracks

### Loading Different Configurations

The shard files of every configuration are under `data/<configuration>/`:

```python
from datasets import load_dataset

def first_shard(config):
    return load_dataset(
        "CERN/ColliderML-Release-1",
        data_files=f"data/{config}/train-00000-of-01000.parquet",
        split="train",
    )

particles = first_shard("ttbar_pu0_particles")        # truth-level particles
tracker_hits = first_shard("ttbar_pu0_tracker_hits")  # tracker hits (detector measurements)
tracks = first_shard("ttbar_pu0_tracks")              # reconstructed tracks
```

## Accessing Event Data

### Single Event

```python
# Get the first event
event = dataset[0]

# Inspect available fields
print("Event fields:", list(event.keys()))

# Access specific fields
for key, value in event.items():
    if hasattr(value, '__len__'):
        print(f"{key}: {len(value)} items")
    else:
        print(f"{key}: {value}")
```

### Batch Loading

Load multiple events at once for efficient processing:

```python
# Load first 10 events as a batch
batch = dataset[:10]

# batch is a dictionary where each value is a list
print("Batch keys:", list(batch.keys()))

# Process batch data
for key, values in batch.items():
    if hasattr(values, '__len__'):
        print(f"{key}: batch of {len(values)} events")
```

### Iteration

Iterate through the dataset:

```python
# Iterate over all events
for event in dataset:
    # Process each event
    print(f"Processing event with {len(event.keys())} fields")

    # Your analysis code here
    break  # Remove this to process all events
```

## Streaming Mode

For large datasets that don't fit in memory, use streaming mode:

```python
from datasets import load_dataset

# Load in streaming mode
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_particles",
    split="train",
    streaming=True  # Enable streaming
)

# Iterate without loading everything into memory
for i, event in enumerate(dataset):
    if i >= 10:  # Process first 10 events
        break
    print(f"Event {i}: {list(event.keys())}")
```

## Available Physics Processes

ColliderML includes multiple physics processes. These examples load the
first shard of each:

### Top Quark Pair Production (ttbar)

```python
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    data_files="data/ttbar_pu0_particles/train-00000-of-01000.parquet",
    split="train",
)
```

### Gluon-Gluon Fusion / Higgs (ggf)

```python
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    data_files="data/ggf_pu0_particles/train-00000-of-01000.parquet",
    split="train",
)
```

### Di-Higgs Production (dihiggs)

```python
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    data_files="data/dihiggs_pu0_particles/train-00000-of-01000.parquet",
    split="train",
)
```

Check the [CERN/ColliderML-Release-1 dataset page](https://huggingface.co/datasets/CERN/ColliderML-Release-1) for a complete list of available configurations.

## Data Inspection Example

Here's a complete example of loading and inspecting ColliderML data:

```python
from datasets import load_dataset
import numpy as np

# Load one shard
dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    data_files="data/ttbar_pu0_particles/train-00000-of-01000.parquet",
    split="train",
)

print(f"\nDataset Information:")
print(f"  Total events: {len(dataset)}")
print(f"  Features: {dataset.features}")

# Inspect first event
event = dataset[0]
print(f"\nFirst Event Structure:")

for key, value in event.items():
    print(f"  {key}:")
    print(f"    Type: {type(value)}")

    # Columns other than event_id hold one list per event
    if isinstance(value, list):
        value = np.asarray(value)
        print(f"    Shape: {value.shape}")
        print(f"    Dtype: {value.dtype}")

        # Print statistics for numeric arrays
        if np.issubdtype(value.dtype, np.number) and value.size > 0:
            print(f"    Range: [{np.min(value):.3f}, {np.max(value):.3f}]")
            print(f"    Mean: {np.mean(value):.3f}")
```

## Using with PyTorch or TensorFlow

The datasets library integrates seamlessly with popular ML frameworks:

### PyTorch

```python
from datasets import load_dataset

dataset = load_dataset(
    "CERN/ColliderML-Release-1",
    data_files="data/ttbar_pu0_particles/train-00000-of-01000.parquet",
    split="train",
)

# Convert to PyTorch format
dataset.set_format(type="torch", columns=["event_id", "px", "py", "pz"])

# Use with PyTorch DataLoader. Events have different numbers of particles,
# so keep each batch as a list of events instead of stacking them.
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=list)
batch = next(iter(dataloader))
print(len(batch), batch[0]["px"].shape)
```

With the default `collate_fn` the `DataLoader` fails on the first batch
(`stack expects each tensor to be equal size`), because every column except
`event_id` has a different length in each event.

### TensorFlow

```python
# Convert to TensorFlow format
tf_dataset = dataset.to_tf_dataset(
    columns=['your_feature_columns'],
    batch_size=32,
    shuffle=True
)
```

## Next Steps

- Load, explode and process the tables with the [library](/library/overview)
- See where downloads are cached in [Cache location](/library/cli#cache-location)
- Work through the [platform tutorial](./tutorial.md) and the [exploration notebook](https://github.com/OpenDataDetector/ColliderML/blob/main/notebooks/colliderml_loader_exploration.ipynb)

## Getting Help

If you encounter issues:

- Visit our [GitHub Issues](https://github.com/OpenDataDetector/ColliderML/issues)
- Consult the [CERN/ColliderML-Release-1 dataset page](https://huggingface.co/datasets/CERN/ColliderML-Release-1)
