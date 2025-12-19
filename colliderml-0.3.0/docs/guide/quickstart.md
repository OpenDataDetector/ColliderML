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
cd colliderml
pip install -e .
```

## Loading Your First Dataset

The ColliderML dataset is hosted on HuggingFace Hub and can be loaded using the standard `datasets` library:

```python
from datasets import load_dataset

# Load the ttbar particles dataset (no pileup)
dataset = load_dataset(
    "CERN/Colliderml-release-1",
    "ttbar_pu0_particles",
    split="train"
)

print(f"Loaded {len(dataset)} events")
```

## Prefer a local CLI + loader workflow?

The HuggingFace `datasets` approach is great for quick access. For analysis workflows, ColliderML also provides a more convenient pattern:

- download Parquet shards once via `colliderml download`
- load locally with `colliderml.core.load_tables`
- explode event tables into flat tables with `colliderml.polars.explode_*`

See the [Library overview](/library/overview), then:

- [CLI download](/library/cli)
- [Local loading](/library/loading)
- [Exploding tables](/library/exploding)

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

```python
from datasets import load_dataset

# Load truth-level particles
particles = load_dataset(
    "CERN/Colliderml-release-1",
    "ttbar_pu0_particles",
    split="train"
)

# Load tracker hits (detector measurements)
tracker_hits = load_dataset(
    "CERN/Colliderml-release-1",
    "ttbar_pu0_tracker_hits",
    split="train"
)

# Load reconstructed tracks
tracks = load_dataset(
    "CERN/Colliderml-release-1",
    "ttbar_pu0_tracks",
    split="train"
)
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
    "CERN/Colliderml-release-1",
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

ColliderML includes multiple physics processes:

### Top Quark Pair Production (ttbar)

```python
dataset = load_dataset(
    "CERN/Colliderml-release-1",
    "ttbar_pu0_particles",
    split="train"
)
```

### Gluon-Gluon Fusion / Higgs (ggf)

```python
dataset = load_dataset(
    "CERN/Colliderml-release-1",
    "ggf_pu0_particles",
    split="train"
)
```

### Di-Higgs Production (dihiggs)

```python
dataset = load_dataset(
    "CERN/Colliderml-release-1",
    "dihiggs_pu0_particles",
    split="train"
)
```

Check the [CERN/Colliderml-release-1 dataset page](https://huggingface.co/datasets/CERN/Colliderml-release-1) for a complete list of available configurations.

## Data Inspection Example

Here's a complete example of loading and inspecting ColliderML data:

```python
from datasets import load_dataset
import numpy as np

# Load dataset
dataset = load_dataset(
    "CERN/Colliderml-release-1",
    "ttbar_pu0_particles",
    split="train"
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

    if hasattr(value, 'shape'):
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
    "CERN/Colliderml-release-1",
    "ttbar_pu0_particles",
    split="train"
)

# Convert to PyTorch format
dataset.set_format(type='torch', columns=['your_feature_columns'])

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

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

- Explore the [Data Structure](./data-structure.md) documentation for detailed field descriptions
- Learn about [Data Management](./data-management.md) for caching and optimization
- Check out the [Examples](../examples/) for complete analysis workflows
- Read about the [Physics Processes](./processes.md) available in ColliderML

## Getting Help

If you encounter issues:

- Check the [FAQ](./faq.md)
- Visit our [GitHub Issues](https://github.com/OpenDataDetector/ColliderML/issues)
- Consult the [CERN/Colliderml-release-1 dataset page](https://huggingface.co/datasets/CERN/Colliderml-release-1)
