# ColliderML

[![Tests](https://github.com/OpenDataDetector/ColliderML/actions/workflows/tests.yml/badge.svg)](https://github.com/OpenDataDetector/ColliderML/actions/workflows/tests.yml)
![Coverage](./coverage.svg)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern machine learning library for high-energy physics data analysis.

## Features

- Efficient parallel data downloading with resume capability
- Support for common HEP data formats
- Machine learning utilities for particle physics
- Visualization tools for physics data

## Installation

### For Users
```bash
# Create and activate environment
conda create -n collider-env python=3.11  # 3.10 or 3.11 recommended
conda activate collider-env

# Install package
pip install colliderml
```

### For Developers
```bash
# Create and activate environment
conda create -n collider-dev python=3.11  # 3.10 or 3.11 recommended
conda activate collider-dev

# Clone repository
git clone https://github.com/OpenDataDetector/ColliderML.git
cd ColliderML

# Install in development mode with extra dependencies
pip install -e ".[dev]"
```

## Quick Start

### CLI

```bash
# Download 100 events from the taster campaign into ./data
colliderml get -c taster -e 100 -O data
```

```python
from colliderml.core.data.manifest import ManifestClient
from colliderml.core.io import DataDownloader

manifest = ManifestClient()
files = manifest.select_files(campaign=None, datasets=["ttbar"], objects=["tracks"], max_events=1000)

downloader = DataDownloader()
results = downloader.download_files([f.path for f in files], local_dir="data", max_workers=4, resume=True)

for path, result in results.items():
    print(path, result.success, result.error)
```

### Features

- **Manifest-driven**: Always selects files from the latest portal manifest
- **Parallel Downloads**: Download multiple files concurrently
- **Resume Capability**: Optionally resume interrupted downloads
- **Progress Tracking**: Real-time progress bars
- **Clear Errors**: Helpful failure messages and HEAD checks

## Development

1. Activate your environment:
   ```bash
   conda activate collider-dev
   ```

2. Run tests:
   ```bash
   # Run unit tests only
   pytest -v -m "not integration"
   
   # Run all tests including integration tests
   pytest -v
   
   # Run with coverage report
   pytest --cov=colliderml
   ```

3. Build documentation:
   ```bash
   mkdocs build
   mkdocs serve  # View at http://127.0.0.1:8000
   ```

## License

[MIT License](LICENSE) 