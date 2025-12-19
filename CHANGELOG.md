# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.3.0] - 2025-12-19

### Added
- [library] CLI commands: `colliderml download` for downloading HuggingFace datasets with local caching, `colliderml list-configs` for discovering available configurations.
- [library] Polars-based data loader with lazy/eager loading support and structured configuration (YAML/dict).
- [library] Physics utilities: pileup subsampling, decay-chain traversal with primary ancestor assignment, calorimeter calibration with region-specific scaling factors.
- [library] Flattening helpers to explode nested Parquet structures into pandas DataFrames for analysis.
- [library] Visualization utilities: coordinate transformations (eta, r), binned energy profile plotting.
- [library] Download timing benchmarks integrated into CI/CD.
- [docs] Reference exploration notebook demonstrating loader, physics utilities, and visualization.

### Changed
- [library] Data loading now uses local Parquet cache populated by CLI, not direct HuggingFace streaming.
- [library] Exploded/flattened data structures now return pandas DataFrames for downstream analysis.
- [library] Physics constants (calorimeter calibration factors) centralized in `colliderml.physics.constants`.

### Fixed
- [library] Decay traversal now correctly identifies root particles by broken parent links, ignoring the `primary` flag.
- [library] Pileup subsampling now correctly filters tracker hits and calorimeter contributions by particle ID.
- [library] Calorimeter calibration now handles both region names and raw detector IDs.

## [0.2.0] - 2025-11-07

### Added
- [dataset] Datasets now hosted on HuggingFace Hub for easier access and distribution.
- [dataset] Support for standard HuggingFace `datasets` library for data loading.
- [docs] Interactive dataset configurator with dynamic channel discovery from HuggingFace API.
- [docs] Updated documentation site with HuggingFace integration examples.

### Changed
- [dataset] Migrated from NERSC manifest-based distribution to HuggingFace datasets.
- [dataset] Data now stored in Parquet format with improved compression and accessibility.
- [docs] Simplified data access workflow using `load_dataset()` instead of custom CLI.

### Removed
- [docs] Removed NERSC manifest.json dependency from documentation build process.

## [0.1.0] - 2025-09-08

### Added
- [dataset] Initial release of ColliderML dataset with ttbar and ggf physics processes.
- [dataset] Four detector hierarchy levels: particles, tracker_hits, calo_hits, and tracks.
- [dataset] Approximately 100,000 simulated events per process with no pileup (pu0).
- [library] Initial `colliderml` Python library with data access utilities.
- [docs] Documentation website with VitePress framework.
- [docs] Dataset configuration modal for exploring available data.

---

Unreleased changes should be added under a `## [Unreleased]` header above with entries marked as `[dataset]`, `[library]`, or `[docs]`.
