# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- [library] `colliderml.simulate(particle_flow=True)` runs the chain past tracking: truth particle tables, calorimeter digitisation and Pandora particle flow (in a second, reco container), and the final parquet tables (`particles`, `generator_particles`, `tracker_simhits`, `tracker_hits`, `tracks`, `truth_tracks`, `vertices`, `calo_cells`, `calo_clusters`, `pfos`) under `<output_dir>/parquet/`. New preset `ttbar-pflow-quick`, new arguments `reco_image`, `threads`, `pandora_processes`, and the CLI flags `--particle-flow`, `--reco-image`, `--threads`, `--pandora-processes`.
- [library] `SimulationResult.parquet_dir` and `SimulationResult.tables()`.
- [library] `colliderml/simulate/pipeline_config.yaml` pins the production-repo ref and both container images in one place (`load_pipeline_config()`), with the overrides `COLLIDERML_PRODUCTION_REF`, `COLLIDERML_SIM_IMAGE`, `COLLIDERML_RECO_IMAGE`.
- [library] `podman-hpc` is accepted as a container runtime; `colliderml simulate --runtime`.
- [docs] Tutorial "Simulate and reconstruct ttbar with particle flow".

### Changed
- [library] The pinned pipeline is `pipeline-v0.3.0` with the `sw:pr-8` sim image. This changes what a local simulation without particle flow writes, compared with 0.4.2:
  - The run directory's `tracker_hits` table now holds the digitised clusters. The Geant4 hits it held before are in the new `tracker_simhits` table. Code that read `tracker_hits` from a 0.4.2 run reads different content under the same name.
  - A `truth_tracks` table is added. The `calo_hits` table is no longer written (calorimeter cells come from the particle-flow stages).
  - The detector description is ODD v6.0.2 (0.4.2: v5.0.0).
  - The MadGraph output stays in `runs/all/<run_id>/events.hepmc.gz`; it is no longer copied into the run directory.
  - Tracking seeds at the 3 T detector field with a seed minimum pT of 0.65 GeV, and AMVF vertexing runs.
- [library] Geant4 and ACTS run with 2 threads by default, and `threads=1` is refused: with one thread the Geant4 step of the pinned sim image writes every event twice.
- [library] `simulate()` refuses an output directory that already holds the run (`FileExistsError`), validates `run_id`, and fails if the production checkout lacks the config of a stage instead of skipping it.
- [library] The per-run stage configs are written to `<output_dir>/configs/<run_id>/` (0.4.2: `.overlays/` in the cached production checkout, shared between runs).
- [library] Every stage's wall time and peak memory are written to `<output_dir>/pipeline_timing.csv`.
- [library] Stages no longer have a two-hour timeout.

### Fixed
- [library] `colliderml simulate` without `--image` passed `None` as the image name; it now uses the pinned image.
- [library] `colliderml.simulate(...)` raised `TypeError: 'module' object is not callable` once anything had imported the `colliderml.simulate` subpackage; it is now callable in any import order.
- [library] A cached production checkout that could not be switched to the requested ref was used as it was; this is now an error.
- [library] The MadGraph output was copied between run directories on the host, which fails with Docker (the directory belongs to root). Pythia now reads it where MadGraph left it.
- [docs] `colliderml.load()` does not read a simulation output directory; the local-simulation guide said it did.

## [0.4.2] - 2026-06-28

### Fixed
- [library] Default backend URL repointed from the dead `api.colliderml.com` (NXDOMAIN) to the live host, so out-of-the-box `colliderml.remote.*` / `colliderml.tasks.submit(...)` calls reach the SaaS backend. Override with `$COLLIDERML_BACKEND` as before.
- [library] Removed the phantom `single_muon` simulation channel: it had no released dataset config and was mis-routed through Pythia generation (it needs an ACTS particle-gun stage). It now errors cleanly instead of silently running a wrong pipeline.

### Changed
- [library] `simulate` uses the native-Arrow OpenDataDetector v5.0.0 image and drops the separate parquet-conversion stage (digitization writes parquet directly).
- [packaging] The `docs/` VitePress site is no longer bundled into the sdist (it had pulled ~5 MB of detector images / event JSON and a local `node_modules` into the source distribution).

## [0.4.1] - 2026-05-27

### Fixed
- [library] `colliderml.simulate(events=N, pileup=M, seed=K)` kwargs now actually take effect. They used to be advertised in the user-facing "Running … (N events, pileup=M)" message, then silently dropped — the static per-stage YAML configs won. The fix writes a per-run overlay under `prod_root/.overlays/<run_id>/` with the kwargs applied; each stage reads the overlay instead of the static config.
- [docs] Landing-page widget (`DataConfig.vue`) now filters channel buttons by the selected pileup level. The widget previously let users pick combinations that don't exist on Hub (e.g. `zee+pu0`, `higgs_portal+pu200`) and copied a CLI command that 404s on download. When the user switches pileup and the current channel isn't available there, the widget falls back to the first valid channel.
- [docs] Tutorial chapter 2 rewritten to show the three simulate knobs (`channel`, `events`, `pileup`) explicitly instead of hiding them behind a preset name. Defaults to a fast `events=2 pileup=5` demo.

## [0.4.0] - 2026-05-27

### Added
- [dataset] HF-native leaderboard score store: `huggingface.co/datasets/CERN/colliderml-benchmark-results` receives one JSON per submission at `results/<task>/<user>/<sha12>.json`. Pairs with the new client-side `colliderml.tasks.submit(..., model_repo_id=...)` which also lands a `.eval_results/colliderml_<task>.yaml` on the user's HF model repo so scores auto-render on the model card.
- [dataset] `CERN/ColliderML-Release-1` configs reorganised: per-pileup/per-object split (`{channel}_{pileup}_{tracker_hits,particles,calo_hits,tracks}`) replaces the prior monolithic per-channel layout. Enables event-range-scoped downloads (the `tracking` eval split is now ~700 MB to pull instead of multiple GB).
- [library] Top-level `colliderml.load(dataset, *, tables, max_events, event_range, …)` — a one-liner that downloads missing parquets on demand, then loads them with the existing Polars backend. Uses the same `<channel>_<pileup>` shorthand as the benchmark task runner.
- [library] `colliderml.simulate` subpackage — run the full Pythia / MadGraph / Geant4 / ACTS pipeline locally inside a Docker or Podman container, or submit to the SaaS backend with `remote=True`. Local simulate auto-clones the companion `colliderml-production` repo at a pinned git ref for pipeline scripts, mirroring the existing ODD and MG5aMC_PY8_interface cache pattern. Ships a bundled preset catalogue (`ttbar-quick`, `higgs-portal-quick`, `ttbar-dev`, `ttbar-benchmark`, …) accessible via `colliderml.simulate.load_presets()`.
- [library] `colliderml.remote` subpackage — HTTP client for the ColliderML SaaS backend service. Exposes `submit`, `status`, `wait_for`, `get_me`, `balance`, and a `RemoteSubmission` dataclass. Authentication is via a HuggingFace token resolved from the environment or the hub's saved credentials.
- [library] `colliderml.tasks` subpackage — registry and runner for six benchmark tasks: `tracking`, `jets`, `anomaly`, `tracking_latency`, `tracking_small`, `data_loading`. Each task defines a dataset, eval event range, input tables, metrics, and a `higher_is_better` direction. Reference baselines (CKF for tracking, scikit-learn GBDT for jets, IsolationForest for anomaly detection) run as `python -m colliderml.tasks.<task>.baselines.<name>`.
- [library] CLI subcommands: `simulate`, `list-presets`, `balance`, `status`. Each handler lazily imports its subsystem so existing `download` / `list-configs` commands stay fast.
- [library] Optional extras: `[sim]`, `[remote]`, `[tasks]`, and an `[all]` meta-extra that bundles everything plus `[dev]`. Base install stays lean.
- [library] `pyarrow>=14.0.0` added to `install_requires` (used by the task scoring code at the polars/arrow boundary).
- [docs] New guide pages: `guide/simulation`, `guide/remote-simulation`, `guide/tasks`. New library reference pages: `library/simulate`, `library/remote`, `library/tasks`. Navigation and sidebar extended to surface the new content.
- [docs] Installation page documents the new optional extras with prerequisites.
- [docs] Quickstart and library overview pages extended with simulate + tasks examples.
- [infra] `docs/.vitepress/config.ts` reads `VITEPRESS_BASE` from the environment so the same config builds both the canonical `/ColliderML/` site and a `/ColliderML/staging/` preview (infrastructure landing in a sibling commit; see B7).

### Changed
- [library] Top-level `colliderml` package now uses a `__getattr__` lazy loader for the optional `simulate`, `remote`, and `tasks` subsystems so `import colliderml` stays cheap when extras aren't installed.

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
