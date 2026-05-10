# ODD bundle

Reference bundle for the [Open Data Detector](https://github.com/OpenDataDetector/OpenDataDetector).

This bundle exists to validate the detector-bundle abstraction itself: the
ColliderML pipeline already runs ODD via hard-coded calls
(`getOpenDataDetectorDirectory()`, `acts.examples.odd.getOpenDataDetector()`).
Running ODD through the generic loader and producing **byte-identical
parquet on a fixed ttbar sample** is the acceptance criterion for Phase 0.

## What's in-tree

| File | Real / stub | Notes |
| :-- | :-- | :-- |
| `detector.yaml`              | real | manifest |
| `acts/tracking.yaml`         | real | maps to the existing ODD ACTS converter call |
| `acts/digi.json`             | pointer | fetched by `setup.sh` from upstream `config/odd-digi-geometric-config.json` |
| `reco/seeding.yaml`          | real | mirrors the seeding cuts hard-coded in `digi_and_reco.py` today |
| `reco/ckf.yaml`              | real | same |
| `output/schema_mapping.yaml` | real | mirrors today's `detector_enums.py` mapping |

## What's fetched

`setup.sh` clones `OpenDataDetector` at the revision pinned in `SOURCES.md`
and symlinks:

- `xml/OpenDataDetector.xml` (and its includes) → `geometry/`
- `data/odd-material-maps.root`                  → `acts/odd-material-maps.root`
- `config/odd-digi-geometric-config.json`        → `acts/digi.json`

The `libOpenDataDetector.so` plugin must already be loadable via
`DD4HEP_LIBRARY_PATH` (provided by the production container or by an
LCG/Spack environment). The bundle does not build it.

## Validation

The Phase-0 regression test is in `colliderml-production` (not yet written):

1. Run the legacy ODD path on a fixed Pythia ttbar seed — write parquet A.
2. Run the same seed through the bundle loader using this bundle — write parquet B.
3. Assert `A == B` row-for-row.

If that passes, this bundle is "reference". The other bundles (`cld`, `lhcb`)
are graded "draft" or "sketch" until they have analogous (looser) checks.
