# Detector bundles

Each subdirectory here is a **detector bundle**: a self-contained set of files
that lets the ColliderML pipeline (Pythia/MadGraph → ddsim → ACTS digi+reco →
parquet) run against one detector. The pipeline reads exactly one file per
bundle directly — `detector.yaml` — and that manifest points at everything
else.

The plan for which detectors are in scope and why this shape was chosen lives
in [`docs/internal/unified-detector-geometry.md`](../docs/internal/unified-detector-geometry.md).

## What's here today

| Bundle | Status | Notes |
| :-- | :-- | :-- |
| [`odd/`](./odd/)   | reference | mirrors what the pipeline does today; used as the regression test for the bundle loader |
| [`cld/`](./cld/)   | draft     | k4geo `CLD_o2_v05`; ACTS digi config is best-effort and labelled DRAFT |
| [`lhcb/`](./lhcb/) | sketch    | Run 3 geometry from `lhcb/Detector`; forward-spectrometer ACTS support is a separate workstream |

Status meanings:

- **reference** — runs end-to-end, byte-identical to pre-bundle pipeline output on a regression sample.
- **draft** — manifest and configs written, geometry fetchable from upstream, ddsim and ACTS reco expected to run but reco quality unvalidated.
- **sketch** — manifest written, key parts of the pipeline (e.g. ACTS forward tracking) known to need new code; do not expect end-to-end runs yet.

## Bundle layout

```
<detector>/
  detector.yaml          ← the manifest (only file the pipeline parses)
  README.md              ← what's in this bundle, what's stub vs real
  SOURCES.md             ← exact upstream URLs + pinned revisions
  setup.sh               ← fetches geometry / plugins / large data
  geometry/              ← DD4hep compact XML(s) — usually populated by setup.sh
  acts/
    tracking.yaml        ← how ACTS converts the DD4hep tree into a TrackingGeometry
    digi.json            ← per-sensor segmentation and smearing
    material_map.root    ← optional, populated by setup.sh
  field/                 ← only if non-uniform; usually a ROOT field map
  reco/
    seeding.yaml         ← seed finder cuts
    ckf.yaml             ← CKF / ambiguity solver tuning
  output/
    schema_mapping.yaml  ← DD4hep volume IDs → ColliderML parquet enums
```

A bundle does **not** vendor large binaries (geometry XML trees, `.so`
plugin libraries, `.root` material/field maps) into this repo. Those come
from the upstream geometry packages, fetched via `setup.sh` and pinned by
revision in `SOURCES.md`. Text configs live in-tree.

## The manifest

`detector.yaml` is validated against [`SCHEMA.json`](./SCHEMA.json) (JSON Schema
2020-12). The schema is the single source of truth for what a manifest may
contain; the per-bundle YAMLs and the loader both validate against it.

A minimal manifest looks like:

```yaml
name: CLD
version: 2024-08-15
provenance:
  upstream: https://github.com/key4hep/k4geo
  upstream_path: FCCee/CLD/compact/CLD_o2_v05/CLD_o2_v05.xml
  upstream_revision: <git-sha>
geometry:
  compact_xml: geometry/CLD_o2_v05.xml
  plugins:
    - libk4geo.so
acts:
  tracking_config: acts/tracking.yaml
  digi_config:     acts/digi.json
field:
  type: uniform_solenoid
  Bz_T: 2.0
machine:
  collider: FCC-ee
  beam: e+e-
  cms_energy_GeV: [91, 160, 240, 365]
output_mapping: output/schema_mapping.yaml
```

See `SCHEMA.json` for the full set of fields.

## Adding a new bundle

1. Copy `odd/` as a starting point; rename in-place.
2. Edit `detector.yaml` to describe the new detector.
3. Replace `SOURCES.md` and `setup.sh` to point at the upstream geometry.
4. Replace `acts/digi.json` and `output/schema_mapping.yaml` for the new sub-detectors.
5. Run the loader's smoke test (lives in `colliderml-production`): it loads the bundle, runs ddsim on 10 events, runs ACTS reco, and writes parquet. If that passes the bundle is at least "draft".
6. Promote to "reference" only after a TDR-figure validation (material budget vs $\eta$, basic resolution plots) signs off.
