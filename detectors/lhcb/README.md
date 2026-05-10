# LHCb bundle

Sketch bundle for the [LHCb](https://lhcb-public.web.cern.ch/) Run 3+ detector.
Geometry comes from [`lhcb/Detector`](https://gitlab.cern.ch/lhcb/Detector),
which is the official DD4hep description LHCb adopted for Run 3.

This bundle is **not runnable end-to-end** through the standard pipeline
yet. It exists to make the LHCb-specific blockers concrete and to give the
forward-tracking workstream a place to land its outputs.

## What's in-tree

| File | Real / sketch | Notes |
| :-- | :-- | :-- |
| `detector.yaml`              | real    | manifest, with `tracking_builder: forward_custom` |
| `acts/tracking.yaml`         | sketch  | parameters the custom builder would consume |
| `acts/digi.json`             | sketch  | per-sub-detector digi entries; numbers from LHCb upgrade TDRs |
| `reco/seeding.yaml`          | sketch  | placeholder; LHCb reco doesn't use ACTS seeding as-is |
| `reco/ckf.yaml`              | sketch  | placeholder |
| `output/schema_mapping.yaml` | sketch  | maps what fits into the canonical enums; flags what doesn't |

## What's fetched

`setup.sh` clones `lhcb/Detector` at the pinned revision and symlinks
`compact/run3/LHCb.xml` into `geometry/`. The plugin
`libDetDescDD4hep.so` must be loadable; this normally means the LHCb stack
container or an LCG environment with LHCb's `Detector` package built.

## Why this is "sketch" not "draft"

Three concrete blockers, ordered by effort:

### 1. ACTS forward tracking — the big one

ACTS' DD4hep auto-converter walks the geometry tree and, for each volume
tagged with the right ACTS extension, produces a barrel layer or an endcap
disc. LHCb has neither. VELO is two retractable silicon halves opening
around the beam line; UT and SciFi are flat planar stations downstream of
the magnet. Wrapping these in cylinders/discs would be physically wrong
and would break the navigator.

ACTS *can* support forward tracking — the framework exposes
`TrackingGeometryBuilder` and `Layer` primitives that don't assume
cylindrical symmetry — but using them for LHCb means **writing a custom
builder**: identify VELO modules, group UT and SciFi planes into
"detector stations", build planar `Layer` objects, configure a navigator
that understands the magnet region. Estimate: 2–3 weeks for a working
prototype.

The bundle declares this in the manifest:
```yaml
acts:
  tracking_builder: forward_custom
  tracking_entry_point: colliderml_lhcb.tracking:build_lhcb_tracking_geometry
```
The Python entry point `colliderml_lhcb.tracking:build_lhcb_tracking_geometry`
does not yet exist. That is the deliverable that promotes this bundle from
"sketch" to "draft".

### 2. Output schema gaps

The canonical `colliderml/physics/detector_enums.py` defines tracker enums
as pixel/short-strip/long-strip × neg-endcap/barrel/pos-endcap. None of
those keys describe VELO, UT, or SciFi:

| LHCb sub-detector | Canonical enum fit | Notes |
| :-- | :-- | :-- |
| VELO              | `pixel_pos_endcap` (forced) | VELO is a forward silicon strip detector; calling it a pixel endcap is an abuse but at least keeps the data flowing |
| UT                | `short_pos_endcap`         | Strip layers, ~250 µm pitch |
| SciFi (T1–T3)     | `long_pos_endcap`          | Scintillating fibre, ~250 µm resolution but not silicon |
| ECAL              | `ecal_pos_endcap`          | Shashlik, fits the calo enum reasonably |
| HCAL              | `hcal_pos_endcap`          | Iron-scintillator |
| RICH1/RICH2       | unmapped                    | Cherenkov rings — needs schema extension |
| M2-M5 muon        | unmapped                    | No muon enum in v1 schema |

This is the closest "lossy v1" can be. Doing it properly needs the schema
extension tracked as an open question in the planning doc (`docs/internal/unified-detector-geometry.md`).

### 3. Conditions and alignment

LHCb in production runs with database-served alignment, calibration, and
dead-channel maps. The bundle assumes ideal alignment, which is reasonable
for ML training samples but **must be surfaced** when a user asks for a
sample-vs-data comparison. The loader should print a banner the first time
this bundle is used pointing at this caveat.
