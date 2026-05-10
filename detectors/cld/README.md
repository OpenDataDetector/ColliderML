# CLD bundle

Draft bundle for [CLD](https://arxiv.org/abs/1911.12230), the FCC-ee detector
concept inherited from CLIC. Geometry comes from
[`key4hep/k4geo`](https://github.com/key4hep/k4geo) at
`FCCee/CLD/compact/CLD_o2_v05/`.

## What's in-tree

| File | Real / draft | Notes |
| :-- | :-- | :-- |
| `detector.yaml`              | real  | manifest |
| `acts/tracking.yaml`         | draft | DD4hep auto-converter; CLD's ACTS extensions are not as exhaustive as ODD's |
| `acts/digi.json`             | draft | seeded from CLD note arXiv:1911.12230 — vertex 3×3 µm pitch, IT/OT 7×572 µm strip-like, ECAL 5×5 mm cells, HCAL 30×30 mm cells |
| `reco/seeding.yaml`          | draft | $p_T$ cut and impact parameter cuts adjusted for FCC-ee Z-pole (lots of soft tracks, small luminous region) |
| `reco/ckf.yaml`              | draft | inherits ODD CKF defaults; should be retuned |
| `output/schema_mapping.yaml` | draft | maps CLD's vertex barrel/endcap onto `pixel_*`, IT/OT onto `short_*`/`long_*`; lossy and documented inline |

## What's fetched

`setup.sh` clones `k4geo` at the pinned revision and symlinks
`FCCee/CLD/compact/CLD_o2_v05/` into `geometry/`. The upstream
`libk4geo.so` plugin must be loadable (k4geo nightly container or local build).

## Known gaps before "reference"

1. **No ACTS material map.** k4geo doesn't ship one for CLD. Generating it
   needs a material-scan run through ddsim and a `MaterialMapper`
   propagator pass. Until that exists, ACTS reconstruction will be
   under-correcting energy loss and overestimating momentum resolution.

2. **Partial ACTS extension coverage.** ODD's `factory/` decorates every
   sensitive volume with the right ACTS extension; k4geo's CLD has these
   on the trackers but coverage on the calorimeters and forward regions is
   incomplete. May need a thin DD4hep plugin in `geometry/` (or upstreamed
   into k4geo) to fill the gaps.

3. **Schema mapping is lossy.** CLD's vertex tracker is a true pixel
   detector; mapping it to `pixel_*` is fine. CLD's IT and OT are silicon
   strip layers but their geometry (large discs and barrels with relatively
   long strips) doesn't align cleanly with ODD's "short strip" / "long
   strip" partition. The current mapping is a best-effort approximation and
   loses information. A proper fix needs schema extension — see the
   "Output schema versioning" open question in the planning doc.

4. **Digi config is from a TDR-style note, not measurements.** CLD has no
   testbeam data; the numbers here are nominal design values.

5. **Tracker volume IDs are stubbed.** The DD4hep volume IDs in
   `output/schema_mapping.yaml` are placeholders. They need to be filled in
   from the actual `CLD_o2_v05.xml` once `setup.sh` has fetched it. The
   loader should fail loudly if it sees an unknown ID rather than silently
   default; we keep the placeholder values explicit so this fails fast.
