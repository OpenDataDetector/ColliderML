# LHCb upstream sources

| Asset | Upstream | Path | Revision |
| :-- | :-- | :-- | :-- |
| Geometry XML tree | https://gitlab.cern.ch/lhcb/Detector | `compact/run3/` | `v1r41` |
| Detector plugin (`libDetDescDD4hep.so`) | provided by LHCb stack containers | — | — |
| Field map | not vendored — see notes | — | — |
| ACTS material map | not available; would be generated locally if and when ACTS forward tracking lands | — | — |

License: [Apache-2.0](https://gitlab.cern.ch/lhcb/Detector/-/blob/master/LICENSE).

## Field map

LHCb's dipole has non-uniform field and operates in two polarities (magnet
up / magnet down). For serious physics samples the analysis-grade field
map (`FieldMap_LHCb.cdf`) is required and is several hundred MB. We do not
ship it in the bundle. Options:

- For most ML sample generation, the in-XML default field is good enough; the manifest declares `field.type: in_geometry`.
- For physics-quality samples, the field map must be staged separately and `detector.yaml` overridden to `field.type: field_map`.

## Access

`gitlab.cern.ch/lhcb/Detector` is publicly readable. No CERN credentials
are required for the geometry; only the LHCb stack container provides the
compiled plugin libraries.
