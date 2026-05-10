# CLD upstream sources

| Asset | Upstream | Path | Revision |
| :-- | :-- | :-- | :-- |
| Geometry XML tree | https://github.com/key4hep/k4geo | `FCCee/CLD/compact/CLD_o2_v05/` | `v00-19-01` |
| Detector plugin (`libk4geo.so`) | provided by `key4hep/key4hep-spack` containers | — | — |
| ACTS material map | not available upstream | — | needs to be generated locally |

License: [GPL-3.0-or-later](https://github.com/key4hep/k4geo/blob/main/LICENSE).
The geometry XML and the bundle's text configs are not derived works of
the GPL'd plugin code; the bundle is licensed under the ColliderML repo
license but a runtime dependency on `libk4geo.so` is GPL.

Reference for design parameters used in `acts/digi.json`:
[N. Bacchetta et al., "CLD — A Detector Concept for the FCC-ee", arXiv:1911.12230](https://arxiv.org/abs/1911.12230).
