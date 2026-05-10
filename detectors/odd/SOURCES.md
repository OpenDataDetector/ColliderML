# ODD upstream sources

| Asset | Upstream | Path | Revision |
| :-- | :-- | :-- | :-- |
| Geometry XML tree | https://github.com/OpenDataDetector/OpenDataDetector | `xml/` | `v3.0.0` |
| ACTS digitization config | https://github.com/OpenDataDetector/OpenDataDetector | `config/odd-digi-geometric-config.json` | `v3.0.0` |
| ACTS material map | https://github.com/OpenDataDetector/OpenDataDetector | `data/odd-material-maps.root` | `v3.0.0` |
| Detector plugin (`libOpenDataDetector.so`) | provided by container | — | — |

License: [MPL-2.0](https://github.com/OpenDataDetector/OpenDataDetector/blob/main/LICENSE).

Pin policy: bumping `upstream_revision` in `detector.yaml` requires re-running
the Phase-0 regression test (see this bundle's `README.md`). If parquet A vs B
differs, either the bump must be reverted or the regression baseline must be
explicitly re-blessed in the PR description.
