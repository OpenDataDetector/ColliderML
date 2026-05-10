---
search: false
---

# Unified detector geometry — plan

::: warning Status
Planning document for the `claude/unified-detector-geometry-cLQqY` branch.
No code yet. The deliverable is the per-detector strategy, the pipeline
contract, and the honest list of where this gets hard.
:::

## Goal

Run the existing ColliderML pipeline (Pythia/MadGraph → ddsim → ACTS digi+reco
→ parquet) for **any major collider detector**, plus IceCube as a special
case, by treating **DD4hep as the single geometry language**. We are *not*
inventing a new description format. We are collecting and, where necessary,
porting existing geometries into DD4hep so one pipeline can drive all of them.

Target detector set:

- **Already DD4hep**: ODD, FCC-ee CLD / IDEA / ALLEGRO, SiD, ILD, MuColl
- **DD4hep, but coupled to its experiment framework**: LHCb (Run 3+), CMS
- **Not DD4hep**: ATLAS (still GeoModel)
- **Wrong physics for DD4hep+Geant4**: IceCube (geometry only — see [§ Why Geant4 cannot simulate IceCube](#why-geant4-cannot-simulate-icecube))

## How to define the pipeline

The instinct — *"isn't it just a YAML pointing to a detector geometry?"* — is
correct in shape but understates what the pipeline currently consumes
implicitly. ddsim, ACTS, and the ColliderML output writer each pull more than
just the compact XML. Today all of that is hard-coded against ODD.

### What ddsim + ACTS actually need at runtime

For a single end-to-end run, a "detector" is not one file. It is a small
bundle:

| Asset | Used by | Per detector? | Source today (ODD) |
| :-- | :-- | :-- | :-- |
| DD4hep compact XML (root + included sub-files) | ddsim, ACTS converter | yes | `OpenDataDetector/xml/OpenDataDetector.xml` |
| DD4hep plugin libraries (C++ detector constructors) | ddsim, ACTS | yes | `OpenDataDetector/factory/*` compiled into `libOpenDataDetector.so`, found via `LD_LIBRARY_PATH` |
| ACTS tracking-geometry conversion config (which volumes are barrel/endcap, layer grouping, surface selection) | ACTS reco | yes | baked into `acts.examples.odd.getOpenDataDetector()` |
| ACTS digitization config (segmentation, smearing, time resolution) | ACTS digi | yes | `OpenDataDetector/config/odd-digi-geometric-config.json` |
| ACTS material map | ACTS reco | recommended | `OpenDataDetector/data/odd-material-maps.root` |
| Magnetic field (uniform → in XML; non-uniform → ROOT field map) | ddsim, ACTS | yes | XML solenoid in `OpenDataDetector.xml` |
| Seeding / CKF / ambiguity tuning | ACTS reco | yes (per detector, possibly per process) | hard-coded in `digi_and_reco.py` |
| Vertex / luminous-region distribution | `merge_and_smear.py` | **per accelerator, not detector** | hard-coded LHC values |
| Output schema mapping (which DD4hep volume IDs map to which ColliderML parquet enum) | output writer | yes | `colliderml/physics/detector_enums.py` |

### The right abstraction: a detector descriptor bundle

The pipeline YAML stays simple. The complexity moves into a versioned
**detector descriptor** — one directory (or pip-installable subpackage) per
detector, with a fixed contract.

```
detectors/
  odd/
    detector.yaml             # entry point, see schema below
    geometry/                 # the DD4hep XML tree
      OpenDataDetector.xml
      ...
    plugins/                  # built shared libs, or a pointer to a wheel
      libOpenDataDetector.so
    acts/
      tracking.json
      digi.json
      material_map.root
    field/                    # only if non-uniform; otherwise omit
      field_map.root
    reco/
      seeding.yaml
      ckf.yaml
    output/
      schema_mapping.yaml     # volume_id → parquet enum
  cld/
    detector.yaml
    geometry/                 # symlink or pointer into k4geo
    ...
```

`detector.yaml` is the only file the pipeline parses directly. It describes
the bundle and declares accelerator intent so the pipeline can refuse
nonsensical combinations (LHC pileup against an FCC-ee detector, etc.):

```yaml
# detectors/cld/detector.yaml
name: CLD
version: 2024-08-15
provenance:
  upstream: https://github.com/key4hep/k4geo
  upstream_path: FCCee/CLD/compact/CLD_o2_v05/CLD_o2_v05.xml
  upstream_revision: <git-sha>
geometry:
  compact_xml: geometry/CLD_o2_v05.xml
  plugins:
    - libk4geo.so          # discovered via DD4HEP_LIBRARY_PATH
acts:
  tracking_config: acts/tracking.json
  digi_config:     acts/digi.json
  material_map:    acts/material_map.root      # optional
field:
  type: uniform_solenoid     # alternatives: field_map, multi_region
  Bz_T: 2.0
machine:
  collider: FCC-ee
  cms_energy_GeV: [91, 160, 240, 365]
  beam: e+e-
  default_vertex_distribution:
    sigma_x_um: 6.6
    sigma_y_um: 0.05
    sigma_z_mm: 1.5
output_mapping: output/schema_mapping.yaml
notes: |
  CLD digitization config is conservative; revisit once k4geo ships official.
```

Then the pipeline YAML the user writes is genuinely small:

```yaml
# pipeline.yaml
detector: cld                      # path or registered name
generator:
  type: pythia
  process: ee_to_zh
  events: 10000
output:
  schema: colliderml-v1
```

### Pipeline-stage changes required

Concrete edits in `colliderml-production`:

1. **`scripts/simulation/ddsim_run.py`** — replace `getOpenDataDetectorDirectory()` with `Detector.from_yaml(path).compact_xml()`; ensure `dd4hep_load_libraries(detector.plugins)` is called before construction.
2. **`scripts/simulation/digi_and_reco.py`** — currently calls `acts.examples.odd.getOpenDataDetector(...)`. Replace with a generic constructor that takes the bundle's tracking/digi/material configs.
3. **`scripts/simulation/merge_and_smear.py`** — read `default_vertex_distribution` from the detector bundle (or a separate accelerator descriptor) instead of hard-coded LHC values.
4. **`Snakefile`** — parametrise rule wildcards on `{detector}` and stage outputs under `data/{detector}/{process}/`.
5. **Output writer** — load `output_mapping.yaml` and translate DD4hep volume IDs to the ColliderML parquet schema. The current `colliderml/physics/detector_enums.py` becomes the *canonical schema* and each detector bundle declares its mapping into it. Detectors with sub-systems that don't fit (e.g. LHCb RICH, FCC dual-readout) need schema extensions — that is genuinely new work, not just plumbing.

### What is and isn't in the YAML

The detector bundle is a small *typed* contract, not a free-form description.
That is the discipline that keeps the pipeline tractable. Things that look
like they belong in the YAML but **shouldn't** be:

- The geometry itself. It belongs in DD4hep XML + plugins. The YAML points at it.
- ACTS internals (`PropagatorOptions`, `Navigator` choice, etc.). Defaults live in the pipeline; the bundle only overrides what is genuinely detector-dependent.
- Per-process tuning (jet algorithm, b-tagging WPs). That belongs in the pipeline YAML, not the detector bundle.

The YAML earns its place by being the **only** thing that varies when a user
asks "run the same workflow against a different detector".

## Per-detector status and plan

### ODD — done

Already shipping. Use as the reference bundle and the regression test for the
new bundle loader. If we can swap ODD's call sites for the new
`Detector.from_yaml()` API and produce byte-identical parquet on a small
ttbar sample, the abstraction is right.

### k4geo detectors (CLD, IDEA, ALLEGRO, SiD, ILD, MuColl) — first phase

[`key4hep/k4geo`](https://github.com/key4hep/k4geo) ships DD4hep XML and
compiled plugins for all of these. Concrete steps:

1. Vendor (or `pip install k4geo` once a Python wheel exists; otherwise depend on the key4hep nightly via cvmfs/Spack) and inspect the available compact files: `FCCee/CLD/compact/`, `FCCee/IDEA/compact/`, `FCCee/ALLEGRO/compact/`, `ILD/compact/`, `SiD/compact/`, `MuColl/compact/`.
2. For each, write a `detector.yaml` against the schema above.
3. The non-trivial item is the **ACTS tracking config**: ACTS reads DD4hep extensions (`ACTS_TYPE`, `acts_layertype` etc.) to know which volumes are tracking. ODD has them; k4geo detectors have varying coverage. CLD and IDEA are reasonably mature; ALLEGRO less so. Where the extensions are missing we either contribute them upstream or write a thin DD4hep `Plugin` that decorates the volumes at load time.
4. Digi configs in k4geo are sparse. We seed reasonable defaults from each detector's TDR (pixel pitch, strip length, calo cell size) and mark them as "draft" in the bundle metadata.
5. Acceptance test per detector: ddsim runs N events without errors; ACTS reco produces a populated tracks table; ColliderML loader reads the parquet.

Estimate: one well-supported detector (CLD) in ~1 week including the bundle-loader refactor; subsequent k4geo detectors in 1–3 days each, mostly digi/material tuning.

### LHCb (Run 3+) — second phase

[`lhcb/Detector`](https://gitlab.cern.ch/lhcb/Detector) is the DD4hep-based
description LHCb adopted for Run 3+. It is the official source. The complications:

- It is intended to be loaded by Gauss/Gaudi inside Moore. Our pipeline uses ddsim directly. Most of the geometry should load — DD4hep is DD4hep — but anything that depends on a Gaudi service (conditions, alignment) needs a stand-in.
- LHCb is forward-spectrometer: very different acceptance from ODD/CMS/ATLAS. The ACTS tracking-volume conventions assume a barrel+endcap collider geometry. ACTS does support forward geometries but the tracking config is non-trivial — likely needs a custom `TrackingGeometryBuilder` rather than the generic DD4hep auto-converter.
- Sub-systems with no analogue in the current ColliderML schema: RICH, hadronic calo with specific transverse segmentation, muon system with strong forward boost. Schema extension required for first-class support; v1 can output what ACTS gives us (tracks + ECAL clusters) and document the gaps.

Estimate: 2–3 weeks for ddsim+ACTS-on-LHCb, plus schema extensions to actually expose RICH/muon/etc. data downstream.

### CMS — third phase

CMS migrated to DD4hep around 2020. The geometry lives inside `cms-sw/cmssw`
under `Geometry/CMSCommonData`, `Geometry/TrackerCommonData`, etc., loaded via
`DetectorDescription/DDCMS`. Two viable extraction paths:

- **(a)** Carve out the geometry XML and the DDCMS plugin into a standalone repo. Tracked by CMS as a side experiment — they have done this for some studies. Big effort, plays well with our model.
- **(b)** Depend on a CMSSW environment for the geometry-load step only, then dump a TGeo + GDML once and feed the dumped artifact to ddsim downstream. Fragile (no live updates) but unblocks a v1 quickly.

Path (b) is the pragmatic v1; path (a) is the right long-term answer. Either
way ACTS already has a CMS-tracker tracking-geometry converter, which removes
one of the bigger risks.

Estimate: 4–8 weeks depending on path.

### ATLAS — fourth phase, riskiest

ATLAS uses [GeoModel](https://geomodel.web.cern.ch/), not DD4hep. There is no
official GeoModel→DD4hep converter. Realistic options:

- **GDML round-trip.** Athena can dump the ATLAS geometry to GDML; DD4hep has `DD4hep_GdmlReader`. The dump captures volumes and materials but **loses sensitive-detector wiring, segmentation, ACTS extensions, and any conditions-driven pieces (alignment, deformations)**. We re-attach those on the DD4hep side via a thin plugin layer. Material accounting is preserved; tracking-geometry conversion needs careful per-subsystem work.
- **Native DD4hep re-implementation.** Write DD4hep detector constructors from the ATLAS TDR/upgrade-TDR drawings. Months of work; benefit is a clean DD4hep tree with proper extensions. Probably overkill for ColliderML's purposes.
- **Use ATLAS GeoModel directly in our simulation.** Means running Athena's simulation, not ddsim. Defeats the unification goal.

Recommended: GDML round-trip for v1, with a clearly-labelled "approximate" tag
in `detector.yaml`. Realistic comparisons with ATLAS official samples will
require validating tracking-volume material budgets against the upstream.

Estimate: 4–6 weeks for a usable GDML-imported ATLAS in the pipeline; longer
to get reconstruction quality close to official.

### IceCube — geometry only

See the next section. IceCube goes into DD4hep as a geometry artifact (DOMs in
an ice cylinder with optical-property tables) and the ColliderML pipeline
*does not* claim to simulate IceCube events end-to-end. The deliverable is
the geometry plus a documented hand-off to CLSim/PPC for photon transport.

## Why Geant4 cannot simulate IceCube {#why-geant4-cannot-simulate-icecube}

This is the single most important honest point in the plan. DD4hep can hold
the IceCube geometry. ddsim — which is just DD4hep wired to Geant4's standard
EM/hadronic physics lists — cannot run a real IceCube event in any usable
amount of time. The reason is photon counts, not geometry.

### The photon-count problem

A relativistic charged particle in glacial ice emits Cherenkov photons at a
rate of **~250 photons per cm** in the 300–600 nm band that IceCube DOMs
detect (Frank–Tamm with $n_\text{ice} \approx 1.32$).

Typical IceCube events:

| Event class | Track / cascade scale | Cherenkov photons per event |
| :-- | :-- | :-- |
| Atmospheric muon, 100 GeV | ~300 m through detector | $\sim 10^{9}$ |
| Through-going $\nu_\mu$ CC, 1 TeV | km-scale muon track + secondaries | $\sim 10^{10}$ |
| Cascade, 1 PeV (e.g. Glashow) | tens of m, very dense | $\sim 10^{11}$ |
| Supernova burst, MeV neutrinos | many low-energy interactions | $\sim 10^{6}$ per neutrino × $10^{4}$/s |

These are not exotic events. They are the population IceCube reconstructs
every day.

### What Geant4 does with optical photons

Geant4 propagates each optical photon as an individual `G4Track`. Each step
runs the optical physics list (`G4OpRayleigh`, `G4OpAbsorption`,
`G4OpBoundaryProcess`, `G4OpMieHG`), queries the navigator for the next
volume, samples a step length, updates the stack, and so on. Empirical
throughput on a single CPU core is roughly **$10^{5}$–$10^{6}$ optical
photons per second**, depending on geometry complexity and whether
photocathode boundary processes fire.

Multiply through:

$$
\frac{10^{10}\ \text{photons / event}}{10^{6}\ \text{photons / CPU-s}}
  \;=\; 10^{4}\ \text{CPU-s / event}
  \;\approx\; 3\ \text{CPU-hours / event}
$$

For the 1 PeV cascade case, ~30 CPU-hours per event. Reconstructing IceCube's
~10⁵ atmospheric events per day at this rate is a >$10^9$ CPU-hour annual
bill. It is not a tuning problem; it is the wrong tool.

### What IceCube actually does

IceCube uses [**CLSim**](https://github.com/claudiok/clsim) and **PPC** —
GPU-resident photon propagators that achieve **$10^{9}$–$10^{10}$ photons
per second per GPU**, four to five orders of magnitude faster than
Geant4-on-CPU. They also implement physics Geant4 doesn't:

- The **SPICE** family of ice models (SPICE-Mie, SPICE-3.2, SPICE-BFR) parameterise scattering and absorption as functions of *depth* (every ~10 m, with explicit dust layers from the South Pole ice cores) and *wavelength*. The scattering kernel is a Henyey–Greenstein with $\langle \cos\theta \rangle \approx 0.94$ plus a flasher anisotropy term. Geant4's `G4Material::OpticalProperties` supports wavelength-dependent absorption length and refractive index, but **does not support depth-stratified scattering models or the SPICE angular distribution out of the box** — you would have to subclass `G4VProcess`.
- **Hole ice**: the refrozen column around each string has different optical properties from the bulk ice and matters for high-precision angular reconstruction. Modelled as a per-DOM cylinder in CLSim; would require careful DD4hep + Geant4 plumbing.
- **DOM angular acceptance**: PMT + cable shadow + glass + gel acceptance is tabulated and applied at the photocathode. Geant4 can do this via `G4OpticalSurface` properties, but the standard IceCube ANGSENS tables are not in a Geant4-native format.

The actual icetray pipeline splits the simulation:

1. Geant4 (in `clsim`'s "particle propagator" mode) handles the **primary interaction** — neutrino-nucleon, muon energy loss producing the secondary cascade structure. This is well within Geant4's strengths and runs in seconds.
2. The secondary charged tracks are handed to **CLSim/PPC on GPU** for Cherenkov emission and photon propagation.
3. PMT response, DOM electronics simulation, and trigger live in icetray modules.

### What this means for our plan

DD4hep can express IceCube as a geometry: 86 strings × 60 DOMs (10" PMTs),
plus IceTop tanks at the surface, all sitting inside an ice cylinder ~1 km
radius / 1 km tall with depth-stratified material properties. This is useful:

- Single source of truth for DOM positions and ice properties, consumable by visualisation and analysis tools that already understand DD4hep.
- A clean export point for CLSim's geometry inputs.
- Does not lie about being simulatable through ddsim.

The IceCube `detector.yaml` therefore declares its simulator explicitly:

```yaml
# detectors/icecube/detector.yaml
name: IceCube
geometry:
  compact_xml: geometry/IceCube.xml
  plugins:
    - libIceCubeGeometry.so
simulation:
  primary: ddsim                        # Geant4 for the neutrino interaction
  optical: clsim                        # GPU photon propagation, external
  photon_handoff:
    format: icetray-i3
    notes: |
      ddsim emits charged-secondary tracks at the boundary of the
      sensitive ice volume. clsim takes over for Cherenkov emission and
      propagation. Full closed-loop simulation lives outside this pipeline.
machine:
  source: cosmic / atmospheric / astrophysical
```

The pipeline writer should refuse to run a "full sim" on IceCube and point
the user at this contract.

## Phasing

| Phase | Work | Outcome |
| :-- | :-- | :-- |
| **0 — bundle abstraction** | `Detector.from_yaml()` loader; refactor ODD into a bundle; pipeline-script edits; regression test for ODD | Same outputs as today, but code is detector-agnostic. |
| **1 — k4geo** | Vendor k4geo, write CLD bundle end-to-end, then IDEA, ALLEGRO, SiD, ILD, MuColl | 6+ collider detectors running through one pipeline. |
| **2 — LHCb** | `lhcb/Detector` integration; ACTS forward-tracking config; schema extensions for RICH/muon | LHCb full sim through the same pipeline. |
| **3 — CMS** | Path (b) first: depend on CMSSW for geometry load, dump GDML/TGeo, run downstream stages on the dump. Path (a) later. | CMS in pipeline, with caveats about update cadence. |
| **4 — ATLAS** | Athena GDML dump → DD4hep import → re-attach sensitive detectors and ACTS extensions | ATLAS in pipeline, labelled "approximate". |
| **5 — IceCube** | DD4hep geometry artifact + CLSim hand-off contract; no ddsim closed loop | Geometry consumable; simulation explicitly out of scope. |

## Open questions

1. **Output schema versioning.** When LHCb adds RICH columns and FCC adds dual-readout columns, do we extend `detector_enums.py` once (and tolerate sparse columns per detector) or version the schema per detector family? The current parquet loader assumes fixed columns; this needs a decision before phase 2.
2. **Container vs cvmfs.** The pipeline today assumes cvmfs / LCG. Multiple detectors mean multiple plugin libraries; either we ship a fat container with all of them (image bloat, conflicting transitive deps), or we split per detector (more images to maintain). Probably one container per detector, sharing a base layer with ddsim/ACTS/Pythia.
3. **Field maps.** ATLAS and CMS have non-uniform field maps that are large ROOT files under restricted access. We need an answer for where the field map lives in the bundle (in-repo, downloaded on first use, mounted from cvmfs).
4. **Conditions / alignment.** Real LHCb/CMS/ATLAS use database-served conditions for alignment, calibration, and dead channels. ColliderML can ignore that for ML-target samples (run with ideal alignment) but it should be a documented assumption in each bundle.
5. **Validation.** For each ported detector we need a sanity check beyond "it runs". For collider detectors, comparing tracking-volume material budgets ($X_0$ vs $\eta$) against published TDR figures is the cheapest validation. Should this be part of the bundle CI or a separate notebook?

## Out of scope (deliberately)

- A new geometry description language. We use DD4hep, full stop.
- Differentiable / ML-surrogate simulation. Compatible with this plan but a separate workstream — once detectors are bundled uniformly, learned surrogates can be trained per detector.
- Astroparticle detectors beyond IceCube (KM3NeT, P-ONE, Auger, CTA). Same physics gap as IceCube; revisit only if a user asks.
