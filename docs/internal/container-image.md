---
search: false
---

# Container Image

The simulation pipeline runs inside a Docker/Shifter image built from
the [OpenDataDetector/sw](https://github.com/OpenDataDetector/sw) repository.

**Current tag:** `ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0`

## Contents

| Software | Version | Notes |
| :-- | :-- | :-- |
| MadGraph | 5.3.9 | NLO generation |
| Pythia | 8.313 | Parton shower + hadronisation |
| DD4hep (ddsim) | — | Geant4-based detector simulation |
| Geant4 | 11.3 | Transport engine |
| ROOT | 6.38 | I/O and analysis framework |
| HepMC3 | 3.3 | Event record format |
| ACTS | main (999.999.999) | Track reconstruction |
| GCC | 13.3.0 | Build toolchain |
| Python | 3.12 | System Python |

## Runtime setup

`scripts/cli/setup_container_env.sh` (sourced automatically by the
pipeline runner) handles first-run setup inside the container:

1. **ACTS Python bindings** — symlink `<prefix>/python/` → `/tmp/acts`
   so `import acts` works (workaround for non-standard install path)
2. **ODD v4.0.4** — clone from CERN GitLab, build `libOpenDataDetector.so`
   via CMake, cache in `.cache/odd-v4-install/`
3. **Geant4 data** — download ~2 GB of physics datasets into `.cache/g4data/`
4. **Pip packages** — install `pyarrow`, `uproot`, `pandas`, `awkward`,
   `h5py`, `tqdm`, `pyhepmc`, `psutil`, `pyedm4hep` into `.cache/pip/`
5. **MG5aMC_PY8_interface** — compile the MadGraph–Pythia8 interface
   with explicit HepMC2 flags (see known issues below)

## Known issues (fix in next image build)

### 1. Missing `bc` command

MadGraph checks for `bc` to calculate shower parameters. Without it,
MadGraph falls back to `noshower` mode and only produces LHE files.

**Fix:** `apt-get install -y bc` in the Dockerfile.

### 2. mg5amc_py8_interface incompatible with Pythia 8.3+

The C++ driver MadGraph uses to steer Pythia showering only works with
Pythia 8.2.x. This blocks NLO+PS generation (ttbar, SUSY, etc.).

**Current workaround** (in `setup_container_env.sh`): compile the
interface directly with explicit HepMC2 flags:

```bash
g++ MG5aMC_PY8_interface.cc -o MG5aMC_PY8_interface \
    -I$PY8/include -I$HEPMC2/include -I$ZLIB/include \
    -O2 -std=c++20 -fPIC -pthread -DGZIP \
    -L$PY8/lib -Wl,-rpath,$PY8/lib -lpythia8 \
    -L$HEPMC2/lib -Wl,-rpath,$HEPMC2/lib -lHepMC \
    -L$ZLIB/lib -Wl,-rpath,$ZLIB/lib -lz -ldl
```

**Long-term fix:** pre-install the interface with `--pythia8_makefile`
during image build, or port the interface to the Pythia 8.3 API.

### 3. Missing `pyedm4hep`

The `convert_all.py` postprocessing script needs `pyedm4hep`. Added to
the pip install list in `setup_container_env.sh` as a workaround.

**Fix:** add `pyedm4hep` to the image's pip install layer.

### 4. ACTS Python bindings path

ACTS installs to `<prefix>/python/` instead of
`<prefix>/lib/pythonX.Y/site-packages/acts/`. The setup script works
around this with a symlink.

**Fix:** adjust the ACTS spack recipe to install into `site-packages`,
or ship an `acts.pth` file.

## Build checklist for next image

- [ ] Add `bc` to system packages
- [ ] Pre-compile `MG5aMC_PY8_interface` with HepMC2 dynamic linking
- [ ] Add `pyedm4hep` to pip install layer
- [ ] Fix ACTS Python install path (or add `.pth` file)
- [ ] Verify `run_pipeline_docker.sh --channel ttbar` completes NLO+PS
      without the `setup_container_env.sh` workaround
- [ ] Tag as `0.3.0` and update `CONTAINER_IMAGE` defaults in backend
      `.env.example` and `colliderml/simulate/docker.py`

## Cache layout

All caches are stored under `<repo>/.cache/` (gitignored) on the host,
bind-mounted into the container:

```
.cache/
├── odd-v4/              # ODD v4.0.4 source (XML geometry)
├── odd-v4-install/      # Built libOpenDataDetector.so
├── g4data/              # Geant4 physics datasets (~2 GB)
├── pip/                 # Python packages for postprocessing
└── MG5aMC_PY8_interface/ # Compiled MG5-Pythia8 interface binary
```
