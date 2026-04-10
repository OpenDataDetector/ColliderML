# Event Display

**Live:** [huggingface.co/spaces/CERN/colliderml-event-display](https://huggingface.co/spaces/CERN/colliderml-event-display)
**Source:** [`spaces/event-display/`](https://github.com/OpenDataDetector/ColliderML/tree/main/spaces/event-display)

Interactive 3D viewer for single events from the ColliderML datasets.
Pick a process and an event ID to see the tracker hits, reconstructed
tracks, and truth particles inside the OpenDataDetector geometry —
rendered with Plotly so you can rotate, zoom, and toggle layers.

## What you can see

- **Blue points** — tracker hits, coloured by detector layer (or
  `volume_id` / `particle_id` if layer is missing).
- **Red lines** — reconstructed tracks, approximated as helical
  polylines from the perigee parameters (`d0`, `z0`, `phi`, `theta`,
  `qop`). This is a visualisation aid, not a physics-accurate
  reconstruction.
- **Yellow dashes** — truth-particle momentum vectors drawn from the
  primary vertex, restricted to primary particles and capped at the 20
  highest-momentum entries so the plot stays readable.

## Datasets

The dropdown ships with a curated subset of the catalogue so the UI
stays responsive on the Space's modest resources:

```
ttbar_pu0
ttbar_pu200
higgs_portal_pu10
zmumu_pu0
diphoton_pu0
single_muon_pu0
```

Each selection loads at most 50 events into memory. If you want a
different process, open an issue — or run the Space locally and edit
`DATASETS` in `app.py`.

## Data loading

The Space prefers its own on-disk cache
(`_cached_events/<dataset>/*.parquet`) if present; `cache_events.py`
populates it by calling `colliderml.load()` once for each dataset and
writing the results to parquet. On a fresh Space with an empty cache
the app falls back to calling `colliderml.load()` live from the
dataset's HuggingFace repo — slower on first hit, then cached in
memory via `functools.lru_cache`.

## Running it locally

```bash
cd spaces/event-display
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

## Customisation

- Add a dataset → append to `DATASETS` in `app.py` and re-run
  `cache_events.py`.
- Add a visualisation layer (e.g. calo cells) → follow the pattern for
  `tracker_hits` in `render_event()` and add a new `colliderml.load`
  table to the list in `_load_dataset()`.
- Change the colour scheme → the Plotly `Scatter3d` traces carry all
  the knobs; search for `Viridis` and `crimson` in `app.py`.
