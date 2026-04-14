"""
ColliderML Event Display

Interactive 3D visualisation of single events from ColliderML datasets.
Loads tracker hits, truth particles, and reconstructed tracks from a
chosen event and renders them with Plotly.

Runs as a Gradio HuggingFace Space.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import colliderml

    HAS_COLLIDERML = True
except ImportError:
    HAS_COLLIDERML = False

# Datasets exposed in the UI. Kept short so the dropdown isn't overwhelming.
DATASETS = [
    "ttbar_pu0",
    "ttbar_pu200",
    "higgs_portal_pu10",
    "zmumu_pu0",
    "diphoton_pu0",
    "single_muon_pu0",
]

# How many events to cache per dataset on the Space host (limits memory).
EVENTS_PER_DATASET = 50

# Local cache directory (populated by cache_events.py at build time).
CACHE_DIR = Path(__file__).parent / "_cached_events"


def _frame_to_arrow(frame: Any) -> pa.Table:
    """Normalise a Polars frame (or already-pyarrow table) into a pyarrow Table.

    ``colliderml.load`` returns Polars frames in v0.4.0+; the rest of
    this Space is pyarrow-native because we want the row-level
    ``filter`` / ``to_pandas`` conveniences.
    """
    if isinstance(frame, pa.Table):
        return frame
    if hasattr(frame, "to_arrow"):
        collected = frame.collect() if hasattr(frame, "collect") else frame
        return collected.to_arrow()
    raise TypeError(f"Cannot convert {type(frame).__name__} to pyarrow.Table")


@lru_cache(maxsize=len(DATASETS))
def _load_dataset(name: str) -> Dict[str, pa.Table]:
    """Return a dict of pyarrow Tables for the given dataset, cached in memory.

    Tries the local cache first (pre-populated by cache_events.py), then
    falls back to ``colliderml.load()`` which downloads on demand.
    """
    local = CACHE_DIR / name
    if local.exists():
        tables: Dict[str, pa.Table] = {}
        for parquet in local.glob("*.parquet"):
            tables[parquet.stem] = pq.read_table(str(parquet))
        if tables:
            return tables

    if not HAS_COLLIDERML:
        return {}

    try:
        polars_frames = colliderml.load(
            name,
            tables=["tracker_hits", "particles", "tracks"],
            max_events=EVENTS_PER_DATASET,
        )
    except Exception as exc:
        print(f"Failed to load {name}: {exc}")
        return {}

    return {key: _frame_to_arrow(frame) for key, frame in polars_frames.items()}


def _filter_event(table, event_id):
    """Return rows where event_id == event_id, as a pandas DataFrame."""
    if table is None:
        return None
    col_names = set(table.column_names)
    if "event_id" not in col_names:
        return table.to_pandas()
    mask = np.asarray(table.column("event_id")) == event_id
    return table.filter(mask).to_pandas()


def _track_polyline(track, n_points=40):
    """Approximate a reconstructed track as a 3D polyline.

    Uses a simple helical extrapolation from the perigee parameters
    (d0, z0, phi, theta, qop). This is a visualisation aid, not physics.
    """
    d0 = track.get("d0", 0.0)
    z0 = track.get("z0", 0.0)
    phi = track.get("phi", 0.0)
    theta = track.get("theta", np.pi / 2)
    qop = track.get("qop", 1e-6)

    # Straight-line projection to ~1 m (good enough for a visual).
    s = np.linspace(0, 1000, n_points)  # mm
    x = -d0 * np.sin(phi) + s * np.sin(theta) * np.cos(phi)
    y = d0 * np.cos(phi) + s * np.sin(theta) * np.sin(phi)
    z = z0 + s * np.cos(theta)

    # Apply a small curvature based on qop (1/GeV momentum magnitude).
    if abs(qop) > 1e-9:
        curvature = float(qop) * 0.3  # rough mm^-1 at 1 T
        x += 0.5 * curvature * (s ** 2) * (-np.sin(phi))
        y += 0.5 * curvature * (s ** 2) * np.cos(phi)

    return x, y, z


def render_event(dataset_name, event_id):
    """Build the 3D figure for the selected dataset/event."""
    tables = _load_dataset(dataset_name)

    if not tables:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for {dataset_name}",
            height=700,
        )
        return fig

    hits_df = _filter_event(tables.get("tracker_hits"), event_id)
    particles_df = _filter_event(tables.get("particles"), event_id)
    tracks_df = _filter_event(tables.get("tracks"), event_id)

    fig = go.Figure()

    # Tracker hits — main point cloud.
    if hits_df is not None and len(hits_df) > 0:
        color_col = None
        for c in ("layer_id", "volume_id", "particle_id"):
            if c in hits_df.columns:
                color_col = c
                break
        fig.add_trace(go.Scatter3d(
            x=hits_df["x"], y=hits_df["y"], z=hits_df["z"],
            mode="markers",
            marker=dict(
                size=1.6,
                color=hits_df[color_col] if color_col else "royalblue",
                colorscale="Viridis",
                opacity=0.85,
                showscale=bool(color_col),
                colorbar=dict(title=color_col) if color_col else None,
            ),
            name=f"Tracker hits ({len(hits_df)})",
            hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>",
        ))

    # Reconstructed tracks — helical polylines.
    if tracks_df is not None and len(tracks_df) > 0:
        for _, track in tracks_df.head(30).iterrows():
            try:
                x, y, z = _track_polyline(track.to_dict())
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="lines",
                    line=dict(color="crimson", width=3),
                    name="track",
                    showlegend=False,
                    hoverinfo="skip",
                ))
            except Exception:
                continue

    # Truth particles — momentum vectors from primary vertex.
    if particles_df is not None and len(particles_df) > 0:
        prim = particles_df
        if "primary" in prim.columns:
            prim = prim[prim["primary"] == True]
        prim = prim.head(20)
        for _, p in prim.iterrows():
            try:
                px, py, pz = p.get("px", 0), p.get("py", 0), p.get("pz", 0)
                pmag = (px ** 2 + py ** 2 + pz ** 2) ** 0.5
                if pmag < 1e-3:
                    continue
                scale = 500.0 / pmag
                fig.add_trace(go.Scatter3d(
                    x=[0, px * scale], y=[0, py * scale], z=[0, pz * scale],
                    mode="lines",
                    line=dict(color="gold", width=2, dash="dash"),
                    name=f"truth pdg={int(p.get('pdg_id', 0))}",
                    showlegend=False,
                    hoverinfo="name",
                ))
            except Exception:
                continue

    fig.update_layout(
        title=f"{dataset_name} — event {event_id}",
        scene=dict(
            xaxis_title="x [mm]",
            yaxis_title="y [mm]",
            zaxis_title="z [mm]",
            aspectmode="data",
            bgcolor="rgb(10, 10, 25)",
        ),
        height=720,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgb(10, 10, 25)",
        font=dict(color="white"),
    )
    return fig


def event_count(dataset_name):
    """How many cached events does this dataset have?"""
    tables = _load_dataset(dataset_name)
    hits = tables.get("tracker_hits")
    if hits is None or "event_id" not in hits.column_names:
        return EVENTS_PER_DATASET - 1
    ids = np.asarray(hits.column("event_id"))
    return int(ids.max()) if len(ids) else 0


def on_dataset_change(dataset_name):
    max_evt = event_count(dataset_name)
    return gr.Slider(minimum=0, maximum=max(max_evt, 0), value=0, step=1, label="Event ID")


with gr.Blocks(
    title="ColliderML Event Display",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container {max-width: 1200px !important;}",
) as demo:
    gr.Markdown(
        """
        # ColliderML Event Display

        Interactive 3D view of single events from the
        [ColliderML datasets](https://huggingface.co/datasets/CERN/ColliderML-Release-1).

        - **Blue points**: tracker hits (coloured by detector layer)
        - **Red lines**: reconstructed tracks (helical approximation)
        - **Yellow dashes**: truth particle momenta from the primary vertex
        """
    )
    with gr.Row():
        dataset = gr.Dropdown(
            DATASETS,
            value=DATASETS[0],
            label="Dataset",
            scale=1,
        )
        event_slider = gr.Slider(
            0, EVENTS_PER_DATASET - 1, value=0, step=1,
            label="Event ID",
            scale=2,
        )
    plot = gr.Plot()

    dataset.change(
        fn=on_dataset_change,
        inputs=dataset,
        outputs=event_slider,
    )

    # Render on any change.
    for comp in (dataset, event_slider):
        comp.change(
            fn=render_event,
            inputs=[dataset, event_slider],
            outputs=plot,
        )

    # Initial render on app load.
    demo.load(
        fn=render_event,
        inputs=[dataset, event_slider],
        outputs=plot,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
