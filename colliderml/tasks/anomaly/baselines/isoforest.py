"""IsolationForest baseline for the anomaly detection task.

Trains on per-event kinematic features from SM channels, then scores a
mixed held-out set of SM and BSM channels. Higher scores are more
anomalous (the script flips sklearn's sign convention to match the
task's ``higher_is_better`` contract).

Usage::

    python -m colliderml.tasks.anomaly.baselines.isoforest \\
        --output iso_preds.parquet --max-events 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

SM_CHANNELS: List[str] = ["ttbar_pu0", "zmumu_pu0", "zee_pu0"]
BSM_CHANNELS: List[str] = [
    "higgs_portal_pu0",
    "susy_gmsb_pu0",
    "hidden_valley_pu0",
    "zprime_pu0",
]

FEATURE_COLUMNS = ["total_pt", "n_tracks", "met", "leading_pt"]


def _event_features(tracks_table) -> "pd.DataFrame":  # type: ignore[name-defined]
    import numpy as np
    import pandas as pd

    df = tracks_table.to_pandas() if hasattr(tracks_table, "to_pandas") else tracks_table
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    px = df.get("px", pd.Series(0.0, index=df.index))
    py = df.get("py", pd.Series(0.0, index=df.index))
    df["pt"] = np.sqrt(px.pow(2) + py.pow(2))
    grouped = df.groupby("event_id")
    out = grouped.agg(
        total_pt=("pt", "sum"),
        n_tracks=("pt", "count"),
        met_x=("px" if "px" in df.columns else "pt", "sum"),
        met_y=("py" if "py" in df.columns else "pt", "sum"),
        leading_pt=("pt", "max"),
    ).reset_index()
    out["met"] = np.sqrt(out["met_x"].pow(2) + out["met_y"].pow(2))
    return out[["event_id", *FEATURE_COLUMNS]]


def run_baseline(output: Path, max_events: int = 500) -> int:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:  # pragma: no cover
        print(
            "scikit-learn is not installed. Run: pip install scikit-learn",
            file=sys.stderr,
        )
        return 1

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from colliderml.tasks._loading import load_task_data

    print("Loading SM channels for training ...")
    train_feats: List[pd.DataFrame] = []
    for channel in SM_CHANNELS:
        try:
            data = load_task_data(channel, tables=["tracks"], max_events=max_events)
        except Exception as exc:  # pragma: no cover - per-channel network/cache miss
            print(f"  skipping {channel}: {exc}", file=sys.stderr)
            continue
        feats = _event_features(data["tracks"])
        if not feats.empty:
            train_feats.append(feats)
    if not train_feats:
        print("No SM training data available.", file=sys.stderr)
        return 1

    x_train = pd.concat(train_feats, ignore_index=True)
    print(f"Training IsolationForest on {len(x_train)} SM events ...")
    iso = IsolationForest(contamination="auto", random_state=42)
    iso.fit(x_train[FEATURE_COLUMNS].values)

    print("Scoring mixed held-out set ...")
    rows = []
    for channel in SM_CHANNELS + BSM_CHANNELS:
        try:
            data = load_task_data(channel, tables=["tracks"], max_events=max_events // 2)
        except Exception:
            continue
        feats = _event_features(data["tracks"])
        if feats.empty:
            continue
        scores = -iso.score_samples(feats[FEATURE_COLUMNS].values)
        for evt, score in zip(feats["event_id"], scores):
            rows.append(
                {"event_id": int(evt), "channel": channel, "anomaly_score": float(score)}
            )

    if not rows:
        print("No held-out events scored.", file=sys.stderr)
        return 1

    table = pa.table(
        {
            "event_id": [r["event_id"] for r in rows],
            "channel": [r["channel"] for r in rows],
            "anomaly_score": [r["anomaly_score"] for r in rows],
        }
    )
    pq.write_table(table, output)
    print(f"Wrote {table.num_rows} anomaly predictions to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IsolationForest anomaly baseline")
    parser.add_argument("--output", type=Path, default=Path("iso_preds.parquet"))
    parser.add_argument("--max-events", type=int, default=500)
    args = parser.parse_args(argv)
    return run_baseline(args.output, args.max_events)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
