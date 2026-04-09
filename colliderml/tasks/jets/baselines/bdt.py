"""Gradient-boosted decision tree baseline for the jet classification task.

Trains a simple scikit-learn ``GradientBoostingClassifier`` on per-event
kinematic features pulled from the tracks table. The truth labels are
synthesised deterministically for this baseline — the backend scorer
uses the canonical truth file when a submission lands via
:func:`colliderml.tasks.submit`.

Usage::

    python -m colliderml.tasks.jets.baselines.bdt \\
        --channel ttbar_pu0 --max-events 1000 --output bdt_preds.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

FEATURE_COLUMNS: List[str] = [
    "pt",
    "eta",
    "phi",
    "mass",
    "n_tracks",
    "sum_track_pt",
]


def _event_features(tracks_table) -> "pd.DataFrame":  # type: ignore[name-defined]
    import numpy as np
    import pandas as pd

    df = tracks_table.to_pandas() if hasattr(tracks_table, "to_pandas") else tracks_table
    if df.empty:
        return pd.DataFrame()
    by_event = df.groupby("event_id")
    rows = []
    for event_id, group in by_event:
        px = group.get("px", pd.Series([0.0]))
        py = group.get("py", pd.Series([0.0]))
        pz = group.get("pz", pd.Series([0.0]))
        e = group.get("energy", pd.Series([0.0]))
        pt_sq = px.pow(2) + py.pow(2)
        pt_sum = float(pt_sq.pow(0.5).sum())
        total_momentum = float((px.pow(2).sum() + py.pow(2).sum() + pz.pow(2).sum()) ** 0.5)
        if total_momentum > 0 and pz.sum() != 0:
            eta = float(np.arctanh(pz.sum() / total_momentum))
        else:
            eta = 0.0
        rows.append(
            {
                "event_id": int(event_id),
                "pt": pt_sum,
                "eta": eta,
                "phi": float(np.arctan2(py.sum(), px.sum())),
                "mass": float(max(e.pow(2).sum() - total_momentum ** 2, 0) ** 0.5),
                "n_tracks": int(len(group)),
                "sum_track_pt": pt_sum,
            }
        )
    return pd.DataFrame(rows)


def run_baseline(channel: str, max_events: int, output: Path) -> int:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:  # pragma: no cover
        print(
            "scikit-learn is not installed. Run: pip install scikit-learn",
            file=sys.stderr,
        )
        return 1

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from colliderml.tasks._loading import load_task_data

    print(f"Loading tracks for {channel} (max_events={max_events}) ...")
    data = load_task_data(channel, tables=["tracks"], max_events=max_events)
    tracks = data["tracks"]

    feats = _event_features(tracks)
    if feats.empty:
        print("No jets found. Is the dataset populated?", file=sys.stderr)
        return 1

    rng = np.random.default_rng(42)
    truth = rng.choice(["b", "c", "light"], size=len(feats), p=[0.2, 0.2, 0.6])

    midpoint = len(feats) // 2
    x_train = feats[FEATURE_COLUMNS].iloc[:midpoint].values
    y_train = truth[:midpoint]
    x_eval = feats[FEATURE_COLUMNS].iloc[midpoint:].values
    eval_event_ids = feats["event_id"].iloc[midpoint:].values

    print(f"Training GBDT on {len(x_train)} jets ...")
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    clf.fit(x_train, y_train)

    print(f"Predicting on {len(x_eval)} jets ...")
    proba = clf.predict_proba(x_eval)
    classes = list(clf.classes_)
    prob_b = proba[:, classes.index("b")] if "b" in classes else np.zeros(len(x_eval))
    prob_c = proba[:, classes.index("c")] if "c" in classes else np.zeros(len(x_eval))
    prob_light = (
        proba[:, classes.index("light")] if "light" in classes else np.zeros(len(x_eval))
    )

    table = pa.table(
        {
            "event_id": eval_event_ids.tolist(),
            "jet_id": [0] * len(eval_event_ids),
            "prob_b": prob_b.tolist(),
            "prob_c": prob_c.tolist(),
            "prob_light": prob_light.tolist(),
        }
    )
    pq.write_table(table, output)
    print(f"Wrote {table.num_rows} jet predictions to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BDT baseline for jet classification")
    parser.add_argument("--channel", default="ttbar_pu0")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("bdt_preds.parquet"))
    args = parser.parse_args(argv)
    return run_baseline(args.channel, args.max_events, args.output)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
