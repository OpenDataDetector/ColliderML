"""Tracking inference latency benchmark.

Users submit a 1-row table with the wall-clock time and event count
they measured on a held-out slice. The backend re-runs the user's code
in a standardised container for canonical numbers when a submission
arrives via :func:`colliderml.tasks.submit`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pyarrow as pa

from colliderml.tasks import register
from colliderml.tasks._base import BenchmarkTask


@register
class TrackingLatencyTask(BenchmarkTask):
    """Wall-clock and throughput constraints on top of the tracking task."""

    name = "tracking_latency"
    dataset = "ttbar_pu200"
    eval_event_range: Tuple[int, int] = (99_000, 100_000)  # 1000 events
    inputs = ["tracker_hits"]
    metrics = ["wallclock_s", "events_per_sec"]
    higher_is_better = {"wallclock_s": False, "events_per_sec": True}

    def load_eval_inputs(self) -> Dict[str, pa.Table]:
        return self.load(tables=["tracker_hits"], event_range=self.eval_event_range)

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"wallclock_s", "n_events"}
        have = set(preds.column_names)
        if not required.issubset(have):
            raise ValueError(
                "Latency submissions must be a 1-row table with columns "
                f"{sorted(required)}; got {sorted(have)}"
            )
        if preds.num_rows != 1:
            raise ValueError(
                f"Latency submissions must be a 1-row table; got {preds.num_rows} rows."
            )

    def score(self, preds: pa.Table) -> Dict[str, float]:
        row = preds.to_pydict()
        wallclock = float(row["wallclock_s"][0])
        n_events = int(row["n_events"][0])
        if wallclock <= 0 or n_events <= 0:
            return {"wallclock_s": float("inf"), "events_per_sec": 0.0}
        return {
            "wallclock_s": round(wallclock, 3),
            "events_per_sec": round(n_events / wallclock, 3),
        }
