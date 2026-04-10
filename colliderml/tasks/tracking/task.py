"""Track reconstruction benchmark task."""

from __future__ import annotations

from typing import Dict, Tuple

import pyarrow as pa

from colliderml.tasks import register
from colliderml.tasks._base import BenchmarkTask
from colliderml.tasks.tracking.metrics import (
    duplicate_rate,
    fake_rate,
    physics_eff_pt1,
    trackml_weighted_efficiency,
)


@register
class TrackingTask(BenchmarkTask):
    """Reconstruct tracks in a ttbar + PU=200 environment."""

    name = "tracking"
    dataset = "ttbar_pu200"
    eval_event_range: Tuple[int, int] = (90_000, 100_000)
    inputs = ["tracker_hits"]
    metrics = ["trackml_eff", "fake_rate", "dup_rate", "physics_eff_pt1"]
    higher_is_better = {
        "trackml_eff": True,
        "fake_rate": False,
        "dup_rate": False,
        "physics_eff_pt1": True,
    }

    def load_eval_inputs(self) -> Dict[str, pa.Table]:
        return self.load(tables=["tracker_hits"], event_range=self.eval_event_range)

    def _load_truth(self) -> Tuple[pa.Table, pa.Table]:
        """Load truth hits and particles for the eval split."""
        hits = self.load(tables=["tracker_hits"], event_range=self.eval_event_range)
        particles = self.load(tables=["particles"], event_range=self.eval_event_range)
        return hits["tracker_hits"], particles["particles"]

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"event_id", "hit_id", "track_id"}
        have = set(preds.column_names)
        if not required.issubset(have):
            raise ValueError(
                f"Tracking predictions must have columns {sorted(required)}; "
                f"got {sorted(have)}"
            )
        events = set(int(e) for e in preds.column("event_id").to_pylist())
        expected = set(range(*self.eval_event_range))
        missing = expected - events
        # Be lenient: require coverage for at least half the eval range.
        if len(missing) > len(expected) * 0.5:
            raise ValueError(
                f"Tracking predictions miss too many eval events: "
                f"{len(missing)}/{len(expected)} missing."
            )

    def score(self, preds: pa.Table) -> Dict[str, float]:
        hits, particles = self._load_truth()
        return {
            "trackml_eff": trackml_weighted_efficiency(preds, hits),
            "fake_rate": fake_rate(preds, hits),
            "dup_rate": duplicate_rate(preds, hits),
            "physics_eff_pt1": physics_eff_pt1(preds, particles),
        }
