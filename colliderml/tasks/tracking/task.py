"""Track reconstruction benchmark task."""

from __future__ import annotations

from typing import Dict, Tuple

import pyarrow as pa

from colliderml.polars import explode_event_table_pyarrow
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
        return {"tracker_hits": self.load_truth_hits()}

    def load_truth_hits(self, *, event_range: Tuple[int, int] | None = None) -> pa.Table:
        """Return tracker hits as a flat row-per-hit pyarrow Table.

        The dataset on disk stores hits as list-per-event (one row per
        event with parallel list columns). Predictions, metrics, and
        baselines all consume the **flat** shape — one row per hit, with
        an explicit ``hit_id`` column equal to the per-event row index.
        This helper performs that explosion and is the canonical way for
        users to load the eval truth.

        Args:
            event_range: Override the eval range. Defaults to
                :attr:`eval_event_range`. Use a small subrange for demos.

        Returns:
            A flat pyarrow Table with columns
            ``event_id, hit_id, particle_id, x, y, z, …``.
        """
        er = event_range or self.eval_event_range
        raw = self.load(tables=["tracker_hits"], event_range=er)["tracker_hits"]
        return explode_event_table_pyarrow(raw, index_name="hit_id")

    def _load_truth(self) -> Tuple[pa.Table, pa.Table]:
        """Load flat truth hits and flat truth particles for the eval split.

        Both are exploded from list-per-event into row-per-object so the
        metrics can iterate over scalar columns directly (they assume
        ``px``/``py``/``primary`` are parallel 1-D arrays).
        """
        hits = self.load_truth_hits()
        particles_raw = self.load(
            tables=["particles"], event_range=self.eval_event_range
        )["particles"]
        particles = explode_event_table_pyarrow(
            particles_raw, index_name="particle_index"
        )
        return hits, particles

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
