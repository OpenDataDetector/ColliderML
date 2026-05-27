"""Track reconstruction benchmark task."""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

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


def _parse_event_range_env(value: str) -> Optional[Tuple[int, int]]:
    """Parse ``"start:end"`` or ``"start,end"`` into ``(start, end)``.

    Returns ``None`` (and lets the caller fall back to the default) if
    the value can't be parsed — tutorials shouldn't crash on a typo in
    an optional env var.
    """
    match = re.match(r"^\s*(\d+)\s*[:,]\s*(\d+)\s*$", value)
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if end <= start:
        return None
    return (start, end)


@register
class TrackingTask(BenchmarkTask):
    """Reconstruct tracks in a ttbar + PU=200 environment.

    Two env vars override the defaults for tutorial / small-machine
    deploys without changing the class-level config:

    - ``COLLIDERML_TRACKING_DATASET`` — e.g. ``"ttbar_pu0"`` to swap
      PU=200 for the much smaller PU=0 split.
    - ``COLLIDERML_TRACKING_EVAL_RANGE`` — ``"start:end"`` (half-open).
      Shrinks the held-out eval slice; useful when paired with PU=0 so
      a fresh user can run chapter 4 of the tutorial without OOM.

    Both default to production values when the env vars are unset.
    """

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

    def __init__(self) -> None:
        # Class-level defaults stay as `dataset` / `eval_event_range`;
        # we shadow them per-instance only when the env vars are set,
        # so unrelated tests and production deploys see exactly the
        # values they used to.
        ds_override = os.environ.get("COLLIDERML_TRACKING_DATASET", "").strip()
        if ds_override:
            self.dataset = ds_override
        er_override = os.environ.get("COLLIDERML_TRACKING_EVAL_RANGE", "").strip()
        if er_override:
            parsed = _parse_event_range_env(er_override)
            if parsed is not None:
                self.eval_event_range = parsed

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
