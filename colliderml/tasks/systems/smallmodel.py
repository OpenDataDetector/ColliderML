"""Small-model tracking challenge: Pareto frontier of efficiency vs. params.

Submissions are regular tracking predictions plus a constant
``n_params`` column. The leaderboard sorts by the smallest tier a model
fits into; within a tier, higher TrackML efficiency wins.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pyarrow as pa

from colliderml.polars import explode_event_table_pyarrow
from colliderml.tasks import register
from colliderml.tasks._base import BenchmarkTask
from colliderml.tasks.tracking.metrics import trackml_weighted_efficiency
from colliderml.tasks.tracking.task import _parse_event_range_env

#: Parameter-count caps for each tier (inclusive upper bound).
PARAM_TIERS: List[int] = [10_000, 100_000, 1_000_000]


def tier_for(n_params: int) -> int:
    """Return the smallest tier (1, 2, 3) a model fits, or 4 if it doesn't qualify."""
    for i, cap in enumerate(PARAM_TIERS, start=1):
        if n_params < cap:
            return i
    return 4


@register
class TrackingSmallModelTask(BenchmarkTask):
    """Tracking with a parameter budget.

    Shares :class:`TrackingTask`'s ``COLLIDERML_TRACKING_DATASET`` and
    ``COLLIDERML_TRACKING_EVAL_RANGE`` env-var overrides so tutorial
    deploys can swap PU=200 for PU=0 without touching code.
    """

    name = "tracking_small"
    dataset = "ttbar_pu200"
    eval_event_range: Tuple[int, int] = (90_000, 100_000)
    inputs = ["tracker_hits"]
    metrics = ["trackml_eff", "n_params", "tier"]
    higher_is_better = {"trackml_eff": True, "n_params": False, "tier": False}

    def __init__(self) -> None:
        import os

        ds_override = os.environ.get("COLLIDERML_TRACKING_DATASET", "").strip()
        if ds_override:
            self.dataset = ds_override
        er_override = os.environ.get("COLLIDERML_TRACKING_EVAL_RANGE", "").strip()
        if er_override:
            parsed = _parse_event_range_env(er_override)
            if parsed is not None:
                self.eval_event_range = parsed

    def load_eval_inputs(self) -> Dict[str, pa.Table]:
        return {"tracker_hits": self._load_flat_hits()}

    def _load_flat_hits(self) -> pa.Table:
        raw = self.load(tables=["tracker_hits"], event_range=self.eval_event_range)
        return explode_event_table_pyarrow(raw["tracker_hits"], index_name="hit_id")

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"event_id", "hit_id", "track_id", "n_params"}
        have = set(preds.column_names)
        if not required.issubset(have):
            raise ValueError(
                "Small-model submissions must be tracking predictions plus a "
                f"constant 'n_params' column; got {sorted(have)}"
            )

    def score(self, preds: pa.Table) -> Dict[str, float]:
        eff = trackml_weighted_efficiency(preds, self._load_flat_hits())
        n_params = int(preds.column("n_params").to_pylist()[0])
        return {
            "trackml_eff": eff,
            "n_params": float(n_params),
            "tier": float(tier_for(n_params)),
        }
