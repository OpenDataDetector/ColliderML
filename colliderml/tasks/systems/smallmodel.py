"""Small-model tracking challenge: Pareto frontier of efficiency vs. params.

Submissions are regular tracking predictions plus a constant
``n_params`` column. The leaderboard sorts by the smallest tier a model
fits into; within a tier, higher TrackML efficiency wins.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pyarrow as pa

from colliderml.tasks import register
from colliderml.tasks._base import BenchmarkTask
from colliderml.tasks.tracking.metrics import trackml_weighted_efficiency

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
    """Tracking with a parameter budget."""

    name = "tracking_small"
    dataset = "ttbar_pu200"
    eval_event_range: Tuple[int, int] = (90_000, 100_000)
    inputs = ["tracker_hits"]
    metrics = ["trackml_eff", "n_params", "tier"]
    higher_is_better = {"trackml_eff": True, "n_params": False, "tier": False}

    def load_eval_inputs(self) -> Dict[str, pa.Table]:
        return self.load(tables=["tracker_hits"], event_range=self.eval_event_range)

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"event_id", "hit_id", "track_id", "n_params"}
        have = set(preds.column_names)
        if not required.issubset(have):
            raise ValueError(
                "Small-model submissions must be tracking predictions plus a "
                f"constant 'n_params' column; got {sorted(have)}"
            )

    def score(self, preds: pa.Table) -> Dict[str, float]:
        truth = self.load(tables=["tracker_hits"], event_range=self.eval_event_range)
        eff = trackml_weighted_efficiency(preds, truth["tracker_hits"])
        n_params = int(preds.column("n_params").to_pylist()[0])
        return {
            "trackml_eff": eff,
            "n_params": float(n_params),
            "tier": float(tier_for(n_params)),
        }
