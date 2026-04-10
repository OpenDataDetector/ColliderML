"""Jet-flavour classification benchmark task.

.. note::
    The current :meth:`score` implementation synthesises a truth label
    from a deterministic RNG. The server-side scorer overrides this
    with the canonical truth file when a user submits via
    :func:`colliderml.tasks.submit`. This placeholder exists so the
    client-side ``evaluate()`` path returns a non-trivial preview
    without requiring the truth file, and so the task contract is
    exercised end-to-end in unit tests.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pyarrow as pa

from colliderml.tasks import register
from colliderml.tasks._base import BenchmarkTask
from colliderml.tasks.jets.metrics import rejection_at_efficiency, roc_auc


@register
class JetClassificationTask(BenchmarkTask):
    """Classify each jet as ``b``, ``c``, or ``light``."""

    name = "jets"
    dataset = "ttbar_pu0"
    eval_event_range: Tuple[int, int] = (90_000, 100_000)
    inputs = ["tracks", "calo_hits"]
    metrics = ["btag_auc", "light_rej_70", "c_rej_70"]
    higher_is_better = {
        "btag_auc": True,
        "light_rej_70": True,
        "c_rej_70": True,
    }

    #: Deterministic seed used for the placeholder truth synthesis in
    #: :meth:`score`. The server-side scorer ignores this.
    _PLACEHOLDER_SEED = 42

    def load_eval_inputs(self) -> Dict[str, pa.Table]:
        return self.load(
            tables=["tracks", "calo_hits"],
            event_range=self.eval_event_range,
        )

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"event_id", "jet_id", "prob_b", "prob_c", "prob_light"}
        have = set(preds.column_names)
        if not required.issubset(have):
            raise ValueError(
                f"Jet predictions must have columns {sorted(required)}; "
                f"got {sorted(have)}"
            )
        prob_b = np.asarray(preds.column("prob_b").to_pylist(), dtype=float)
        prob_c = np.asarray(preds.column("prob_c").to_pylist(), dtype=float)
        prob_light = np.asarray(preds.column("prob_light").to_pylist(), dtype=float)
        sums = prob_b + prob_c + prob_light
        if not np.allclose(sums, 1.0, atol=0.02):
            raise ValueError(
                "prob_b + prob_c + prob_light must sum to 1 (±0.02) per row."
            )

    def score(self, preds: pa.Table) -> Dict[str, float]:
        cols = preds.to_pydict()
        n = len(cols["event_id"])
        rng = np.random.default_rng(self._PLACEHOLDER_SEED)
        truth_flavour = rng.choice(["b", "c", "light"], size=n, p=[0.2, 0.2, 0.6])
        is_b = (truth_flavour == "b").astype(int)
        is_c = (truth_flavour == "c").astype(int)

        prob_b = np.asarray(cols["prob_b"], dtype=float)
        prob_c = np.asarray(cols["prob_c"], dtype=float)

        return {
            "btag_auc": roc_auc(is_b, prob_b),
            "light_rej_70": rejection_at_efficiency(is_b, prob_b, 0.70),
            "c_rej_70": rejection_at_efficiency(is_b + is_c, prob_b + prob_c, 0.70),
        }
