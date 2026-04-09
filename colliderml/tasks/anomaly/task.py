"""Anomaly detection benchmark: SM events versus a mix of BSM signals.

Treats the four Standard Model channels (ttbar, zmumu, zee at ``pu0``)
as "normal" and the four BSM channels (higgs_portal, susy_gmsb,
hidden_valley, zprime at ``pu0``) as anomalies. Users submit per-event
anomaly scores; we compute AUROC and signal efficiency at 1 % false
positive rate.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa

from colliderml.tasks import register
from colliderml.tasks._base import BenchmarkTask
from colliderml.tasks.jets.metrics import roc_auc


def _signal_eff_at_fpr(
    labels: np.ndarray, scores: np.ndarray, target_fpr: float
) -> float:
    """Signal efficiency at a fixed background false-positive rate."""
    bg_scores = np.sort(scores[labels == 0])[::-1]
    if len(bg_scores) == 0:
        return 0.0
    idx = int(len(bg_scores) * target_fpr)
    idx = min(idx, len(bg_scores) - 1)
    threshold = float(bg_scores[idx])
    sig_scores = scores[labels == 1]
    if len(sig_scores) == 0:
        return 0.0
    return round(float((sig_scores >= threshold).mean()), 6)


@register
class AnomalyDetectionTask(BenchmarkTask):
    """Flag BSM events against a background of SM processes."""

    name = "anomaly"
    # Virtual dataset — this task draws from several underlying datasets
    # instead of a single one.
    dataset = "mixed_sm_bsm"
    eval_event_range: Tuple[int, int] = (0, 10_000)
    inputs = ["tracks", "calo_hits"]
    metrics = ["auroc", "sig_eff_1fpr"]
    higher_is_better = {"auroc": True, "sig_eff_1fpr": True}

    SM_CHANNELS: List[str] = ["ttbar_pu0", "zmumu_pu0", "zee_pu0"]
    BSM_CHANNELS: List[str] = [
        "higgs_portal_pu0",
        "susy_gmsb_pu0",
        "hidden_valley_pu0",
        "zprime_pu0",
    ]

    def load_eval_inputs(self) -> Dict[str, pa.Table]:
        """Return the first 1000 events from every participating channel.

        Channels that aren't available locally (e.g. on a developer's
        laptop that hasn't downloaded the BSM datasets yet) are
        silently skipped. The server-side scorer always has every
        channel available and never hits this fallback.
        """
        out: Dict[str, pa.Table] = {}
        for ch in self.SM_CHANNELS + self.BSM_CHANNELS:
            try:
                ch_tables = self.load(dataset=ch, tables=["tracks"], max_events=1000)
            except Exception:
                continue
            if "tracks" in ch_tables:
                out[ch] = ch_tables["tracks"]
        return out

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"event_id", "channel", "anomaly_score"}
        have = set(preds.column_names)
        if not required.issubset(have):
            raise ValueError(
                f"Anomaly predictions must have columns {sorted(required)}; "
                f"got {sorted(have)}"
            )

    def score(self, preds: pa.Table) -> Dict[str, float]:
        cols = preds.to_pydict()
        channels = list(cols["channel"])
        scores = np.asarray(cols["anomaly_score"], dtype=float)
        labels = np.asarray(
            [1 if c in self.BSM_CHANNELS else 0 for c in channels], dtype=int
        )
        return {
            "auroc": roc_auc(labels, scores),
            "sig_eff_1fpr": _signal_eff_at_fpr(labels, scores, target_fpr=0.01),
        }
