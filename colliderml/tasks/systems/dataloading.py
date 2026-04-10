"""Data loading throughput benchmark.

Users report how fast they could load + iterate a fixed held-out slice
into numpy/torch tensors. The task exists to surface contributions to
:func:`colliderml.load` and upstream libraries (pyarrow, huggingface_hub).
"""

from __future__ import annotations

from typing import Dict, Tuple

import pyarrow as pa

from colliderml.tasks import register
from colliderml.tasks._base import BenchmarkTask


@register
class DataLoadingTask(BenchmarkTask):
    """Measure end-to-end loader throughput in events / second."""

    name = "data_loading"
    dataset = "ttbar_pu200"
    eval_event_range: Tuple[int, int] = (0, 10_000)
    inputs = ["tracker_hits"]
    metrics = ["events_per_sec_local", "events_per_sec_streaming"]
    higher_is_better = {
        "events_per_sec_local": True,
        "events_per_sec_streaming": True,
    }

    def load_eval_inputs(self) -> Dict[str, pa.Table]:
        """This task's "input" is the user's code — nothing to hand back."""
        return {}

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"local_seconds", "streaming_seconds", "n_events"}
        have = set(preds.column_names)
        if not required.issubset(have):
            raise ValueError(
                "data_loading submissions must be a 1-row table with columns "
                f"{sorted(required)}; got {sorted(have)}"
            )
        if preds.num_rows != 1:
            raise ValueError(
                f"data_loading submissions must be a 1-row table; got {preds.num_rows}."
            )

    def score(self, preds: pa.Table) -> Dict[str, float]:
        row = preds.to_pydict()
        n_events = int(row["n_events"][0])
        local = float(row["local_seconds"][0])
        streaming = float(row["streaming_seconds"][0])
        return {
            "events_per_sec_local": round(n_events / max(local, 1e-9), 3),
            "events_per_sec_streaming": round(n_events / max(streaming, 1e-9), 3),
        }
