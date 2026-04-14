"""Reference baselines for the tracking task.

* :func:`~colliderml.tasks.tracking.baselines.oracle.perfect_oracle_predictions`
  and :func:`~colliderml.tasks.tracking.baselines.oracle.noised_oracle_predictions`
  — pedagogical "no real tracking" baselines built from truth.
* :mod:`~colliderml.tasks.tracking.baselines.ckf` — convert a pipeline
  CKF run into the leaderboard schema.
"""

from colliderml.tasks.tracking.baselines.oracle import (
    noised_oracle_predictions,
    perfect_oracle_predictions,
)

__all__ = ["perfect_oracle_predictions", "noised_oracle_predictions"]
