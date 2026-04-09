"""Abstract base class for benchmark tasks.

Every task defines:

* a **name** (used in URLs, the CLI, the leaderboard, and the registry),
* a **dataset** key of the form ``"<channel>_<pileup>"`` (e.g.
  ``"ttbar_pu200"``) that locates the data on HuggingFace,
* an **eval_event_range** half-open tuple that withholds events from users
  for training and defines what counts as a valid scoring submission,
* the **inputs** a model may read, and
* a list of **metrics** with a per-metric ``higher_is_better`` direction.

Subclasses implement :meth:`load_eval_inputs`, :meth:`validate_predictions`,
and :meth:`score`. The base class supplies the :meth:`load` helper that
task implementations use to pull their eval data, and :meth:`is_better`
which the leaderboard uses to decide whether a submission beats the
current best.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Tuple

import pyarrow as pa


class BenchmarkTask(ABC):
    """Base class for a benchmark task.

    See the module docstring for an overview of the contract.
    """

    #: Identifier used in URLs, the CLI, and the registry.
    name: ClassVar[str]
    #: ``"<channel>_<pileup>"`` key (e.g. ``"ttbar_pu200"``) the task
    #: pulls data from. Used by :meth:`load`.
    dataset: ClassVar[str]
    #: ``(start, end)`` half-open range of event IDs that count for
    #: scoring. Events outside this range are training data.
    eval_event_range: ClassVar[Tuple[int, int]]
    #: Table names the task exposes to the model as inputs.
    inputs: ClassVar[List[str]]
    #: Metric names in display order. The first metric is the primary.
    metrics: ClassVar[List[str]]
    #: Per-metric direction. Missing keys default to ``True``
    #: (higher is better).
    higher_is_better: ClassVar[Dict[str, bool]] = {}

    # ------------------------------------------------------------------
    # Data loading helper — used by task implementations' load_eval_inputs
    # and score() methods.
    # ------------------------------------------------------------------

    def load(
        self,
        *,
        tables: List[str],
        dataset: str | None = None,
        max_events: int | None = None,
        event_range: Tuple[int, int] | None = None,
    ) -> Dict[str, pa.Table]:
        """Load tables for this task as pyarrow Tables.

        Thin wrapper around :func:`colliderml.tasks._loading.load_task_data`
        that defaults ``dataset`` to :attr:`self.dataset`. Task subclasses
        normally call :meth:`load` with just ``tables=[...]``.

        Args:
            tables: Object types to load (e.g. ``["tracker_hits"]``).
            dataset: Override the task's default dataset. Used by
                multi-dataset tasks like anomaly detection.
            max_events: Limit to the first ``max_events`` events.
            event_range: Restrict to events in ``[start, end)``. Useful
                for scoring only the held-out eval split.

        Returns:
            Mapping of table name → pyarrow Table.
        """
        from colliderml.tasks._loading import load_task_data

        return load_task_data(
            dataset or self.dataset,
            tables=tables,
            max_events=max_events,
            event_range=event_range,
        )

    # ------------------------------------------------------------------
    # Abstract surface — every task must implement these.
    # ------------------------------------------------------------------

    @abstractmethod
    def load_eval_inputs(self) -> Dict[str, pa.Table]:
        """Return the eval inputs as a dict of pyarrow Tables.

        Clients use this to build a model's input pipeline; the backend
        uses it to run baselines on-demand.
        """

    @abstractmethod
    def validate_predictions(self, preds: pa.Table) -> None:
        """Raise :class:`ValueError` if ``preds`` doesn't match the task schema."""

    @abstractmethod
    def score(self, preds: pa.Table) -> Dict[str, float]:
        """Compute every metric on ``preds``. Returns ``{metric: value}``."""

    # ------------------------------------------------------------------
    # Default helpers
    # ------------------------------------------------------------------

    def is_better(self, metric: str, new: float, current: float) -> bool:
        """Return ``True`` if ``new`` beats ``current`` for ``metric``.

        Used by the leaderboard to decide whether to award credits for a
        submission.
        """
        higher = self.higher_is_better.get(metric, True)
        return new > current if higher else new < current
