"""ColliderML benchmark task registry and runner.

Exposes the public surface for discovering, inspecting, and scoring
benchmark tasks::

    import colliderml.tasks

    colliderml.tasks.list_tasks()          # -> ['tracking', 'jets', ...]
    task = colliderml.tasks.get('tracking')
    colliderml.tasks.evaluate('tracking', 'preds.parquet')
    colliderml.tasks.submit('tracking', 'preds.parquet')

The individual task modules under ``tracking/``, ``jets/``, ``anomaly/``
and ``systems/`` register themselves with :func:`register` at import
time via the decorator pattern. Importing this package is therefore the
only thing a user needs to do to populate the registry.

To avoid ambiguity with the root-level ``benchmarks/`` directory in this
repository (which hosts a download-timing harness and its JSON results),
the physics + systems task registry lives under ``colliderml.tasks``
rather than ``colliderml.benchmarks``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type, Union

import pyarrow as pa

from colliderml.tasks._base import BenchmarkTask
from colliderml.tasks._loading import load_task_data, parse_dataset_name

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASKS: Dict[str, Type[BenchmarkTask]] = {}


def register(cls: Type[BenchmarkTask]) -> Type[BenchmarkTask]:
    """Class decorator that registers a :class:`BenchmarkTask` subclass.

    Raises :class:`ValueError` on duplicate names so that silent shadowing
    cannot happen when two modules accidentally claim the same task name.
    """
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"Task {cls.__name__} is missing a class-level 'name' attribute.")
    if name in _TASKS and _TASKS[name] is not cls:
        raise ValueError(
            f"Task name '{name}' is already registered by {_TASKS[name].__name__}. "
            "Pick a different name."
        )
    _TASKS[name] = cls
    return cls


def list_tasks() -> List[str]:
    """Return every registered task name, sorted alphabetically."""
    return sorted(_TASKS.keys())


def get(name: str) -> BenchmarkTask:
    """Return a fresh instance of the task registered under ``name``.

    Raises:
        ValueError: If ``name`` is not registered. The error lists every
            available task name so users can self-correct.
    """
    if name not in _TASKS:
        available = ", ".join(list_tasks()) or "(none registered)"
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    return _TASKS[name]()


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def _coerce_predictions(predictions: Any) -> pa.Table:
    """Normalise ``predictions`` into a pyarrow Table.

    Accepts:
    * a path (``str`` or ``pathlib.Path``) to a parquet file
    * an existing :class:`pyarrow.Table`
    * a pandas :class:`~pandas.DataFrame`

    Anything else raises :class:`TypeError`.
    """
    if isinstance(predictions, pa.Table):
        return predictions
    # `os.PathLike` handles pathlib.Path without an explicit Path import.
    if isinstance(predictions, (str, bytes)) or hasattr(predictions, "__fspath__"):
        import pyarrow.parquet as pq

        return pq.read_table(str(predictions))
    if hasattr(predictions, "to_parquet") and hasattr(predictions, "columns"):
        return pa.Table.from_pandas(predictions)
    raise TypeError(
        "predictions must be a path, pyarrow Table, or pandas DataFrame; "
        f"got {type(predictions).__name__}"
    )


def evaluate(task_name: str, predictions: Any) -> Dict[str, float]:
    """Score a prediction artefact against a registered task.

    Args:
        task_name: Registered task name.
        predictions: Path to a parquet file, a pyarrow Table, or a pandas
            DataFrame.

    Returns:
        Mapping of metric name → float value.
    """
    preds = _coerce_predictions(predictions)
    task = get(task_name)
    task.validate_predictions(preds)
    return task.score(preds)


def submit(
    task_name: str,
    predictions: Any,
    *,
    backend_url: str | None = None,
) -> Dict[str, Any]:
    """Upload predictions to the backend leaderboard and return its response.

    Scores the predictions locally first (so users see a preview even when
    the backend is unreachable), then posts the parquet bytes to
    ``POST /v1/benchmark/<task_name>/submit``. The backend re-scores with
    its canonical truth set and may award credits for new bests.

    Args:
        task_name: Registered task name.
        predictions: Path to a parquet file, a pyarrow Table, or a pandas
            DataFrame.
        backend_url: Override the default backend URL.

    Returns:
        The backend's JSON response, which typically includes ``scores``
        and ``credits_earned``.

    Raises:
        ImportError: If the ``remote`` extra is not installed.
        RuntimeError: If the backend returns a non-2xx status code.
    """
    preds = _coerce_predictions(predictions)
    task = get(task_name)
    task.validate_predictions(preds)
    local_scores = task.score(preds)

    # Remote dependencies are lazy so users who only need local scoring
    # don't need the `remote` extra.
    try:
        from colliderml.remote._auth import auth_headers, require_hf_token
        from colliderml.remote._client import DEFAULT_BACKEND_URL, _require_requests
    except ImportError as exc:  # pragma: no cover - exercised in CI envs without extras
        raise ImportError(
            "colliderml.tasks.submit() requires the 'remote' extra: "
            "pip install 'colliderml[remote]'"
        ) from exc

    import json
    from io import BytesIO

    import pyarrow.parquet as pq

    requests = _require_requests()
    token = require_hf_token()
    url = (backend_url or DEFAULT_BACKEND_URL).rstrip("/")

    buf = BytesIO()
    pq.write_table(preds, buf)
    files = {
        "predictions": ("predictions.parquet", buf.getvalue(), "application/octet-stream"),
    }
    data = {"local_scores": json.dumps(local_scores)}

    response = requests.post(
        f"{url}/v1/benchmark/{task_name}/submit",
        files=files,
        data=data,
        headers=auth_headers(token),
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Leaderboard submission for '{task_name}' failed: "
            f"{response.status_code} {response.text}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected response body from leaderboard: {body!r}")
    return body


# ---------------------------------------------------------------------------
# Registration — importing these modules runs their @register decorators
# ---------------------------------------------------------------------------
# Imported at the bottom so `from colliderml.tasks import register` above
# is already defined when the task modules execute.

from colliderml.tasks.anomaly.task import AnomalyDetectionTask  # noqa: E402,F401
from colliderml.tasks.jets.task import JetClassificationTask  # noqa: E402,F401
from colliderml.tasks.systems.dataloading import DataLoadingTask  # noqa: E402,F401
from colliderml.tasks.systems.latency import TrackingLatencyTask  # noqa: E402,F401
from colliderml.tasks.systems.smallmodel import TrackingSmallModelTask  # noqa: E402,F401
from colliderml.tasks.tracking.task import TrackingTask  # noqa: E402,F401


__all__ = [
    "BenchmarkTask",
    "register",
    "list_tasks",
    "get",
    "evaluate",
    "submit",
    "load_task_data",
    "parse_dataset_name",
]
