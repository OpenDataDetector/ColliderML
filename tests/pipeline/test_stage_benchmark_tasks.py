"""Pipeline stage: benchmark-tasks / model-zoo surface.

Asserts the backend task catalogue is well-formed and that the client's
registered task metadata matches the backend's — catching client/server
registry skew (the backend scores against the library's task definitions).
"""

from __future__ import annotations

import pytest

import colliderml
from tests.pipeline._http import http_get

pytestmark = pytest.mark.pipeline


@pytest.mark.live
def test_benchmark_tasks_well_formed(live_backend):
    status, body = http_get(f"{live_backend}/v1/benchmark/tasks", timeout=30)
    assert status == 200, f"HTTP {status} — {body}"
    assert isinstance(body, list) and body
    for t in body:
        assert {"name", "metrics", "higher_is_better", "eval_event_range"}.issubset(t)
        assert len(t["eval_event_range"]) == 2


@pytest.mark.hermetic
def test_client_metadata_matches_backend(hermetic_backend):
    """The backend's per-task metrics/higher_is_better match the client registry."""
    status, body = http_get(f"{hermetic_backend}/v1/benchmark/tasks", timeout=20)
    assert status == 200, f"HTTP {status} — {body}"
    for t in body:
        name = t["name"]
        client = colliderml.tasks.get(name)
        assert list(t["metrics"]) == list(client.metrics), f"{name}: metrics differ"
        assert dict(t["higher_is_better"]) == dict(client.higher_is_better), (
            f"{name}: higher_is_better differs between server and client"
        )
