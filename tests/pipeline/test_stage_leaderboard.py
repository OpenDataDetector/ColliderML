"""Pipeline stage: leaderboard read.

Hermetic GREEN proves the route + SQL are sound on a freshly-migrated DB.
Live RED catches the deployment/data defect: ``/v1/leaderboard/{task}`` 500s
for every task on the deployed backend while ``/v1/benchmark/tasks`` is 200 —
the contrast localizes the break to deployment, not code.
"""

from __future__ import annotations

import pytest

from tests.pipeline._http import http_get

pytestmark = pytest.mark.pipeline

TASKS = ["tracking", "jets", "anomaly", "tracking_latency", "tracking_small", "data_loading"]


@pytest.mark.hermetic
def test_leaderboard_hermetic_returns_list(hermetic_backend):
    """Every task's leaderboard returns 200 + a JSON list on the compose backend."""
    for task in TASKS:
        status, body = http_get(f"{hermetic_backend}/v1/leaderboard/{task}", timeout=20)
        assert status == 200, f"{task}: HTTP {status} — {body}"
        assert isinstance(body, list), f"{task}: expected a list, got {type(body)}"


@pytest.mark.live
def test_benchmark_tasks_live_200(live_backend):
    """The known-good live route — proves the backend is up and lists the tasks."""
    status, body = http_get(f"{live_backend}/v1/benchmark/tasks", timeout=30)
    assert status == 200, f"HTTP {status} — {body}"
    names = {t["name"] for t in body}
    assert {"tracking", "jets", "anomaly"}.issubset(names)


@pytest.mark.live
def test_leaderboard_live_all_tasks_200(live_backend):
    """RED: the live leaderboard read must return 200 for every task. Today it 500s."""
    status, tasks = http_get(f"{live_backend}/v1/benchmark/tasks", timeout=30)
    names = [t["name"] for t in tasks] if status == 200 and isinstance(tasks, list) else TASKS
    failures = {}
    for task in names:
        s, body = http_get(f"{live_backend}/v1/leaderboard/{task}", timeout=30)
        if s != 200:
            failures[task] = (s, str(body)[:120])
    assert not failures, (
        f"GET /v1/leaderboard/<task> not 200 for: {failures} — "
        f"fix the deployed leaderboard read path (backend/app/leaderboard.py)."
    )
