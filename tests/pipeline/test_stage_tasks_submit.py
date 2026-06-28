"""Pipeline stage: tasks submit (remote scoring + leaderboard write).

GREEN (mock): submit() posts the parquet + local scores to
``/v1/benchmark/<task>/submit`` with a bearer token, and surfaces backend
errors. Hermetic: a real round-trip against the compose backend.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import colliderml

pytestmark = pytest.mark.pipeline


def _jets_table(n: int = 12) -> pa.Table:
    prob_b = [0.05 + 0.9 * (i / (n - 1)) for i in range(n)]
    prob_c = [(1.0 - p) * 0.4 for p in prob_b]
    prob_light = [1.0 - b - c for b, c in zip(prob_b, prob_c)]
    return pa.table({
        "event_id": list(range(n)), "jet_id": list(range(n)),
        "prob_b": prob_b, "prob_c": prob_c, "prob_light": prob_light,
    })


def test_submit_posts_multipart(fake_requests):
    fake_requests.post_response = type(fake_requests.post_response)(
        200, {"submission_id": "s-1", "scores": {"btag_auc": 0.9}, "credits_earned": 0.0}
    )
    body = colliderml.tasks.submit("jets", _jets_table(), backend_url="http://test")
    assert body["submission_id"] == "s-1"
    call = fake_requests.post_calls[-1]
    assert call["url"].endswith("/v1/benchmark/jets/submit")
    assert "predictions" in (call["files"] or {})
    assert "local_scores" in (call["data"] or {})
    assert "Authorization" in (call["headers"] or {})


def test_submit_raises_on_backend_error(fake_requests):
    fake_requests.post_response = type(fake_requests.post_response)(500, {}, text="boom")
    with pytest.raises(RuntimeError):
        colliderml.tasks.submit("jets", _jets_table(), backend_url="http://test")


@pytest.mark.hermetic
def test_submit_hermetic_roundtrip(hermetic_backend, hf_token, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", hf_token)
    body = colliderml.tasks.submit("jets", _jets_table(), backend_url=hermetic_backend)
    assert "submission_id" in body and "scores" in body
