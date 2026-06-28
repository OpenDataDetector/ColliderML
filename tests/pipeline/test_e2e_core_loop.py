"""End-to-end: the whole core loop against the CI compose backend (mock-SFAPI).

load -> remote.submit -> wait_for(completed) -> balance dropped -> tasks.evaluate
-> tasks.submit -> leaderboard read reflects the submission. Green once the
backend test-auth seam (ALLOW_TEST_TOKEN) lands; skips cleanly without a token
or a hermetic backend.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import colliderml
from tests.pipeline._http import http_get, http_post

pytestmark = [pytest.mark.pipeline, pytest.mark.e2e, pytest.mark.hermetic]


def _jets_table(n: int = 12) -> pa.Table:
    prob_b = [0.05 + 0.9 * (i / (n - 1)) for i in range(n)]
    prob_c = [(1.0 - p) * 0.4 for p in prob_b]
    prob_light = [1.0 - b - c for b, c in zip(prob_b, prob_c)]
    return pa.table({
        "event_id": list(range(n)), "jet_id": list(range(n)),
        "prob_b": prob_b, "prob_c": prob_c, "prob_light": prob_light,
    })


def test_core_loop_end_to_end(hermetic_backend, hf_token, admin_token, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", hf_token)

    # --- learn + fund the CI test user (created on first /v1/me hit) ---
    bearer = {"Authorization": f"Bearer {hf_token}"}
    st, me = http_get(f"{hermetic_backend}/v1/me", headers=bearer, timeout=20)
    assert st == 200, f"/v1/me -> {st}: {me}"
    username = me["hf_username"]
    st, _ = http_post(
        f"{hermetic_backend}/admin/grant",
        json_body={"hf_username": username, "delta": 1000, "reason": "ci-e2e"},
        headers={"X-Admin-Token": admin_token}, timeout=20,
    )
    assert st == 200, f"/admin/grant -> {st}"

    # --- 1. load public mini-data ---
    frames = colliderml.load("ttbar_pu0", tables=["particles"], max_events=2)
    assert frames["particles"].height >= 1

    # --- 2-4. remote submit -> poll -> balance dropped ---
    from colliderml.remote import balance, submit, wait_for

    before = balance(backend_url=hermetic_backend)
    sub = submit(channel="higgs_portal", events=10, pileup=0, seed=11, backend_url=hermetic_backend)
    assert sub.request_id
    final = wait_for(sub.request_id, backend_url=hermetic_backend, timeout=60, poll_interval=1.0)
    assert final.state == "completed"
    assert balance(backend_url=hermetic_backend) <= before

    # --- 5. tasks evaluate (local preview) ---
    table = _jets_table()
    assert "btag_auc" in colliderml.tasks.evaluate("jets", table)

    # --- 6. tasks submit ---
    body = colliderml.tasks.submit("jets", table, backend_url=hermetic_backend)
    assert "submission_id" in body

    # --- 7. leaderboard reflects the submission ---
    st, lb = http_get(f"{hermetic_backend}/v1/leaderboard/jets", timeout=20)
    assert st == 200 and isinstance(lb, list)
    assert body["submission_id"] in {r.get("id") for r in lb} or any(
        r.get("hf_username") == username for r in lb
    )
