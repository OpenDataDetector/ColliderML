"""Pipeline stage: SFAPI / NERSC dispatch health (live drift detector).

Real simulation breaks silently if the deployed backend's egress IP drifts out
of the NERSC-registered allowlist, or its SFAPI credentials expire. CI can't
test SFAPI directly — a GitHub runner's IP isn't allowlisted, so it would 403.
Instead we ask the backend's `/admin/sfapi/health`, which runs the check from
the only allowlisted vantage point (the backend itself).

Needs the prod admin token (`$ADMIN_TOKEN`); skips cleanly without it.
"""

from __future__ import annotations

import os

import pytest

from tests.pipeline._http import http_get

pytestmark = [pytest.mark.pipeline, pytest.mark.live]


@pytest.fixture
def admin_token() -> str:
    t = os.environ.get("ADMIN_TOKEN")
    if not t:
        pytest.skip("ADMIN_TOKEN not set (needed for /admin/sfapi/health)")
    return t


def test_sfapi_egress_in_allowlist_and_reachable(live_backend, admin_token):
    status, body = http_get(
        f"{live_backend}/admin/sfapi/health",
        headers={"X-Admin-Token": admin_token},
        timeout=45,
    )
    assert status == 200, f"HTTP {status} — {body}"
    assert isinstance(body, dict), body
    assert body.get("mock_mode") is False, (
        "backend is in mock mode — SFAPI credentials are not configured, so "
        "submissions never reach Perlmutter."
    )
    assert body.get("in_allowlist") is True, (
        f"backend egress IP {body.get('egress_ip')} is outside the NERSC allowlist "
        f"{body.get('allowlist_cidr')} — update the IRIS SFAPI client's allowlist "
        f"(or the backend's outbound IP). SFAPI calls will 403 until fixed."
    )
    assert body.get("sfapi_ok") is True, (
        f"backend cannot reach SFAPI: {body.get('detail')} "
        f"(credentials revoked/expired, or egress not allowlisted)."
    )
