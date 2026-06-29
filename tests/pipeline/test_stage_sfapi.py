"""Pipeline stage: SFAPI / NERSC dispatch health (live drift detector).

Real simulation breaks silently if the deployed backend's egress IP drifts out
of the NERSC-registered allowlist, or its SFAPI credentials expire. CI can't
test SFAPI directly — a GitHub runner's IP isn't allowlisted, so it would 403.
Instead we ask the backend's public ``/v1/sfapi/health``, which runs the check
from the only allowlisted vantage point (the backend itself).

The endpoint is intentionally unauthenticated and returns only non-sensitive
health signals, so this needs **no token** — safe for the public repo's CI.
"""

from __future__ import annotations

import pytest

from tests.pipeline._http import http_get

pytestmark = [pytest.mark.pipeline, pytest.mark.live]


def test_sfapi_egress_in_allowlist_and_reachable(live_backend):
    status, body = http_get(f"{live_backend}/v1/sfapi/health", timeout=45)
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
