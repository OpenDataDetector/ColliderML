"""Pipeline stage: simulate remotely (SaaS client).

Mock tier asserts the submit/poll HTTP contract. The RED test asserts the
*shipped default* backend host resolves. Hermetic tier drives a real
submit→wait_for against the CI compose backend (mock-SFAPI).
"""

from __future__ import annotations

import importlib
import socket
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.pipeline


def test_submit_posts_to_v1_simulate(fake_requests):
    """Mock: remote.submit() POSTs /v1/simulate with the request payload + bearer."""
    from colliderml.remote import submit

    fake_requests.post_response = type(fake_requests.post_response)(
        200,
        {
            "request_id": "r-1",
            "state": "submitted",
            "channel": "ttbar",
            "events": 10,
            "pileup": 0,
            "credits_charged": 1.0,
        },
    )
    sub = submit(channel="ttbar", events=10, pileup=0, seed=42, backend_url="http://test")
    assert sub.request_id == "r-1"
    call = fake_requests.post_calls[-1]
    assert call["url"].endswith("/v1/simulate")
    assert call["json"] == {"channel": "ttbar", "events": 10, "pileup": 0, "seed": 42}
    assert "Authorization" in (call["headers"] or {})


def test_default_backend_is_reachable(monkeypatch):
    """RED: the backend URL the library ships by default must resolve via DNS.

    Re-derive the *shipped* default (env unset) so a CI COLLIDERML_BACKEND
    override can't mask the dead host. Today it is api.colliderml.com (NXDOMAIN).
    """
    import colliderml.remote._client as client_mod

    monkeypatch.delenv("COLLIDERML_BACKEND", raising=False)
    importlib.reload(client_mod)
    try:
        host = urlparse(client_mod.DEFAULT_BACKEND_URL).hostname
        try:
            socket.getaddrinfo(host, 443)
        except socket.gaierror as e:
            pytest.fail(
                f"shipped default backend host {host!r} does not resolve ({e}); "
                f"repoint DEFAULT_BACKEND_URL in colliderml/remote/_client.py."
            )
    finally:
        importlib.reload(client_mod)  # restore env-derived value for other tests


@pytest.mark.hermetic
def test_submit_wait_for_hermetic(hermetic_backend, hf_token, monkeypatch):
    """Real submit→wait_for against the compose backend (mock-SFAPI completes ~2s)."""
    monkeypatch.setenv("HF_TOKEN", hf_token)
    from colliderml.remote import submit, wait_for

    sub = submit(channel="higgs_portal", events=10, pileup=0, seed=7, backend_url=hermetic_backend)
    assert sub.request_id
    final = wait_for(sub.request_id, backend_url=hermetic_backend, timeout=60, poll_interval=1.0)
    assert final.state == "completed"
