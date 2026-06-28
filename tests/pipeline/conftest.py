"""Shared fixtures for the pipeline-health suite.

Reuses the existing mock harnesses (``_FakeRequests`` from
``tests/test_remote_client.py``, the ``stub_loader`` pattern from
``tests/test_load_unified.py``) and adds backend fixtures for the two
targeting modes:

* **hermetic** — a CI-stood-up backend at ``$COLLIDERML_BACKEND``
  (docker-compose, mock-SFAPI). Tests skip cleanly if it isn't set.
* **live** — the deployed backend (``colliderml-backend.onrender.com``),
  used by the drift-detector lane; warmed for Render free-tier cold start.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Dict, List, Optional

import pytest

from tests.pipeline._http import LIVE_BACKEND_URL, http_get, wait_for_health  # noqa: F401


# ---------------------------------------------------------------------------
# Backend target fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hermetic_backend() -> str:
    """URL of a CI-stood-up backend; skip if not provided.

    The pipeline-health workflow sets COLLIDERML_BACKEND=http://localhost:8000
    after `docker compose up`. Locally: run backend/docker-compose.yml (or
    scripts/setup_tutorial_env.sh) and export COLLIDERML_BACKEND.
    """
    url = os.environ.get("COLLIDERML_BACKEND", "").rstrip("/")
    if not url or "onrender.com" in url:
        pytest.skip("no hermetic backend ($COLLIDERML_BACKEND points at localhost compose)")
    if not wait_for_health(url, deadline_s=60.0):
        pytest.skip(f"hermetic backend {url} did not become healthy")
    return url


@pytest.fixture(scope="session")
def live_backend() -> str:
    """URL of the deployed backend, warmed for cold start."""
    url = LIVE_BACKEND_URL.rstrip("/")
    if not wait_for_health(url, deadline_s=90.0):
        pytest.skip(f"live backend {url} unreachable / did not wake")
    return url


@pytest.fixture
def hf_token() -> str:
    """A real HF bearer token, or skip cleanly (for `needs_hf_token` tests)."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        pytest.skip("HF_TOKEN not set")
    return tok


@pytest.fixture
def docker_available() -> str:
    rt = shutil.which("docker") or shutil.which("podman")
    if not rt:
        pytest.skip("no docker/podman runtime on PATH")
    return rt


@pytest.fixture(scope="session")
def admin_token() -> str:
    return os.environ.get("ADMIN_TOKEN", "dev-admin-token")


# ---------------------------------------------------------------------------
# Reused mock harnesses
# ---------------------------------------------------------------------------
# Lifted from tests/test_remote_client.py so the pipeline suite is self-contained.

class _FakeResponse:
    def __init__(self, status_code: int = 200, json_body: Optional[Dict[str, Any]] = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_body = json_body if json_body is not None else {}
        self.text = text or str(self._json_body)

    def json(self) -> Dict[str, Any]:
        return self._json_body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _FakeHTTPError(f"status {self.status_code}")


class _FakeHTTPError(Exception):
    pass


class _FakeRequestException(Exception):
    pass


class _FakeRequests:
    """Minimal stand-in for the `requests` module."""

    RequestException = _FakeRequestException

    def __init__(self) -> None:
        self.post_calls: List[Dict[str, Any]] = []
        self.get_calls: List[Dict[str, Any]] = []
        self.post_response: _FakeResponse = _FakeResponse(200, {"request_id": "stub"})
        self.get_response: _FakeResponse = _FakeResponse(200, {})
        self.raise_on_post: Optional[Exception] = None

    def post(self, url, *, json=None, data=None, files=None, headers=None, timeout=None):
        if self.raise_on_post is not None:
            raise self.raise_on_post
        self.post_calls.append(
            {"url": url, "json": json, "data": data, "files": files, "headers": headers, "timeout": timeout}
        )
        return self.post_response

    def get(self, url, *, headers=None, timeout=None):
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        return self.get_response


@pytest.fixture
def fake_requests(monkeypatch: pytest.MonkeyPatch) -> _FakeRequests:
    """Install a fake `requests` module + stub HF auth; return the fake."""
    from colliderml.remote import _auth as auth_mod
    from colliderml.remote import _client as client_mod

    fake = _FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake)  # type: ignore[arg-type]
    monkeypatch.setattr(auth_mod, "require_hf_token", lambda: "test-token")
    monkeypatch.setattr(client_mod, "require_hf_token", lambda: "test-token")
    return fake


@pytest.fixture
def stub_loader(monkeypatch: pytest.MonkeyPatch):
    """Replace the core loader + cache-ensurance so load tests are pure-Python."""
    import polars as pl

    from colliderml import _load as load_mod

    calls: Dict[str, object] = {}

    def fake_ensure(channel, pileup, tables, *, dataset_id, revision=None, max_events=None, event_range=None):
        calls["ensure"] = (channel, pileup, list(tables), dataset_id, revision)
        calls["ensure_bounds"] = (max_events, event_range)

    stub_frame = pl.DataFrame({"event_id": list(range(10)), "hit_id": list(range(10))})

    def fake_load_tables(cfg):
        calls["cfg"] = cfg
        return {"tracker_hits": stub_frame.clone(), "particles": stub_frame.clone()}

    monkeypatch.setattr(load_mod, "_ensure_local", fake_ensure)
    monkeypatch.setattr(load_mod, "_core_load_tables", fake_load_tables)
    return calls
