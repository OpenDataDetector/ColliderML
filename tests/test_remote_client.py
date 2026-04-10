"""Unit tests for :mod:`colliderml.remote._client`.

These tests never hit the network — they inject a fake ``requests``
module via ``monkeypatch`` so each endpoint can be exercised
independently. Authentication is similarly stubbed through
:mod:`colliderml.remote._auth`.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import pytest

from colliderml.remote import (
    RemoteSubmission,
    balance,
    get_me,
    status,
    submit,
    wait_for,
)
from colliderml.remote import _auth as auth_mod
from colliderml.remote import _client as client_mod


# ---------------------------------------------------------------------------
# Fake `requests` module
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_body: Optional[Dict[str, Any]] = None,
        text: str = "",
    ) -> None:
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
    """A minimal stand-in for the real :mod:`requests` module."""

    RequestException = _FakeRequestException

    def __init__(self) -> None:
        self.post_calls: List[Dict[str, Any]] = []
        self.get_calls: List[Dict[str, Any]] = []
        self.post_response: _FakeResponse = _FakeResponse(200, {"request_id": "stub"})
        self.get_response: _FakeResponse = _FakeResponse(200, {})
        self.raise_on_post: Optional[Exception] = None

    def post(
        self,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> _FakeResponse:
        if self.raise_on_post is not None:
            raise self.raise_on_post
        self.post_calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return self.post_response

    def get(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> _FakeResponse:
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        return self.get_response


@pytest.fixture
def fake_requests(monkeypatch: pytest.MonkeyPatch) -> _FakeRequests:
    """Install a fake requests module and return the fake for assertions."""
    fake = _FakeRequests()
    # client_mod._require_requests() imports `requests` dynamically — stash
    # our fake in sys.modules so the dynamic import picks it up.
    monkeypatch.setitem(sys.modules, "requests", fake)  # type: ignore[arg-type]
    # Stub authentication in BOTH the module that defines the symbol and
    # the module that has it bound via `from ... import require_hf_token`.
    # Patching only _auth has no effect because _client already holds a
    # local reference to the original function.
    monkeypatch.setattr(auth_mod, "require_hf_token", lambda: "test-token")
    monkeypatch.setattr(client_mod, "require_hf_token", lambda: "test-token")
    return fake


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_get_hf_token_prefers_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert auth_mod.get_hf_token() == "env-token"


def test_get_hf_token_fallback_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "alt-token")
    assert auth_mod.get_hf_token() == "alt-token"


def test_require_hf_token_raises_with_helpful_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(auth_mod, "get_hf_token", lambda: None)
    with pytest.raises(RuntimeError, match="huggingface-cli login"):
        auth_mod.require_hf_token()


def test_auth_headers_format() -> None:
    assert auth_mod.auth_headers("abc") == {"Authorization": "Bearer abc"}


# ---------------------------------------------------------------------------
# submit()
# ---------------------------------------------------------------------------


def test_submit_posts_expected_payload(fake_requests: _FakeRequests) -> None:
    fake_requests.post_response = _FakeResponse(
        200,
        {
            "request_id": "req-123",
            "state": "submitted",
            "credits_charged": 1.5,
            "estimated_node_hours": 0.8,
            "estimated_completion_seconds": 600,
        },
    )
    submission = submit(channel="ttbar", events=10, pileup=0)
    assert isinstance(submission, RemoteSubmission)
    assert submission.request_id == "req-123"
    assert submission.channel == "ttbar"
    assert submission.events == 10
    assert submission.pileup == 0
    assert submission.credits_charged == 1.5
    assert submission.estimated_node_hours == 0.8
    assert len(fake_requests.post_calls) == 1
    call = fake_requests.post_calls[0]
    assert call["url"].endswith("/v1/simulate")
    assert call["json"] == {"channel": "ttbar", "events": 10, "pileup": 0, "seed": 42}
    assert call["headers"] == {"Authorization": "Bearer test-token"}


def test_submit_respects_backend_url_override(fake_requests: _FakeRequests) -> None:
    fake_requests.post_response = _FakeResponse(
        200, {"request_id": "req-xyz", "state": "submitted"}
    )
    submit(
        channel="ttbar",
        events=1,
        pileup=0,
        backend_url="https://staging.example.com/",
    )
    # Trailing slash must be stripped, /v1/simulate appended exactly once.
    assert fake_requests.post_calls[0]["url"] == "https://staging.example.com/v1/simulate"


def test_submit_raises_on_401(fake_requests: _FakeRequests) -> None:
    fake_requests.post_response = _FakeResponse(401, {"detail": "bad token"})
    with pytest.raises(RuntimeError, match="Authentication failed"):
        submit(channel="ttbar", events=1, pileup=0)


def test_submit_raises_on_402_insufficient_credits(fake_requests: _FakeRequests) -> None:
    fake_requests.post_response = _FakeResponse(402, {"detail": "need 3 more credits"})
    with pytest.raises(RuntimeError, match="need 3 more credits"):
        submit(channel="ttbar", events=1000, pileup=200)


def test_submit_raises_on_other_400(fake_requests: _FakeRequests) -> None:
    fake_requests.post_response = _FakeResponse(
        500, {"detail": "oops"}, text='{"detail":"oops"}'
    )
    with pytest.raises(RuntimeError, match="Backend error 500"):
        submit(channel="ttbar", events=1, pileup=0)


def test_submit_wraps_connection_errors(fake_requests: _FakeRequests) -> None:
    fake_requests.raise_on_post = _FakeRequestException("boom")
    with pytest.raises(RuntimeError, match="Could not reach"):
        submit(channel="ttbar", events=1, pileup=0)


# ---------------------------------------------------------------------------
# status() / wait_for()
# ---------------------------------------------------------------------------


def test_status_returns_remote_submission(fake_requests: _FakeRequests) -> None:
    fake_requests.get_response = _FakeResponse(
        200,
        {
            "request_id": "req-xyz",
            "state": "running",
            "channel": "higgs_portal",
            "events": 10,
            "pileup": 10,
            "credits_charged": 0.2,
        },
    )
    snap = status("req-xyz")
    assert snap.request_id == "req-xyz"
    assert snap.state == "running"
    assert snap.is_terminal is False
    assert snap.channel == "higgs_portal"


def test_wait_for_polls_until_terminal(
    fake_requests: _FakeRequests, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        [
            _FakeResponse(200, {"request_id": "r", "state": "running"}),
            _FakeResponse(200, {"request_id": "r", "state": "running"}),
            _FakeResponse(200, {"request_id": "r", "state": "completed"}),
        ]
    )

    def fake_get(url: str, **_: Any) -> _FakeResponse:
        return next(responses)

    monkeypatch.setattr(fake_requests, "get", fake_get)
    monkeypatch.setattr(client_mod.time, "sleep", lambda _x: None)

    polls: List[str] = []
    snap = wait_for("r", poll_interval=0.0, on_poll=lambda s: polls.append(s.state))
    assert snap.state == "completed"
    assert snap.is_terminal
    assert polls == ["running", "running"]


def test_wait_for_raises_on_failed_state(
    fake_requests: _FakeRequests, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_requests.get_response = _FakeResponse(
        200, {"request_id": "r", "state": "failed"}
    )
    monkeypatch.setattr(client_mod.time, "sleep", lambda _x: None)
    with pytest.raises(RuntimeError, match="failed on the backend"):
        wait_for("r", poll_interval=0.0)


def test_wait_for_raises_on_timeout(
    fake_requests: _FakeRequests, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_requests.get_response = _FakeResponse(
        200, {"request_id": "r", "state": "running"}
    )
    monkeypatch.setattr(client_mod.time, "sleep", lambda _x: None)

    # Drive monotonic so the first post-poll tick exceeds the timeout.
    ticks = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: next(ticks))

    with pytest.raises(TimeoutError):
        wait_for("r", poll_interval=0.0, timeout=1.0)


# ---------------------------------------------------------------------------
# get_me() / balance()
# ---------------------------------------------------------------------------


def test_get_me_returns_body(fake_requests: _FakeRequests) -> None:
    fake_requests.get_response = _FakeResponse(
        200, {"hf_username": "alice", "credits": 42.5, "created_at": "2025-01-01"}
    )
    me = get_me()
    assert me["hf_username"] == "alice"
    assert fake_requests.get_calls[-1]["url"].endswith("/v1/me")


def test_balance_pulls_credits_field(fake_requests: _FakeRequests) -> None:
    fake_requests.get_response = _FakeResponse(200, {"credits": 7.0})
    assert balance() == 7.0


def test_balance_defaults_to_zero(fake_requests: _FakeRequests) -> None:
    fake_requests.get_response = _FakeResponse(200, {})
    assert balance() == 0.0


# ---------------------------------------------------------------------------
# RemoteSubmission
# ---------------------------------------------------------------------------


def test_remote_submission_is_terminal() -> None:
    assert RemoteSubmission(
        request_id="r", state="completed", channel="ttbar", events=1, pileup=0
    ).is_terminal
    assert not RemoteSubmission(
        request_id="r", state="running", channel="ttbar", events=1, pileup=0
    ).is_terminal


def test_remote_submission_refresh_updates_state(
    fake_requests: _FakeRequests, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = RemoteSubmission(
        request_id="r",
        state="submitted",
        channel="ttbar",
        events=1,
        pileup=0,
    )
    fake_requests.get_response = _FakeResponse(
        200,
        {
            "request_id": "r",
            "state": "completed",
            "channel": "ttbar",
            "events": 1,
            "pileup": 0,
            "output_hf_repo": "ColliderML/req-r",
        },
    )
    snap.refresh()
    assert snap.state == "completed"
    assert snap.output_hf_repo == "ColliderML/req-r"
