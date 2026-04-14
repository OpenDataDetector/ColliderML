"""HTTP client for the ColliderML SaaS backend.

Keeps the HTTP layer separate from authentication (in
:mod:`colliderml.remote._auth`) so each half can be tested with
``monkeypatch`` in isolation.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from colliderml.remote._auth import auth_headers, require_hf_token

#: Default backend URL. Overridable with ``$COLLIDERML_BACKEND`` or the
#: ``backend_url=`` keyword on every client entry point.
DEFAULT_BACKEND_URL = os.environ.get(
    "COLLIDERML_BACKEND", "https://api.colliderml.com"
)

#: Interval between poll cycles in :func:`wait_for`.
POLL_INTERVAL_SECONDS = 30

#: Timeout (seconds) for submit / status / me calls.
_HTTP_TIMEOUT_SECONDS = 60

#: States that terminate a simulation request.
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


@dataclass
class RemoteSubmission:
    """Snapshot of a remote simulation request.

    The ``state`` and ``output_hf_repo`` fields mutate over time — use
    :func:`status` (or :meth:`refresh`) to refresh them.

    Attributes:
        request_id: Backend identifier for the request.
        state: One of ``submitted``, ``queued``, ``running``,
            ``completed``, ``failed``, ``cancelled``.
        channel: Physics channel that was requested.
        events: Event count.
        pileup: Pileup level.
        credits_charged: Credits deducted from the user's balance for
            this request (``0`` if the backend reused a cached run).
        estimated_node_hours: Backend estimate for the wall-clock cost.
        estimated_completion_seconds: Backend estimate for time-to-done.
        output_hf_repo: HuggingFace dataset repo holding the outputs,
            populated once the job completes.
        backend_url: Base URL that served this request (for subsequent
            :meth:`refresh` calls).
    """

    request_id: str
    state: str
    channel: str
    events: int
    pileup: int
    credits_charged: float = 0.0
    estimated_node_hours: float = 0.0
    estimated_completion_seconds: int = 0
    output_hf_repo: Optional[str] = None
    backend_url: str = DEFAULT_BACKEND_URL
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` if the request is no longer running."""
        return self.state in _TERMINAL_STATES

    def refresh(self) -> "RemoteSubmission":
        """Mutate in-place with the latest state from the backend."""
        latest = status(self.request_id, backend_url=self.backend_url)
        self.state = latest.state
        self.output_hf_repo = latest.output_hf_repo
        self.raw = latest.raw
        return self


def _require_requests() -> Any:
    """Return the :mod:`requests` module or raise a helpful error.

    ``requests`` is only installed under ``colliderml[remote]`` so callers
    who only need local simulation can keep their dependency footprint
    small. The import is deferred into each function so ``import
    colliderml.remote`` itself stays cheap.
    """
    try:
        import requests  # noqa: WPS433  (deferred import by design)
    except ImportError as exc:  # pragma: no cover - exercised in CI envs without extras
        raise ImportError(
            "The colliderml.remote client requires the 'remote' extra.\n"
            "Install it with: pip install 'colliderml[remote]'"
        ) from exc
    return requests


def _resolve_backend_url(backend_url: Optional[str]) -> str:
    return (backend_url or DEFAULT_BACKEND_URL).rstrip("/")


def _parse_submission(
    payload: Dict[str, Any],
    *,
    backend_url: str,
    channel: str,
    events: int,
    pileup: int,
) -> RemoteSubmission:
    """Build a :class:`RemoteSubmission` from a backend JSON body."""
    return RemoteSubmission(
        request_id=str(payload["request_id"]),
        state=str(payload.get("state", "submitted")),
        channel=channel,
        events=events,
        pileup=pileup,
        credits_charged=float(payload.get("credits_charged", 0) or 0),
        estimated_node_hours=float(payload.get("estimated_node_hours", 0) or 0),
        estimated_completion_seconds=int(payload.get("estimated_completion_seconds", 0) or 0),
        output_hf_repo=payload.get("output_hf_repo"),
        backend_url=backend_url,
        raw=dict(payload),
    )


def submit(
    *,
    channel: str,
    events: int,
    pileup: int,
    seed: int = 42,
    backend_url: Optional[str] = None,
) -> RemoteSubmission:
    """Submit a simulation request to the ColliderML backend.

    Args:
        channel: Physics channel (e.g. ``"ttbar"``).
        events: Number of events requested.
        pileup: Pileup level.
        seed: Random seed forwarded to the backend.
        backend_url: Override ``$COLLIDERML_BACKEND`` or the default.

    Returns:
        A :class:`RemoteSubmission` handle. Call :func:`wait_for` (or
        :meth:`RemoteSubmission.refresh`) to track progress.

    Raises:
        RuntimeError: If authentication fails, credits are insufficient,
            or the backend returns a non-2xx status code.
        ImportError: If the ``remote`` extra is not installed.
    """
    requests = _require_requests()
    token = require_hf_token()
    url = _resolve_backend_url(backend_url)

    try:
        response = requests.post(
            f"{url}/v1/simulate",
            json={"channel": channel, "events": events, "pileup": pileup, "seed": seed},
            headers=auth_headers(token),
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach ColliderML backend at {url}: {exc}") from exc

    _raise_for_submit_error(response, url=url)

    payload = response.json()
    submission = _parse_submission(
        payload,
        backend_url=url,
        channel=channel,
        events=events,
        pileup=pileup,
    )

    if payload.get("cached"):
        print(
            f"Reused cached dataset: {submission.output_hf_repo}",
            file=sys.stderr,
        )
    else:
        print(
            f"Submitted request {submission.request_id} "
            f"(est {submission.estimated_node_hours:.2f} credits, "
            f"~{submission.estimated_completion_seconds // 60} min).",
            file=sys.stderr,
        )
    return submission


def _raise_for_submit_error(response: Any, *, url: str) -> None:
    """Translate non-2xx responses from /v1/simulate into actionable errors."""
    if response.status_code == 401:
        raise RuntimeError(
            "Authentication failed. Your HF token may be invalid or expired.\n"
            "Run: huggingface-cli login"
        )
    if response.status_code == 402:
        detail = _safe_json_detail(response, default="Insufficient credits")
        raise RuntimeError(str(detail))
    if response.status_code == 409:
        detail = _safe_json_detail(response, default="Duplicate request")
        raise RuntimeError(f"Duplicate request: {detail}")
    if response.status_code >= 400:
        raise RuntimeError(f"Backend error {response.status_code} at {url}: {response.text}")


def _safe_json_detail(response: Any, *, default: str) -> Any:
    """Return ``response.json()['detail']`` or ``default`` on any failure."""
    try:
        body = response.json()
    except ValueError:
        return default
    if isinstance(body, dict):
        return body.get("detail", default)
    return default


def status(request_id: str, *, backend_url: Optional[str] = None) -> RemoteSubmission:
    """Fetch the current state of a submitted request.

    Returns a :class:`RemoteSubmission` with freshly-populated fields.
    The returned object is a snapshot; it does not share identity with
    any previous :class:`RemoteSubmission` for the same ``request_id``.
    """
    requests = _require_requests()
    token = require_hf_token()
    url = _resolve_backend_url(backend_url)

    response = requests.get(
        f"{url}/v1/requests/{request_id}",
        headers=auth_headers(token),
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    return RemoteSubmission(
        request_id=str(payload.get("request_id", request_id)),
        state=str(payload.get("state", "unknown")),
        channel=str(payload.get("channel", "")),
        events=int(payload.get("events", 0) or 0),
        pileup=int(payload.get("pileup", 0) or 0),
        credits_charged=float(payload.get("credits_charged", 0) or 0),
        estimated_node_hours=float(payload.get("estimated_node_hours", 0) or 0),
        estimated_completion_seconds=int(payload.get("estimated_completion_seconds", 0) or 0),
        output_hf_repo=payload.get("output_hf_repo"),
        backend_url=url,
        raw=dict(payload),
    )


def wait_for(
    request_id: str,
    *,
    backend_url: Optional[str] = None,
    timeout: Optional[float] = None,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    on_poll: Optional[Any] = None,
) -> RemoteSubmission:
    """Poll ``request_id`` until it reaches a terminal state.

    Args:
        request_id: Backend request identifier.
        backend_url: Override the default backend URL.
        timeout: Maximum wall-clock seconds to wait before raising
            :class:`TimeoutError`. ``None`` means wait forever.
        poll_interval: Seconds between polls.
        on_poll: Optional callback ``(submission,) -> None`` invoked on
            every non-terminal poll. Useful for progress reporting.

    Returns:
        The :class:`RemoteSubmission` in its final (terminal) state.

    Raises:
        TimeoutError: If ``timeout`` elapses before the request finishes.
        RuntimeError: If the request terminates in ``failed`` or
            ``cancelled``.
    """
    start = time.monotonic()
    snapshot = status(request_id, backend_url=backend_url)
    while not snapshot.is_terminal:
        if timeout is not None and (time.monotonic() - start) > timeout:
            raise TimeoutError(
                f"Request {request_id} did not complete within {timeout:.0f}s "
                f"(last state: {snapshot.state})"
            )
        if on_poll is not None:
            on_poll(snapshot)
        time.sleep(poll_interval)
        snapshot = status(request_id, backend_url=backend_url)

    if snapshot.state == "failed":
        raise RuntimeError(f"Request {request_id} failed on the backend.")
    if snapshot.state == "cancelled":
        raise RuntimeError(f"Request {request_id} was cancelled.")
    return snapshot


def get_me(*, backend_url: Optional[str] = None) -> Dict[str, Any]:
    """Return the authenticated user's profile from ``/v1/me``.

    The shape of the returned dict is defined by the backend; at minimum
    it includes ``hf_username``, ``credits``, and ``created_at``.
    """
    requests = _require_requests()
    token = require_hf_token()
    url = _resolve_backend_url(backend_url)

    response = requests.get(
        f"{url}/v1/me",
        headers=auth_headers(token),
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected response from {url}/v1/me: {body!r}")
    return body


def balance(*, backend_url: Optional[str] = None) -> float:
    """Return the authenticated user's current credit balance.

    Thin wrapper around :func:`get_me` that pulls out the ``credits``
    field and coerces it to :class:`float`. Used by the top-level
    ``colliderml.balance`` convenience and the ``colliderml balance``
    CLI command.
    """
    me = get_me(backend_url=backend_url)
    return float(me.get("credits", 0.0) or 0.0)
