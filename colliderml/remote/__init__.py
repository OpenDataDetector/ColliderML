"""Client for the ColliderML SaaS backend.

Wraps the FastAPI service that runs in front of NERSC Perlmutter. Used by
:func:`colliderml.simulate` when called with ``remote=True``, and directly
by callers that want to poll or query outside a single ``simulate()``
invocation.

Requires ``pip install 'colliderml[remote]'`` for the ``requests``
dependency. Authentication is via a HuggingFace token (env var or
``huggingface-cli login``).

Public surface::

    from colliderml.remote import (
        submit,       # POST /v1/simulate
        status,       # GET  /v1/requests/{id}
        wait_for,     # submit + poll until terminal
        get_me,       # GET  /v1/me
        balance,      # convenience wrapper around get_me()["credits"]
        RemoteSubmission,
    )
"""

from __future__ import annotations

from colliderml.remote._client import (
    DEFAULT_BACKEND_URL,
    POLL_INTERVAL_SECONDS,
    RemoteSubmission,
    balance,
    get_me,
    status,
    submit,
    wait_for,
)

__all__ = [
    "submit",
    "status",
    "wait_for",
    "get_me",
    "balance",
    "RemoteSubmission",
    "DEFAULT_BACKEND_URL",
    "POLL_INTERVAL_SECONDS",
]
