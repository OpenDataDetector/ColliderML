"""HuggingFace token resolution.

Separated from :mod:`colliderml.remote._client` so tests can monkey-patch
token lookup without touching HTTP code.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

#: Environment variables checked in order of preference.
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def get_hf_token() -> Optional[str]:
    """Return a HuggingFace API token, or ``None`` if none is configured.

    Resolution order:

    1. ``$HF_TOKEN`` environment variable
    2. ``$HUGGING_FACE_HUB_TOKEN`` environment variable
    3. :func:`huggingface_hub.get_token` (the modern helper, 0.20+)
    4. :class:`huggingface_hub.HfFolder` (legacy path for older hub versions)

    Any missing package or import error in steps 3–4 is silently ignored
    so that ``huggingface_hub`` remains a soft dependency for
    authentication — it is already in :mod:`colliderml`'s base
    ``install_requires``, but users sometimes end up with a partially
    broken install.
    """
    for env_var in _HF_TOKEN_ENV_VARS:
        token = os.environ.get(env_var)
        if token:
            return token

    try:
        from huggingface_hub import get_token as _hf_get_token  # type: ignore
    except ImportError:
        _hf_get_token = None  # type: ignore[assignment]

    if _hf_get_token is not None:
        try:
            saved = _hf_get_token()
        except Exception:
            saved = None
        if saved:
            return saved

    try:
        from huggingface_hub import HfFolder  # type: ignore

        return HfFolder.get_token()
    except Exception:
        return None


def require_hf_token() -> str:
    """Return a HuggingFace token or raise a helpful :class:`RuntimeError`.

    This is the function every HTTP call should use at the top of its
    body — it centralises the error message pointing users at
    ``huggingface-cli login``.
    """
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "The ColliderML remote client requires a HuggingFace token.\n"
            "Get one at https://huggingface.co/settings/tokens and then run one of:\n"
            "  huggingface-cli login           # interactive\n"
            "  export HF_TOKEN=<your-token>    # env var"
        )
    return token


def auth_headers(token: str) -> Dict[str, str]:
    """Build the ``Authorization`` header for a bearer token."""
    return {"Authorization": f"Bearer {token}"}
