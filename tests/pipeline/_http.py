"""Tiny stdlib HTTP helpers for the pipeline suite (no `requests` dependency)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

LIVE_BACKEND_URL = os.environ.get(
    "COLLIDERML_LIVE_BACKEND", "https://colliderml-backend.onrender.com"
).rstrip("/")


def _maybe_json(body: str):
    try:
        return json.loads(body)
    except (ValueError, TypeError):
        return body


def http_get(url: str, *, timeout: float = 15.0, headers: Optional[Dict[str, str]] = None):
    """GET a URL; return (status_code, parsed_json_or_text)."""
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _maybe_json(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        return e.code, _maybe_json(body)


def http_post(url: str, *, json_body=None, timeout: float = 30.0, headers: Optional[Dict[str, str]] = None):
    """POST JSON to a URL; return (status_code, parsed_json_or_text)."""
    data = json.dumps(json_body or {}).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _maybe_json(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        return e.code, _maybe_json(body)


def wait_for_health(base_url: str, *, deadline_s: float = 90.0) -> bool:
    """Poll ``{base_url}/healthz`` until 200 or the deadline (cold-start safe)."""
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            status, _ = http_get(f"{base_url}/healthz", timeout=10.0)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(3.0)
    return False
