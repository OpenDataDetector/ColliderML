"""Syntax-check the python code blocks of the user docs and the README (no network, nothing is run).

API signature listings such as ``colliderml.simulate(*, preset: str | None = None, ...) -> SimulationResult``
are checked as ``def`` statements, and ``>>>`` blocks with the prompts removed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    sorted((ROOT / "docs" / "guide").glob("*.md"))
    + sorted((ROOT / "docs" / "library").glob("*.md"))
    + [ROOT / "docs" / "index.md", ROOT / "README.md"]
)
FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>\w*)[^\n]*\n(?P<body>.*?)^(?P=indent)```", re.M | re.S)
SIGNATURE = re.compile(r"^[A-Za-z_][\w.]*\(\s*$")


def _python_blocks():
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        for m in FENCE.finditer(text):
            if m.group("lang") != "python":
                continue
            line = text.count("\n", 0, m.start()) + 1
            indent = m.group("indent")
            body = "\n".join(
                l[len(indent):] if l.startswith(indent) else l for l in m.group("body").splitlines()
            )
            yield pytest.param(body, id=f"{page.relative_to(ROOT)}:{line}")


def _as_source(body: str) -> str:
    lines = body.splitlines()
    if lines and lines[0].startswith(">>>"):
        return "\n".join(l[4:] for l in lines if l.startswith((">>> ", "... ")))
    if lines and SIGNATURE.match(lines[0]):
        return "def _signature(\n" + "\n".join(lines[1:]) + ": ...\n"
    return body


def test_pages_have_python_blocks():
    assert len(list(_python_blocks())) > 50


@pytest.mark.parametrize("body", list(_python_blocks()))
def test_python_block_compiles(body):
    compile(_as_source(body), "<docs>", "exec")
