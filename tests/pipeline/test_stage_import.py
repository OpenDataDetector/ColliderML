"""Pipeline stage: import / install.

Contract: ``import colliderml`` is cheap (heavy extras lazy-loaded), the
public surface resolves, and ``__version__`` is populated.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.pipeline


def test_bare_import_is_lazy():
    """`import colliderml` must not eagerly import the heavy extras.

    Run in a fresh subprocess so other tests in this session (which import
    colliderml.tasks etc.) don't pollute sys.modules.
    """
    code = (
        "import sys, colliderml; "
        "assert 'colliderml.tasks' not in sys.modules, 'tasks imported eagerly'; "
        "assert 'colliderml.simulate' not in sys.modules, 'simulate imported eagerly'; "
        "print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


def test_lazy_attrs_resolve():
    import colliderml

    assert callable(colliderml.load)
    # Accessing each lazy attribute resolves without raising.
    assert colliderml.tasks is not None
    assert colliderml.remote is not None
    assert callable(colliderml.tasks.list_tasks)


def test_unknown_attr_raises_attributeerror():
    import colliderml

    with pytest.raises(AttributeError):
        _ = colliderml.does_not_exist


def test_version_present():
    import colliderml

    assert isinstance(colliderml.__version__, str)
    assert colliderml.__version__
