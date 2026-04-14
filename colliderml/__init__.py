"""ColliderML: a modern machine-learning library for HEP data.

This is the top-level package. It re-exports the four existing
subpackages (``core``, ``physics``, ``polars``, ``utils`` — all v0.3.1
API is preserved) plus the v0.4.0 additions:

* :func:`load` — one-liner for downloading and loading a dataset
* :mod:`colliderml.simulate` — run the pipeline locally (via Docker)
  or remotely (via the SaaS backend)
* :mod:`colliderml.remote` — direct client for the SaaS backend
* :mod:`colliderml.tasks` — benchmark task registry + reference
  baselines

Heavyweight features are gated behind optional extras so the base
install stays lean:

.. code-block:: bash

    pip install colliderml                 # core + load + viz
    pip install 'colliderml[sim]'          # + local Docker/Podman pipeline
    pip install 'colliderml[remote]'       # + SaaS client (requests)
    pip install 'colliderml[tasks]'        # + reference baselines (scikit-learn)
    pip install 'colliderml[all]'          # everything above + dev tools

The ``simulate``, ``remote``, and ``tasks`` subpackages are imported
lazily on first access so that ``import colliderml`` stays cheap even
when the extras are not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.4.0"

# Subpackages that were in v0.3.1 and are always available.
from . import core
from . import physics
from . import polars
from . import utils

# Top-level convenience function — canonical one-liner for dataset access.
from colliderml._load import DEFAULT_OBJECT_TABLES, load

#: Names the top-level package exposes. ``simulate``, ``balance``,
#: ``tasks`` and ``remote`` are resolved lazily via ``__getattr__`` below
#: so that ``import colliderml`` stays cheap for users of the base install.
__all__ = [
    "__version__",
    "core",
    "physics",
    "polars",
    "utils",
    "load",
    "DEFAULT_OBJECT_TABLES",
    "simulate",
    "balance",
    "tasks",
    "remote",
]


if TYPE_CHECKING:  # pragma: no cover - hints for IDEs and type checkers
    from colliderml import remote as remote  # noqa: F401
    from colliderml import simulate as simulate_module  # noqa: F401
    from colliderml import tasks as tasks  # noqa: F401
    from colliderml.remote import balance as balance  # noqa: F401
    from colliderml.simulate import simulate as simulate  # noqa: F401


def __getattr__(name: str) -> Any:
    """Lazy-import the optional v0.4.0 subpackages and their convenience callables.

    Importing ``colliderml.simulate`` pulls in ``pyyaml`` and the
    Docker orchestration code; ``colliderml.remote`` wants
    ``requests``; ``colliderml.tasks`` pulls in the whole registry
    and numpy/scipy metrics. Users who only need ``colliderml.load``
    shouldn't pay for any of that at import time, so we defer each
    one until it's first accessed.

    Raises:
        AttributeError: For attributes that aren't part of the public
            surface listed in ``__all__``.
    """
    if name == "simulate":
        from colliderml.simulate.api import simulate as _simulate

        return _simulate
    if name == "balance":
        from colliderml.remote import balance as _balance

        return _balance
    if name == "tasks":
        import colliderml.tasks as _tasks

        return _tasks
    if name == "remote":
        import colliderml.remote as _remote

        return _remote
    raise AttributeError(f"module 'colliderml' has no attribute {name!r}")
