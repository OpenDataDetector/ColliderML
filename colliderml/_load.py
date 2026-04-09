"""Top-level ``colliderml.load()`` convenience wrapper.

This is the single function most users should reach for. It bundles
the download → load → filter steps into a one-liner while still
honouring the Polars-native contract of the rest of the library.

The canonical, lower-level entry points are still available for
callers that need them:

* :func:`colliderml.core.hf_download.download_config` —
  explicit download with a :class:`DownloadSpec`.
* :func:`colliderml.core.loader.load_tables` —
  loader driven by a :class:`LoaderConfig`.
* :func:`colliderml.core.tables.select_events` — low-level event
  filter.

:func:`load` is just those three composed behind a simple string API,
auto-downloading missing parquets before handing back Polars frames.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import polars as pl

from colliderml.core.data.loader_config import LoaderConfig
from colliderml.core.hf_download import (
    DEFAULT_DATASET_ID,
    DownloadSpec,
    download_config,
    list_local_configs,
)
from colliderml.core.loader import load_tables as _core_load_tables
from colliderml.core.tables import select_events

#: The set of object tables every release ships for a given channel
#: and pileup. Used as the default when the caller doesn't say which
#: tables they want.
DEFAULT_OBJECT_TABLES: List[str] = [
    "particles",
    "tracker_hits",
    "calo_hits",
    "tracks",
]


_PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]


def _parse_dataset_name(dataset: str) -> Tuple[str, str]:
    """Split ``<channel>_<pileup>`` into ``(channel, pileup)``.

    Mirrors :func:`colliderml.tasks._loading.parse_dataset_name` — a thin
    convenience so ``colliderml.load`` can offer the same shorthand as
    the benchmark task runner without a cross-package import.
    """
    import re

    match = re.match(r"^(?P<channel>.+?)_(?P<pileup>pu\d+)$", dataset)
    if not match:
        raise ValueError(
            f"Cannot parse dataset name '{dataset}'. Expected "
            f"'<channel>_pu<N>', e.g. 'ttbar_pu0' or 'higgs_portal_pu200'."
        )
    return match.group("channel"), match.group("pileup")


def _ensure_local(
    channel: str,
    pileup: str,
    tables: List[str],
    *,
    dataset_id: str,
    revision: Optional[str] = None,
) -> None:
    """Download any missing per-object configs into the local cache."""
    local = set(list_local_configs(None, dataset_id))
    for obj in tables:
        cfg_name = f"{channel}_{pileup}_{obj}"
        if cfg_name in local:
            continue
        download_config(
            DownloadSpec(
                dataset_id=dataset_id,
                config=cfg_name,
                revision=revision,
            )
        )


def _apply_event_range(
    frames: Dict[str, _PolarsFrame], event_range: Tuple[int, int]
) -> Dict[str, _PolarsFrame]:
    start, end = event_range
    if end < start:
        raise ValueError(f"event_range end ({end}) must be >= start ({start}).")
    event_ids = list(range(int(start), int(end)))
    return select_events(frames, event_ids)


def _apply_max_events(
    frames: Dict[str, _PolarsFrame], max_events: int
) -> Dict[str, _PolarsFrame]:
    n = int(max_events)
    out: Dict[str, _PolarsFrame] = {}
    for key, frame in frames.items():
        out[key] = frame.head(n) if isinstance(frame, pl.DataFrame) else frame.limit(n)
    return out


def load(
    dataset: str,
    *,
    tables: Optional[List[str]] = None,
    max_events: Optional[int] = None,
    event_range: Optional[Tuple[int, int]] = None,
    dataset_id: Optional[str] = None,
    revision: Optional[str] = None,
    lazy: bool = False,
    auto_download: bool = True,
) -> Dict[str, _PolarsFrame]:
    """Download (if needed) and load a ColliderML dataset.

    ``load`` is a thin convenience over
    :func:`colliderml.core.loader.load_tables` with the following
    differences:

    * The ``dataset`` argument uses the same ``<channel>_<pileup>``
      shorthand as the rest of the library and the benchmark tasks
      (``"ttbar_pu0"``, ``"higgs_portal_pu200"``). ``load`` parses it
      into the ``channels`` and ``pileup`` fields of a
      :class:`LoaderConfig` internally.
    * Missing objects are auto-downloaded on demand when
      ``auto_download=True`` (the default) so notebooks can call
      ``colliderml.load("ttbar_pu0")`` on a fresh machine and have the
      right thing happen.
    * ``event_range=(start, end)`` selects a half-open range of event
      IDs after loading — useful for grabbing a held-out eval split
      without manually composing :func:`colliderml.core.select_events`.

    Args:
        dataset: Dataset key of the form ``"<channel>_<pileup>"``.
        tables: Object tables to load. Defaults to
            :data:`DEFAULT_OBJECT_TABLES`.
        max_events: Cap the result at the first ``max_events`` events
            per table. Applied after ``event_range``.
        event_range: Half-open range ``(start, end)`` of event IDs to
            keep.
        dataset_id: Override the HuggingFace dataset repo. Defaults to
            :data:`colliderml.core.hf_download.DEFAULT_DATASET_ID`
            (currently ``"CERN/ColliderML-Release-1"``).
        revision: Optional HF revision/tag/commit for reproducibility.
        lazy: If ``True``, return :class:`polars.LazyFrame` instead of
            :class:`polars.DataFrame`.
        auto_download: If ``False``, don't attempt to pull missing
            parquets — raise :class:`FileNotFoundError` instead.

    Returns:
        Dict keyed by object name (e.g. ``"tracker_hits"``) whose
        values are Polars frames.

    Raises:
        ValueError: If ``dataset`` does not match ``<channel>_pu<N>``.
        FileNotFoundError: If ``auto_download=False`` and a required
            config isn't cached yet.

    Example::

        import colliderml

        # Default tables, auto-download on first call.
        frames = colliderml.load("ttbar_pu0", max_events=100)
        print(frames["particles"])

        # Held-out eval slice for a benchmark task.
        frames = colliderml.load(
            "ttbar_pu200",
            tables=["tracker_hits"],
            event_range=(90_000, 100_000),
        )
    """
    channel, pileup = _parse_dataset_name(dataset)
    repo = dataset_id or DEFAULT_DATASET_ID
    object_tables = list(tables) if tables is not None else list(DEFAULT_OBJECT_TABLES)

    if auto_download:
        _ensure_local(channel, pileup, object_tables, dataset_id=repo, revision=revision)

    config = LoaderConfig(
        dataset_id=repo,
        channels=channel,
        pileup=pileup,
        objects=object_tables,
        split="train",
        lazy=lazy,
    )
    frames: Dict[str, _PolarsFrame] = _core_load_tables(config)

    if event_range is not None:
        frames = _apply_event_range(frames, event_range)
    if max_events is not None:
        frames = _apply_max_events(frames, max_events)
    return frames


__all__ = ["load", "DEFAULT_OBJECT_TABLES"]
