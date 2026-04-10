"""Shared data-loading helper for benchmark tasks.

Wraps :func:`colliderml.core.loader.load_tables` and
:func:`colliderml.core.hf_download.download_config` with a simple
``<channel>_<pileup>`` string API so task implementations don't have to
construct a :class:`LoaderConfig` for every call. Also converts the
Polars frames the loader returns into pyarrow Tables at the boundary —
task scoring metrics in this repo are pyarrow-first, while the rest of
the library is Polars-first, so we isolate the format bridge here.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import polars as pl
import pyarrow as pa

from colliderml.core.data.loader_config import LoaderConfig
from colliderml.core.hf_download import (
    DEFAULT_DATASET_ID,
    DownloadSpec,
    download_config,
    list_local_configs,
)
from colliderml.core.loader import load_tables
from colliderml.core.tables import select_events

#: Regex that splits ``<channel>_<pileup>`` into its two halves. The
#: pileup segment always starts with ``pu`` so we can reliably split on
#: it even when ``channel`` itself contains underscores (e.g.
#: ``higgs_portal_pu10``).
_DATASET_NAME_PATTERN = re.compile(r"^(?P<channel>.+?)_(?P<pileup>pu\d+)$")


def parse_dataset_name(name: str) -> Tuple[str, str]:
    """Split a ``"<channel>_<pileup>"`` string into ``(channel, pileup)``.

    Args:
        name: A dataset name like ``"ttbar_pu0"`` or ``"higgs_portal_pu200"``.

    Returns:
        ``(channel, pileup)`` tuple, e.g. ``("higgs_portal", "pu200")``.

    Raises:
        ValueError: If ``name`` does not match ``<channel>_pu<digits>``.
    """
    match = _DATASET_NAME_PATTERN.match(name)
    if not match:
        raise ValueError(
            f"Cannot parse dataset name '{name}'. Expected the form "
            f"'<channel>_pu<N>', e.g. 'ttbar_pu0' or 'higgs_portal_pu200'."
        )
    return match.group("channel"), match.group("pileup")


def _ensure_downloaded(
    channel: str,
    pileup: str,
    tables: List[str],
    *,
    dataset_id: str,
) -> None:
    """Populate the local cache with every per-object config this task needs.

    The loader can only read configs that are already on disk, so this
    pre-flight step calls :func:`download_config` for any missing piece.
    """
    local = set(list_local_configs(None, dataset_id))
    for obj in tables:
        cfg_name = f"{channel}_{pileup}_{obj}"
        if cfg_name in local:
            continue
        download_config(DownloadSpec(dataset_id=dataset_id, config=cfg_name))


def _expand_event_range(event_range: Tuple[int, int]) -> List[int]:
    start, end = event_range
    if end < start:
        raise ValueError(f"event_range end ({end}) must be >= start ({start}).")
    return list(range(int(start), int(end)))


def _collect_and_convert(
    frames: Dict[str, "pl.DataFrame | pl.LazyFrame"],
) -> Dict[str, pa.Table]:
    """Collect any lazy frames and convert every frame to a pyarrow Table."""
    out: Dict[str, pa.Table] = {}
    for key, frame in frames.items():
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()
        out[key] = frame.to_arrow()
    return out


def load_task_data(
    dataset: str,
    *,
    tables: List[str],
    max_events: Optional[int] = None,
    event_range: Optional[Tuple[int, int]] = None,
    dataset_id: Optional[str] = None,
) -> Dict[str, pa.Table]:
    """Load tables for a ``<channel>_<pileup>`` dataset key.

    Args:
        dataset: Dataset key like ``"ttbar_pu200"``.
        tables: Object types to load (e.g. ``["tracker_hits"]``).
        max_events: Optional cap on the number of events.
        event_range: Optional half-open range of event IDs; events outside
            are dropped. Applied before ``max_events``.
        dataset_id: Override the HuggingFace dataset repo. Defaults to
            :data:`colliderml.core.hf_download.DEFAULT_DATASET_ID`.

    Returns:
        Mapping of object name → pyarrow Table.
    """
    # Validate the event range *before* touching the network so callers
    # get a fast failure on bad inputs.
    if event_range is not None:
        _expand_event_range(event_range)

    channel, pileup = parse_dataset_name(dataset)
    repo = dataset_id or DEFAULT_DATASET_ID

    _ensure_downloaded(channel, pileup, tables, dataset_id=repo)

    config = LoaderConfig(
        dataset_id=repo,
        channels=channel,
        pileup=pileup,
        objects=list(tables),
        split="train",
        lazy=False,
    )
    polars_frames = load_tables(config)

    if event_range is not None:
        polars_frames = select_events(polars_frames, _expand_event_range(event_range))

    if max_events is not None:
        # Simple head-based limit; tasks rarely need precise per-event semantics
        # once event_range has already narrowed the set.
        polars_frames = {
            key: (frame.head(int(max_events)) if isinstance(frame, pl.DataFrame) else frame.limit(int(max_events)))
            for key, frame in polars_frames.items()
        }

    return _collect_and_convert(polars_frames)
