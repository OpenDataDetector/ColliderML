"""Oracle baselines for the tracking task.

These baselines have no relationship to a real tracking algorithm — they
construct ``track_id`` assignments directly from the truth
``particle_id`` and are useful as pedagogical anchors:

* :func:`perfect_oracle_predictions` assigns every hit to its truth
  particle. It scores ``trackml_eff == 1.0`` and represents the metric
  upper bound.

* :func:`noised_oracle_predictions` starts from the perfect assignment
  and deliberately degrades it by **splitting** a fraction of tracks
  into two halves and **merging** a fraction of track pairs. Splits
  attack the "the track owns ≥50% of its particle's hits" half of the
  TrackML double-majority rule; merges attack the "one particle owns
  ≥50% of the track's hits" half. A researcher can dial the two knobs
  to see each failure mode drop the efficiency independently.

Both functions accept the same ``truth_hits`` shape as
:func:`colliderml.tasks.tracking.metrics.trackml_weighted_efficiency`:
a pyarrow Table with columns ``event_id``, ``hit_id``, and either
``particle_id`` or ``majority_particle_id``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa


def _resolve_particle_column(table: pa.Table) -> str:
    """Return whichever of ``particle_id`` / ``majority_particle_id`` is present."""
    for candidate in ("particle_id", "majority_particle_id"):
        if candidate in table.column_names:
            return candidate
    raise KeyError(
        "truth_hits must contain 'particle_id' or 'majority_particle_id'; "
        f"got {table.column_names}"
    )


def perfect_oracle_predictions(truth_hits: pa.Table) -> pa.Table:
    """Assign each hit's ``track_id`` to its truth ``particle_id``.

    Args:
        truth_hits: Flat pyarrow Table with columns ``event_id``,
            ``hit_id``, and ``particle_id`` (or ``majority_particle_id``).

    Returns:
        A pyarrow Table with columns ``event_id``, ``hit_id``,
        ``track_id`` suitable for
        :meth:`colliderml.tasks.tracking.task.TrackingTask.score`. Scores
        ``trackml_eff == 1.0`` by construction.
    """
    pid_col = _resolve_particle_column(truth_hits)
    return pa.table(
        {
            "event_id": truth_hits.column("event_id"),
            "hit_id": truth_hits.column("hit_id"),
            "track_id": truth_hits.column(pid_col),
        }
    )


def _group_hits_by_track(
    events: List[int], hits: List[int], particles: List[Any]
) -> Dict[Tuple[int, Any], List[int]]:
    """Build ``(event_id, particle_id) -> [hit_indices]`` map (row-indices into the input arrays)."""
    groups: Dict[Tuple[int, Any], List[int]] = defaultdict(list)
    for i, (e, _, pid) in enumerate(zip(events, hits, particles)):
        groups[(int(e), pid)].append(i)
    return groups


def noised_oracle_predictions(
    truth_hits: pa.Table,
    *,
    split_fraction: float = 0.1,
    merge_fraction: float = 0.1,
    seed: Optional[int] = None,
) -> pa.Table:
    """Start from the perfect oracle and corrupt a fraction of the assignments.

    Splitting keeps the ``(particle → track)`` majority but breaks the
    reverse ``(track → particle)`` majority, so the corresponding track
    is scored as a fake. Merging keeps neither majority, so **both**
    contributing tracks lose efficiency.

    Args:
        truth_hits: Flat pyarrow Table with ``event_id``, ``hit_id``,
            ``particle_id`` (or ``majority_particle_id``).
        split_fraction: Fraction of tracks (i.e. distinct ``(event_id,
            particle_id)`` groups) to split in half. ``0.0`` disables
            splitting; ``1.0`` splits every track.
        merge_fraction: Fraction of tracks to merge pairwise within
            their event. Tracks are paired greedily in iteration order;
            pairing an odd number of selected tracks leaves the last one
            alone. ``0.0`` disables merging.
        seed: Optional RNG seed for determinism.

    Returns:
        A pyarrow Table with columns ``event_id``, ``hit_id``,
        ``track_id``. Degradation scales monotonically with both
        fractions.

    Raises:
        ValueError: If ``split_fraction`` or ``merge_fraction`` is
            outside ``[0, 1]``.
    """
    if not 0.0 <= split_fraction <= 1.0:
        raise ValueError(f"split_fraction must be in [0, 1]; got {split_fraction}")
    if not 0.0 <= merge_fraction <= 1.0:
        raise ValueError(f"merge_fraction must be in [0, 1]; got {merge_fraction}")

    rng = np.random.default_rng(seed)
    pid_col = _resolve_particle_column(truth_hits)

    events = truth_hits.column("event_id").to_pylist()
    hits = truth_hits.column("hit_id").to_pylist()
    particles = truth_hits.column(pid_col).to_pylist()
    track_ids: List[Any] = list(particles)

    groups = _group_hits_by_track(events, hits, particles)
    all_tracks = list(groups.keys())

    # Derive a fresh-id generator that cannot collide with any existing particle_id.
    existing_ids = {pid for _, pid in all_tracks}
    try:
        max_existing = max(int(p) for p in existing_ids)
    except (TypeError, ValueError):
        max_existing = 0
    fresh_id_counter = max(max_existing + 1, 1)

    # ---- splits ---------------------------------------------------------
    if split_fraction > 0 and all_tracks:
        n_split = int(round(split_fraction * len(all_tracks)))
        split_targets = rng.choice(len(all_tracks), size=n_split, replace=False)
        for idx in split_targets:
            key = all_tracks[idx]
            row_indices = groups[key]
            n = len(row_indices)
            if n < 2:
                continue  # nothing to split
            # Split asymmetrically: the new-id fragment gets the smaller share
            # (at most ~1/3 of hits, min 1), so that fragment's particle
            # contribution is < 50% and is counted as a fake track. An even
            # split would leave both halves at the 50% threshold and the
            # metric would score both as correct.
            cut = max(1, n // 3)
            new_id = fresh_id_counter
            fresh_id_counter += 1
            for ri in row_indices[:cut]:
                track_ids[ri] = new_id

    # ---- merges ---------------------------------------------------------
    if merge_fraction > 0 and all_tracks:
        # Group tracks by event so merges stay within the same event.
        tracks_by_event: Dict[int, List[Tuple[int, Any]]] = defaultdict(list)
        for key in all_tracks:
            tracks_by_event[key[0]].append(key)
        for event_id, event_tracks in tracks_by_event.items():
            n_merge = int(round(merge_fraction * len(event_tracks)))
            if n_merge < 2:
                continue
            merge_targets = rng.choice(
                len(event_tracks), size=n_merge, replace=False
            )
            selected = [event_tracks[i] for i in merge_targets]
            for left, right in zip(selected[::2], selected[1::2]):
                # Re-tag every hit in `right` with `left`'s current id.
                merged_id = track_ids[groups[left][0]]
                for ri in groups[right]:
                    track_ids[ri] = merged_id

    return pa.table(
        {
            "event_id": pa.array(events),
            "hit_id": pa.array(hits),
            "track_id": pa.array(track_ids),
        }
    )
