"""Tracking metrics — TrackML weighted efficiency, fake rate, duplicate rate.

Pure-Python / numpy / pyarrow implementations suitable for both
client-side evaluation (this package) and server-side re-scoring (the
backend). Nothing imports scikit-learn, so the tracking task runs in a
minimal environment.

The primary metric is **TrackML weighted efficiency**: the sum of hit
weights across all correctly reconstructed tracks divided by the total
hit weight in the event's predictions. A reconstructed track is
"correct" if (a) a majority of its hits come from a single particle,
and (b) that particle contributes at least 50% of its own hits to the
track. This mirrors the TrackML challenge definition and penalises both
fragmented tracks and merged tracks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa


def _resolve_truth_pid_column(cols: Dict[str, Any]) -> List[Any]:
    """Return the per-hit truth particle id column.

    Different pipeline versions emit either ``particle_id`` or
    ``majority_particle_id``; we accept both so this metric works on
    old and new releases.
    """
    pid = cols.get("particle_id") or cols.get("majority_particle_id")
    if pid is None:
        raise KeyError(
            "Truth hits table must contain 'particle_id' or 'majority_particle_id'."
        )
    return list(pid)


def _build_truth_map(truth_hits: pa.Table) -> Tuple[Dict[Tuple[int, int], Tuple[Any, float]], Dict[Tuple[int, Any], float]]:
    """Pre-compute the (event, hit) → (particle, weight) map and the per-particle total weight."""
    cols = truth_hits.to_pydict()
    pids = _resolve_truth_pid_column(cols)
    weights = cols.get("weight") or [1.0] * len(cols["event_id"])
    event_col = cols["event_id"]
    hit_col = cols["hit_id"]

    hit_map: Dict[Tuple[int, int], Tuple[Any, float]] = {}
    particle_totals: Dict[Tuple[int, Any], float] = defaultdict(float)
    for e, h, pid, w in zip(event_col, hit_col, pids, weights):
        wf = float(w)
        hit_map[(int(e), int(h))] = (pid, wf)
        particle_totals[(int(e), pid)] += wf
    return hit_map, particle_totals


def trackml_weighted_efficiency(preds: pa.Table, truth_hits: pa.Table) -> float:
    """Weighted reconstruction efficiency in [0, 1].

    Args:
        preds: Columns ``event_id``, ``hit_id``, ``track_id`` (and
            optional ``weight``).
        truth_hits: Columns ``event_id``, ``hit_id``, ``particle_id``
            (or ``majority_particle_id``) and optional ``weight``.
    """
    if truth_hits.num_rows == 0 or preds.num_rows == 0:
        return 0.0
    truth_map, particle_totals = _build_truth_map(truth_hits)

    pred_cols = preds.to_pydict()
    # Aggregate predicted tracks: (event, track) -> list of (particle, weight)
    tracks: Dict[Tuple[int, Any], List[Tuple[Any, float]]] = defaultdict(list)
    total_weight = 0.0
    for e, h, t in zip(pred_cols["event_id"], pred_cols["hit_id"], pred_cols["track_id"]):
        match = truth_map.get((int(e), int(h)))
        if match is None:
            continue
        pid, w = match
        tracks[(int(e), t)].append((pid, w))
        total_weight += w

    correct_weight = 0.0
    for (event, _track), members in tracks.items():
        if not members:
            continue
        by_particle: Dict[Any, float] = defaultdict(float)
        for pid, w in members:
            by_particle[pid] += w
        majority_pid, majority_weight = max(by_particle.items(), key=lambda kv: kv[1])
        track_total = sum(w for _, w in members)
        particle_total = particle_totals.get((event, majority_pid), 0.0)
        if particle_total == 0:
            continue
        if (
            majority_weight / track_total >= 0.5
            and majority_weight / particle_total >= 0.5
        ):
            correct_weight += majority_weight

    if total_weight == 0:
        return 0.0
    return round(correct_weight / total_weight, 6)


def fake_rate(preds: pa.Table, truth_hits: pa.Table) -> float:
    """Fraction of reconstructed tracks with no clear majority particle."""
    if truth_hits.num_rows == 0 or preds.num_rows == 0:
        return 0.0
    truth_map, _ = _build_truth_map(truth_hits)
    pred_cols = preds.to_pydict()

    # (event, track) -> list of particle ids from the truth side
    track_members: Dict[Tuple[int, Any], List[Any]] = defaultdict(list)
    for e, h, t in zip(pred_cols["event_id"], pred_cols["hit_id"], pred_cols["track_id"]):
        info = truth_map.get((int(e), int(h)))
        if info is not None:
            track_members[(int(e), t)].append(info[0])

    if not track_members:
        return 0.0
    fakes = 0
    for members in track_members.values():
        top = Counter(members).most_common(1)
        if not top or top[0][1] / len(members) < 0.5:
            fakes += 1
    return round(fakes / len(track_members), 6)


def duplicate_rate(preds: pa.Table, truth_hits: pa.Table) -> float:
    """Fraction of truth particles matched to more than one reco track."""
    if truth_hits.num_rows == 0 or preds.num_rows == 0:
        return 0.0
    truth_map, _ = _build_truth_map(truth_hits)
    pred_cols = preds.to_pydict()

    particle_tracks: Dict[Tuple[int, Any], set] = defaultdict(set)
    for e, h, t in zip(pred_cols["event_id"], pred_cols["hit_id"], pred_cols["track_id"]):
        info = truth_map.get((int(e), int(h)))
        if info is None:
            continue
        particle_tracks[(int(e), info[0])].add(t)
    if not particle_tracks:
        return 0.0
    dupes = sum(1 for tracks in particle_tracks.values() if len(tracks) > 1)
    return round(dupes / len(particle_tracks), 6)


def physics_eff_pt1(preds: pa.Table, particles: pa.Table) -> float:
    """Fraction of pT>1 GeV truth particles with a reconstructed track.

    "Reconstructed" is judged by ``majority_particle_id`` on the
    predictions side, which the CKF baseline and most ML trackers
    populate. If the predictions don't include that column the metric
    collapses to 0 — this is intentional, it forces models to surface
    the particle association explicitly.
    """
    truth_cols = particles.to_pydict()
    px = np.asarray(truth_cols.get("px", []), dtype=float)
    py = np.asarray(truth_cols.get("py", []), dtype=float)
    pt = np.sqrt(px * px + py * py) if len(px) else np.array([])
    pids = truth_cols.get("particle_id", [])
    primary = truth_cols.get("primary", [True] * len(pids))

    high_pt: set = set()
    for i, (pid, is_primary) in enumerate(zip(pids, primary)):
        if not is_primary:
            continue
        if i < len(pt) and pt[i] > 1.0:
            high_pt.add(pid)
    if not high_pt:
        return 0.0

    pred_cols = preds.to_pydict()
    if "majority_particle_id" not in pred_cols:
        return 0.0
    matched = {int(p) for p in pred_cols["majority_particle_id"] if p in high_pt}
    return round(len(matched) / len(high_pt), 6)


__all__ = [
    "trackml_weighted_efficiency",
    "fake_rate",
    "duplicate_rate",
    "physics_eff_pt1",
]
