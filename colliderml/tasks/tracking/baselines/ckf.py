"""CKF baseline for the tracking benchmark.

The ACTS Combinatorial Kalman Filter (CKF) is already part of the
ColliderML simulation pipeline: every ``digi_and_reco`` stage writes the
CKF tracks as a ``tracks`` table. This script converts that table into
the parquet schema expected by :class:`TrackingTask` so users have a
concrete, working baseline to compare against.

Usage::

    # Convert an existing pipeline run:
    python -m colliderml.tasks.tracking.baselines.ckf \\
        --run-dir /path/to/runs/0 --output preds.parquet

    # Or run the pipeline first, then convert (requires the [sim] extra):
    python -m colliderml.tasks.tracking.baselines.ckf --simulate \\
        --channel ttbar --events 1000 --pileup 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _predictions_from_track_tables(track_files: list) -> pa.Table:
    """One row per (track, hit) from event tables (one row per event, list columns).

    ``hit_ids`` are positions in the event's ``tracker_hits`` lists, the ``hit_id``
    convention of :class:`TrackingTask`.
    """
    cols: dict = {"event_id": [], "hit_id": [], "track_id": [], "majority_particle_id": []}
    for f in track_files:
        tracks = pq.read_table(f, columns=["event_id", "track_id", "hit_ids", "majority_particle_id"])
        for ev in tracks.to_pylist():
            for tid, hit_ids, pid in zip(ev["track_id"], ev["hit_ids"], ev["majority_particle_id"]):
                for h in hit_ids:
                    cols["event_id"].append(ev["event_id"])
                    cols["hit_id"].append(h)
                    cols["track_id"].append(tid)
                    cols["majority_particle_id"].append(pid)
    return pa.table(
        {
            "event_id": pa.array(cols["event_id"], pa.uint32()),
            "hit_id": pa.array(cols["hit_id"], pa.uint32()),
            "track_id": pa.array(cols["track_id"], pa.int64()),
            "majority_particle_id": pa.array(cols["majority_particle_id"], pa.uint64()),
        }
    )


def convert_tracks_to_predictions(run_dir: Path, output: Path) -> None:
    """Read a pipeline run's outputs and emit tracking predictions as parquet.

    Reads the ``tracks/*.parquet`` event tables that ``colliderml simulate``
    writes into the run directory, or flat ``tracker_hits.parquet`` and
    ``tracks.parquet`` files from older pipeline versions.
    """
    run_dir = Path(run_dir)
    track_files = sorted((run_dir / "tracks").glob("*.parquet"))
    if track_files:
        preds = _predictions_from_track_tables(track_files)
        pq.write_table(preds, output)
        print(f"Wrote {preds.num_rows} tracking predictions to {output}")
        return

    hits_parquet = run_dir / "tracker_hits.parquet"
    tracks_parquet = run_dir / "tracks.parquet"
    if not (hits_parquet.exists() and tracks_parquet.exists()):
        raise FileNotFoundError(
            f"Expected tracks/*.parquet (or {hits_parquet.name} and {tracks_parquet.name}) "
            f"in {run_dir}. Pass the runs/<run_id> directory of a `colliderml simulate` run."
        )

    hits = pq.read_table(hits_parquet)
    columns = set(hits.column_names)
    event_id = hits.column("event_id") if "event_id" in columns else pa.array([0] * hits.num_rows)
    hit_id = (
        hits.column("hit_id")
        if "hit_id" in columns
        else pa.array(list(range(hits.num_rows)))
    )
    track_id = (
        hits.column("track_id")
        if "track_id" in columns
        else pa.array([-1] * hits.num_rows)
    )

    preds = pa.table(
        {
            "event_id": event_id,
            "hit_id": hit_id,
            "track_id": track_id,
        }
    )
    pq.write_table(preds, output)
    print(f"Wrote {preds.num_rows} tracking predictions to {output}")


def run_full_baseline(channel: str, events: int, pileup: int, output: Path) -> None:
    """Convenience: drive :func:`colliderml.simulate` then convert its output."""
    try:
        from colliderml.simulate import simulate
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CKF baseline --simulate requires the 'sim' extra: "
            "pip install 'colliderml[sim]'"
        ) from exc

    print(f"Running simulation: {channel} × {events} events, pileup={pileup} ...")
    result = simulate(channel=channel, events=events, pileup=pileup)
    print(f"Simulation done. Run dir: {result.run_dir}")
    convert_tracks_to_predictions(Path(result.run_dir), output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CKF baseline for the tracking benchmark")
    parser.add_argument("--run-dir", type=Path, help="Existing pipeline run directory")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run the full simulation pipeline first (requires the [sim] extra).",
    )
    parser.add_argument("--channel", default="ttbar")
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--pileup", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("ckf_preds.parquet"))
    args = parser.parse_args(argv)

    if args.simulate:
        run_full_baseline(args.channel, args.events, args.pileup, args.output)
        return 0
    if args.run_dir:
        convert_tracks_to_predictions(args.run_dir, args.output)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
