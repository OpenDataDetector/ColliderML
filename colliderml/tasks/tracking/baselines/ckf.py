"""CKF baseline for the tracking benchmark.

The ACTS Combinatorial Kalman Filter (CKF) is already part of the
ColliderML simulation pipeline — every ``digi_and_reco`` stage writes a
``tracksummary_ambi.root`` file. This script converts that output into
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


def convert_tracks_to_predictions(run_dir: Path, output: Path) -> None:
    """Read a pipeline run's outputs and emit tracking predictions as parquet.

    Expects either ``tracker_hits.parquet`` and ``tracks.parquet`` to exist
    in ``run_dir`` (the usual output of the ``convert_all`` stage).
    """
    run_dir = Path(run_dir)
    hits_parquet = run_dir / "tracker_hits.parquet"
    tracks_parquet = run_dir / "tracks.parquet"
    if not (hits_parquet.exists() and tracks_parquet.exists()):
        raise FileNotFoundError(
            f"Expected {hits_parquet.name} and {tracks_parquet.name} in {run_dir}. "
            "Run the pipeline with the `convert_all` stage enabled."
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
