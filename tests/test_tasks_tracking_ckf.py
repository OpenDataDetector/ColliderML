"""CKF baseline converts the tracks event table of a simulate run (no network)."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from colliderml.tasks.tracking.baselines.ckf import convert_tracks_to_predictions


def test_ckf_reads_tracks_table(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "0"
    (run / "tracks").mkdir(parents=True)
    tracks = pa.table(
        {
            "event_id": pa.array([0, 1], pa.uint32()),
            "track_id": pa.array([[0, 1], [0]], pa.list_(pa.uint16())),
            "hit_ids": pa.array([[[3, 5], [7]], [[2]]], pa.list_(pa.list_(pa.uint32()))),
            "majority_particle_id": pa.array([[11, 12], [21]], pa.list_(pa.uint64())),
        }
    )
    pq.write_table(tracks, run / "tracks" / "tracks_000000-000002.parquet")

    out = tmp_path / "preds.parquet"
    convert_tracks_to_predictions(run, out)
    preds = pq.read_table(out).to_pydict()
    assert preds == {
        "event_id": [0, 0, 0, 1],
        "hit_id": [3, 5, 7, 2],
        "track_id": [0, 0, 1, 0],
        "majority_particle_id": [11, 11, 12, 21],
    }
