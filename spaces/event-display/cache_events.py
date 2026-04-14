"""Pre-cache a small set of events per dataset for instant Space load.

Run this once at build time (or in a HF Space build hook) to populate
``_cached_events/<dataset>/<table>.parquet`` with ~50 events each.

Usage::

    python cache_events.py
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import colliderml

from app import CACHE_DIR, DATASETS, EVENTS_PER_DATASET, _frame_to_arrow


def main() -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    for dataset in DATASETS:
        out_dir = CACHE_DIR / dataset
        out_dir.mkdir(exist_ok=True)
        print(f"Caching {dataset} ...")
        try:
            frames = colliderml.load(
                dataset,
                tables=["tracker_hits", "particles", "tracks"],
                max_events=EVENTS_PER_DATASET,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        for name, frame in frames.items():
            table: pa.Table = _frame_to_arrow(frame)
            out_path = out_dir / f"{name}.parquet"
            pq.write_table(table, str(out_path))
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  {name}: {table.num_rows} rows ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
