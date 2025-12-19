#!/usr/bin/env python
"""Download speed benchmark for ColliderML datasets on HuggingFace.

This is intentionally focused on *download/caching*, not compute performance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from huggingface_hub import hf_hub_download, list_repo_files

from colliderml.core.hf_download import DEFAULT_DATASET_ID, default_data_dir, local_config_dir


DEFAULT_CONFIGS = [
    "ttbar_pu0_particles",
    "ttbar_pu0_tracker_hits",
    "ttbar_pu0_calo_hits",
    "ttbar_pu0_tracks",
]


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _list_shards(dataset_id: str, config: str, split: str, revision: Optional[str]) -> list[str]:
    prefix = f"data/{config}/"
    files = list_repo_files(repo_id=dataset_id, repo_type="dataset", revision=revision)
    shards = [
        f
        for f in files
        if f.startswith(prefix) and f.endswith(".parquet") and f[len(prefix) :].startswith(f"{split}-")
    ]
    return sorted(shards)


@dataclass(frozen=True)
class DownloadTiming:
    """Timing record for one config/shard."""

    config: str
    split: str
    filename: str
    local_path: str
    size_bytes: int
    initial_seconds: float
    cached_seconds: float


def benchmark_download_one(
    *,
    dataset_id: str,
    config: str,
    split: str,
    revision: Optional[str],
    out_dir: Path,
    shard_index: int = 0,
    force_initial: bool = True,
) -> DownloadTiming:
    """Benchmark download + cached re-load for one shard."""
    shards = _list_shards(dataset_id, config, split, revision)
    if not shards:
        raise FileNotFoundError(f"No parquet shards found for config={config} split={split}")
    if shard_index < 0 or shard_index >= len(shards):
        raise IndexError(f"shard_index={shard_index} out of range (0..{len(shards)-1})")

    filename = shards[shard_index]
    cfg_dir = local_config_dir(out_dir, dataset_id, config)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Initial download (force to reflect network time, even if cache exists).
    t0 = time.perf_counter()
    path0 = hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        local_dir=str(cfg_dir),
        local_dir_use_symlinks=False,
        force_download=force_initial,
    )
    t1 = time.perf_counter()

    local_path = Path(path0)
    size_bytes = local_path.stat().st_size if local_path.exists() else 0

    # Cached re-load (should be fast).
    t2 = time.perf_counter()
    path1 = hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        local_dir=str(cfg_dir),
        local_dir_use_symlinks=False,
        force_download=False,
    )
    t3 = time.perf_counter()

    return DownloadTiming(
        config=config,
        split=split,
        filename=filename,
        local_path=str(path1),
        size_bytes=size_bytes,
        initial_seconds=t1 - t0,
        cached_seconds=t3 - t2,
    )


def load_thresholds(path: Path) -> dict:
    """Load thresholds YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("thresholds.yaml must parse to a dict")
    return payload


def check_thresholds(
    timings: List[DownloadTiming], thresholds: dict, *, warn_only: bool = True
) -> int:
    """Compare results to thresholds; prints warnings and returns exit code."""
    cfg_thresholds = {c["name"]: c for c in thresholds.get("configs", []) if isinstance(c, dict) and "name" in c}
    exit_code = 0
    for t in timings:
        th = cfg_thresholds.get(t.config)
        if not th:
            continue
        max_init = float(th.get("max_initial_seconds", 0))
        max_cached = float(th.get("max_cached_seconds", 0))
        if max_init and t.initial_seconds > max_init:
            msg = (
                f"[WARN] download initial_seconds exceeded for {t.config}: "
                f"{t.initial_seconds:.2f}s > {max_init:.2f}s"
            )
            print(msg)
            if not warn_only:
                exit_code = 1
        if max_cached and t.cached_seconds > max_cached:
            msg = (
                f"[WARN] download cached_seconds exceeded for {t.config}: "
                f"{t.cached_seconds:.2f}s > {max_cached:.2f}s"
            )
            print(msg)
            if not warn_only:
                exit_code = 1
    return exit_code


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ColliderML download benchmark (HuggingFace -> local)")
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--split", default="train")
    p.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    p.add_argument("--revision", default=None)
    p.add_argument("--out", default=None, help="Base output dir (defaults to cache/env).")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--thresholds", default="benchmarks/thresholds.yaml")
    p.add_argument("--warn-only", action="store_true", help="Never fail; only print warnings.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve() if args.out else default_data_dir()
    configs = _parse_csv(args.configs)
    thresholds_path = Path(args.thresholds).expanduser().resolve()
    thresholds = load_thresholds(thresholds_path) if thresholds_path.exists() else {}

    timings: List[DownloadTiming] = []
    for cfg in configs:
        print(f"[bench] downloading: {cfg} (shard_index={args.shard_index})")
        t = benchmark_download_one(
            dataset_id=args.dataset_id,
            config=cfg,
            split=args.split,
            revision=args.revision,
            out_dir=out_dir,
            shard_index=args.shard_index,
            force_initial=True,
        )
        print(
            f"[bench] {cfg}: initial={t.initial_seconds:.2f}s cached={t.cached_seconds:.2f}s "
            f"size={t.size_bytes/1e6:.1f}MB"
        )
        timings.append(t)

    payload = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "revision": args.revision,
        "out_dir": str(out_dir),
        "shard_index": args.shard_index,
        "timestamp_unix": int(time.time()),
        "results": [t.__dict__ for t in timings],
    }

    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime(payload["timestamp_unix"]))
    results_path = Path("benchmarks/results") / f"download_benchmark_{ts}.json"
    write_results(results_path, payload)
    print(f"[bench] wrote: {results_path}")

    return check_thresholds(timings, thresholds, warn_only=bool(args.warn_only))


if __name__ == "__main__":
    raise SystemExit(main())


