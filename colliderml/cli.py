"""ColliderML command-line interface.

This CLI is intentionally limited to:
- Downloading datasets/configs from HuggingFace to local disk
- Listing locally cached configs (and optionally remote config discovery)

All data loading/manipulation is done separately via the Polars loader.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

from colliderml.core.hf_download import (
    DEFAULT_DATASET_ID,
    DATA_DIR_ENV,
    DownloadSpec,
    default_data_dir,
    discover_remote_configs,
    download_config,
    list_local_configs,
)


def _parse_csv(value: str) -> list[str]:
    """Parse a comma-separated list argument."""
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def _build_config_names(channels: Iterable[str], pileup: str, objects: Iterable[str]) -> list[str]:
    """Build config names like `{channel}_{pileup}_{object}`."""
    configs: list[str] = []
    for ch in channels:
        for obj in objects:
            configs.append(f"{ch}_{pileup}_{obj}")
    return configs


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f'HuggingFace dataset repo id (default: "{DEFAULT_DATASET_ID}")',
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            f"Output base directory for downloads (overrides ${DATA_DIR_ENV}; "
            f"default: {default_data_dir()})"
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HuggingFace revision (tag/branch/commit SHA) for reproducibility.",
    )


def _cmd_download(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).expanduser().resolve() if args.out else None

    if args.config:
        configs = _parse_csv(args.config)
    else:
        channels = _parse_csv(args.channels)
        objects = _parse_csv(args.objects)
        configs = _build_config_names(channels, args.pileup, objects)

    if not configs:
        raise SystemExit("No configs specified. Use --config or (--channels, --pileup, --objects).")

    for cfg in configs:
        spec = DownloadSpec(
            dataset_id=args.dataset_id,
            config=cfg,
            split=args.split,
            revision=args.revision,
            max_events=args.max_events,
        )
        result = download_config(spec, out_dir=out_dir, force=args.force)
        print(f"Downloaded: {result.config} -> {result.local_dir}")
    return 0


def _cmd_list_configs(args: argparse.Namespace) -> int:
    data_dir = Path(args.out).expanduser().resolve() if args.out else None

    if args.remote:
        configs = discover_remote_configs(args.dataset_id, revision=args.revision)
        for c in configs:
            print(c)
        return 0

    configs = list_local_configs(data_dir, args.dataset_id)
    for c in configs:
        print(c)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: parser for the `colliderml` entrypoint.
    """
    parser = argparse.ArgumentParser(prog="colliderml", description="ColliderML CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download dataset parquet shards to local disk")
    _add_common_args(p_dl)
    p_dl.add_argument("--split", default="train", help='Dataset split name (default: "train")')
    p_dl.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Download only enough shards to cover at least N events (heuristic; loader enforces exact).",
    )
    p_dl.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist locally.",
    )
    p_dl.add_argument(
        "--config",
        default=None,
        help='Comma-separated explicit config name(s), e.g. "ttbar_pu0_particles".',
    )
    p_dl.add_argument(
        "--channels",
        default="ttbar",
        help='Comma-separated physics channels/processes, e.g. "ttbar,ggf".',
    )
    p_dl.add_argument("--pileup", default="pu0", help='Pileup token, e.g. "pu0" or "pu200".')
    p_dl.add_argument(
        "--objects",
        default="particles,tracker_hits,calo_hits,tracks",
        help='Comma-separated object types, e.g. "particles,calo_hits".',
    )
    p_dl.set_defaults(func=_cmd_download)

    p_ls = sub.add_parser("list-configs", help="List configs (local cache by default)")
    _add_common_args(p_ls)
    p_ls.add_argument(
        "--remote",
        action="store_true",
        help="List configs by inspecting the remote dataset repo (requires network).",
    )
    p_ls.set_defaults(func=_cmd_list_configs)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argv override for testing.

    Returns:
        int: process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


