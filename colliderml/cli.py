"""ColliderML command-line interface.

Subcommands:

* ``download`` — download dataset configs from HuggingFace to local disk
* ``list-configs`` — list locally cached configs (``--remote`` to query HF)
* ``simulate`` — run the full simulation pipeline locally or remotely
  (requires the ``[sim]`` or ``[remote]`` extras respectively)
* ``list-presets`` — print the bundled simulation preset catalogue
* ``balance`` — show the authenticated user's credit balance on the
  SaaS backend (requires the ``[remote]`` extra)
* ``status <request-id>`` — check a remote simulation request's state

Each subcommand's handler lazily imports its subsystem so that
``colliderml download`` does not pay the cost of importing Docker or
``requests`` code when it only needs the HuggingFace loader.
"""

from __future__ import annotations

import argparse
import json
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


# ---------------------------------------------------------------------------
# v0.4.0 subcommands — simulate / list-presets / balance / status
#
# Each handler uses a lazy import so that users who only run
# `colliderml download` don't pay the cost of importing Docker or
# requests-related code.
# ---------------------------------------------------------------------------


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Run the simulation pipeline locally or submit to the SaaS backend."""
    from colliderml.simulate import simulate  # deferred: pulls pyyaml + docker helpers

    if args.remote and args.local:
        raise SystemExit("--local and --remote are mutually exclusive.")
    use_remote = bool(args.remote)
    try:
        result = simulate(
            preset=args.preset,
            channel=args.channel,
            events=args.events,
            pileup=args.pileup,
            seed=args.seed,
            output_dir=args.output,
            image=args.image,
            remote=use_remote,
            run_id=args.run_id,
            quiet=args.quiet,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    if use_remote:
        print(f"Remote request submitted: {result.remote_request_id}")
        print(f"Poll with: colliderml status {result.remote_request_id}")
    else:
        print(f"Output: {result.run_dir}")
    return 0


def _cmd_list_presets(args: argparse.Namespace) -> int:
    """Print the bundled simulation preset catalogue."""
    from colliderml.simulate import load_presets  # deferred

    presets = load_presets()
    if not presets:
        print("(no presets registered)")
        return 0
    name_width = max(len(name) for name in presets) + 2
    channel_width = max(len(p.channel) for p in presets.values()) + 2
    for name in sorted(presets):
        p = presets[name]
        line = (
            f"{name:<{name_width}} "
            f"channel={p.channel:<{channel_width}} "
            f"events={p.events:<6} "
            f"pileup={p.pileup:<4} "
            f"{p.description}"
        )
        print(line.rstrip())
    return 0


def _cmd_balance(args: argparse.Namespace) -> int:
    """Print the authenticated user's credit balance."""
    try:
        from colliderml.remote import balance, get_me  # deferred
    except ImportError as exc:
        print(
            "The 'balance' subcommand requires the [remote] extra:\n"
            "  pip install 'colliderml[remote]'",
            file=__import__("sys").stderr,
        )
        raise SystemExit(1) from exc

    try:
        me = get_me(backend_url=args.backend_url)
    except RuntimeError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1

    credits = float(me.get("credits", 0.0) or 0.0)
    username = me.get("hf_username", "(unknown)")
    print(f"user:    {username}")
    print(f"credits: {credits:.2f}")
    if args.json:
        print(json.dumps(me, indent=2, sort_keys=True))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Fetch and print the status of a remote simulation request."""
    try:
        from colliderml.remote import status  # deferred
    except ImportError as exc:
        print(
            "The 'status' subcommand requires the [remote] extra:\n"
            "  pip install 'colliderml[remote]'",
            file=__import__("sys").stderr,
        )
        raise SystemExit(1) from exc

    try:
        snap = status(args.request_id, backend_url=args.backend_url)
    except Exception as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1

    print(f"request_id:  {snap.request_id}")
    print(f"state:       {snap.state}")
    print(f"channel:     {snap.channel}")
    print(f"events:      {snap.events}")
    print(f"pileup:      {snap.pileup}")
    if snap.output_hf_repo:
        print(f"output:      {snap.output_hf_repo}")
    if args.json:
        print(json.dumps(snap.raw, indent=2, sort_keys=True, default=str))
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

    # --- simulate ---------------------------------------------------------
    p_sim = sub.add_parser(
        "simulate",
        help="Run the simulation pipeline locally (Docker/Podman) or remotely.",
    )
    p_sim.add_argument("--preset", default=None, help="Named preset (e.g. 'ttbar-quick').")
    p_sim.add_argument(
        "--channel", default=None, help="Physics channel (e.g. 'ttbar', 'higgs_portal')."
    )
    p_sim.add_argument("--events", type=int, default=None, help="Number of events to simulate.")
    p_sim.add_argument("--pileup", type=int, default=None, help="Pileup level (0, 40, 200, …).")
    p_sim.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p_sim.add_argument("--output", default=None, help="Host output directory.")
    p_sim.add_argument(
        "--image",
        default=None,
        help="Container image override. Defaults to the pinned ODD sw image.",
    )
    exec_mode = p_sim.add_mutually_exclusive_group()
    exec_mode.add_argument(
        "--local", action="store_true", help="Run locally (requires Docker or Podman)."
    )
    exec_mode.add_argument(
        "--remote",
        action="store_true",
        help="Submit to the SaaS backend (requires an HF token).",
    )
    p_sim.add_argument(
        "--run-id", default="0", help="Run subdirectory name (default: '0')."
    )
    p_sim.add_argument(
        "--quiet", action="store_true", help="Suppress per-stage progress messages."
    )
    p_sim.set_defaults(func=_cmd_simulate)

    # --- list-presets -----------------------------------------------------
    p_lp = sub.add_parser("list-presets", help="Print the bundled simulation preset catalogue.")
    p_lp.set_defaults(func=_cmd_list_presets)

    # --- balance ----------------------------------------------------------
    p_bal = sub.add_parser("balance", help="Show your SaaS backend credit balance.")
    p_bal.add_argument(
        "--backend-url",
        default=None,
        help="Override the backend URL (default: $COLLIDERML_BACKEND or https://api.colliderml.com).",
    )
    p_bal.add_argument(
        "--json",
        action="store_true",
        help="Also print the full /v1/me response as JSON.",
    )
    p_bal.set_defaults(func=_cmd_balance)

    # --- status -----------------------------------------------------------
    p_stat = sub.add_parser("status", help="Check the state of a remote simulation request.")
    p_stat.add_argument("request_id", help="Backend request ID to query.")
    p_stat.add_argument(
        "--backend-url",
        default=None,
        help="Override the backend URL.",
    )
    p_stat.add_argument(
        "--json",
        action="store_true",
        help="Also print the full /v1/requests response as JSON.",
    )
    p_stat.set_defaults(func=_cmd_status)

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


