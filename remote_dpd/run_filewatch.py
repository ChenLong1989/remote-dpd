"""Command-line entry point compatible with the old supplier argument."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .service import RemoteDPDService, ServiceOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MATLAB-free remote ILC DPD file watcher")
    parser.add_argument("supplier_name", help="supplier directory name, e.g. Zilink")
    parser.add_argument("--watch-root", default="/opt/SharePoint", help="root containing supplier directories")
    parser.add_argument("--path", help="watch this exact directory instead of <watch-root>/<supplier>")
    parser.add_argument(
        "--engine",
        default="ilc",
        choices=(
            "ilc",
            "legacy_ilc",
            "linear_ilc",
            "instantaneous_gain_ilc",
            "model_vjp_ilc",
            "model_lm_ilc",
        ),
        help="DPD/ILC engine",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=1800.0)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="[remote-dpd] %(asctime)s %(levelname)s %(message)s")
    directory = Path(args.path) if args.path else Path(args.watch_root) / args.supplier_name
    service = RemoteDPDService(
        directory,
        supplier_name=args.supplier_name,
        engine_name=args.engine,
        options=ServiceOptions(heartbeat_seconds=args.heartbeat_seconds),
    )
    service.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
