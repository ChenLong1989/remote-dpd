"""Command-line entry point for the versioned MAT command service."""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from .file_interface import FileCommandService
from .storage import (
    DEFAULT_CLEANUP_INTERVAL_SECONDS,
    DEFAULT_RETENTION_SECONDS,
    RunStore,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Device-driven DPD service with a versioned MAT command inbox"
    )
    parser.add_argument(
        "--exchange-root",
        required=True,
        help="directory containing the inbox and outbox subdirectories",
    )
    parser.add_argument(
        "--runtime-root",
        help="temporary run-storage root (default: <exchange-root>/runtime)",
    )
    parser.add_argument(
        "--retention-days",
        type=float,
        default=DEFAULT_RETENTION_SECONDS / 86400.0,
        help="temporary run retention in days",
    )
    parser.add_argument(
        "--cleanup-interval-seconds",
        type=float,
        default=float(DEFAULT_CLEANUP_INTERVAL_SECONDS),
    )
    parser.add_argument("--status-poll-seconds", type=float, default=0.02)
    parser.add_argument(
        "--once",
        action="store_true",
        help="scan all current commands synchronously and exit",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def run(args: argparse.Namespace, *, stop_event: threading.Event | None = None) -> int:
    exchange_root = Path(args.exchange_root).expanduser()
    runtime_root = (
        Path(args.runtime_root).expanduser()
        if args.runtime_root
        else exchange_root / "runtime"
    )
    retention_seconds = float(args.retention_days) * 86400.0
    if retention_seconds < 0.0:
        raise ValueError("retention_days must not be negative")

    store = RunStore(
        runtime_root,
        retention_seconds=retention_seconds,
        cleanup_interval_seconds=float(args.cleanup_interval_seconds),
    )
    service = FileCommandService(
        exchange_root,
        run_store=store,
        status_poll_seconds=float(args.status_poll_seconds),
    )
    store.start_cleanup()
    try:
        if args.once:
            service.scan(background=False)
            return 0

        service.start()
        shutdown = stop_event or threading.Event()
        while not shutdown.wait(0.5):
            pass
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        service.close()
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[remote-dpd] %(asctime)s %(levelname)s %(message)s",
    )
    try:
        return run(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
