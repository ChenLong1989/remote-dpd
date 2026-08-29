"""Command-line entry point for the versioned MAT command service."""

from __future__ import annotations

import argparse
import ipaddress
import logging
import threading
from pathlib import Path
from typing import Any

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
        "--mode",
        choices=("file", "web"),
        default="file",
        help="run the MAT inbox service or the trusted-network Web console",
    )
    parser.add_argument(
        "--waveform-root",
        help="Web waveform library root (default: <exchange-root>/waveforms)",
    )
    parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help=("IPv4 bind address (default: 127.0.0.1; use 0.0.0.0 for trusted LAN)"),
    )
    parser.add_argument(
        "--web-allowed-host",
        action="append",
        default=[],
        help=("trusted private IPv4 Host header; repeat for multiple LAN addresses"),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8000,
        help="Web console port",
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
    if args.mode == "web" and args.once:
        raise ValueError("--once is only valid in file mode")
    if isinstance(args.web_port, bool) or not 1 <= int(args.web_port) <= 65535:
        raise ValueError("web_port must be between 1 and 65535")
    web_host, web_allowed_hosts = _validate_web_network(
        args.web_host,
        args.web_allowed_host,
    )

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

        if args.mode == "web":
            import uvicorn

            from .web import create_web_app

            waveform_root = (
                Path(args.waveform_root).expanduser()
                if args.waveform_root
                else exchange_root / "waveforms"
            )
            service.start()
            app = create_web_app(
                command_service=service,
                run_store=store,
                waveform_root=waveform_root,
                allowed_hosts=web_allowed_hosts,
            )
            config = uvicorn.Config(
                app,
                host=web_host,
                port=int(args.web_port),
                log_level=args.log_level.lower(),
                workers=1,
                proxy_headers=False,
                timeout_graceful_shutdown=10.0,
            )
            server = uvicorn.Server(config)
            if stop_event is not None:
                watcher = threading.Thread(
                    target=_stop_web_server,
                    args=(stop_event, server),
                    name="remote-dpd-web-stop",
                    daemon=True,
                )
                watcher.start()
            server.run()
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


def _stop_web_server(stop_event: threading.Event, server: Any) -> None:
    stop_event.wait()
    server.should_exit = True


def _validate_web_network(
    host: object,
    allowed_hosts: object,
) -> tuple[str, tuple[str, ...]]:
    bind_address = _parse_bind_address(host)
    if isinstance(allowed_hosts, (str, bytes)) or not isinstance(
        allowed_hosts,
        (list, tuple),
    ):
        raise TypeError("web_allowed_host must be a sequence of IPv4 addresses")
    normalized_allowed: list[str] = []
    allowed_addresses: list[ipaddress.IPv4Address] = []
    for value in allowed_hosts:
        address = _parse_allowed_address(value)
        normalized = str(address)
        if normalized not in normalized_allowed:
            normalized_allowed.append(normalized)
            allowed_addresses.append(address)
    if not bind_address.is_loopback and not any(
        not address.is_loopback for address in allowed_addresses
    ):
        raise ValueError(
            "non-loopback web_host requires at least one non-loopback web_allowed_host"
        )
    if (
        not bind_address.is_loopback
        and not bind_address.is_unspecified
        and str(bind_address) not in normalized_allowed
    ):
        raise ValueError("a specific LAN web_host must also be a web_allowed_host")
    return str(bind_address), tuple(normalized_allowed)


def _parse_bind_address(value: object) -> ipaddress.IPv4Address:
    address = _parse_ipv4(value, "web_host")
    if address.is_loopback and address != ipaddress.IPv4Address("127.0.0.1"):
        raise ValueError("loopback web_host must be 127.0.0.1")
    if not (address.is_loopback or address.is_private or address.is_unspecified):
        raise ValueError("web_host must be loopback, private, or 0.0.0.0")
    if address.is_multicast or address.is_reserved:
        raise ValueError("web_host must be a unicast IPv4 address or 0.0.0.0")
    return address


def _parse_allowed_address(value: object) -> ipaddress.IPv4Address:
    address = _parse_ipv4(value, "web_allowed_host")
    if (
        address.is_unspecified
        or address.is_multicast
        or address.is_reserved
        or not (address.is_loopback or address.is_private)
    ):
        raise ValueError(
            "web_allowed_host must be a private or loopback unicast IPv4 address"
        )
    return address


def _parse_ipv4(value: object, name: str) -> ipaddress.IPv4Address:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be an IPv4 address string")
    try:
        return ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as exc:
        raise ValueError(f"{name} must be an IPv4 address") from exc


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
