import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import uvicorn
from scipy.io import loadmat, savemat

from remote_dpd.cli import build_parser, run


class CLITests(unittest.TestCase):
    def test_parser_uses_new_exchange_contract_without_supplier_argument(self):
        args = build_parser().parse_args(["--exchange-root", "/tmp/exchange", "--once"])

        self.assertEqual(args.exchange_root, "/tmp/exchange")
        self.assertTrue(args.once)
        self.assertEqual(args.retention_days, 7.0)
        self.assertEqual(args.mode, "file")

    def test_once_creates_service_directories_and_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_parser().parse_args(
                [
                    "--exchange-root",
                    str(root / "exchange"),
                    "--runtime-root",
                    str(root / "temporary"),
                    "--once",
                ]
            )

            result = run(args)

            self.assertEqual(result, 0)
            self.assertTrue((root / "exchange" / "inbox").is_dir())
            self.assertTrue((root / "exchange" / "outbox").is_dir())
            self.assertTrue((root / "temporary" / "runs").is_dir())

    def test_resident_service_honors_an_existing_stop_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_parser().parse_args(
                ["--exchange-root", str(root / "exchange")]
            )
            stop_event = threading.Event()
            stop_event.set()

            self.assertEqual(run(args, stop_event=stop_event), 0)

    def test_web_mode_is_fixed_to_loopback_and_honors_stop_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            args = build_parser().parse_args(
                [
                    "--exchange-root",
                    str(root / "exchange"),
                    "--mode",
                    "web",
                    "--web-port",
                    str(port),
                ]
            )
            stop_event = threading.Event()
            stop_event.set()

            with patch("uvicorn.Config", wraps=uvicorn.Config) as config:
                self.assertEqual(run(args, stop_event=stop_event), 0)

            self.assertTrue((root / "exchange" / "waveforms").is_dir())
            self.assertEqual(config.call_args.kwargs["host"], "127.0.0.1")
            self.assertEqual(config.call_args.kwargs["workers"], 1)
            self.assertFalse(config.call_args.kwargs["proxy_headers"])
            self.assertEqual(
                config.call_args.kwargs["timeout_graceful_shutdown"],
                10.0,
            )

    def test_once_is_rejected_in_web_mode(self):
        args = build_parser().parse_args(
            ["--exchange-root", "/tmp/exchange", "--mode", "web", "--once"]
        )

        with self.assertRaisesRegex(ValueError, "file mode"):
            run(args)

    def test_once_processes_a_complete_simulated_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exchange = root / "exchange"
            inbox = exchange / "inbox"
            inbox.mkdir(parents=True)
            sample_count = 32
            samples = np.arange(sample_count)
            x = 0.2 * np.exp(2j * np.pi * 3 * samples / sample_count)
            config = {
                "device_type": "simulated",
                "device_config": {
                    "sample_rate_hz": 1.0e6,
                    "average_segment_count": 2,
                    "target_power_dbm": -10.0,
                    "safety_power_limit_dbm": 0.0,
                    "initial_attenuation_db": 30.0,
                    "min_attenuation_db": 0.0,
                    "max_attenuation_db": 60.0,
                    "settle_seconds": 0.0,
                    "max_adjustments": 100,
                    "call_timeout_seconds": 1.0,
                    "device_options": {
                        "max_capture_samples": sample_count * 2,
                        "noise_dbfs": -100.0,
                    },
                },
                "runtime_name": "basic_ilc",
                "runtime_config": {"mu": 0.5},
                "max_iterations": 1,
            }
            savemat(
                inbox / "command_cli-run.mat",
                {
                    "schema_version": 1,
                    "command_id": "cli-run",
                    "action": "run",
                    "x": x,
                    "config_json": json.dumps(config),
                },
            )
            args = build_parser().parse_args(
                ["--exchange-root", str(exchange), "--once"]
            )

            self.assertEqual(run(args), 0)

            status = loadmat(
                exchange / "outbox" / "status_cli-run.mat",
                squeeze_me=True,
            )
            self.assertEqual(str(status["state"]), "completed")
            self.assertEqual(int(status["iteration"]), 1)
            self.assertTrue((exchange / "outbox" / "result_cli-run.mat").is_file())
            self.assertTrue(
                (exchange / "runtime" / "runs" / "cli-run" / "manifest.json").is_file()
            )

    def test_negative_retention_is_rejected_before_service_start(self):
        args = build_parser().parse_args(
            ["--exchange-root", "/tmp/exchange", "--retention-days", "-1"]
        )

        with self.assertRaisesRegex(ValueError, "retention_days"):
            run(args)


if __name__ == "__main__":
    unittest.main()
