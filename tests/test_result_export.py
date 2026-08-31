import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from remote_dpd.controller import (
    ClosedLoopConfig,
    ClosedLoopController,
    ControllerState,
)
from remote_dpd.device import DeviceConfig, create_rf_bench
from remote_dpd.exceptions import MatProtocolError
from remote_dpd.file_interface import _parse_config_json
from remote_dpd.power_control import PowerController
from remote_dpd.protocol import load_mat
from remote_dpd.result_export import (
    FINAL_RESULT_SCHEMA_VERSION,
    ResultExportError,
    build_final_payload,
    export_final_mat,
    load_final_payload,
)


class FinalResultExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sample_count = 48
        samples = np.arange(sample_count)
        x = 0.22 * np.exp(2j * np.pi * 3 * samples / sample_count) + 0.08 * np.exp(
            2j * np.pi * 7 * samples / sample_count
        )
        config = ClosedLoopConfig(
            device_config=DeviceConfig(
                sample_rate_hz=1.0e6,
                average_segment_count=3,
                target_power_dbm=-10.0,
                safety_power_limit_dbm=0.0,
                initial_attenuation_db=30.0,
                min_attenuation_db=0.0,
                max_attenuation_db=60.0,
                settle_seconds=0.0,
                max_adjustments=100,
                call_timeout_seconds=1.0,
                device_options={
                    "max_capture_samples": sample_count * 2,
                    "noise_dbfs": -100.0,
                    "random_seed": 9,
                    "power_reference_dbm": 10.0,
                },
            ),
            runtime_config={},
            max_iterations=2,
        )
        controller = ClosedLoopController(
            create_rf_bench("simulated"),
            power_controller=PowerController(sleep_fn=lambda _: None),
        )
        controller.connect()
        controller.apply_config(config)
        controller.load_reference(x)
        cls.snapshot = controller.run_auto()

    def test_payload_uses_final_evaluated_record_and_is_detached(self):
        snapshot_x = self.snapshot.x.copy()
        final_y = self.snapshot.records[-1].y.copy()
        final_z = self.snapshot.records[-1].z.copy()

        payload = build_final_payload(self.snapshot)

        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "x",
                "y",
                "z",
                "metrics",
                "config",
                "status",
                "completed_at",
            },
        )
        self.assertEqual(payload["schema_version"], FINAL_RESULT_SCHEMA_VERSION)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["completed_at"], self.snapshot.completed_at)
        self.assertEqual(payload["x"].shape, (snapshot_x.size, 1))
        self.assertEqual(payload["y"].shape, (snapshot_x.size, 1))
        self.assertEqual(payload["z"].shape, (snapshot_x.size, 1))
        np.testing.assert_array_equal(payload["x"][:, 0], snapshot_x)
        np.testing.assert_array_equal(payload["y"][:, 0], final_y)
        np.testing.assert_array_equal(payload["z"][:, 0], final_z)
        self.assertFalse(
            np.array_equal(payload["y"][:, 0], self.snapshot.records[-2].y)
        )

        metrics = payload["metrics"]
        final_record = self.snapshot.records[-1]
        self.assertEqual(metrics["iteration"], final_record.iteration)
        self.assertEqual(metrics["nmse_db"], final_record.preprocessing.nmse_db)
        self.assertEqual(
            metrics["digital_rms"], final_record.digital_safety.candidate_rms
        )
        self.assertEqual(
            metrics["digital_peak"], final_record.digital_safety.candidate_peak
        )
        self.assertEqual(metrics["power_dbm"], final_record.power_dbm)
        self.assertEqual(metrics["attenuation_db"], final_record.attenuation_db)
        self.assertEqual(metrics["gain_correction"], self.snapshot.gain_correction)
        self.assertEqual(metrics["capture_segment_count"], 3)
        self.assertEqual(metrics["capture_batch_count"], 2)

        config = json.loads(payload["config"])
        self.assertEqual(config["device_type"], "simulated")
        self.assertEqual(config["runtime_name"], "basic_ilc")
        self.assertEqual(config["runtime_config"], {"mu": 0.5})
        self.assertEqual(config["max_iterations"], 2)
        self.assertIn("pa_coefficients", config["device_config"]["device_options"])
        self.assertIn("system_gain_db", config["device_config"]["device_options"])
        parsed = _parse_config_json(payload["config"])
        self.assertEqual(parsed.device_type, "simulated")
        parsed_config = parsed.closed_loop.to_dict()
        expected_config = self.snapshot.config.to_dict()
        expected_config["runtime_config"] = {"mu": 0.5}
        self.assertEqual(parsed_config, expected_config)

        payload["x"][0, 0] = 99.0 + 1.0j
        payload["y"][1, 0] = 98.0 + 2.0j
        payload["z"][2, 0] = 97.0 + 3.0j
        np.testing.assert_array_equal(self.snapshot.x, snapshot_x)
        np.testing.assert_array_equal(self.snapshot.records[-1].y, final_y)
        np.testing.assert_array_equal(self.snapshot.records[-1].z, final_z)

    def test_mat_round_trip_preserves_contract_and_column_vectors(self):
        completed_at = self.snapshot.completed_at
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "result.mat"
            returned = export_final_mat(
                target,
                self.snapshot,
                completed_at=completed_at,
            )

            self.assertEqual(returned, target)
            from scipy.io import loadmat

            raw = loadmat(target, squeeze_me=False, struct_as_record=False)
            self.assertEqual(raw["x"].shape, (self.snapshot.x.size, 1))
            self.assertEqual(raw["y"].shape, (self.snapshot.x.size, 1))
            self.assertEqual(raw["z"].shape, (self.snapshot.x.size, 1))

            loaded = load_mat(target)
            self.assertEqual(loaded["schema_version"], FINAL_RESULT_SCHEMA_VERSION)
            self.assertEqual(loaded["status"], "completed")
            self.assertEqual(loaded["completed_at"], completed_at)
            self.assertEqual(loaded["metrics"]["iteration"], 2)
            self.assertEqual(loaded["metrics"]["capture_segment_count"], 3)
            self.assertEqual(loaded["metrics"]["capture_batch_count"], 2)
            self.assertEqual(json.loads(loaded["config"])["runtime_config"]["mu"], 0.5)
            np.testing.assert_array_equal(loaded["y"], self.snapshot.records[-1].y)
            np.testing.assert_array_equal(loaded["z"], self.snapshot.records[-1].z)

            validated = load_final_payload(target)
            self.assertEqual(validated["completed_at"], self.snapshot.completed_at)
            self.assertFalse(validated["x"].flags.writeable)
            self.assertEqual(
                json.loads(validated["config"])["device_type"],
                "simulated",
            )

    def test_invalid_completed_snapshots_are_rejected_without_files(self):
        nan_preprocessing = replace(
            self.snapshot.records[-1].preprocessing,
            nmse_db=float("nan"),
        )
        nan_record = replace(
            self.snapshot.records[-1],
            preprocessing=nan_preprocessing,
            z=nan_preprocessing.z,
        )
        cases = {
            "not completed": replace(self.snapshot, state=ControllerState.STOPPED),
            "missing x": replace(self.snapshot, x=None),
            "missing records": replace(self.snapshot, records=(), iteration=None),
            "missing completion time": replace(self.snapshot, completed_at=None),
            "missing device type": replace(self.snapshot, device_type=""),
            "nan metric": replace(
                self.snapshot,
                records=(*self.snapshot.records[:-1], nan_record),
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (name, snapshot) in enumerate(cases.items()):
                with self.subTest(case=name):
                    target = root / f"result_{index}.mat"
                    temporary = root / f".result_{index}.tmp.mat"
                    with self.assertRaises(ResultExportError):
                        export_final_mat(target, snapshot)
                    self.assertFalse(target.exists())
                    self.assertFalse(temporary.exists())

            directory_target = root / "directory.mat"
            directory_target.mkdir()
            with self.assertRaisesRegex(ResultExportError, "directory"):
                export_final_mat(directory_target, self.snapshot)
            self.assertFalse((root / ".directory.tmp.mat").exists())

    def test_atomic_write_failure_preserves_old_result_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.mat"
            target.write_bytes(b"existing-result")
            temporary = target.with_name(".result.tmp.mat")

            def fail_after_partial_write(path, *_args, **_kwargs):
                Path(path).write_bytes(b"partial-result")
                raise OSError("injected write failure")

            with (
                patch("scipy.io.savemat", side_effect=fail_after_partial_write),
                self.assertRaises(MatProtocolError),
            ):
                export_final_mat(target, self.snapshot)

            self.assertEqual(target.read_bytes(), b"existing-result")
            self.assertFalse(temporary.exists())

    def test_completed_at_requires_an_aware_iso_timestamp(self):
        naive_datetime = datetime(2026, 8, 28)  # noqa: DTZ001 - invalid test input
        for value in (naive_datetime, "2026-08-28T01:02:03", "invalid"):
            with self.subTest(value=value), self.assertRaises(ResultExportError):
                build_final_payload(self.snapshot, completed_at=value)

        different_time = datetime(
            2026,
            8,
            28,
            9,
            10,
            11,
            123456,
            tzinfo=timezone(timedelta(hours=8)),
        )
        with self.assertRaisesRegex(ResultExportError, "terminal timestamp"):
            build_final_payload(self.snapshot, completed_at=different_time)

    def test_load_final_payload_rejects_tampered_contracts(self):
        payload = build_final_payload(self.snapshot)
        cases = {
            "missing device type": {
                **payload,
                "config": json.dumps(
                    {
                        key: value
                        for key, value in json.loads(payload["config"]).items()
                        if key != "device_type"
                    }
                ),
            },
            "length mismatch": {**payload, "z": payload["z"][:-1]},
            "non-finite waveform": {
                **payload,
                "y": np.full(payload["y"].shape, complex(float("nan"), 0.0)),
            },
            "wrong status": {**payload, "status": "failed"},
        }
        with tempfile.TemporaryDirectory() as directory:
            from scipy.io import savemat

            root = Path(directory)
            for name, invalid in cases.items():
                with self.subTest(case=name):
                    target = root / f"{name.replace(' ', '_')}.mat"
                    savemat(target, invalid)
                    with self.assertRaises(ResultExportError):
                        load_final_payload(target)


if __name__ == "__main__":
    unittest.main()
