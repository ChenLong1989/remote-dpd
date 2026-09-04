import json
import os
import tempfile
import time
import unittest

from platform_guards import FD_ANCHORED_SEMANTICS, SYMLINKS_SUPPORTED
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from remote_dpd.controller import (
    ClosedLoopConfig,
    ControllerErrorInfo,
    ControllerSnapshot,
    ControllerState,
    IterationRecord,
    ReferenceNormalizationReport,
    _generate_seed_waveform,
)
from remote_dpd.device import DeviceConfig
from remote_dpd.power_control import PowerAdjustment
from remote_dpd.preprocessing import CaptureBatch, FeedbackPreprocessor
from remote_dpd.protocol import load_mat
from remote_dpd.safety import validate_candidate, validate_reference
from remote_dpd.storage import (
    RUN_SCHEMA_VERSION,
    RunConflictError,
    RunNotFoundError,
    RunRecorder,
    RunStorageError,
    RunStore,
)


class RunStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.x = np.asarray(
            [
                0.20 + 0.05j,
                -0.10 + 0.15j,
                -0.18 - 0.04j,
                0.08 - 0.16j,
            ],
            dtype=np.complex128,
        )
        self.config = ClosedLoopConfig(
            device_config=DeviceConfig(
                sample_rate_hz=245.76e6,
                average_segment_count=2,
                target_power_dbm=-10.0,
                safety_power_limit_dbm=0.0,
                settle_seconds=0.0,
                call_timeout_seconds=1.0,
                device_options={"random_seed": 7, "labels": ["a", "b"]},
            ),
            normalize_reference_rms=False,
            runtime_config={"mu": 0.5},
            max_iterations=2,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_complete_snapshot_saves_all_artifacts_as_strict_json_and_npy(self):
        store = RunStore(self.root)
        original_x = self.x.copy()
        recorder = store.create_run(self.config, self.x, run_id="complete-run")
        self.x[:] = 0.0

        snapshot = self._snapshot(ControllerState.COMPLETED)
        self.assertTrue(recorder.record_snapshot(snapshot))
        self.assertFalse(recorder.owns_active_guard)

        manifest = recorder.read_manifest()
        self.assertEqual(manifest["schema_version"], RUN_SCHEMA_VERSION)
        self.assertEqual(manifest["run_id"], "complete-run")
        self.assertEqual(manifest["status"], "completed")
        self.assertTrue(manifest["created"].endswith("Z"))
        self.assertTrue(manifest["updated"].endswith("Z"))
        self.assertTrue(manifest["completed"].endswith("Z"))
        self.assertEqual(
            [item["iteration"] for item in manifest["iterations"]],
            [0, 1, 2],
        )
        self.assertEqual(manifest["device_type"], "simulated")
        self.assertEqual(manifest["final_result"], "final_result.mat")
        self.assertEqual(manifest["completed"], snapshot.completed_at)
        final_result_path = recorder.final_result_path
        self.assertIsNotNone(final_result_path)
        final_result = load_mat(final_result_path)
        self.assertEqual(final_result["completed_at"], snapshot.completed_at)
        self.assertEqual(json.loads(final_result["config"])["device_type"], "simulated")

        np.testing.assert_array_equal(recorder.read_reference(), original_x)
        stored = recorder.read_iteration(0)
        np.testing.assert_array_equal(stored["y"], snapshot.records[0].y)
        np.testing.assert_array_equal(stored["z"], snapshot.records[0].z)
        np.testing.assert_array_equal(
            stored["aligned_average"],
            snapshot.records[0].preprocessing.aligned_average,
        )
        self.assertFalse(stored["y"].flags.writeable)
        metadata = stored["metadata"]
        self.assertEqual(metadata["digital_safety"]["passed"], True)
        self.assertIn("batch_diagnostics", metadata["preprocessing"])
        self.assertEqual(metadata["runtime_metrics"]["array"], [1, 2])
        self.assertEqual(
            metadata["runtime_metrics"]["complex"],
            {"$type": "complex", "real": 1.0, "imag": 2.0},
        )
        self.assertEqual(len(recorder.read_power_trace()), 2)
        self.assertEqual(recorder.read_latest_snapshot()["gain_correction"], 1.25)
        self.assertEqual(recorder.read_events()[-1]["kind"], "state")

        for json_path in recorder.path.rglob("*.json"):
            value = json.loads(json_path.read_text(encoding="utf-8"))
            json.dumps(value, allow_nan=False)
            self.assertNotIn("raw_capture", json_path.read_text(encoding="utf-8"))
        self.assertFalse(
            any(path.name.endswith(".tmp") for path in recorder.path.rglob("*"))
        )

        listed = store.list_runs()
        self.assertEqual([item["run_id"] for item in listed], ["complete-run"])
        run_data = store.read_run("complete-run")
        self.assertEqual(run_data["config"], self.config.to_dict())
        self.assertEqual(run_data["manifest"], manifest)

    def test_repeated_snapshot_is_idempotent_and_conflicting_round_is_rejected(self):
        store = RunStore(self.root)
        recorder = store.create_run(self.config, self.x, run_id="idempotent")
        snapshot = self._snapshot(ControllerState.COMPLETED)
        recorder.record_snapshot(snapshot)
        manifest_path = recorder.path / "manifest.json"
        record_path = recorder.path / "iterations" / "000000" / "record.json"
        manifest_time = manifest_path.stat().st_mtime_ns
        record_time = record_path.stat().st_mtime_ns
        time.sleep(0.01)

        self.assertFalse(recorder.record_snapshot(snapshot))
        self.assertEqual(manifest_path.stat().st_mtime_ns, manifest_time)
        self.assertEqual(record_path.stat().st_mtime_ns, record_time)
        self.assertEqual(len(recorder.read_events()), 1)

        conflicting = self._snapshot(
            ControllerState.COMPLETED,
            feedback_scale=0.65,
        )
        manifest_before = recorder.read_manifest()
        with self.assertRaises(RunConflictError):
            recorder.record_snapshot(conflicting)
        self.assertEqual(recorder.read_manifest(), manifest_before)

    def test_failed_snapshot_records_structured_error_and_completes_run(self):
        store = RunStore(self.root)
        recorder = store.create_run(self.config, self.x, run_id="failed-run")
        error = ControllerErrorInfo(
            operation="step",
            code="safety_limit_exceeded",
            exception_type="PowerControlError",
            message="unsafe power",
            shutdown_error="shutdown timed out",
        )

        recorder.record_snapshot(self._snapshot(ControllerState.FAILED, error=error))

        manifest = recorder.read_manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertIsNotNone(manifest["completed"])
        error_event = next(
            item for item in recorder.read_events() if item["kind"] == "error"
        )
        self.assertEqual(error_event["details"]["code"], "safety_limit_exceeded")
        self.assertEqual(error_event["details"]["shutdown_error"], "shutdown timed out")

    def test_atomic_write_failure_leaves_manifest_valid_and_no_temp_file(self):
        store = RunStore(self.root)
        recorder = store.create_run(self.config, self.x, run_id="atomic-run")
        manifest_before = recorder.read_manifest()
        real_replace = os.replace

        def fail_manifest_replace(source, destination):
            if Path(destination).name == "manifest.json":
                raise OSError("injected manifest replace failure")
            return real_replace(source, destination)

        with (
            patch(
                "remote_dpd.storage.os.replace",
                side_effect=fail_manifest_replace,
            ),
            self.assertRaises(OSError),
        ):
            recorder.record_error(RuntimeError("device failed"), operation="capture")

        self.assertEqual(recorder.read_manifest(), manifest_before)
        json.dumps(recorder.read_manifest(), allow_nan=False)
        self.assertFalse(
            any(path.name.endswith(".tmp") for path in recorder.path.rglob("*"))
        )
        self.assertTrue(
            recorder.record_error(RuntimeError("device failed"), operation="capture")
        )
        self.assertFalse(
            recorder.record_error(RuntimeError("device failed"), operation="capture")
        )

    def test_active_and_export_guards_protect_expired_run(self):
        store = RunStore(self.root, retention_seconds=1.0)
        recorder = store.create_run(self.config, self.x, run_id="guarded")
        created = _parse_timestamp(recorder.read_manifest()["created"])
        future = created + timedelta(seconds=10)

        self.assertEqual(store.cleanup_expired(now=future), ())
        recorder.close()
        with recorder.export_guard():
            self.assertEqual(store.cleanup_expired(now=future), ())
            self.assertTrue(recorder.path.exists())
        self.assertEqual(store.cleanup_expired(now=future), ("guarded",))
        with self.assertRaises(RunNotFoundError):
            store.open_run("guarded")

    def test_guards_and_read_run_are_shared_by_stores_for_the_same_root(self):
        first = RunStore(self.root, retention_seconds=1.0)
        second = RunStore(self.root, retention_seconds=1.0)
        recorder = first.create_run(self.config, self.x, run_id="shared-guard")
        recorder.close()
        future = _parse_timestamp(recorder.read_manifest()["created"]) + timedelta(
            seconds=10
        )

        with first.active_run("shared-guard"):
            self.assertEqual(second.cleanup_expired(now=future), ())
        with second.export_guard("shared-guard"):
            self.assertEqual(first.cleanup_expired(now=future), ())

        real_read_config = RunRecorder.read_config

        def read_config_while_cleanup_is_requested(current):
            self.assertEqual(second.cleanup_expired(now=future), ())
            return real_read_config(current)

        with patch.object(
            RunRecorder,
            "read_config",
            new=read_config_while_cleanup_is_requested,
        ):
            run_data = first.read_run("shared-guard")
        self.assertEqual(run_data["manifest"]["run_id"], "shared-guard")
        self.assertEqual(second.cleanup_expired(now=future), ("shared-guard",))

    def test_expiration_respects_retention_boundary(self):
        store = RunStore(self.root, retention_seconds=100.0)
        recorder = store.create_run(self.config, self.x, run_id="retention")
        recorder.close()
        updated = _parse_timestamp(recorder.read_manifest()["updated"])

        self.assertEqual(
            store.cleanup_expired(now=updated + timedelta(seconds=99.999)), ()
        )
        self.assertEqual(
            store.cleanup_expired(now=updated + timedelta(seconds=100)),
            ("retention",),
        )

    @unittest.skipUnless(SYMLINKS_SUPPORTED, "creating symlinks requires privileges on this host")
    def test_symlinks_uncontrolled_directories_and_unsafe_ids_are_rejected(self):
        store = RunStore(self.root, retention_seconds=0.0)
        external = self.root / "external"
        external.mkdir()
        marker = external / "must-remain.txt"
        marker.write_text("safe", encoding="utf-8")
        (store.runs_root / "linked").symlink_to(external, target_is_directory=True)
        uncontrolled = store.runs_root / "uncontrolled"
        uncontrolled.mkdir()
        (uncontrolled / "keep.txt").write_text("safe", encoding="utf-8")

        self.assertEqual(store.cleanup_expired(now=time.time() + 1000), ())
        self.assertTrue(marker.exists())
        self.assertTrue((uncontrolled / "keep.txt").exists())
        self.assertEqual(store.list_runs(), ())
        with self.assertRaises(RunNotFoundError):
            store.open_run("linked")

        for unsafe in ("../escape", "/absolute", ".hidden", "a/b", "含中文"):
            with (
                self.subTest(run_id=unsafe),
                self.assertRaises((TypeError, ValueError)),
            ):
                store.create_run(self.config, self.x, run_id=unsafe)
        self.assertFalse((self.root / "escape").exists())

    @unittest.skipUnless(SYMLINKS_SUPPORTED, "creating symlinks requires privileges on this host")
    def test_replaced_runs_root_is_rejected_without_touching_target(self):
        store = RunStore(self.root)
        original_runs = store.runs_root
        moved_runs = self.root / "original-runs"
        original_runs.rename(moved_runs)
        external = self.root / "replacement-target"
        external.mkdir()
        marker = external / "must-remain.txt"
        marker.write_text("safe", encoding="utf-8")
        original_runs.symlink_to(external, target_is_directory=True)

        with self.assertRaises(RunStorageError):
            store.cleanup_expired(now=time.time() + 1000)
        self.assertTrue(marker.exists())

    @unittest.skipUnless(SYMLINKS_SUPPORTED, "creating symlinks requires privileges on this host")
    def test_iteration_directory_symlink_cannot_escape_run(self):
        store = RunStore(self.root)
        recorder = store.create_run(self.config, self.x, run_id="artifact-symlink")
        external = self.root / "artifact-target"
        external.mkdir()
        (recorder.path / "iterations").symlink_to(external, target_is_directory=True)
        manifest_before = recorder.read_manifest()

        with self.assertRaises(RunStorageError):
            recorder.record_snapshot(self._snapshot(ControllerState.COMPLETED))

        self.assertEqual(tuple(external.iterdir()), ())
        self.assertEqual(recorder.read_manifest(), manifest_before)

    def test_cleanup_thread_is_daemon_periodic_and_repeated_start_stop_is_safe(self):
        store = RunStore(
            self.root,
            retention_seconds=0.0,
            cleanup_interval_seconds=0.02,
        )
        recorder = store.create_run(self.config, self.x, run_id="periodic")
        run_path = recorder.path
        recorder.close()

        self.assertTrue(store.start_cleanup())
        self.assertFalse(store.start_cleanup())
        deadline = time.monotonic() + 2.0
        while run_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        with self.assertRaises(RunNotFoundError):
            store.open_run("periodic")
        self.assertTrue(store.stop_cleanup(timeout=1.0))
        self.assertFalse(store.stop_cleanup(timeout=1.0))

    def test_mark_terminal_does_not_require_a_controller_transition(self):
        store = RunStore(self.root)
        recorder = store.create_run(self.config, self.x, run_id="superseded")

        self.assertTrue(
            recorder.mark_terminal(
                ControllerState.STOPPED,
                message="configuration superseded",
                error_code="superseded",
            )
        )
        self.assertFalse(recorder.owns_active_guard)
        manifest = recorder.read_manifest()
        self.assertEqual(manifest["status"], "stopped")
        self.assertIsNotNone(manifest["completed"])
        terminal_event = recorder.read_events()[-1]
        self.assertEqual(terminal_event["kind"], "terminal")
        self.assertEqual(terminal_event["details"]["error_code"], "superseded")

        self.assertFalse(
            recorder.mark_terminal(
                "stopped",
                message="configuration superseded",
                error_code="superseded",
            )
        )
        with self.assertRaises(RunConflictError):
            recorder.mark_terminal("failed")

    def test_precommit_final_cache_can_recover_a_completed_manifest(self):
        store = RunStore(self.root)
        recorder = store.create_run(self.config, self.x, run_id="recoverable")
        snapshot = self._snapshot(ControllerState.COMPLETED)

        from remote_dpd.result_export import export_final_mat

        def crash_after_cache(path, current_snapshot):
            export_final_mat(path, current_snapshot)
            raise RuntimeError("simulated crash after final cache")

        with (
            patch(
                "remote_dpd.result_export.export_final_mat",
                side_effect=crash_after_cache,
            ),
            self.assertRaisesRegex(RuntimeError, "simulated crash"),
        ):
            recorder.record_snapshot(snapshot)

        precommit = recorder.read_manifest()
        self.assertEqual(precommit["status"], "finalizing")
        self.assertEqual(
            [entry["iteration"] for entry in precommit["iterations"]],
            [0, 1, 2],
        )
        self.assertEqual(recorder.read_iteration(2)["metadata"]["iteration"], 2)
        self.assertIsNotNone(recorder.final_result_path)

        recorder.close()
        recovered = RunStore(self.root).open_run("recoverable")
        self.assertTrue(recovered.mark_terminal(ControllerState.COMPLETED))
        manifest = recovered.read_manifest()
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["final_result"], "final_result.mat")
        self.assertEqual(manifest["completed"], snapshot.completed_at)
        self.assertEqual(recovered.read_iteration(2)["metadata"]["iteration"], 2)

    def test_completed_recovery_rejects_cache_from_different_config(self):
        store = RunStore(self.root)
        recorder = store.create_run(self.config, self.x, run_id="wrong-cache")
        snapshot = self._snapshot(ControllerState.COMPLETED)
        from remote_dpd.result_export import export_final_mat, load_final_payload

        def crash_after_cache(path, current_snapshot):
            export_final_mat(path, current_snapshot)
            raise RuntimeError("simulated crash after final cache")

        with (
            patch(
                "remote_dpd.result_export.export_final_mat",
                side_effect=crash_after_cache,
            ),
            self.assertRaises(RuntimeError),
        ):
            recorder.record_snapshot(snapshot)

        cache = recorder.final_result_path
        payload = load_final_payload(cache)
        config = json.loads(payload["config"])
        config["device_config"]["target_power_dbm"] = -9.0
        payload["config"] = json.dumps(config, separators=(",", ":"))
        from scipy.io import savemat

        savemat(cache, payload)
        with self.assertRaisesRegex(RunConflictError, "config conflicts"):
            recorder.mark_terminal(ControllerState.COMPLETED)

    def test_schema_one_precommit_cache_remains_recoverable(self):
        store = RunStore(self.root)
        recorder = store.create_run(self.config, self.x, run_id="legacy-cache")
        snapshot = self._snapshot(ControllerState.COMPLETED)
        from remote_dpd.result_export import export_final_mat, load_final_payload

        def crash_after_cache(path, current_snapshot):
            export_final_mat(path, current_snapshot)
            raise RuntimeError("simulated legacy precommit crash")

        with (
            patch(
                "remote_dpd.result_export.export_final_mat",
                side_effect=crash_after_cache,
            ),
            self.assertRaises(RuntimeError),
        ):
            recorder.record_snapshot(snapshot)

        cache = recorder.final_result_path
        payload = load_final_payload(cache)
        config = json.loads(payload["config"])
        config.pop("normalize_reference_rms")
        config.pop("reference_target_rms_dbfs")
        config.pop("seed_noise_enabled")
        config.pop("seed_noise_psd_db")
        config.pop("seed_noise_bandwidth_hz")
        config.pop("seed_noise_seed")
        metrics = dict(payload["metrics"])
        metrics.pop("source_rms_dbfs")
        metrics.pop("reference_rms_dbfs")
        metrics.pop("reference_scale_db")
        legacy_payload = {
            **payload,
            "schema_version": 1,
            "config": json.dumps(config, separators=(",", ":"), sort_keys=True),
            "metrics": metrics,
        }
        from scipy.io import savemat

        savemat(cache, legacy_payload)
        stored_config = self.config.to_dict()
        stored_config.pop("normalize_reference_rms")
        stored_config.pop("reference_target_rms_dbfs")
        stored_config.pop("seed_noise_enabled")
        stored_config.pop("seed_noise_psd_db")
        stored_config.pop("seed_noise_bandwidth_hz")
        stored_config.pop("seed_noise_seed")
        (recorder.path / "config.json").write_text(
            json.dumps(stored_config, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )

        recorder.close()
        recovered = RunStore(self.root).open_run("legacy-cache")
        self.assertTrue(recovered.mark_terminal(ControllerState.COMPLETED))
        self.assertEqual(recovered.read_manifest()["status"], "completed")

    def _snapshot(
        self,
        state,
        *,
        feedback_scale=0.8,
        error=None,
    ):
        x = np.asarray(
            [
                0.20 + 0.05j,
                -0.10 + 0.15j,
                -0.18 - 0.04j,
                0.08 - 0.16j,
            ],
            dtype=np.complex128,
        )
        phase = np.exp(0.2j)
        batch = CaptureBatch(
            iq=np.tile(x * feedback_scale * phase, 2),
            segment_length=x.size,
            segment_count=2,
            sample_rate_hz=self.config.device_config.sample_rate_hz,
        )
        preprocessing = FeedbackPreprocessor(
            x, self.config.device_config.sample_rate_hz
        ).process((batch,))
        safety = validate_reference(x)
        seed = _generate_seed_waveform(x, self.config)
        reference_rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        reference_rms_dbfs = float(20.0 * np.log10(reference_rms))
        normalization = ReferenceNormalizationReport(
            enabled=False,
            source_rms=reference_rms,
            source_rms_dbfs=reference_rms_dbfs,
            target_rms_dbfs=-15.0,
            scale=1.0,
            scale_db=0.0,
            effective_rms=reference_rms,
            effective_rms_dbfs=reference_rms_dbfs,
        )
        record = IterationRecord(
            iteration=0,
            y=seed,
            z=preprocessing.z,
            power_dbm=-10.1,
            attenuation_db=25.0,
            digital_safety=safety,
            preprocessing=preprocessing,
            runtime_metrics={
                "array": np.asarray([1, 2]),
                "complex": 1.0 + 2.0j,
            },
        )
        records = (record,)
        if state is ControllerState.COMPLETED:
            candidate_safety = validate_candidate(x, seed)
            records = (
                record,
                replace(record, iteration=1, digital_safety=candidate_safety),
                replace(record, iteration=2, digital_safety=candidate_safety),
            )
        terminal = state in {
            ControllerState.COMPLETED,
            ControllerState.FAILED,
            ControllerState.STOPPED,
        }
        return ControllerSnapshot(
            state=state,
            connected=True,
            configured=True,
            reference_loaded=True,
            transmitting=False,
            stop_requested=False,
            active_operation=None,
            iteration=records[-1].iteration,
            max_iterations=self.config.max_iterations,
            gain_correction=1.25,
            locked_attenuation_db=25.0,
            latest_power_dbm=-10.1,
            config=self.config,
            device_type="simulated",
            completed_at="2026-08-28T01:02:03Z" if terminal else None,
            reference_safety=safety,
            reference_normalization=normalization,
            x=x,
            records=records,
            power_trace=(
                PowerAdjustment(30.0, -15.0, 5.0),
                PowerAdjustment(25.0, -10.1, 0.1),
            ),
            last_error=error,
        )


def _parse_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


if __name__ == "__main__":
    unittest.main()
