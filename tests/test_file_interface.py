import json
import os
import tempfile
import threading
import time
import unittest

from platform_guards import FD_ANCHORED_SEMANTICS, SYMLINKS_SUPPORTED
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat, savemat

from remote_dpd.controller import (
    ClosedLoopController,
    ControllerErrorInfo,
    ControllerSnapshot,
    ControllerState,
)
from remote_dpd.file_interface import (
    CommandStatus,
    FileCommandError,
    FileCommandProcessor,
    FileCommandService,
    _parse_config_json,
)
from remote_dpd.simulation import SimulatedRFBench
from remote_dpd.storage import RunStore


def _reference(sample_count: int = 64) -> np.ndarray:
    samples = np.arange(sample_count)
    return (
        0.24 * np.exp(2j * np.pi * 3 * samples / sample_count)
        + 0.08 * np.exp(2j * np.pi * 9 * samples / sample_count)
    ).astype(np.complex128)


def _configuration(
    sample_count: int = 64,
    *,
    max_iterations: int = 2,
) -> dict[str, object]:
    return {
        "device_type": "simulated",
        "normalize_reference_rms": False,
        "reference_target_rms_dbfs": -15.0,
        "device_config": {
            "center_frequency_hz": 3.5e9,
            "sample_rate_hz": 245.76e6,
            "tx_channel": "0",
            "rx_channel": "0",
            "trigger": "immediate",
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
                "noise_dbfs": -100.0,
                "random_seed": 7,
                "power_reference_dbm": 10.0,
                "max_capture_samples": sample_count * 2,
            },
        },
        "runtime_name": "basic_ilc",
        "runtime_config": {"mu": 0.5},
        "max_iterations": max_iterations,
    }


def _write_command(
    service: FileCommandService,
    command_id: str,
    action: str,
    *,
    x: np.ndarray | None = None,
    config: dict[str, object] | str | None = None,
    payload_updates: dict[str, object] | None = None,
    atomic: bool = False,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "command_id": command_id,
        "action": action,
    }
    if x is not None:
        payload["x"] = x
    if config is not None:
        payload["config_json"] = (
            config if isinstance(config, str) else json.dumps(config)
        )
    if payload_updates:
        payload.update(payload_updates)
    path = service.inbox / f"command_{command_id}.mat"
    if not atomic:
        savemat(path, payload)
        return path
    temporary = service.inbox / f".incoming-{command_id}.mat"
    savemat(temporary, payload)
    os.replace(temporary, path)
    return path


class _GateSimulatedBench(SimulatedRFBench):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self._entered = entered
        self._release = release

    def capture(self, request, timeout_seconds):
        self._entered.set()
        if not self._release.wait(5.0):
            raise TimeoutError("test capture gate timed out")
        return super().capture(request, timeout_seconds)


class _GateDisconnectBench(SimulatedRFBench):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self._entered = entered
        self._release = release

    def disconnect(self, timeout_seconds):
        self._entered.set()
        if not self._release.wait(5.0):
            raise TimeoutError("test disconnect gate timed out")
        return super().disconnect(timeout_seconds)


class _DispatchRaceProcessor(FileCommandProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.observe = threading.Event()
        self.observed_active_ids = []
        self.service = None

    def request_stop(self):
        self.entered.set()
        if not self.observe.wait(5.0):
            raise TimeoutError("test stop dispatch gate timed out")
        self.observed_active_ids.append(self.service.active_command_id)


class FileCommandServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.service = FileCommandService(
            self.root / "exchange",
            status_poll_seconds=0.002,
        )
        self.x = _reference()

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def process(
        self,
        command_id: str,
        action: str,
        *,
        x: np.ndarray | None = None,
        config: dict[str, object] | str | None = None,
    ):
        path = _write_command(
            self.service,
            command_id,
            action,
            x=x,
            config=config,
        )
        return self.service.process_file(path)

    def test_run_command_completes_simulation_and_writes_final_result(self):
        status = self.process(
            "run-e2e",
            "run",
            x=self.x,
            config=_configuration(),
        )

        self.assertTrue(status.accepted)
        self.assertEqual(status.state, "completed")
        self.assertEqual(status.iteration, 2)
        self.assertEqual(status.error_code, "")
        result_path = self.service.result_path("run-e2e")
        self.assertTrue(result_path.is_file())
        payload = loadmat(result_path, squeeze_me=True, struct_as_record=False)
        self.assertEqual(int(payload["schema_version"]), 2)
        np.testing.assert_allclose(np.asarray(payload["x"]).reshape(-1), self.x)
        self.assertEqual(np.asarray(payload["y"]).size, self.x.size)
        self.assertEqual(np.asarray(payload["z"]).size, self.x.size)
        self.assertEqual(str(payload["status"]), "completed")
        self.assertIn("metrics", payload)
        self.assertIn("config", payload)

    def test_run_replaces_configuration_and_reference_as_one_input_pair(self):
        old_reference = _reference(128)
        new_reference = _reference(32)
        self.process("paired-old-load", "load", x=old_reference)
        self.process(
            "paired-old-configure",
            "configure",
            config=_configuration(sample_count=128, max_iterations=1),
        )

        status = self.process(
            "paired-new-run",
            "run",
            x=new_reference,
            config=_configuration(sample_count=32, max_iterations=1),
        )

        self.assertEqual(status.state, "completed")
        result = loadmat(
            self.service.result_path("paired-new-run"),
            squeeze_me=True,
        )
        np.testing.assert_allclose(
            np.asarray(result["x"]).reshape(-1),
            new_reference,
        )

    def test_run_normalizes_source_before_storage_runtime_and_export(self):
        source = self.x * 4.0
        config = _configuration(max_iterations=1)
        config["normalize_reference_rms"] = True
        config["reference_target_rms_dbfs"] = -15.0

        status = self.process(
            "normalized-run",
            "run",
            x=source,
            config=config,
        )
        payload = loadmat(
            self.service.result_path("normalized-run"),
            squeeze_me=True,
            struct_as_record=False,
        )
        effective = np.asarray(payload["x"]).reshape(-1)

        self.assertEqual(status.state, "completed")
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.abs(effective) ** 2))),
            10.0 ** (-15.0 / 20.0),
        )
        self.assertLessEqual(float(np.max(np.abs(effective))), 1.0)
        np.testing.assert_array_equal(source, self.x * 4.0)
        exported_config = json.loads(str(payload["config"]))
        self.assertTrue(exported_config["normalize_reference_rms"])
        self.assertEqual(exported_config["reference_target_rms_dbfs"], -15.0)

    def test_all_stepwise_commands_use_the_same_controller(self):
        loaded = self.process("manual-load", "load", x=self.x)
        configured = self.process(
            "manual-configure", "configure", config=_configuration()
        )
        power = self.process("manual-power", "power_tune")
        calibration = self.process("manual-calibration", "calibrate")
        first = self.process("manual-step-1", "step")
        second = self.process("manual-step-2", "step")
        exported = self.process("manual-export", "export")

        self.assertEqual(loaded.state, "loaded")
        self.assertEqual(configured.state, "ready")
        self.assertEqual(power.state, "power_ready")
        self.assertEqual(calibration.state, "calibrated")
        self.assertEqual(calibration.iteration, 0)
        self.assertEqual(first.state, "calibrated")
        self.assertEqual(first.iteration, 1)
        self.assertEqual(second.state, "completed")
        self.assertEqual(second.iteration, 2)
        self.assertEqual(exported.state, "completed")
        self.assertTrue(self.service.result_path("manual-export").is_file())

        reset = self.process("manual-reset", "reset")
        self.assertEqual(reset.state, "idle")
        snapshot = self.service.processor.snapshot()
        self.assertIsNotNone(snapshot)
        self.assertFalse(snapshot.configured)
        self.assertFalse(snapshot.reference_loaded)

    def test_manual_connection_and_transmission_commands_are_exposed(self):
        self.process("lifecycle-load", "load", x=self.x)
        self.process("lifecycle-configure", "configure", config=_configuration())

        started = self.process("lifecycle-start", "start_transmission")
        self.assertEqual(started.state, "ready")
        self.assertTrue(self.service.processor.snapshot().transmitting)

        stopped = self.process("lifecycle-stop-tx", "stop_transmission")
        self.assertEqual(stopped.state, "ready")
        self.assertFalse(self.service.processor.snapshot().transmitting)

        disconnected = self.process("lifecycle-disconnect", "disconnect")
        self.assertEqual(disconnected.state, "idle")
        self.assertFalse(self.service.processor.snapshot().connected)

        connected = self.process("lifecycle-connect", "connect")
        self.assertEqual(connected.state, "idle")
        self.assertTrue(self.service.processor.snapshot().connected)
        self.assertFalse(self.service.processor.snapshot().configured)

    def test_stop_transmission_retry_and_disconnect_preserve_run_consistency(self):
        self.service.close()
        store = RunStore(self.root / "lifecycle-runs")
        self.service = FileCommandService(
            self.root / "lifecycle-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        self.process("stored-lifecycle-load", "load", x=self.x)
        configured = self.process(
            "stored-lifecycle-configure",
            "configure",
            config=_configuration(),
        )
        self.process("stored-lifecycle-power", "power_tune")
        self.process("stored-lifecycle-calibration", "calibrate")
        stop_path = _write_command(
            self.service,
            "stored-lifecycle-stop-tx",
            "stop_transmission",
        )

        first_stop = self.service.process_file(stop_path)
        second_stop = self.service.process_file(stop_path)
        disconnected = self.process("stored-lifecycle-disconnect", "disconnect")

        self.assertEqual(first_stop.state, "calibrated")
        self.assertEqual(second_stop, first_stop)
        self.assertEqual(disconnected.state, "idle")
        self.assertFalse(self.service.processor.snapshot().connected)
        self.assertIsNone(self.service.processor.run_id)
        manifest = store.open_run(configured.run_id).read_manifest()
        self.assertEqual(manifest["status"], "stopped")
        self.assertEqual(
            [entry["iteration"] for entry in manifest["iterations"]],
            [0],
        )
        store.close()

    def test_run_store_records_the_complete_automatic_run(self):
        self.service.close()
        store = RunStore(self.root / "temporary-runs")
        self.service = FileCommandService(
            self.root / "stored-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )

        status = self.process(
            "stored-run",
            "run",
            x=self.x,
            config=_configuration(),
        )

        self.assertEqual(status.state, "completed")
        manifest = store.open_run("stored-run").read_manifest()
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(
            [entry["iteration"] for entry in manifest["iterations"]],
            [0, 1, 2],
        )

    def test_completed_run_can_be_detached_then_reset_or_reconfigured(self):
        self.service.close()
        store = RunStore(self.root / "completed-lifecycle-runs")
        self.service = FileCommandService(
            self.root / "completed-lifecycle-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        completed = self.process(
            "completed-lifecycle",
            "run",
            x=self.x,
            config=_configuration(),
        )
        self.assertEqual(completed.state, "completed")

        loaded = self.process("after-completed-load", "load", x=self.x * 0.9)
        configured = self.process(
            "after-completed-config",
            "configure",
            config=_configuration(),
        )
        reset = self.process("after-completed-reset", "reset")

        self.assertEqual(loaded.state, "ready")
        self.assertEqual(configured.state, "ready")
        self.assertEqual(reset.state, "idle")
        self.assertEqual(
            store.open_run("completed-lifecycle").read_manifest()["status"],
            "completed",
        )
        self.assertEqual(
            store.open_run("after-completed-load").read_manifest()["status"],
            "stopped",
        )
        self.assertEqual(
            store.open_run("after-completed-config").read_manifest()["status"],
            "stopped",
        )

    def test_cleanup_of_completed_run_does_not_block_reset(self):
        self.service.close()
        store = RunStore(
            self.root / "cleaned-completed-runs",
            retention_seconds=0.0,
        )
        self.service = FileCommandService(
            self.root / "cleaned-completed-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        completed = self.process(
            "cleaned-completed",
            "run",
            x=self.x,
            config=_configuration(),
        )
        self.assertEqual(completed.state, "completed")
        self.assertEqual(
            store.cleanup_expired(now=time.time() + 1.0),
            ("cleaned-completed",),
        )

        reset = self.process("reset-after-cleanup", "reset")

        self.assertEqual(reset.state, "idle")
        self.assertIsNone(self.service.processor.run_id)

    def test_completed_snapshot_exports_after_temporary_run_cleanup(self):
        self.service.close()
        store = RunStore(
            self.root / "cleaned-export-runs",
            retention_seconds=0.0,
        )
        self.service = FileCommandService(
            self.root / "cleaned-export-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        self.process("cleaned-export-load", "load", x=self.x)
        self.process("cleaned-export-config", "configure", config=_configuration())
        self.process("cleaned-export-power", "power_tune")
        self.process("cleaned-export-calibrate", "calibrate")
        self.process("cleaned-export-step-1", "step")
        completed = self.process("cleaned-export-step-2", "step")
        self.assertEqual(completed.state, "completed")
        self.assertEqual(
            store.cleanup_expired(now=time.time() + 1.0),
            ("cleaned-export-config",),
        )

        exported = self.process("export-after-cleanup", "export")

        self.assertEqual(exported.state, "completed")
        self.assertTrue(self.service.result_path("export-after-cleanup").is_file())
        self.assertIsNone(self.service.processor.run_id)

    def test_duplicate_completed_run_keeps_persisted_status_unchanged(self):
        self.service.close()
        store = RunStore(self.root / "completed-idempotence-runs")
        self.service = FileCommandService(
            self.root / "completed-idempotence-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        command_path = _write_command(
            self.service,
            "completed-idempotence",
            "run",
            x=self.x,
            config=_configuration(),
        )
        first = self.service.process_file(command_path)
        status_path = self.service.status_path("completed-idempotence")
        first_bytes = status_path.read_bytes()
        first_mtime = status_path.stat().st_mtime_ns
        time.sleep(0.01)

        second = self.service.process_file(command_path)

        self.assertEqual(second, first)
        self.assertEqual(status_path.read_bytes(), first_bytes)
        self.assertEqual(status_path.stat().st_mtime_ns, first_mtime)

    def test_run_status_never_points_to_a_previous_recorder_during_setup(self):
        self.service.close()
        store = RunStore(self.root / "run-identity-runs")
        exchange = self.root / "run-identity-exchange"
        entered = threading.Event()
        release = threading.Event()
        factory_calls = 0

        def factory(_device_type):
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 2:
                entered.set()
                if not release.wait(5.0):
                    raise TimeoutError("test factory gate timed out")
            return ClosedLoopController(SimulatedRFBench())

        self.service = FileCommandService(
            exchange,
            run_store=store,
            controller_factory=factory,
            status_poll_seconds=0.002,
        )
        self.process("identity-load", "load", x=self.x)
        self.process("identity-config", "configure", config=_configuration())
        command_path = _write_command(
            self.service,
            "identity-run",
            "run",
            config=_configuration(),
        )
        results = []
        thread = threading.Thread(
            target=lambda: results.append(self.service.process_file(command_path))
        )
        thread.start()
        self.assertTrue(entered.wait(5.0))

        setup_status = self.service.read_status("identity-run")
        self.assertEqual(setup_status.run_id, "identity-run")
        self.assertNotEqual(setup_status.run_id, "identity-config")

        release.set()
        thread.join(5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].state, "completed")
        self.assertEqual(results[0].run_id, "identity-run")

    def test_service_close_persists_a_nonterminal_run_as_stopped(self):
        self.service.close()
        store = RunStore(self.root / "close-runs")
        self.service = FileCommandService(
            self.root / "close-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        self.process("close-load", "load", x=self.x)
        self.process("close-config", "configure", config=_configuration())

        self.service.close()

        manifest = store.open_run("close-config").read_manifest()
        self.assertEqual(manifest["status"], "stopped")
        self.assertIsNotNone(manifest["completed"])

    def test_reset_and_input_replacement_terminate_previous_run_manifests(self):
        self.service.close()
        store = RunStore(self.root / "replacement-runs")
        self.service = FileCommandService(
            self.root / "replacement-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        self.process("replace-load", "load", x=self.x)
        self.process("replace-config", "configure", config=_configuration())

        self.process("replacement-load", "load", x=self.x * 0.9)
        self.assertEqual(
            store.open_run("replace-config").read_manifest()["status"],
            "stopped",
        )

        self.process("replacement-config", "configure", config=_configuration())
        self.assertEqual(
            store.open_run("replacement-load").read_manifest()["status"],
            "stopped",
        )

        self.process("replacement-reset", "reset")
        self.assertEqual(
            store.open_run("replacement-config").read_manifest()["status"],
            "stopped",
        )

    def test_busy_rejection_and_stop_interrupt_an_in_flight_run(self):
        self.service.close()
        entered = threading.Event()
        release = threading.Event()

        def factory(device_type: str) -> ClosedLoopController:
            self.assertEqual(device_type, "simulated")
            return ClosedLoopController(_GateSimulatedBench(entered, release))

        self.service = FileCommandService(
            self.root / "gated-exchange",
            controller_factory=factory,
            status_poll_seconds=0.002,
        )
        run_path = _write_command(
            self.service,
            "slow-run",
            "run",
            x=self.x,
            config=_configuration(max_iterations=5),
        )
        results: list[object] = []
        run_thread = threading.Thread(
            target=lambda: results.append(self.service.process_file(run_path))
        )
        run_thread.start()
        self.assertTrue(entered.wait(5.0))

        busy_path = _write_command(
            self.service,
            "busy-load",
            "load",
            x=self.x,
        )
        busy = self.service.process_file(busy_path)
        stop_path = _write_command(self.service, "stop-now", "stop")
        stopped = self.service.process_file(stop_path)

        self.assertFalse(busy.accepted)
        self.assertEqual(busy.state, "busy")
        self.assertEqual(busy.error_code, "busy")
        self.assertTrue(stopped.accepted)
        self.assertIn(stopped.state, {"stopping", "stopped"})
        release.set()
        run_thread.join(5.0)
        self.assertFalse(run_thread.is_alive())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].state, "stopped")
        deadline = time.monotonic() + 5.0
        while self.service.read_status("stop-now").state == "stopping":
            if time.monotonic() >= deadline:
                self.fail("stop command status did not reach a terminal state")
            time.sleep(0.01)
        self.assertEqual(self.service.read_status("stop-now").state, "stopped")
        self.assertFalse(self.service.result_path("slow-run").exists())

    def test_stop_latch_cancels_run_while_controller_factory_is_blocked(self):
        self.service.close()
        entered = threading.Event()
        release = threading.Event()

        def factory(_device_type: str) -> ClosedLoopController:
            entered.set()
            if not release.wait(5.0):
                raise TimeoutError("test factory gate timed out")
            return ClosedLoopController(SimulatedRFBench())

        self.service = FileCommandService(
            self.root / "factory-gated-exchange",
            controller_factory=factory,
            status_poll_seconds=0.002,
        )
        run_path = _write_command(
            self.service,
            "factory-run",
            "run",
            x=self.x,
            config=_configuration(),
        )
        results = []
        run_thread = threading.Thread(
            target=lambda: results.append(self.service.process_file(run_path))
        )
        run_thread.start()
        self.assertTrue(entered.wait(5.0))

        stop_path = _write_command(self.service, "factory-stop", "stop")
        stop_status = self.service.process_file(stop_path)
        release.set()
        run_thread.join(5.0)

        self.assertEqual(stop_status.state, "stopping")
        self.assertFalse(run_thread.is_alive())
        self.assertEqual(results[0].state, "failed")
        self.assertEqual(results[0].error_code, "cancelled")
        deadline = time.monotonic() + 5.0
        while self.service.read_status("factory-stop").state == "stopping":
            if time.monotonic() >= deadline:
                self.fail("factory stop status did not reach a terminal state")
            time.sleep(0.01)
        self.assertEqual(self.service.read_status("factory-stop").state, "stopped")
        self.assertFalse(self.service.result_path("factory-run").exists())

    def test_stop_monitor_recognizes_disconnect_idle_as_completed(self):
        self.service.close()
        entered = threading.Event()
        release = threading.Event()

        def factory(_device_type):
            return ClosedLoopController(_GateDisconnectBench(entered, release))

        self.service = FileCommandService(
            self.root / "disconnect-gated-exchange",
            controller_factory=factory,
            status_poll_seconds=0.002,
        )
        self.process("disconnect-gated-load", "load", x=self.x)
        self.process(
            "disconnect-gated-configure",
            "configure",
            config=_configuration(),
        )
        disconnect_path = _write_command(
            self.service,
            "disconnect-gated-target",
            "disconnect",
        )
        results = []
        disconnect_thread = threading.Thread(
            target=lambda: results.append(self.service.process_file(disconnect_path))
        )
        disconnect_thread.start()
        self.assertTrue(entered.wait(5.0))

        stop_path = _write_command(
            self.service,
            "disconnect-gated-stop",
            "stop",
        )
        stop_status = self.service.process_file(stop_path)
        release.set()
        disconnect_thread.join(5.0)

        self.assertFalse(disconnect_thread.is_alive())
        self.assertEqual(results[0].state, "idle")
        self.assertEqual(stop_status.state, "stopping")
        deadline = time.monotonic() + 5.0
        while self.service.read_status("disconnect-gated-stop").state == "stopping":
            if time.monotonic() >= deadline:
                self.fail("disconnect stop status did not become terminal")
            time.sleep(0.01)
        completed_stop = self.service.read_status("disconnect-gated-stop")
        self.assertEqual(completed_stop.state, "completed")
        self.assertEqual(completed_stop.error_code, "")

    def test_stop_monitor_never_masks_a_failed_shutdown_snapshot(self):
        target = CommandStatus(
            command_id="shutdown-target",
            accepted=True,
            state="ready",
            iteration=-1,
            message="reference transmission started",
            error_code="",
            timestamp="2026-08-28T00:00:00+00:00",
        )
        self.service._write_status(target)
        failed_snapshot = ControllerSnapshot(
            state=ControllerState.FAILED,
            connected=True,
            configured=True,
            reference_loaded=True,
            transmitting=True,
            stop_requested=True,
            active_operation=None,
            iteration=None,
            max_iterations=2,
            gain_correction=None,
            locked_attenuation_db=None,
            latest_power_dbm=None,
            config=None,
            device_type="simulated",
            completed_at="2026-08-28T00:00:00+00:00",
            last_error=ControllerErrorInfo(
                operation="request_stop",
                code="shutdown_failed",
                exception_type="RuntimeError",
                message="RF transmission could not be stopped",
            ),
        )

        status = self.service._stop_status_from_target(
            "shutdown-stop",
            "shutdown-target",
            target_action="start_transmission",
            requested_snapshot=failed_snapshot,
        )

        self.assertEqual(status.state, "failed")
        self.assertEqual(status.error_code, "shutdown_failed")
        self.assertIn("could not be stopped", status.message)

    def test_stop_reaches_processor_even_when_status_write_fails(self):
        stop_path = _write_command(self.service, "unsafe-status-stop", "stop")

        with (
            patch.object(
                self.service.processor,
                "request_stop",
                return_value=None,
            ) as request_stop,
            patch.object(
                self.service,
                "_write_status",
                side_effect=FileCommandError(
                    "status_write_failed", "injected status failure"
                ),
            ),
            self.assertRaisesRegex(FileCommandError, "status failure"),
        ):
            self.service.process_file(stop_path)

        request_stop.assert_called_once_with()

    def test_stop_reaches_pending_controller_before_snapshot_persistence(self):
        processor = FileCommandProcessor()
        current = ClosedLoopController(SimulatedRFBench())
        pending = ClosedLoopController(SimulatedRFBench())
        processor._controller = current
        processor._pending_controller = pending

        with (
            patch.object(
                processor,
                "record_snapshot",
                side_effect=OSError("injected persistence failure"),
            ),
            self.assertRaisesRegex(OSError, "persistence failure"),
        ):
            processor.request_stop()

        self.assertEqual(current.snapshot().state.value, "stopped")
        self.assertEqual(pending.snapshot().state.value, "stopped")

    def test_post_command_snapshot_failure_stops_active_transmission(self):
        self.service.close()
        store = RunStore(self.root / "snapshot-failure-runs")
        self.service = FileCommandService(
            self.root / "snapshot-failure-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        self.process("snapshot-failure-load", "load", x=self.x)
        self.process("snapshot-failure-config", "configure", config=_configuration())
        processor = self.service.processor
        record_snapshot = processor.record_snapshot

        def fail_power_ready(snapshot):
            if snapshot is not None and snapshot.state is ControllerState.POWER_READY:
                raise OSError("injected snapshot persistence failure")
            return record_snapshot(snapshot)

        with patch.object(
            processor,
            "record_snapshot",
            side_effect=fail_power_ready,
        ):
            failed = self.process("snapshot-failure-power", "power_tune")

        after = processor.snapshot()
        self.assertEqual(failed.state, "stopped")
        self.assertEqual(failed.error_code, "o_s")
        self.assertFalse(after.transmitting)
        self.assertEqual(after.state, ControllerState.STOPPED)

    def test_final_status_write_failure_stops_active_transmission(self):
        self.process("status-failure-load", "load", x=self.x)
        self.process("status-failure-config", "configure", config=_configuration())
        command_path = _write_command(
            self.service,
            "status-failure-power",
            "power_tune",
        )
        write_status = self.service._write_status

        def fail_power_ready_status(status):
            if (
                status.command_id == "status-failure-power"
                and status.state == "power_ready"
            ):
                raise FileCommandError(
                    "status_write_failed",
                    "injected final status failure",
                )
            return write_status(status)

        with (
            patch.object(
                self.service,
                "_write_status",
                side_effect=fail_power_ready_status,
            ),
            self.assertRaisesRegex(FileCommandError, "final status failure"),
        ):
            self.service.process_file(command_path)

        after = self.service.processor.snapshot()
        self.assertFalse(after.transmitting)
        self.assertEqual(after.state, ControllerState.STOPPED)

    def test_run_progress_snapshot_failure_requests_stop(self):
        self.service.close()
        entered = threading.Event()
        release = threading.Event()
        persistence_failed = threading.Event()

        def factory(_device_type):
            return ClosedLoopController(_GateSimulatedBench(entered, release))

        self.service = FileCommandService(
            self.root / "progress-snapshot-failure-exchange",
            controller_factory=factory,
            status_poll_seconds=0.002,
        )
        processor = self.service.processor
        record_snapshot = processor.record_snapshot

        def fail_active_snapshot(snapshot):
            if snapshot is not None and snapshot.state in {
                ControllerState.CALIBRATING,
                ControllerState.RUNNING,
            }:
                persistence_failed.set()
                raise OSError("injected progress snapshot failure")
            return record_snapshot(snapshot)

        command_path = _write_command(
            self.service,
            "progress-snapshot-failure",
            "run",
            x=self.x,
            config=_configuration(max_iterations=5),
        )
        results = []
        with patch.object(
            processor,
            "record_snapshot",
            side_effect=fail_active_snapshot,
        ):
            thread = threading.Thread(
                target=lambda: results.append(self.service.process_file(command_path))
            )
            thread.start()
            self.assertTrue(entered.wait(5.0))
            self.assertTrue(persistence_failed.wait(5.0))
            release.set()
            thread.join(5.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].state, "stopped")
        self.assertEqual(results[0].error_code, "o_s")
        self.assertFalse(processor.snapshot().transmitting)
        self.assertFalse(self.service.result_path("progress-snapshot-failure").exists())

    def test_run_progress_status_failure_is_reported_after_safe_stop(self):
        self.service.close()
        entered = threading.Event()
        release = threading.Event()
        status_failed = threading.Event()

        def factory(_device_type):
            return ClosedLoopController(_GateSimulatedBench(entered, release))

        self.service = FileCommandService(
            self.root / "progress-status-failure-exchange",
            controller_factory=factory,
            status_poll_seconds=0.002,
        )
        write_status = self.service._write_status

        def fail_active_status(status):
            if status.command_id == "progress-status-failure" and status.state in {
                "calibrating",
                "running",
            }:
                status_failed.set()
                raise FileCommandError(
                    "status_write_failed",
                    "injected progress status failure",
                )
            return write_status(status)

        command_path = _write_command(
            self.service,
            "progress-status-failure",
            "run",
            x=self.x,
            config=_configuration(max_iterations=5),
        )
        results = []
        with patch.object(
            self.service,
            "_write_status",
            side_effect=fail_active_status,
        ):
            thread = threading.Thread(
                target=lambda: results.append(self.service.process_file(command_path))
            )
            thread.start()
            self.assertTrue(entered.wait(5.0))
            self.assertTrue(status_failed.wait(5.0))
            release.set()
            thread.join(5.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].state, "stopped")
        self.assertEqual(results[0].error_code, "status_write_failed")
        self.assertFalse(self.service.processor.snapshot().transmitting)

    def test_terminal_status_waits_for_a_blocked_progress_monitor(self):
        self.service.close()
        capture_entered = threading.Event()
        capture_release = threading.Event()
        progress_entered = threading.Event()
        progress_release = threading.Event()

        def factory(_device_type):
            return ClosedLoopController(
                _GateSimulatedBench(capture_entered, capture_release)
            )

        self.service = FileCommandService(
            self.root / "blocked-progress-exchange",
            controller_factory=factory,
            status_poll_seconds=0.002,
        )
        write_status = self.service._write_status

        def block_progress_status(status):
            if status.command_id == "blocked-progress" and status.state in {
                "calibrating",
                "running",
            }:
                progress_entered.set()
                if not progress_release.wait(5.0):
                    raise TimeoutError("test progress status gate timed out")
            return write_status(status)

        command_path = _write_command(
            self.service,
            "blocked-progress",
            "run",
            x=self.x,
            config=_configuration(max_iterations=1),
        )
        results = []
        with patch.object(
            self.service,
            "_write_status",
            side_effect=block_progress_status,
        ):
            thread = threading.Thread(
                target=lambda: results.append(self.service.process_file(command_path))
            )
            thread.start()
            self.assertTrue(capture_entered.wait(5.0))
            self.assertTrue(progress_entered.wait(5.0))
            capture_release.set()
            deadline = time.monotonic() + 5.0
            while (
                self.service.processor.snapshot().state is not ControllerState.COMPLETED
            ):
                if time.monotonic() >= deadline:
                    self.fail("controller did not complete while progress was blocked")
                time.sleep(0.01)
            time.sleep(1.1)
            self.assertTrue(thread.is_alive())
            progress_release.set()
            thread.join(5.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].state, "completed")
        self.assertEqual(
            self.service.read_status("blocked-progress").state, "completed"
        )

    def test_stop_target_capture_is_atomic_with_the_cancellation_request(self):
        self.service.close()
        processor = _DispatchRaceProcessor()
        self.service = FileCommandService(
            self.root / "stop-dispatch-exchange",
            processor=processor,
            status_poll_seconds=0.002,
        )
        processor.service = self.service
        self.service._write_status(
            CommandStatus(
                command_id="command-a",
                accepted=True,
                state="stopped",
                iteration=0,
                message="command A stopped",
                error_code="",
                timestamp="2026-08-28T00:00:00Z",
            )
        )
        with self.service._dispatch_lock:
            self.service._active_command_id = "command-a"
            self.service._idle.clear()

        stop_path = _write_command(self.service, "atomic-stop", "stop")
        stop_results = []
        stop_thread = threading.Thread(
            target=lambda: stop_results.append(self.service.process_file(stop_path))
        )
        stop_thread.start()
        self.assertTrue(processor.entered.wait(5.0))

        switch_attempted = threading.Event()
        switch_completed = threading.Event()

        def switch_to_command_b():
            switch_attempted.set()
            self.service._release_claim("command-a")
            with self.service._dispatch_lock:
                self.service._active_command_id = "command-b"
                self.service._idle.clear()
            switch_completed.set()

        switch_thread = threading.Thread(target=switch_to_command_b)
        switch_thread.start()
        self.assertTrue(switch_attempted.wait(5.0))
        self.assertFalse(switch_completed.wait(0.05))

        processor.observe.set()
        stop_thread.join(5.0)
        switch_thread.join(5.0)

        self.assertFalse(stop_thread.is_alive())
        self.assertFalse(switch_thread.is_alive())
        self.assertEqual(processor.observed_active_ids, ["command-a"])
        self.assertEqual(self.service.active_command_id, "command-b")
        self.assertEqual(stop_results[0].state, "stopping")
        self.service._release_claim("command-b")

    def test_stop_monitor_uses_its_bound_terminal_snapshot_for_a_stale_target_status(
        self,
    ):
        self.service._write_status(
            CommandStatus(
                command_id="stale-target",
                accepted=True,
                state="ready",
                iteration=-1,
                message="target command completed",
                error_code="",
                timestamp="2026-08-28T00:00:00Z",
            )
        )
        controller = ClosedLoopController(SimulatedRFBench())
        stopped_snapshot = controller.request_stop()

        status = self.service._stop_status_from_target(
            "stale-target-stop",
            "stale-target",
            requested_snapshot=stopped_snapshot,
        )

        self.assertEqual(status.state, "stopped")
        self.assertEqual(status.error_code, "")

    def test_service_stop_ends_transmission_between_stepwise_commands(self):
        self.process("service-stop-load", "load", x=self.x)
        self.process("service-stop-config", "configure", config=_configuration())
        self.process("service-stop-power", "power_tune")
        self.assertTrue(self.service.processor.snapshot().transmitting)

        self.service.stop()

        snapshot = self.service.processor.snapshot()
        self.assertFalse(snapshot.transmitting)
        self.assertEqual(snapshot.state, ControllerState.STOPPED)

    def test_duplicate_command_id_returns_persisted_status_without_reexecution(self):
        path = _write_command(self.service, "same-id", "load", x=self.x)
        first = self.service.process_file(path)
        first_mtime = self.service.status_path("same-id").stat().st_mtime_ns
        replacement = self.x * 0.5
        savemat(
            path,
            {
                "schema_version": 1,
                "command_id": "same-id",
                "action": "load",
                "x": replacement,
            },
        )

        second = self.service.process_file(path)

        self.assertEqual(first, second)
        self.assertEqual(
            self.service.status_path("same-id").stat().st_mtime_ns,
            first_mtime,
        )
        self.assertIsNone(self.service.processor.controller)

    def test_restart_marks_an_orphaned_nonterminal_status_failed_without_rerun(self):
        command_path = _write_command(
            self.service,
            "orphaned-run",
            "run",
            x=self.x,
            config=_configuration(),
        )
        self.service._write_status(
            CommandStatus(
                command_id="orphaned-run",
                accepted=True,
                state="running",
                iteration=0,
                message="automatic run in progress",
                error_code="",
                timestamp="2026-08-28T00:00:00Z",
            )
        )

        recovered = self.service.process_file(command_path)

        self.assertTrue(recovered.accepted)
        self.assertEqual(recovered.state, "failed")
        self.assertEqual(recovered.error_code, "service_restarted")
        self.assertIsNone(self.service.processor.controller)
        self.assertFalse(self.service.result_path("orphaned-run").exists())

    def test_restart_recovers_result_when_final_status_write_was_interrupted(self):
        self.service.close()
        store = RunStore(self.root / "status-recovery-runs")
        exchange = self.root / "status-recovery-exchange"
        self.service = FileCommandService(
            exchange,
            run_store=store,
            status_poll_seconds=0.002,
        )
        command_path = _write_command(
            self.service,
            "status-recovery",
            "run",
            x=self.x,
            config=_configuration(),
        )
        write_status = self.service._write_status

        def fail_completed_status(status):
            if status.command_id == "status-recovery" and status.state == "completed":
                raise FileCommandError(
                    "status_write_failed",
                    "injected final status failure",
                )
            return write_status(status)

        with (
            patch.object(
                self.service, "_write_status", side_effect=fail_completed_status
            ),
            self.assertRaisesRegex(FileCommandError, "final status failure"),
        ):
            self.service.process_file(command_path)

        self.assertTrue(self.service.result_path("status-recovery").is_file())
        self.assertEqual(
            store.open_run("status-recovery").read_manifest()["status"],
            "completed",
        )
        self.service.close()

        def forbidden_factory(_device_type):
            raise AssertionError("recovery must not recreate a controller")

        self.service = FileCommandService(
            exchange,
            run_store=store,
            controller_factory=forbidden_factory,
            status_poll_seconds=0.002,
        )
        recovered = self.service.process_file(command_path)

        self.assertEqual(recovered.state, "completed")
        self.assertEqual(recovered.iteration, 2)
        self.assertEqual(recovered.run_id, "status-recovery")

    def test_restart_republishes_cached_result_after_outbox_failure(self):
        self.service.close()
        store = RunStore(self.root / "publish-recovery-runs")
        exchange = self.root / "publish-recovery-exchange"
        self.service = FileCommandService(
            exchange,
            run_store=store,
            status_poll_seconds=0.002,
        )
        command_path = _write_command(
            self.service,
            "publish-recovery",
            "run",
            x=self.x,
            config=_configuration(),
        )
        with patch(
            "remote_dpd.file_interface._atomic_publish_file",
            side_effect=FileCommandError(
                "result_publish_failed",
                "injected outbox failure",
            ),
        ):
            failed = self.service.process_file(command_path)

        self.assertEqual(failed.state, "failed")
        self.assertEqual(failed.error_code, "result_publish_failed")
        self.assertFalse(self.service.result_path("publish-recovery").exists())
        recorder = store.open_run("publish-recovery")
        self.assertEqual(recorder.read_manifest()["status"], "completed")
        self.assertIsNotNone(recorder.final_result_path)
        self.service.close()

        def forbidden_factory(_device_type):
            raise AssertionError("recovery must not recreate a controller")

        self.service = FileCommandService(
            exchange,
            run_store=store,
            controller_factory=forbidden_factory,
            status_poll_seconds=0.002,
        )
        recovered = self.service.process_file(command_path)

        self.assertEqual(recovered.state, "completed")
        self.assertTrue(self.service.result_path("publish-recovery").is_file())

    def test_restart_terminalizes_an_interrupted_run_manifest_without_replay(self):
        self.service.close()
        store = RunStore(self.root / "orphan-manifest-runs")
        exchange = self.root / "orphan-manifest-exchange"
        parsed = _parse_config_json(json.dumps(_configuration()))
        store.create_run(
            parsed.closed_loop,
            self.x,
            run_id="orphan-manifest",
        )

        def forbidden_factory(_device_type):
            raise AssertionError("recovery must not recreate a controller")

        self.service = FileCommandService(
            exchange,
            run_store=store,
            controller_factory=forbidden_factory,
            status_poll_seconds=0.002,
        )
        command_path = _write_command(
            self.service,
            "orphan-manifest",
            "run",
            x=self.x,
            config=_configuration(),
        )
        self.service._write_status(
            CommandStatus(
                command_id="orphan-manifest",
                accepted=True,
                state="running",
                iteration=0,
                message="automatic run in progress",
                error_code="",
                timestamp="2026-08-28T00:00:00Z",
                run_id="orphan-manifest",
            )
        )

        recovered = self.service.process_file(command_path)

        self.assertEqual(recovered.state, "failed")
        self.assertEqual(recovered.error_code, "service_restarted")
        manifest = store.open_run("orphan-manifest").read_manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertIsNotNone(manifest["completed"])

    def test_completed_stepwise_lineage_does_not_change_configure_status_shape(self):
        self.service.close()
        store = RunStore(self.root / "lineage-recovery-runs")
        exchange = self.root / "lineage-recovery-exchange"
        self.service = FileCommandService(
            exchange,
            run_store=store,
            status_poll_seconds=0.002,
        )
        self.process("lineage-load", "load", x=self.x)
        configure_path = _write_command(
            self.service,
            "lineage-config",
            "configure",
            config=_configuration(max_iterations=1),
        )
        self.assertEqual(self.service.process_file(configure_path).state, "ready")
        self.process("lineage-power", "power_tune")
        self.process("lineage-calibrate", "calibrate")
        self.assertEqual(self.process("lineage-step", "step").state, "completed")
        self.service.status_path("lineage-config").unlink()
        self.service.close()

        def forbidden_factory(_device_type):
            raise AssertionError("lineage recovery must not recreate a controller")

        self.service = FileCommandService(
            exchange,
            run_store=store,
            controller_factory=forbidden_factory,
            status_poll_seconds=0.002,
        )
        first = self.service.process_file(configure_path)
        status_path = self.service.status_path("lineage-config")
        first_mtime = status_path.stat().st_mtime_ns
        second = self.service.process_file(configure_path)

        self.assertEqual(first.state, "ready")
        self.assertEqual(first.iteration, -1)
        self.assertEqual(first.run_id, "lineage-config")
        self.assertEqual(second, first)
        self.assertEqual(status_path.stat().st_mtime_ns, first_mtime)

    def test_automatic_run_gets_a_same_id_self_contained_recorder(self):
        self.service.close()
        store = RunStore(self.root / "continued-run-storage")
        self.service = FileCommandService(
            self.root / "continued-run-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        self.process("continued-load", "load", x=self.x)
        self.process("continued-config", "configure", config=_configuration())
        self.process("continued-power", "power_tune")
        self.process("continued-calibrate", "calibrate")

        completed = self.process("continued-auto", "run")

        self.assertEqual(completed.state, "completed")
        self.assertEqual(completed.run_id, "continued-auto")
        self.assertEqual(
            store.open_run("continued-config").read_manifest()["status"],
            "stopped",
        )
        manifest = store.open_run("continued-auto").read_manifest()
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(
            [entry["iteration"] for entry in manifest["iterations"]],
            [0, 1, 2],
        )

    def test_detaching_a_finalizing_run_commits_its_valid_result_cache(self):
        self.service.close()
        store = RunStore(self.root / "finalizing-detach-runs")
        self.service = FileCommandService(
            self.root / "finalizing-detach-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        from remote_dpd.result_export import export_final_mat

        def crash_after_cache(path, snapshot):
            export_final_mat(path, snapshot)
            raise RuntimeError("injected crash after final cache")

        with patch(
            "remote_dpd.result_export.export_final_mat",
            side_effect=crash_after_cache,
        ):
            failed = self.process(
                "finalizing-detach",
                "run",
                x=self.x,
                config=_configuration(),
            )

        recorder = store.open_run("finalizing-detach")
        self.assertEqual(failed.state, "failed")
        self.assertEqual(recorder.read_manifest()["status"], "finalizing")
        self.assertIsNotNone(recorder.final_result_path)

        reset = self.process("finalizing-reset", "reset")

        self.assertEqual(reset.state, "idle")
        self.assertEqual(recorder.read_manifest()["status"], "completed")

    def test_valid_result_without_status_is_recovered_without_reexecution(self):
        self.service.close()
        store = RunStore(self.root / "missing-status-runs")
        exchange = self.root / "missing-status-exchange"
        self.service = FileCommandService(
            exchange,
            run_store=store,
            status_poll_seconds=0.002,
        )
        command_path = _write_command(
            self.service,
            "missing-status",
            "run",
            x=self.x,
            config=_configuration(),
        )
        completed = self.service.process_file(command_path)
        self.assertEqual(completed.state, "completed")
        self.service.status_path("missing-status").unlink()
        self.service.close()

        def forbidden_factory(_device_type):
            raise AssertionError("recovery must not recreate a controller")

        self.service = FileCommandService(
            exchange,
            run_store=store,
            controller_factory=forbidden_factory,
            status_poll_seconds=0.002,
        )
        recovered = self.service.process_file(command_path)

        self.assertEqual(recovered.state, "completed")
        self.assertEqual(recovered.run_id, "missing-status")

    def test_corrupt_outbox_result_is_replaced_from_valid_run_cache(self):
        self.service.close()
        store = RunStore(self.root / "cache-repair-runs")
        exchange = self.root / "cache-repair-exchange"
        self.service = FileCommandService(
            exchange,
            run_store=store,
            status_poll_seconds=0.002,
        )
        command_path = _write_command(
            self.service,
            "cache-repair",
            "run",
            x=self.x,
            config=_configuration(),
        )
        self.assertEqual(self.service.process_file(command_path).state, "completed")
        result_path = self.service.result_path("cache-repair")
        result_path.write_bytes(b"corrupt outbox")

        recovered = self.service.process_file(command_path)

        self.assertEqual(recovered.state, "completed")
        payload = loadmat(result_path, squeeze_me=True, struct_as_record=False)
        self.assertEqual(str(payload["status"]), "completed")

    def test_two_corrupt_result_copies_fail_recovery_without_reexecution(self):
        self.service.close()
        store = RunStore(self.root / "invalid-cache-runs")
        exchange = self.root / "invalid-cache-exchange"
        self.service = FileCommandService(
            exchange,
            run_store=store,
            status_poll_seconds=0.002,
        )
        command_path = _write_command(
            self.service,
            "invalid-cache",
            "run",
            x=self.x,
            config=_configuration(),
        )
        self.assertEqual(self.service.process_file(command_path).state, "completed")
        result_path = self.service.result_path("invalid-cache")
        cache_path = store.open_run("invalid-cache").final_result_path
        self.assertIsNotNone(cache_path)
        result_path.write_bytes(b"corrupt outbox")
        cache_path.write_bytes(b"corrupt cache")
        self.service.close()

        def forbidden_factory(_device_type):
            raise AssertionError("recovery must not recreate a controller")

        self.service = FileCommandService(
            exchange,
            run_store=store,
            controller_factory=forbidden_factory,
            status_poll_seconds=0.002,
        )
        recovered = self.service.process_file(command_path)

        self.assertEqual(recovered.state, "failed")
        self.assertEqual(recovered.error_code, "recovery_artifact_invalid")

    def test_fresh_reset_is_immediately_idempotent(self):
        path = _write_command(self.service, "fresh-reset", "reset")

        first = self.service.process_file(path)
        second = self.service.process_file(path)

        self.assertEqual(first.state, "idle")
        self.assertEqual(second, first)

    def test_load_after_reset_has_a_terminal_loaded_status(self):
        self.process("reload-config", "configure", config=_configuration())
        self.process("reload-reset", "reset")
        path = _write_command(self.service, "reload-reference", "load", x=self.x)

        first = self.service.process_file(path)
        second = self.service.process_file(path)

        self.assertEqual(first.state, "loaded")
        self.assertEqual(second, first)
        self.assertTrue(self.service.processor.snapshot().reference_loaded)

    def test_reset_stops_transmission_even_when_run_storage_fails(self):
        self.service.close()
        store = RunStore(self.root / "reset-storage-failure-runs")
        self.service = FileCommandService(
            self.root / "reset-storage-failure-exchange",
            run_store=store,
            status_poll_seconds=0.002,
        )
        self.process("reset-failure-load", "load", x=self.x)
        self.process("reset-failure-config", "configure", config=_configuration())
        self.process("reset-failure-power", "power_tune")
        before = self.service.processor.snapshot()
        self.assertTrue(before.transmitting)
        recorder = self.service.processor._recorder

        with patch.object(
            recorder,
            "read_manifest",
            side_effect=OSError("injected run storage failure"),
        ):
            failed = self.process("reset-storage-failure", "reset")

        after = self.service.processor.snapshot()
        self.assertEqual(failed.state, "failed")
        self.assertEqual(failed.error_code, "o_s")
        self.assertFalse(after.transmitting)
        self.assertEqual(after.state.value, "idle")
        self.assertIsNone(self.service.processor.run_id)

    def test_start_scans_existing_commands_and_watches_atomic_renames(self):
        _write_command(self.service, "preexisting", "load", x=self.x)
        self.service.start()
        self.assertTrue(self.service.wait_for_idle(5.0))
        self.assertEqual(self.service.read_status("preexisting").state, "loaded")

        _write_command(
            self.service,
            "renamed",
            "load",
            x=self.x * 0.9,
            atomic=True,
        )
        deadline = time.monotonic() + 5.0
        while not self.service.status_path("renamed").exists():
            if time.monotonic() >= deadline:
                self.fail("watchdog did not process the atomically renamed command")
            time.sleep(0.01)
        self.assertTrue(self.service.wait_for_idle(5.0))
        self.assertEqual(self.service.read_status("renamed").state, "loaded")

    def test_stop_prevents_a_delayed_watchdog_dispatch_recreating_executor(self):
        command_path = _write_command(
            self.service,
            "late-watchdog",
            "load",
            x=self.x,
        )
        ensure_entered = threading.Event()
        allow_ensure = threading.Event()
        original_ensure = self.service._ensure_executor

        def delayed_ensure():
            ensure_entered.set()
            if not allow_ensure.wait(5.0):
                raise TimeoutError("test executor gate timed out")
            return original_ensure()

        with patch.object(self.service, "_ensure_executor", delayed_ensure):
            callback = threading.Thread(
                target=self.service._process_event_path,
                args=(command_path,),
            )
            callback.start()
            self.assertTrue(ensure_entered.wait(5.0))
            self.service.stop(wait=False)
            allow_ensure.set()
            callback.join(timeout=5.0)

        self.assertFalse(callback.is_alive())
        self.assertTrue(self.service.wait_for_idle(5.0))
        self.assertIsNone(self.service._executor)
        status = self.service.read_status("late-watchdog")
        self.assertEqual(status.state, "failed")
        self.assertEqual(status.error_code, "service_stopping")

    def test_invalid_commands_get_rejected_status_and_service_continues(self):
        mismatch = _write_command(
            self.service,
            "filename-id",
            "load",
            x=self.x,
            payload_updates={"command_id": "payload-id"},
        )
        mismatch_status = self.service.process_file(mismatch)
        matrix = _write_command(
            self.service,
            "matrix-x",
            "load",
            x=np.ones((2, 2), dtype=np.complex128),
        )
        matrix_status = self.service.process_file(matrix)
        invalid_config = _configuration()
        invalid_config["unknown"] = True
        config_path = _write_command(
            self.service,
            "bad-config",
            "configure",
            config=invalid_config,
        )
        config_status = self.service.process_file(config_path)

        self.assertFalse(mismatch_status.accepted)
        self.assertEqual(mismatch_status.error_code, "command_id_mismatch")
        self.assertFalse(matrix_status.accepted)
        self.assertEqual(matrix_status.error_code, "invalid_reference")
        self.assertFalse(config_status.accepted)
        self.assertEqual(config_status.error_code, "invalid_config")

        valid = self.process("after-errors", "load", x=self.x)
        self.assertTrue(valid.accepted)
        self.assertEqual(valid.state, "loaded")

    @unittest.skipUnless(SYMLINKS_SUPPORTED, "creating symlinks requires privileges on this host")
    def test_path_and_filename_validation_prevent_traversal(self):
        outside = self.root / "command_outside.mat"
        savemat(
            outside,
            {
                "schema_version": 1,
                "command_id": "outside",
                "action": "load",
                "x": self.x,
            },
        )
        malformed = self.service.inbox / "command_../escape.mat"

        with self.assertRaisesRegex(FileCommandError, "direct child"):
            self.service.process_file(outside)
        with self.assertRaises(FileCommandError):
            self.service.process_file(malformed)

        symlink = self.service.inbox / "command_link.mat"
        symlink.symlink_to(outside)
        with self.assertRaisesRegex(FileCommandError, "symlink"):
            self.service.process_file(symlink)

    def test_status_replacements_are_always_readable_and_leave_no_temp_files(self):
        status = self.process(
            "atomic-run",
            "run",
            x=self.x,
            config=_configuration(max_iterations=3),
        )
        self.assertEqual(status.state, "completed")
        for _ in range(20):
            payload = loadmat(self.service.status_path("atomic-run"), squeeze_me=True)
            self.assertEqual(str(payload["command_id"]), "atomic-run")
            self.assertEqual(int(payload["schema_version"]), 1)
        self.assertEqual(list(self.service.outbox.glob(".*.tmp.mat")), [])

    def test_typed_runtime_values_decode_closed_loop_to_dict_contract(self):
        config = _configuration()
        config["runtime_config"] = {
            "coefficient": {"$type": "complex", "real": 0.5, "imag": -0.25},
            "taps": {
                "$type": "ndarray",
                "dtype": "<f8",
                "shape": [2],
                "data": [1.0, 0.5],
            },
        }

        parsed = _parse_config_json(json.dumps(config))

        self.assertEqual(parsed.closed_loop.runtime_config["coefficient"], 0.5 - 0.25j)
        np.testing.assert_array_equal(
            parsed.closed_loop.runtime_config["taps"], np.asarray([1.0, 0.5])
        )

    def test_missing_reference_normalization_fields_use_new_defaults(self):
        config = _configuration()
        config.pop("normalize_reference_rms")
        config.pop("reference_target_rms_dbfs")

        parsed = _parse_config_json(json.dumps(config))

        self.assertTrue(parsed.closed_loop.normalize_reference_rms)
        self.assertEqual(parsed.closed_loop.reference_target_rms_dbfs, -15.0)

    def test_strict_scalar_and_action_validation(self):
        version_path = _write_command(
            self.service,
            "bad-version",
            "load",
            x=self.x,
            payload_updates={"schema_version": 2},
        )
        action_path = _write_command(
            self.service,
            "bad-action",
            "RUN",
            x=self.x,
        )
        scalar_path = _write_command(
            self.service,
            "bad-scalar",
            "load",
            x=self.x,
            payload_updates={"schema_version": np.asarray([1, 1])},
        )

        version = self.service.process_file(version_path)
        action = self.service.process_file(action_path)
        scalar = self.service.process_file(scalar_path)

        self.assertEqual(version.error_code, "unsupported_schema")
        self.assertEqual(action.error_code, "unsupported_action")
        self.assertEqual(scalar.error_code, "invalid_scalar")


if __name__ == "__main__":
    unittest.main()
