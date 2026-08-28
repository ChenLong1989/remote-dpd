import json
import threading
import unittest
from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from remote_dpd.controller import (
    ClosedLoopConfig,
    ClosedLoopController,
    ControllerBusyError,
    ControllerState,
    ControllerStateError,
)
from remote_dpd.device import (
    CaptureRequest,
    DeviceConfig,
    DeviceParameterSchema,
    PowerSensor,
    Receiver,
    RFBench,
    Transmitter,
)
from remote_dpd.power_control import PowerController
from remote_dpd.preprocessing import CaptureBatch
from remote_dpd.runtime import (
    DPDRuntime,
    RuntimeStepInput,
    RuntimeStepResult,
    register_runtime,
)
from remote_dpd.safety import DigitalSafetyError


class FakeRFBench(RFBench, Transmitter, Receiver, PowerSensor):
    def __init__(self, max_capture_samples: int) -> None:
        self._max_capture_samples = max_capture_samples
        self.connected = False
        self.configured = False
        self.transmitting = False
        self.config: DeviceConfig | None = None
        self.waveform: np.ndarray | None = None
        self.attenuation_db = 30.0
        self.capture_requests: list[CaptureRequest] = []
        self.uploaded_waveforms: list[np.ndarray] = []
        self.calls: list[tuple[str, float]] = []
        self.power_readings: list[float] = []
        self.measure_count = 0
        self.safe_shutdown_count = 0
        self.block_shutdown = False
        self.shutdown_entered = threading.Event()
        self.shutdown_release = threading.Event()
        self.feedback_scale = 1.0
        self.block_capture = False
        self.capture_entered = threading.Event()
        self.capture_release = threading.Event()

    @property
    def transmitter(self) -> Transmitter:
        return self

    @property
    def receiver(self) -> Receiver:
        return self

    @property
    def power_sensor(self) -> PowerSensor:
        return self

    @property
    def parameter_schema(self) -> DeviceParameterSchema:
        return DeviceParameterSchema("fake", 1)

    @property
    def max_capture_samples(self) -> int:
        return self._max_capture_samples

    def connect(self, timeout_seconds: float) -> None:
        self.calls.append(("connect", timeout_seconds))
        self.connected = True

    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        self.calls.append(("configure", timeout_seconds))
        if not self.connected or self.transmitting:
            raise RuntimeError("invalid configure lifecycle")
        self.config = config
        self.configured = True
        self.attenuation_db = config.initial_attenuation_db

    def safe_shutdown(self, timeout_seconds: float) -> None:
        self.calls.append(("safe_shutdown", timeout_seconds))
        self.safe_shutdown_count += 1
        self.shutdown_entered.set()
        if self.block_shutdown and not self.shutdown_release.wait(timeout=5.0):
            raise TimeoutError("test shutdown was not released")
        self.transmitting = False

    def disconnect(self, timeout_seconds: float) -> None:
        self.calls.append(("disconnect", timeout_seconds))
        self.transmitting = False
        self.connected = False
        self.configured = False
        self.config = None
        self.waveform = None

    def upload_waveform(self, waveform: np.ndarray, timeout_seconds: float) -> None:
        self.calls.append(("upload", timeout_seconds))
        if not self.connected or not self.configured or self.transmitting:
            raise RuntimeError("invalid upload lifecycle")
        self.waveform = np.array(waveform, dtype=np.complex128, copy=True)
        self.uploaded_waveforms.append(self.waveform.copy())

    def start_transmission(self, timeout_seconds: float) -> None:
        self.calls.append(("start", timeout_seconds))
        if self.waveform is None:
            raise RuntimeError("no waveform")
        self.transmitting = True

    def stop_transmission(self, timeout_seconds: float) -> None:
        self.calls.append(("stop", timeout_seconds))
        if not self.connected:
            raise RuntimeError("not connected")
        self.transmitting = False

    def get_attenuation_db(self, timeout_seconds: float) -> float:
        self.calls.append(("get_attenuation", timeout_seconds))
        return self.attenuation_db

    def set_attenuation_db(self, attenuation_db: float, timeout_seconds: float) -> None:
        self.calls.append(("set_attenuation", timeout_seconds))
        self.attenuation_db = float(attenuation_db)

    def capture(self, request: CaptureRequest, timeout_seconds: float) -> CaptureBatch:
        self.calls.append(("capture", timeout_seconds))
        if not self.transmitting or self.waveform is None or self.config is None:
            raise RuntimeError("capture requires transmission")
        self.capture_requests.append(request)
        self.capture_entered.set()
        if self.block_capture and not self.capture_release.wait(timeout=5.0):
            raise TimeoutError("test capture was not released")

        waveform = self.waveform
        nonlinear = waveform * (1.0 + 0.4 * np.abs(waveform) ** 2)
        feedback = self.feedback_scale * nonlinear
        packed = np.tile(feedback, request.segment_count)
        return CaptureBatch(
            iq=packed,
            segment_length=request.segment_length,
            segment_count=request.segment_count,
            sample_rate_hz=self.config.sample_rate_hz,
            coherent_within_batch=True,
        )

    def measure_power_dbm(self, timeout_seconds: float) -> float:
        self.calls.append(("measure_power", timeout_seconds))
        self.measure_count += 1
        if self.power_readings:
            return self.power_readings.pop(0)
        return 20.0 - self.attenuation_db


class _ReleaseGateLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.release_entered = threading.Event()
        self.allow_release = threading.Event()

    def acquire(self, blocking: bool = True) -> bool:
        return self._lock.acquire(blocking=blocking)

    def release(self) -> None:
        self.release_entered.set()
        if not self.allow_release.wait(timeout=5.0):
            raise TimeoutError("test operation-lock release was not allowed")
        self._lock.release()


class UnsafeRuntime(DPDRuntime):
    name = "controller_test_unsafe"

    def _step(
        self,
        step_input: RuntimeStepInput,
        config: Mapping[str, object],
    ) -> RuntimeStepResult:
        return RuntimeStepResult(
            y_candidate=step_input.x * 10.0,
            metrics={"unsafe": True},
        )


class CloseFailsOnceRuntime(DPDRuntime):
    name = "controller_test_close_fails_once"
    instances: ClassVar[list["CloseFailsOnceRuntime"]] = []

    def __init__(self) -> None:
        super().__init__()
        self.instance_index = len(self.instances)
        self.close_attempts = 0
        self.instances.append(self)

    def _step(
        self,
        step_input: RuntimeStepInput,
        config: Mapping[str, object],
    ) -> RuntimeStepResult:
        return RuntimeStepResult(step_input.y_current, {})

    def _on_close(self) -> None:
        self.close_attempts += 1
        if self.instance_index == 0 and self.close_attempts == 1:
            raise RuntimeError("transient runtime close failure")


class ClosedLoopControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_runtime(UnsafeRuntime.name, UnsafeRuntime, replace=True)
        register_runtime(
            CloseFailsOnceRuntime.name,
            CloseFailsOnceRuntime,
            replace=True,
        )

    def setUp(self) -> None:
        CloseFailsOnceRuntime.instances.clear()
        samples = np.arange(96)
        self.x = (
            0.16 * np.exp(2j * np.pi * samples / 23)
            + 0.06 * np.exp(2j * np.pi * samples / 11)
        ).astype(np.complex128)

    def make_config(
        self,
        *,
        average_segment_count: int = 3,
        max_iterations: int = 2,
        runtime_name: str = "basic_ilc",
        runtime_config: Mapping[str, object] | None = None,
        target_power_dbm: float = -10.0,
        max_adjustments: int = 10,
    ) -> ClosedLoopConfig:
        return ClosedLoopConfig(
            device_config=DeviceConfig(
                sample_rate_hz=245.76e6,
                average_segment_count=average_segment_count,
                target_power_dbm=target_power_dbm,
                safety_power_limit_dbm=0.0,
                initial_attenuation_db=30.0,
                min_attenuation_db=0.0,
                max_attenuation_db=60.0,
                settle_seconds=0.0,
                max_adjustments=max_adjustments,
                call_timeout_seconds=2.5,
            ),
            runtime_name=runtime_name,
            runtime_config={"mu": 0.5} if runtime_config is None else runtime_config,
            max_iterations=max_iterations,
        )

    def make_controller(
        self,
        *,
        max_capture_samples: int | None = None,
        config: ClosedLoopConfig | None = None,
    ) -> tuple[ClosedLoopController, FakeRFBench, ClosedLoopConfig]:
        actual_config = config or self.make_config()
        bench = FakeRFBench(max_capture_samples or self.x.size * 100)
        controller = ClosedLoopController(
            bench,
            power_controller=PowerController(sleep_fn=lambda _: None),
        )
        controller.connect()
        controller.apply_config(actual_config)
        controller.load_reference(self.x)
        return controller, bench, actual_config

    @staticmethod
    def calibrate_manually(controller: ClosedLoopController) -> None:
        controller.start_reference_transmission()
        controller.tune_power()
        controller.calibrate()

    def test_config_is_deeply_frozen_and_validates_iteration_count(self):
        nested = {"values": [1, {"array": np.asarray([2.0, 3.0])}]}
        config = self.make_config(runtime_config=nested)
        nested["values"][1]["array"][0] = 99.0
        nested["values"].append(4)

        self.assertEqual(len(config.runtime_config["values"]), 2)
        self.assertEqual(config.runtime_config["values"][1]["array"][0], 2.0)
        with self.assertRaises(TypeError):
            config.runtime_config["new"] = True
        with self.assertRaises(ValueError):
            self.make_config(max_iterations=0)
        with self.assertRaises(TypeError):
            ClosedLoopConfig(DeviceConfig(), max_iterations=True)

    def test_config_has_a_detached_strict_json_representation(self):
        config = ClosedLoopConfig(
            DeviceConfig(),
            runtime_name="custom",
            runtime_config={
                "coefficient": 0.5 + 0.25j,
                "taps": np.asarray([1.0, 0.5]),
            },
            max_iterations=2,
        )

        serialized = config.to_dict()

        json.dumps(serialized, allow_nan=False)
        self.assertEqual(
            serialized["runtime_config"]["coefficient"]["$type"], "complex"
        )
        self.assertEqual(serialized["runtime_config"]["taps"]["$type"], "ndarray")
        serialized["runtime_config"]["taps"]["data"][0] = 99.0
        self.assertEqual(config.runtime_config["taps"][0], 1.0)

        for values in (
            np.asarray([True, False]),
            np.asarray([1, 2], dtype=np.int16),
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([1.0 + 2.0j]),
            np.asarray(["a", "b"]),
        ):
            with self.subTest(dtype=values.dtype):
                current = ClosedLoopConfig(
                    DeviceConfig(),
                    runtime_config={"values": values},
                )
                json.dumps(current.to_dict(), allow_nan=False)

        structured_dtype = np.dtype([("p", np.int32), ("value", np.float64)])
        invalid_arrays = [
            np.asarray([b"a"]),
            np.asarray(["2026-08-28"], dtype="datetime64[D]"),
            np.asarray([(1, 0.5)], dtype=structured_dtype),
        ]
        if np.dtype(np.longdouble).itemsize > 8:
            invalid_arrays.append(np.asarray([1.0], dtype=np.longdouble))
        if np.dtype(np.clongdouble).itemsize > 16:
            invalid_arrays.append(np.asarray([1.0 + 1.0j], dtype=np.clongdouble))
        for values in invalid_arrays:
            with (
                self.subTest(dtype=values.dtype),
                self.assertRaisesRegex(TypeError, "dtype"),
            ):
                ClosedLoopConfig(
                    DeviceConfig(),
                    runtime_config={"values": values},
                )

    def test_illegal_state_transitions_do_not_call_devices(self):
        bench = FakeRFBench(self.x.size * 2)
        controller = ClosedLoopController(
            bench,
            power_controller=PowerController(sleep_fn=lambda _: None),
        )
        with self.assertRaises(ControllerStateError):
            controller.tune_power()
        self.assertEqual(bench.calls, [])

        controller.connect()
        controller.apply_config(self.make_config())
        controller.load_reference(self.x)
        with self.assertRaises(ControllerStateError):
            controller.calibrate()
        with self.assertRaises(ControllerStateError):
            controller.tune_power()
        self.assertEqual(controller.snapshot().state, ControllerState.READY)
        self.assertEqual(controller.snapshot().device_type, "fake")
        self.assertIsNone(controller.snapshot().completed_at)

    def test_capture_requests_split_at_complete_segment_boundaries(self):
        config = self.make_config(average_segment_count=5)
        controller, bench, _ = self.make_controller(
            max_capture_samples=self.x.size * 2 + 7,
            config=config,
        )
        self.calibrate_manually(controller)

        self.assertEqual(
            [request.segment_count for request in bench.capture_requests],
            [2, 2, 1],
        )
        self.assertTrue(
            all(
                request.segment_length == self.x.size
                for request in bench.capture_requests
            )
        )
        record = controller.snapshot().current_record
        self.assertIsNotNone(record)
        self.assertEqual(record.preprocessing.segment_count, 5)

    def test_power_ready_reference_can_be_stopped_and_restarted(self):
        controller, bench, _ = self.make_controller()
        controller.start_reference_transmission()
        controller.tune_power()
        controller.stop_transmission()

        restarted = controller.start_reference_transmission()
        self.assertEqual(restarted.state, ControllerState.POWER_READY)
        self.assertIsNone(restarted.active_operation)
        self.assertTrue(bench.transmitting)
        record = controller.calibrate()
        self.assertEqual(record.iteration, 0)

    def test_calibration_monitors_power_immediately_before_capture(self):
        controller, bench, _ = self.make_controller()
        controller.start_reference_transmission()
        controller.tune_power()
        capture_count = len(bench.capture_requests)
        bench.power_readings = [1.0]

        with self.assertRaisesRegex(RuntimeError, "safety limit"):
            controller.calibrate()

        self.assertEqual(len(bench.capture_requests), capture_count)
        self.assertEqual(controller.snapshot().state, ControllerState.FAILED)
        self.assertFalse(bench.transmitting)

    def test_manual_and_auto_paths_commit_identical_final_results(self):
        manual, manual_bench, _ = self.make_controller()
        automatic, automatic_bench, _ = self.make_controller()

        self.calibrate_manually(manual)
        manual.step()
        final_manual = manual.step()
        automatic_snapshot = automatic.run_auto()
        final_automatic = automatic_snapshot.current_record

        self.assertEqual(manual.snapshot().state, ControllerState.COMPLETED)
        self.assertEqual(automatic_snapshot.state, ControllerState.COMPLETED)
        self.assertIsNotNone(manual.snapshot().completed_at)
        self.assertIsNotNone(automatic_snapshot.completed_at)
        self.assertIsNotNone(final_automatic)
        np.testing.assert_allclose(final_manual.y, final_automatic.y, atol=1e-12)
        np.testing.assert_allclose(final_manual.z, final_automatic.z, atol=1e-12)
        self.assertEqual(final_manual.iteration, 2)
        self.assertFalse(manual_bench.transmitting)
        self.assertFalse(automatic_bench.transmitting)
        self.assertEqual(len(manual.snapshot().records), 3)
        with self.assertRaises(ControllerStateError):
            manual.step()

    def test_candidate_safety_failure_never_uploads_or_monitors_candidate(self):
        config = self.make_config(
            runtime_name=UnsafeRuntime.name,
            runtime_config={},
            max_iterations=1,
        )
        controller, bench, _ = self.make_controller(config=config)
        self.calibrate_manually(controller)
        upload_count = len(bench.uploaded_waveforms)
        measure_count = bench.measure_count
        capture_count = len(bench.capture_requests)

        with self.assertRaises(DigitalSafetyError):
            controller.step()

        snapshot = controller.snapshot()
        self.assertEqual(snapshot.state, ControllerState.FAILED)
        self.assertIsNotNone(snapshot.completed_at)
        self.assertEqual(len(bench.uploaded_waveforms), upload_count)
        self.assertEqual(bench.measure_count, measure_count)
        self.assertEqual(len(bench.capture_requests), capture_count)
        self.assertFalse(bench.transmitting)
        self.assertEqual(snapshot.last_error.operation, "step")

    def test_power_failure_after_upload_does_not_capture_feedback(self):
        controller, bench, _ = self.make_controller()
        self.calibrate_manually(controller)
        calibration_capture_count = len(bench.capture_requests)
        bench.power_readings = [1.0]

        with self.assertRaisesRegex(RuntimeError, "safety limit"):
            controller.step()

        snapshot = controller.snapshot()
        self.assertEqual(snapshot.state, ControllerState.FAILED)
        self.assertEqual(len(bench.capture_requests), calibration_capture_count)
        self.assertEqual(snapshot.last_error.code, "safety_limit_exceeded")
        self.assertEqual(snapshot.latest_power_dbm, 1.0)
        self.assertIn("1.0", snapshot.last_error.message)
        self.assertFalse(bench.transmitting)

    def test_stop_request_and_single_operation_lock_are_thread_safe(self):
        controller, bench, _ = self.make_controller()
        bench.block_capture = True
        errors: list[Exception] = []

        def run() -> None:
            try:
                controller.run_auto()
            except Exception as exc:  # noqa: BLE001 - assertion captures worker errors
                errors.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(bench.capture_entered.wait(timeout=2.0))

        with self.assertRaises(ControllerBusyError):
            controller.apply_config(self.make_config(max_iterations=3))
        stop_snapshot = controller.request_stop()
        self.assertEqual(stop_snapshot.state, ControllerState.STOPPING)
        bench.capture_release.set()
        worker.join(timeout=5.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(controller.snapshot().state, ControllerState.STOPPED)
        self.assertIsNotNone(controller.snapshot().completed_at)
        self.assertFalse(bench.transmitting)
        self.assertGreaterEqual(bench.safe_shutdown_count, 1)

    def test_stop_request_cannot_be_lost_during_operation_lock_release(self):
        controller, bench, _ = self.make_controller()
        controller.start_reference_transmission()
        release_gate = _ReleaseGateLock()
        controller._operation_lock = release_gate

        worker = threading.Thread(target=controller.stop_transmission)
        worker.start()
        self.assertTrue(release_gate.release_entered.wait(timeout=2.0))

        stopper = threading.Thread(target=controller.request_stop)
        stopper.start()
        self.assertTrue(stopper.is_alive())
        release_gate.allow_release.set()
        worker.join(timeout=5.0)
        stopper.join(timeout=5.0)

        self.assertFalse(worker.is_alive())
        self.assertFalse(stopper.is_alive())
        self.assertEqual(controller.snapshot().state, ControllerState.STOPPED)
        self.assertGreaterEqual(bench.safe_shutdown_count, 1)

    def test_snapshot_and_repeated_stop_do_not_block_on_slow_shutdown(self):
        controller, bench, _ = self.make_controller()
        controller.start_reference_transmission()
        bench.block_shutdown = True

        first_stop = threading.Thread(target=controller.request_stop)
        first_stop.start()
        self.assertTrue(bench.shutdown_entered.wait(timeout=2.0))

        snapshot = controller.snapshot()
        repeated_results = []
        repeated_stop = threading.Thread(
            target=lambda: repeated_results.append(controller.request_stop())
        )
        repeated_stop.start()
        repeated_stop.join(timeout=0.5)

        self.assertEqual(snapshot.state, ControllerState.STOPPING)
        self.assertFalse(repeated_stop.is_alive())
        self.assertEqual(repeated_results[0].state, ControllerState.STOPPING)
        bench.shutdown_release.set()
        first_stop.join(timeout=5.0)
        self.assertFalse(first_stop.is_alive())
        self.assertEqual(controller.snapshot().state, ControllerState.STOPPED)

    def test_new_config_invalidates_calibration_and_iteration_state(self):
        controller, bench, _ = self.make_controller()
        self.calibrate_manually(controller)
        self.assertEqual(controller.snapshot().iteration, 0)
        self.assertTrue(bench.transmitting)

        snapshot = controller.apply_config(self.make_config(max_iterations=4))

        self.assertEqual(snapshot.state, ControllerState.READY)
        self.assertIsNone(snapshot.iteration)
        self.assertIsNone(snapshot.gain_correction)
        self.assertIsNone(snapshot.locked_attenuation_db)
        self.assertEqual(snapshot.records, ())
        self.assertFalse(snapshot.transmitting)
        self.assertFalse(bench.transmitting)

        replacement = self.x * 0.9
        reloaded = controller.load_reference(replacement)
        self.assertEqual(reloaded.state, ControllerState.READY)
        self.assertEqual(reloaded.records, ())
        self.assertIsNone(reloaded.gain_correction)
        np.testing.assert_array_equal(reloaded.x, replacement)

    def test_new_reference_restores_initial_attenuation_before_transmission(self):
        config = self.make_config(target_power_dbm=-5.0, max_adjustments=20)
        controller, bench, _ = self.make_controller(config=config)
        self.calibrate_manually(controller)
        self.assertEqual(controller.snapshot().locked_attenuation_db, 25.2)
        self.assertEqual(bench.attenuation_db, 25.2)

        controller.load_reference(self.x * 0.9)
        call_count = len(bench.calls)
        controller.start_reference_transmission()
        new_calls = bench.calls[call_count:]

        self.assertEqual(bench.attenuation_db, 30.0)
        self.assertEqual(
            [name for name, _ in new_calls[:3]],
            ["set_attenuation", "upload", "start"],
        )

    def test_all_controller_device_calls_use_configured_timeout(self):
        controller, bench, _ = self.make_controller()
        controller.run_auto()

        configured_calls = [
            timeout for name, timeout in bench.calls if name not in {"connect"}
        ]
        self.assertTrue(configured_calls)
        self.assertTrue(all(timeout == 2.5 for timeout in configured_calls))
        self.assertEqual(bench.calls[0], ("connect", 10.0))
        self.assertIsNone(controller.snapshot().active_operation)

    def test_stop_request_does_not_rewrite_a_terminal_state(self):
        controller, _, _ = self.make_controller()
        completed = controller.run_auto()

        after_stop = controller.request_stop()

        self.assertEqual(completed.state, ControllerState.COMPLETED)
        self.assertEqual(after_stop.state, ControllerState.COMPLETED)
        self.assertEqual(after_stop.completed_at, completed.completed_at)
        self.assertFalse(after_stop.stop_requested)

    def test_runtime_replacement_closes_both_instances_when_old_close_fails(self):
        config = self.make_config(
            runtime_name=CloseFailsOnceRuntime.name,
            runtime_config={},
        )
        bench = FakeRFBench(self.x.size * 2)
        controller = ClosedLoopController(
            bench,
            power_controller=PowerController(sleep_fn=lambda _: None),
        )
        controller.connect()
        controller.load_reference(self.x)
        controller.apply_config(config)
        old_runtime = CloseFailsOnceRuntime.instances[0]

        with self.assertRaisesRegex(RuntimeError, "close failure"):
            controller.load_reference(self.x * 0.9)

        replacement = CloseFailsOnceRuntime.instances[1]
        self.assertTrue(old_runtime.closed)
        self.assertEqual(old_runtime.close_attempts, 2)
        self.assertTrue(replacement.closed)
        self.assertEqual(controller.snapshot().state, ControllerState.FAILED)
        self.assertIsNone(controller._runtime)


if __name__ == "__main__":
    unittest.main()
