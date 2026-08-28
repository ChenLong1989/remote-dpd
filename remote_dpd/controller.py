"""Single-task orchestration for the device-driven DPD closed loop."""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Event, Lock, RLock
from types import MappingProxyType
from typing import Any

import numpy as np

from .device import CaptureRequest, DeviceConfig, RFBench
from .power_control import (
    PowerAdjustment,
    PowerControlCancelled,
    PowerController,
    PowerControlResult,
)
from .preprocessing import CaptureBatch, FeedbackPreprocessor, PreprocessingResult
from .runtime import DPDRuntime, RuntimeStepInput, create_runtime
from .safety import (
    DigitalSafetyReport,
    validate_candidate,
    validate_reference,
)

MAX_CAPTURE_WORKING_SAMPLES = 10_000_000
MAX_RETAINED_ROUND_SAMPLES = 20_000_000


class ControllerState(str, Enum):
    """Externally visible states of one closed-loop controller."""

    IDLE = "idle"
    READY = "ready"
    POWER_TUNING = "power_tuning"
    POWER_READY = "power_ready"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class ControllerError(RuntimeError):
    """Base error raised by the application controller."""


class ControllerStateError(ControllerError):
    """A command is not valid in the controller's current state."""


class ControllerBusyError(ControllerError):
    """Another mutating controller command currently owns the operation lock."""


class ControllerStoppedError(ControllerError):
    """An operation ended because a manual stop was requested."""


@dataclass(frozen=True, slots=True)
class ClosedLoopConfig:
    """Immutable device, runtime, and iteration configuration."""

    device_config: DeviceConfig
    runtime_name: str = "basic_ilc"
    runtime_config: Mapping[str, Any] = field(default_factory=lambda: {"mu": 0.5})
    max_iterations: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.device_config, DeviceConfig):
            raise TypeError("device_config must be a DeviceConfig")
        if not isinstance(self.runtime_name, str):
            raise TypeError("runtime_name must be a string")
        runtime_name = self.runtime_name.strip().lower()
        if not runtime_name:
            raise ValueError("runtime_name must not be empty")
        if isinstance(self.max_iterations, (bool, np.bool_)) or not isinstance(
            self.max_iterations, numbers.Integral
        ):
            raise TypeError("max_iterations must be an integer")
        max_iterations = int(self.max_iterations)
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero")
        runtime_config = _freeze_mapping(self.runtime_config, "runtime_config")

        object.__setattr__(self, "runtime_name", runtime_name)
        object.__setattr__(self, "runtime_config", runtime_config)
        object.__setattr__(self, "max_iterations", max_iterations)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return {
            "device_config": self.device_config.to_dict(),
            "runtime_name": self.runtime_name,
            "runtime_config": _json_config_value(self.runtime_config),
            "max_iterations": self.max_iterations,
        }


@dataclass(frozen=True, slots=True)
class ControllerErrorInfo:
    """Structured details retained after a controller operation fails."""

    operation: str
    code: str
    exception_type: str
    message: str
    shutdown_error: str | None = None


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """One fully transmitted, measured, captured, and evaluated round."""

    iteration: int
    y: np.ndarray = field(repr=False)
    z: np.ndarray = field(repr=False)
    power_dbm: float
    attenuation_db: float
    digital_safety: DigitalSafetyReport
    preprocessing: PreprocessingResult
    runtime_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.iteration, (bool, np.bool_)) or not isinstance(
            self.iteration, numbers.Integral
        ):
            raise TypeError("iteration must be a non-negative integer")
        iteration = int(self.iteration)
        if iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        y = _readonly_signal(self.y, "y")
        z = _readonly_signal(self.z, "z")
        if y.size != z.size:
            raise ValueError("y and z must have the same length")
        power_dbm = _finite_real(self.power_dbm, "power_dbm")
        attenuation_db = _finite_real(self.attenuation_db, "attenuation_db")
        if not isinstance(self.digital_safety, DigitalSafetyReport):
            raise TypeError("digital_safety must be a DigitalSafetyReport")
        if not isinstance(self.preprocessing, PreprocessingResult):
            raise TypeError("preprocessing must be a PreprocessingResult")
        if self.preprocessing.z.size != z.size or not np.array_equal(
            self.preprocessing.z, z
        ):
            raise ValueError("preprocessing.z must equal z")

        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "power_dbm", power_dbm)
        object.__setattr__(self, "attenuation_db", attenuation_db)
        object.__setattr__(
            self,
            "runtime_metrics",
            _freeze_mapping(self.runtime_metrics, "runtime_metrics"),
        )


@dataclass(frozen=True, slots=True)
class ControllerSnapshot:
    """Thread-safe immutable view of controller state and committed results."""

    state: ControllerState
    connected: bool
    configured: bool
    reference_loaded: bool
    transmitting: bool
    stop_requested: bool
    active_operation: str | None
    iteration: int | None
    max_iterations: int | None
    gain_correction: float | None
    locked_attenuation_db: float | None
    latest_power_dbm: float | None
    config: ClosedLoopConfig | None
    device_type: str
    completed_at: str | None
    reference_safety: DigitalSafetyReport | None = None
    x: np.ndarray | None = field(default=None, repr=False)
    records: tuple[IterationRecord, ...] = ()
    power_trace: tuple[PowerAdjustment, ...] = ()
    last_error: ControllerErrorInfo | None = None

    @property
    def current_record(self) -> IterationRecord | None:
        """Return the latest fully committed round, if one exists."""
        return self.records[-1] if self.records else None

    @property
    def y(self) -> np.ndarray | None:
        """Return the latest fully evaluated transmit waveform."""
        record = self.current_record
        return None if record is None else record.y

    @property
    def z(self) -> np.ndarray | None:
        """Return the latest fully evaluated preprocessed feedback."""
        record = self.current_record
        return None if record is None else record.z


class _StopRequested(Exception):
    """Internal control-flow signal used at device-call boundaries."""


class ClosedLoopController:
    """Serialize commands and own the complete device-driven DPD state machine."""

    _ACTIVE_STATES = frozenset(
        {
            ControllerState.POWER_TUNING,
            ControllerState.CALIBRATING,
            ControllerState.RUNNING,
            ControllerState.STOPPING,
        }
    )

    def __init__(
        self,
        bench: RFBench,
        *,
        power_controller: PowerController | None = None,
        default_timeout_seconds: float = 10.0,
    ) -> None:
        if not isinstance(bench, RFBench):
            raise TypeError("bench must be an RFBench")
        if power_controller is not None and not isinstance(
            power_controller, PowerController
        ):
            raise TypeError("power_controller must be a PowerController or None")
        timeout = _finite_real(default_timeout_seconds, "default_timeout_seconds")
        if timeout <= 0.0:
            raise ValueError("default_timeout_seconds must be greater than zero")

        self._bench = bench
        self._power_controller = power_controller or PowerController()
        self._default_timeout_seconds = timeout
        self._operation_lock = Lock()
        self._state_lock = RLock()
        self._stop_event = Event()

        self._state = ControllerState.IDLE
        self._connected = False
        self._configured = False
        self._transmitting = False
        self._active_operation: str | None = None
        self._config: ClosedLoopConfig | None = None
        self._x: np.ndarray | None = None
        self._reference_safety: DigitalSafetyReport | None = None
        self._runtime: DPDRuntime | None = None
        self._preprocessor: FeedbackPreprocessor | None = None
        self._power_result: PowerControlResult | None = None
        self._power_trace: tuple[PowerAdjustment, ...] = ()
        self._latest_power_dbm: float | None = None
        self._gain_correction: float | None = None
        self._records: list[IterationRecord] = []
        self._tx_iteration: int | None = None
        self._last_error: ControllerErrorInfo | None = None
        self._completed_at: str | None = None

    @property
    def bench(self) -> RFBench:
        """Return the controller-owned bench adapter."""
        return self._bench

    def connect(self, timeout_seconds: float | None = None) -> ControllerSnapshot:
        """Connect the bench without implicitly applying configuration."""
        with self._exclusive("connect"):
            if self._connected:
                raise ControllerStateError("the RF bench is already connected")
            if self._state in self._ACTIVE_STATES:
                raise ControllerStateError(
                    "connect is not allowed while a task is active"
                )
            timeout = self._resolve_timeout(timeout_seconds)
            try:
                self._check_stop()
                self._bench.connect(timeout_seconds=timeout)
                self._connected = True
                self._check_stop()
                self._last_error = None
                self._refresh_idle_or_ready()
            except _StopRequested as exc:
                self._finish_stopped("connect")
                raise ControllerStoppedError("connect was stopped") from exc
            except Exception as exc:
                self._finish_failed("connect", exc)
                raise
        return self.snapshot()

    def disconnect(self, timeout_seconds: float | None = None) -> ControllerSnapshot:
        """Safely stop and disconnect the bench, invalidating applied state."""
        with self._exclusive("disconnect"):
            if not self._connected:
                raise ControllerStateError("the RF bench is not connected")
            timeout = self._resolve_timeout(timeout_seconds)
            try:
                self._safe_shutdown_or_raise(timeout)
                self._bench.disconnect(timeout_seconds=timeout)
                self._connected = False
                self._configured = False
                self._close_runtime_or_raise()
                self._config = None
                self._clear_run_state()
                self._stop_event.clear()
                self._last_error = None
                self._state = ControllerState.IDLE
            except Exception as exc:
                self._finish_failed("disconnect", exc)
                raise
        return self.snapshot()

    def apply_config(self, config: ClosedLoopConfig) -> ControllerSnapshot:
        """Apply a complete configuration and invalidate prior calibration."""
        if not isinstance(config, ClosedLoopConfig):
            raise TypeError("config must be a ClosedLoopConfig")
        with self._exclusive("apply_config"):
            self._require_modifiable("apply_config")
            if not self._connected:
                raise ControllerStateError("connect the RF bench before configuration")

            candidate_runtime: DPDRuntime | None = None
            try:
                self._check_stop()
                effective_config = self._effective_config(config)
                timeout = effective_config.device_config.call_timeout_seconds
                candidate_runtime = create_runtime(effective_config.runtime_name)
                candidate_runtime.initialize(effective_config.runtime_config)
                self._stop_tx_if_needed(timeout)
                self._check_stop()
                self._bench.configure(
                    effective_config.device_config,
                    timeout_seconds=timeout,
                )
                self._check_stop()
                self._validate_capture_capacity_for(effective_config, self._x)

                self._close_runtime_or_raise()
                self._runtime = candidate_runtime
                candidate_runtime = None
                self._config = effective_config
                self._configured = True
                self._clear_run_state()
                self._stop_event.clear()
                self._last_error = None
                self._refresh_idle_or_ready()
            except _StopRequested as exc:
                if candidate_runtime is not None:
                    self._close_detached_runtime(candidate_runtime)
                self._finish_stopped("apply_config")
                raise ControllerStoppedError("configuration was stopped") from exc
            except Exception as exc:
                if candidate_runtime is not None:
                    self._close_detached_runtime(candidate_runtime)
                self._finish_failed("apply_config", exc)
                raise
        return self.snapshot()

    def load_reference(self, reference: np.ndarray) -> ControllerSnapshot:
        """Validate and defensively store the unmodified periodic reference."""
        with self._exclusive("load_reference"):
            self._require_modifiable("load_reference")
            try:
                safety_report = validate_reference(reference)
                x = _readonly_signal(reference, "reference")
                timeout = self._current_timeout()
                self._stop_tx_if_needed(timeout)
                self._check_stop()
                self._validate_capture_capacity_for(self._config, x)
                self._reinitialize_runtime_or_raise()

                self._x = x
                self._clear_run_state()
                self._reference_safety = safety_report
                self._stop_event.clear()
                self._last_error = None
                self._refresh_idle_or_ready()
            except _StopRequested as exc:
                self._finish_stopped("load_reference")
                raise ControllerStoppedError("reference loading was stopped") from exc
            except Exception as exc:
                self._finish_failed("load_reference", exc)
                raise
        return self.snapshot()

    def start_reference_transmission(self) -> ControllerSnapshot:
        """Upload the exact reference and begin cyclic transmission."""
        with self._exclusive("start_reference_transmission"):
            self._require_state(ControllerState.READY, ControllerState.POWER_READY)
            preserve_power_ready = self._state is ControllerState.POWER_READY
            try:
                self._start_reference_internal(
                    preserve_power_ready=preserve_power_ready
                )
            except _StopRequested as exc:
                self._finish_stopped("start_reference_transmission")
                raise ControllerStoppedError(
                    "reference transmission was stopped"
                ) from exc
            except Exception as exc:
                self._finish_failed("start_reference_transmission", exc)
                raise
        return self.snapshot()

    def stop_transmission(self) -> ControllerSnapshot:
        """Stop RF output without discarding a valid calibration."""
        with self._exclusive("stop_transmission"):
            if not self._connected:
                raise ControllerStateError("the RF bench is not connected")
            if self._state in self._ACTIVE_STATES:
                raise ControllerStateError(
                    "use request_stop while a controller operation is active"
                )
            try:
                self._stop_tx_if_needed(self._current_timeout())
            except Exception as exc:
                self._finish_failed("stop_transmission", exc)
                raise
        return self.snapshot()

    def tune_power(self) -> PowerControlResult:
        """Tune initial attenuation while the exact reference is transmitting."""
        with self._exclusive("tune_power"):
            self._require_state(ControllerState.READY)
            if not self._transmitting or self._tx_iteration != 0:
                raise ControllerStateError(
                    "power tuning requires the reference waveform to be transmitting"
                )
            try:
                return self._tune_power_internal()
            except (PowerControlCancelled, _StopRequested) as exc:
                self._retain_power_trace(exc)
                self._finish_stopped("tune_power")
                raise ControllerStoppedError("power tuning was stopped") from exc
            except Exception as exc:
                self._finish_failed("tune_power", exc)
                raise

    def calibrate(self) -> IterationRecord:
        """Capture round zero and establish the fixed amplitude correction."""
        with self._exclusive("calibrate"):
            self._require_state(ControllerState.POWER_READY)
            try:
                return self._calibrate_internal()
            except (PowerControlCancelled, _StopRequested) as exc:
                self._finish_stopped("calibrate")
                raise ControllerStoppedError("calibration was stopped") from exc
            except Exception as exc:
                self._finish_failed("calibrate", exc)
                raise

    def step(self) -> IterationRecord:
        """Execute and atomically commit one complete ILC iteration."""
        with self._exclusive("step"):
            self._require_state(ControllerState.CALIBRATED)
            try:
                return self._step_internal()
            except (PowerControlCancelled, _StopRequested) as exc:
                self._retain_power_trace(exc)
                self._finish_stopped("step")
                raise ControllerStoppedError("iteration was stopped") from exc
            except Exception as exc:
                self._finish_failed("step", exc)
                raise

    def run_auto(self) -> ControllerSnapshot:
        """Fill missing safe prerequisites and run through max_iterations."""
        with self._exclusive("run_auto"):
            if self._state not in {
                ControllerState.READY,
                ControllerState.POWER_READY,
                ControllerState.CALIBRATED,
            }:
                raise ControllerStateError(
                    f"run_auto is not allowed in state {self._state.value!r}"
                )
            try:
                if self._state is ControllerState.READY:
                    if not self._transmitting or self._tx_iteration != 0:
                        self._start_reference_internal()
                    self._tune_power_internal()
                if self._state is ControllerState.POWER_READY:
                    if not self._transmitting or self._tx_iteration != 0:
                        self._start_reference_internal(preserve_power_ready=True)
                    self._calibrate_internal()
                while self._state is ControllerState.CALIBRATED:
                    self._step_internal()
            except (PowerControlCancelled, _StopRequested) as exc:
                self._retain_power_trace(exc)
                self._finish_stopped("run_auto")
            except Exception as exc:
                self._finish_failed("run_auto", exc)
                raise
        return self.snapshot()

    def request_stop(self) -> ControllerSnapshot:
        """Request cancellation without competing with an in-flight device call."""
        with self._state_lock:
            if self._active_operation is None and self._state in {
                ControllerState.COMPLETED,
                ControllerState.STOPPED,
                ControllerState.FAILED,
            }:
                return self.snapshot()
            self._stop_event.set()
            if self._active_operation is not None:
                self._state = ControllerState.STOPPING

        if self._operation_lock.acquire(blocking=False):
            try:
                with self._state_lock:
                    self._active_operation = "request_stop"
                self._finish_stopped("request_stop")
            finally:
                with self._state_lock:
                    self._active_operation = None
                    self._operation_lock.release()
        return self.snapshot()

    def reset(self) -> ControllerSnapshot:
        """Safely clear task configuration and results while keeping the connection."""
        with self._exclusive("reset"):
            timeout = self._current_timeout()
            try:
                if self._connected:
                    self._safe_shutdown_or_raise(timeout)
                self._close_runtime_or_raise()
                self._config = None
                self._configured = False
                self._x = None
                self._reference_safety = None
                self._clear_run_state()
                self._stop_event.clear()
                self._last_error = None
                self._state = ControllerState.IDLE
            except Exception as exc:
                self._finish_failed("reset", exc)
                raise
        return self.snapshot()

    def snapshot(self) -> ControllerSnapshot:
        """Return a consistent immutable snapshot without taking the operation lock."""
        with self._state_lock:
            power_trace = self._power_trace
            current_record = self._records[-1] if self._records else None
            return ControllerSnapshot(
                state=self._state,
                connected=self._connected,
                configured=self._configured,
                reference_loaded=self._x is not None,
                transmitting=self._transmitting,
                stop_requested=self._stop_event.is_set(),
                active_operation=self._active_operation,
                iteration=None if current_record is None else current_record.iteration,
                max_iterations=(
                    None if self._config is None else self._config.max_iterations
                ),
                gain_correction=self._gain_correction,
                locked_attenuation_db=(
                    None
                    if self._power_result is None
                    else self._power_result.attenuation_db
                ),
                latest_power_dbm=(self._latest_power_dbm),
                config=self._config,
                device_type=self._bench.parameter_schema.device_type,
                completed_at=self._completed_at,
                reference_safety=self._reference_safety,
                x=self._x,
                records=tuple(self._records),
                power_trace=power_trace,
                last_error=self._last_error,
            )

    @contextmanager
    def _exclusive(self, operation: str) -> Iterator[None]:
        if not self._operation_lock.acquire(blocking=False):
            raise ControllerBusyError(
                f"controller is busy with {self._active_operation or 'another operation'}"
            )
        with self._state_lock:
            self._active_operation = operation
        try:
            yield
        finally:
            cleanup_required = False
            with self._state_lock:
                if self._stop_event.is_set() and self._state not in {
                    ControllerState.STOPPED,
                    ControllerState.FAILED,
                }:
                    self._state = ControllerState.STOPPING
                    cleanup_required = True
                else:
                    self._active_operation = None
                    self._operation_lock.release()
            if cleanup_required:
                self._finish_stopped(operation)
                with self._state_lock:
                    self._active_operation = None
                    self._operation_lock.release()

    def _start_reference_internal(self, *, preserve_power_ready: bool = False) -> None:
        config, x = self._require_ready_components()
        timeout = config.device_config.call_timeout_seconds
        self._check_stop()
        validate_reference(x)
        if not preserve_power_ready:
            self._stop_tx_if_needed(timeout)
            self._check_stop()
            self._bench.transmitter.set_attenuation_db(
                config.device_config.initial_attenuation_db,
                timeout_seconds=timeout,
            )
            self._check_stop()
        self._switch_waveform(x, iteration=0, timeout=timeout)
        self._check_stop()
        self._state = (
            ControllerState.POWER_READY
            if preserve_power_ready
            else ControllerState.READY
        )

    def _tune_power_internal(self) -> PowerControlResult:
        config, _ = self._require_ready_components()
        self._state = ControllerState.POWER_TUNING
        self._check_stop()
        result = self._power_controller.tune(
            self._bench.transmitter,
            self._bench.power_sensor,
            config.device_config,
            cancel_requested=self._stop_event.is_set,
        )
        self._check_stop()
        self._power_result = result
        self._power_trace = result.trace
        self._latest_power_dbm = result.power_dbm
        self._state = ControllerState.POWER_READY
        return result

    def _calibrate_internal(self) -> IterationRecord:
        config, x = self._require_ready_components()
        if not self._transmitting or self._tx_iteration != 0:
            raise ControllerStateError(
                "calibration requires the power-tuned reference to be transmitting"
            )
        if self._power_result is None:
            raise ControllerStateError("calibration requires a completed power tune")

        self._state = ControllerState.CALIBRATING
        self._check_stop()
        power_dbm = self._power_controller.monitor(
            self._bench.power_sensor,
            config.device_config,
            cancel_requested=self._stop_event.is_set,
        )
        self._latest_power_dbm = power_dbm
        self._check_stop()
        batches = self._capture_batches(config, x.size)
        self._check_stop()
        preprocessor = FeedbackPreprocessor(x, config.device_config.sample_rate_hz)
        result = preprocessor.process(batches, gain_correction=None)
        self._check_stop()
        safety = validate_reference(x)
        record = IterationRecord(
            iteration=0,
            y=x,
            z=result.z,
            power_dbm=power_dbm,
            attenuation_db=self._power_result.attenuation_db,
            digital_safety=safety,
            preprocessing=result,
            runtime_metrics={},
        )

        self._preprocessor = preprocessor
        self._gain_correction = result.gain_correction
        self._records = [record]
        self._state = ControllerState.CALIBRATED
        return record

    def _step_internal(self) -> IterationRecord:
        config, x = self._require_ready_components()
        if self._runtime is None or self._preprocessor is None:
            raise ControllerStateError(
                "calibration and runtime initialization are required"
            )
        if self._gain_correction is None or not self._records:
            raise ControllerStateError("fixed gain calibration is unavailable")
        current = self._records[-1]
        next_iteration = current.iteration + 1
        if next_iteration > config.max_iterations:
            raise ControllerStateError("max_iterations has already been reached")

        self._state = ControllerState.RUNNING
        self._check_stop()
        runtime_result = self._runtime.step(
            RuntimeStepInput(
                x=x,
                y_current=current.y,
                z_current=current.z,
                iteration=next_iteration,
                config=config.runtime_config,
            )
        )
        self._check_stop()

        # Safety is intentionally evaluated before stopping the current valid signal.
        safety = validate_candidate(x, runtime_result.y_candidate)
        candidate = _readonly_signal(runtime_result.y_candidate, "y_candidate")
        self._check_stop()
        self._switch_waveform(
            candidate,
            iteration=next_iteration,
            timeout=config.device_config.call_timeout_seconds,
        )
        self._check_stop()
        power_dbm = self._power_controller.monitor(
            self._bench.power_sensor,
            config.device_config,
            cancel_requested=self._stop_event.is_set,
        )
        self._latest_power_dbm = power_dbm
        self._check_stop()
        batches = self._capture_batches(config, x.size)
        self._check_stop()
        preprocessing = self._preprocessor.process(
            batches,
            gain_correction=self._gain_correction,
        )
        self._check_stop()

        record = IterationRecord(
            iteration=next_iteration,
            y=candidate,
            z=preprocessing.z,
            power_dbm=power_dbm,
            attenuation_db=self._power_result.attenuation_db,
            digital_safety=safety,
            preprocessing=preprocessing,
            runtime_metrics=runtime_result.metrics,
        )
        self._records.append(record)

        if next_iteration == config.max_iterations:
            self._safe_shutdown_or_raise(config.device_config.call_timeout_seconds)
            self._check_stop()
            self._close_runtime_or_raise()
            self._check_stop()
            self._set_terminal_state(ControllerState.COMPLETED)
        else:
            self._state = ControllerState.CALIBRATED
        return record

    def _capture_batches(
        self,
        config: ClosedLoopConfig,
        segment_length: int,
    ) -> tuple[CaptureBatch, ...]:
        max_samples = self._validate_max_capture_samples(
            self._bench.receiver.max_capture_samples
        )
        segments_per_batch = max_samples // segment_length
        if segments_per_batch < 1:
            raise ValueError(
                "receiver.max_capture_samples cannot hold one complete reference period"
            )

        remaining = config.device_config.average_segment_count
        batches: list[CaptureBatch] = []
        timeout = config.device_config.call_timeout_seconds
        while remaining:
            self._check_stop()
            segment_count = min(segments_per_batch, remaining)
            request = CaptureRequest(
                segment_length=segment_length,
                segment_count=segment_count,
            )
            batch = self._bench.receiver.capture(
                request,
                timeout_seconds=timeout,
            )
            self._check_stop()
            if not isinstance(batch, CaptureBatch):
                raise TypeError("receiver.capture must return a CaptureBatch")
            if batch.segment_length != request.segment_length:
                raise ValueError("receiver returned an unexpected segment length")
            if batch.segment_count != request.segment_count:
                raise ValueError("receiver returned an unexpected segment count")
            batches.append(batch)
            remaining -= segment_count
        return tuple(batches)

    def _switch_waveform(
        self,
        waveform: np.ndarray,
        *,
        iteration: int,
        timeout: float,
    ) -> None:
        self._stop_tx_if_needed(timeout)
        self._check_stop()
        self._bench.transmitter.upload_waveform(
            waveform,
            timeout_seconds=timeout,
        )
        self._check_stop()
        self._bench.transmitter.start_transmission(timeout_seconds=timeout)
        self._transmitting = True
        self._tx_iteration = iteration

    def _stop_tx_if_needed(self, timeout: float) -> None:
        if not self._transmitting:
            return
        self._bench.transmitter.stop_transmission(timeout_seconds=timeout)
        self._transmitting = False
        self._tx_iteration = None

    def _safe_shutdown_or_raise(self, timeout: float) -> None:
        self._bench.safe_shutdown(timeout_seconds=timeout)
        self._transmitting = False
        self._tx_iteration = None

    def _finish_stopped(self, operation: str) -> None:
        self._state = ControllerState.STOPPING
        shutdown_error = self._cleanup_after_terminal()
        if shutdown_error is not None:
            error = RuntimeError(shutdown_error)
            self._record_failure(operation, error, shutdown_error=shutdown_error)
            self._set_terminal_state(ControllerState.FAILED)
            return
        self._last_error = None
        self._set_terminal_state(ControllerState.STOPPED)

    def _finish_failed(self, operation: str, exc: Exception) -> None:
        self._retain_power_trace(exc)
        shutdown_error = self._cleanup_after_terminal()
        self._record_failure(operation, exc, shutdown_error=shutdown_error)
        self._set_terminal_state(ControllerState.FAILED)

    def _retain_power_trace(self, exc: Exception) -> None:
        measured_power_dbm = getattr(exc, "measured_power_dbm", None)
        if isinstance(measured_power_dbm, numbers.Real) and math.isfinite(
            float(measured_power_dbm)
        ):
            self._latest_power_dbm = float(measured_power_dbm)
        trace = getattr(exc, "trace", None)
        if isinstance(trace, tuple) and all(
            isinstance(item, PowerAdjustment) for item in trace
        ):
            self._power_trace = trace
            if trace and self._latest_power_dbm is None:
                self._latest_power_dbm = trace[-1].power_dbm

    def _cleanup_after_terminal(self) -> str | None:
        errors: list[str] = []
        if self._connected:
            try:
                self._safe_shutdown_or_raise(self._current_timeout())
            except Exception as exc:  # noqa: BLE001 - aggregate cleanup errors
                errors.append(f"safe shutdown failed: {exc}")
        try:
            self._close_runtime_or_raise()
        except Exception as exc:  # noqa: BLE001 - aggregate terminal cleanup errors
            errors.append(f"runtime close failed: {exc}")
        return "; ".join(errors) or None

    def _record_failure(
        self,
        operation: str,
        exc: Exception,
        *,
        shutdown_error: str | None,
    ) -> None:
        raw_code = getattr(exc, "code", None)
        if isinstance(raw_code, Enum):
            raw_code = raw_code.value
        code = (
            str(raw_code)
            if raw_code is not None
            else _exception_code(type(exc).__name__)
        )
        self._last_error = ControllerErrorInfo(
            operation=operation,
            code=code,
            exception_type=type(exc).__name__,
            message=str(exc),
            shutdown_error=shutdown_error,
        )

    def _clear_run_state(self) -> None:
        self._preprocessor = None
        self._power_result = None
        self._power_trace = ()
        self._latest_power_dbm = None
        self._gain_correction = None
        self._records = []
        self._tx_iteration = None
        self._completed_at = None

    def _set_terminal_state(self, state: ControllerState) -> None:
        if state not in {
            ControllerState.COMPLETED,
            ControllerState.STOPPED,
            ControllerState.FAILED,
        }:
            raise ValueError("state must be terminal")
        self._state = state
        self._completed_at = _utc_timestamp()

    def _reinitialize_runtime_or_raise(self) -> None:
        if self._config is None:
            return
        replacement = create_runtime(self._config.runtime_name)
        try:
            replacement.initialize(self._config.runtime_config)
        except Exception:
            self._close_detached_runtime(replacement)
            raise
        try:
            self._close_runtime_or_raise()
        except Exception:
            self._close_detached_runtime(replacement)
            raise
        self._runtime = replacement

    def _close_runtime_or_raise(self) -> None:
        runtime = self._runtime
        if runtime is not None:
            runtime.close()
            self._runtime = None

    @staticmethod
    def _close_detached_runtime(runtime: DPDRuntime) -> None:
        try:
            runtime.close()
        except Exception:  # noqa: BLE001 - cleanup must not mask the primary failure
            return

    def _refresh_idle_or_ready(self) -> None:
        self._state = (
            ControllerState.READY
            if self._connected and self._configured and self._x is not None
            else ControllerState.IDLE
        )
        self._completed_at = None

    def _require_ready_components(self) -> tuple[ClosedLoopConfig, np.ndarray]:
        if not self._connected or not self._configured or self._config is None:
            raise ControllerStateError("the RF bench is not connected and configured")
        if self._x is None:
            raise ControllerStateError("a reference waveform has not been loaded")
        return self._config, self._x

    def _require_state(self, *allowed: ControllerState) -> None:
        if self._state not in allowed:
            names = ", ".join(state.value for state in allowed)
            raise ControllerStateError(
                f"command requires state {names}; current state is {self._state.value}"
            )

    def _require_modifiable(self, operation: str) -> None:
        if self._state in self._ACTIVE_STATES:
            raise ControllerStateError(
                f"{operation} is not allowed while a task is active"
            )

    def _validate_capture_capacity_for(
        self,
        config: ClosedLoopConfig | None,
        reference: np.ndarray | None,
    ) -> None:
        if config is None or reference is None:
            return
        max_samples = self._validate_max_capture_samples(
            self._bench.receiver.max_capture_samples
        )
        if max_samples < reference.size:
            raise ValueError(
                "receiver.max_capture_samples cannot hold one complete reference period"
            )
        capture_samples = (
            int(reference.size) * config.device_config.average_segment_count
        )
        if capture_samples > MAX_CAPTURE_WORKING_SAMPLES:
            raise ValueError(
                "reference length times average_segment_count exceeds the "
                f"capture working limit of {MAX_CAPTURE_WORKING_SAMPLES} samples"
            )
        retained_samples = int(reference.size) * (config.max_iterations + 1)
        if retained_samples > MAX_RETAINED_ROUND_SAMPLES:
            raise ValueError(
                "reference length times recorded round count exceeds the "
                f"retained-history limit of {MAX_RETAINED_ROUND_SAMPLES} samples"
            )

    def _effective_config(self, config: ClosedLoopConfig) -> ClosedLoopConfig:
        options = self._bench.parameter_schema.validate_options(
            config.device_config.device_options
        )
        device_values = config.device_config.to_dict()
        device_values["device_options"] = options
        return ClosedLoopConfig(
            device_config=DeviceConfig(**device_values),
            runtime_name=config.runtime_name,
            runtime_config=config.runtime_config,
            max_iterations=config.max_iterations,
        )

    @staticmethod
    def _validate_max_capture_samples(value: object) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral
        ):
            raise TypeError("receiver.max_capture_samples must be an integer")
        result = int(value)
        if result <= 0:
            raise ValueError("receiver.max_capture_samples must be positive")
        return result

    def _check_stop(self) -> None:
        if self._stop_event.is_set():
            raise _StopRequested

    def _resolve_timeout(self, value: float | None) -> float:
        if value is None:
            return self._current_timeout()
        timeout = _finite_real(value, "timeout_seconds")
        if timeout <= 0.0:
            raise ValueError("timeout_seconds must be greater than zero")
        return timeout

    def _current_timeout(self) -> float:
        if self._config is None:
            return self._default_timeout_seconds
        return self._config.device_config.call_timeout_seconds


def _readonly_signal(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain numeric samples")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.array(array, dtype=np.complex128, order="C", copy=True)
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{name} must contain only finite samples")
    return np.frombuffer(copied.tobytes(), dtype=np.complex128)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _freeze_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        result[key] = _freeze_value(item, f"{name}.{key}")
    return MappingProxyType(result)


def _freeze_value(value: object, name: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return _finite_real(value, name)
    if isinstance(value, numbers.Complex):
        result = complex(value)
        if not math.isfinite(result.real) or not math.isfinite(result.imag):
            raise ValueError(f"{name} must be finite")
        return result
    if isinstance(value, Mapping):
        return _freeze_mapping(value, name)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{name} arrays must not contain Python objects")
        if value.dtype.kind not in "biufcU":
            raise TypeError(
                f"{name} arrays must use boolean, numeric, complex, or Unicode dtype"
            )
        if (value.dtype.kind == "f" and value.dtype.itemsize > 8) or (
            value.dtype.kind == "c" and value.dtype.itemsize > 16
        ):
            raise TypeError(
                f"{name} arrays must use JSON-compatible numeric dtype precision"
            )
        copied = np.array(value, order="C", copy=True)
        if np.issubdtype(copied.dtype, np.number) and not np.all(np.isfinite(copied)):
            raise ValueError(f"{name} arrays must be finite")
        return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_value(item, f"{name}[{index}]") for index, item in enumerate(value)
        )
    raise TypeError(f"unsupported value type {type(value).__name__!r} in {name}")


def _json_config_value(value: object) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return {
            "$type": "complex",
            "real": value.real,
            "imag": value.imag,
        }
    if isinstance(value, Mapping):
        return {str(key): _json_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "$type": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": _json_config_value(value.tolist()),
        }
    raise TypeError(f"unsupported frozen config value {type(value).__name__!r}")


def _exception_code(name: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return words.removesuffix("_error") or "controller_error"


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
