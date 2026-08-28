"""Versioned MAT-file command interface for the device-driven controller."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, RLock, Thread, current_thread
from typing import TYPE_CHECKING, Any

import numpy as np

from .controller import (
    ClosedLoopConfig,
    ClosedLoopController,
    ControllerSnapshot,
    ControllerState,
)
from .device import DeviceConfig, create_rf_bench

if TYPE_CHECKING:
    from .storage import RunRecorder, RunStore


FILE_COMMAND_SCHEMA_VERSION = 1
COMMAND_ACTIONS = frozenset(
    {
        "load",
        "configure",
        "power_tune",
        "calibrate",
        "step",
        "run",
        "stop",
        "reset",
        "export",
    }
)
COMMAND_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
COMMAND_FILE_PATTERN = re.compile(r"^command_([A-Za-z0-9][A-Za-z0-9_-]{0,63})\.mat$")
_DEVICE_CONFIG_FIELDS = frozenset(field.name for field in fields(DeviceConfig))
_CONFIG_FIELDS = frozenset(
    {
        "device_type",
        "device_config",
        "runtime_name",
        "runtime_config",
        "max_iterations",
    }
)
_COMMAND_FIELDS = frozenset(
    {"schema_version", "command_id", "action", "x", "config_json"}
)
_MAX_ARRAY_ELEMENTS = 100_000_000
_LOG = logging.getLogger(__name__)


class FileCommandError(ValueError):
    """A command file is invalid or cannot be executed safely."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ParsedConfiguration:
    """A validated device selector and controller configuration."""

    device_type: str
    closed_loop: ClosedLoopConfig


@dataclass(frozen=True, slots=True)
class FileCommand:
    """One strictly parsed command payload."""

    command_id: str
    action: str
    x: np.ndarray | None = None
    configuration: ParsedConfiguration | None = None


@dataclass(frozen=True, slots=True)
class CommandStatus:
    """Persistent status representation shared with MAT-file consumers."""

    command_id: str
    accepted: bool
    state: str
    iteration: int
    message: str
    error_code: str
    timestamp: str
    run_id: str = ""
    schema_version: int = FILE_COMMAND_SCHEMA_VERSION

    def to_mat(self) -> dict[str, Any]:
        """Return scalar values suitable for ``scipy.io.savemat``."""

        return {
            "schema_version": np.int64(self.schema_version),
            "command_id": self.command_id,
            "accepted": np.uint8(self.accepted),
            "state": self.state,
            "iteration": np.int64(self.iteration),
            "message": self.message,
            "error_code": self.error_code,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class CommandExecution:
    """Internal result returned after one accepted command finishes."""

    snapshot: ControllerSnapshot | None
    state: str
    iteration: int
    message: str
    result_path: Path | None = None


ControllerFactory = Callable[[str], ClosedLoopController]


def _default_controller_factory(device_type: str) -> ClosedLoopController:
    return ClosedLoopController(create_rf_bench(device_type))


class FileCommandProcessor:
    """Own one controller lifecycle and execute already parsed commands."""

    def __init__(
        self,
        *,
        controller_factory: ControllerFactory | None = None,
        run_store: RunStore | Any | None = None,
    ) -> None:
        self._controller_factory = controller_factory or _default_controller_factory
        self._run_store = run_store
        self._lock = RLock()
        self._stop_requested = Event()
        self._controller: ClosedLoopController | None = None
        self._pending_controller: ClosedLoopController | None = None
        self._device_type: str | None = None
        self._configuration: ClosedLoopConfig | None = None
        self._x: np.ndarray | None = None
        self._recorder: RunRecorder | Any | None = None
        self._run_id: str | None = None

    @property
    def controller(self) -> ClosedLoopController | None:
        """Return the current controller without transferring ownership."""

        with self._lock:
            return self._controller

    @property
    def run_id(self) -> str | None:
        """Return the active temporary-run identifier, if storage is enabled."""

        with self._lock:
            return self._run_id

    @property
    def run_store(self) -> RunStore | Any | None:
        """Return the configured run store for durable recovery."""

        return self._run_store

    def snapshot(self) -> ControllerSnapshot | None:
        """Return the current controller snapshot, if configured."""

        controller = self.controller
        return None if controller is None else controller.snapshot()

    def execute(self, command: FileCommand, result_path: Path) -> CommandExecution:
        """Execute one non-stop command against the shared controller."""

        if command.action == "stop":
            raise ValueError("stop must be dispatched through request_stop")
        self._raise_if_stop_requested()

        if command.action == "load":
            assert command.x is not None
            self._load_reference(command.x)
            self._raise_if_stop_requested()
            self._ensure_recorder(command.command_id)
            snapshot = self.snapshot()
            self.record_snapshot(snapshot)
            return self._execution(snapshot, "reference loaded")

        if command.action == "configure":
            assert command.configuration is not None
            self._apply_configuration(command.configuration)
            self._raise_if_stop_requested()
            self._ensure_recorder(command.command_id)
            snapshot = self.snapshot()
            self.record_snapshot(snapshot)
            return self._execution(snapshot, "configuration applied")

        if command.action == "run":
            if command.configuration is not None:
                self._apply_configuration(command.configuration)
                self._raise_if_stop_requested()
            if command.x is not None:
                self._load_reference(command.x)
                self._raise_if_stop_requested()
            controller = self._require_controller()
            self._require_ready_inputs()
            snapshot = controller.snapshot()
            self._start_run_recorder(command.command_id, snapshot)
            self._raise_if_stop_requested()
            self.record_snapshot(snapshot)
            with self._recorder_context():
                snapshot = controller.run_auto()
            self.record_snapshot(snapshot)
            exported: Path | None = None
            if snapshot.state is ControllerState.COMPLETED:
                exported = self._export_snapshot(result_path, snapshot)
            message = (
                "automatic run completed"
                if snapshot.state is ControllerState.COMPLETED
                else f"automatic run ended in {snapshot.state.value}"
            )
            return self._execution(snapshot, message, exported)

        if command.action == "reset":
            snapshot = self._reset_session()
            if snapshot is None:
                return CommandExecution(
                    snapshot=None,
                    state=ControllerState.IDLE.value,
                    iteration=-1,
                    message="controller reset",
                )
            return self._execution(snapshot, "controller reset")

        controller = self._require_controller()
        if command.action == "power_tune":
            snapshot = controller.snapshot()
            if snapshot.state is ControllerState.READY and not snapshot.transmitting:
                controller.start_reference_transmission()
            with self._recorder_context():
                controller.tune_power()
            snapshot = controller.snapshot()
            message = "initial output power tuned"
        elif command.action == "calibrate":
            snapshot = controller.snapshot()
            if (
                snapshot.state is ControllerState.POWER_READY
                and not snapshot.transmitting
            ):
                controller.start_reference_transmission()
            with self._recorder_context():
                controller.calibrate()
            snapshot = controller.snapshot()
            message = "round-zero calibration completed"
        elif command.action == "step":
            with self._recorder_context():
                controller.step()
            snapshot = controller.snapshot()
            message = "one ILC iteration completed"
        elif command.action == "export":
            snapshot = controller.snapshot()
            exported = self._export_snapshot(result_path, snapshot)
            self.record_snapshot(snapshot)
            return self._execution(snapshot, "final result exported", exported)
        else:  # pragma: no cover - parser owns the action allowlist
            raise FileCommandError(
                "unsupported_action", f"unsupported action {command.action!r}"
            )

        self.record_snapshot(snapshot)
        return self._execution(snapshot, message)

    def request_stop(self) -> ControllerSnapshot | None:
        """Forward a cancellation request without waiting for the active command."""

        self._stop_requested.set()
        with self._lock:
            current = self._controller
            controllers = tuple(
                controller
                for controller in (current, self._pending_controller)
                if controller is not None
            )
        snapshots: dict[ClosedLoopController, ControllerSnapshot] = {}
        errors: list[Exception] = []
        for controller in dict.fromkeys(controllers):
            try:
                snapshots[controller] = controller.request_stop()
            except Exception as exc:  # noqa: BLE001 - every controller must be stopped
                errors.append(exc)

        current_snapshot = None if current is None else snapshots.get(current)
        if current_snapshot is not None:
            try:
                self.record_snapshot(current_snapshot)
            except Exception as exc:  # noqa: BLE001 - persist only after hardware stops
                errors.append(exc)
        if errors:
            for additional in errors[1:]:
                _LOG.error(
                    "additional error while stopping processor controllers",
                    exc_info=(
                        type(additional),
                        additional,
                        additional.__traceback__,
                    ),
                )
            raise errors[0]
        if current_snapshot is not None:
            return current_snapshot
        return next(iter(snapshots.values()), None)

    def begin_command(self) -> None:
        """Clear the cancellation latch before one newly claimed command."""
        self._stop_requested.clear()

    def _raise_if_stop_requested(self) -> None:
        if self._stop_requested.is_set():
            raise FileCommandError("cancelled", "command was cancelled by stop")

    def record_snapshot(self, snapshot: ControllerSnapshot | None) -> None:
        """Persist a snapshot through an optional duck-typed run recorder."""

        if snapshot is None:
            return
        with self._lock:
            recorder = self._recorder
        if recorder is not None:
            from .storage import RunNotFoundError

            try:
                recorder.record_snapshot(snapshot)
            except RunNotFoundError:
                recorder.close()
                with self._lock:
                    if self._recorder is recorder:
                        self._recorder = None
                        self._run_id = None

    def close(self) -> None:
        """Best-effort stop and release of the current controller."""

        with self._lock:
            recorder = self._recorder
            controller = self._controller
        stopped_snapshot: ControllerSnapshot | None = None
        if controller is not None:
            try:
                stopped_snapshot = controller.request_stop()
            except Exception:  # noqa: BLE001 - shutdown must continue
                _LOG.debug("failed to stop controller while closing", exc_info=True)
                stopped_snapshot = controller.snapshot()
        if recorder is not None and stopped_snapshot is not None:
            try:
                recorder.record_snapshot(stopped_snapshot)
            except Exception:  # noqa: BLE001 - process shutdown is best effort
                _LOG.debug("failed to persist stopped controller", exc_info=True)
        if recorder is not None:
            recorder.close()
        with self._lock:
            self._recorder = None
            self._run_id = None
        if controller is not None:
            try:
                if controller.snapshot().connected:
                    controller.disconnect()
            except Exception:  # noqa: BLE001 - process shutdown is best effort
                _LOG.debug(
                    "failed to disconnect controller while closing", exc_info=True
                )

    def _reset_session(self) -> ControllerSnapshot | None:
        with self._lock:
            controller = self._controller
            recorder = self._recorder
        snapshot: ControllerSnapshot | None = None
        reset_error: Exception | None = None
        storage_error: Exception | None = None
        try:
            if controller is not None:
                snapshot = controller.reset()
        except Exception as exc:  # noqa: BLE001 - storage cleanup must still run
            reset_error = exc
            snapshot = controller.snapshot()

        try:
            if (
                reset_error is not None
                and recorder is not None
                and snapshot is not None
            ):
                recorder.record_snapshot(snapshot)
                recorder.close()
            else:
                self._finalize_recorder(recorder, reason="controller reset")
        except Exception as exc:  # noqa: BLE001 - hardware reset already completed
            storage_error = exc
        finally:
            if recorder is not None:
                recorder.close()
            with self._lock:
                self._configuration = None
                self._x = None
                self._recorder = None
                self._run_id = None

        if reset_error is not None:
            if storage_error is not None:
                _LOG.error(
                    "failed to persist a controller reset failure",
                    exc_info=(
                        type(storage_error),
                        storage_error,
                        storage_error.__traceback__,
                    ),
                )
            raise reset_error
        if storage_error is not None:
            raise storage_error
        return snapshot

    @staticmethod
    def _finalize_recorder(
        recorder: RunRecorder | Any | None,
        *,
        reason: str,
    ) -> None:
        if recorder is None:
            return
        from .storage import RunNotFoundError, RunStorageError

        try:
            manifest_status = recorder.read_manifest()["status"]
        except RunNotFoundError:
            recorder.close()
            return
        if manifest_status == "finalizing":
            try:
                recorder.mark_terminal(
                    ControllerState.COMPLETED,
                    message="completed result finalized while detaching recorder",
                )
            except RunStorageError:
                recorder.mark_terminal(
                    ControllerState.FAILED,
                    message="final result cache was invalid while detaching recorder",
                    error_code="recovery_artifact_invalid",
                )
            recorder.close()
            return
        if manifest_status in {
            ControllerState.COMPLETED.value,
            ControllerState.STOPPED.value,
            ControllerState.FAILED.value,
        }:
            recorder.close()
            return
        recorder.mark_terminal(
            ControllerState.STOPPED,
            message=reason,
            error_code="superseded",
        )
        recorder.close()

    def _apply_configuration(self, parsed: ParsedConfiguration) -> None:
        self._raise_if_stop_requested()
        with self._lock:
            previous = self._controller
            previous_recorder = self._recorder
            reference = self._x

        candidate = self._controller_factory(parsed.device_type)
        if not isinstance(candidate, ClosedLoopController):
            raise TypeError("controller_factory must return ClosedLoopController")
        with self._lock:
            self._pending_controller = candidate
        try:
            self._raise_if_stop_requested()
            candidate.connect()
            self._raise_if_stop_requested()
            candidate.apply_config(parsed.closed_loop)
            self._raise_if_stop_requested()
            if reference is not None:
                candidate.load_reference(reference)
                self._raise_if_stop_requested()
            effective_config = candidate.snapshot().config
            if effective_config is None:  # pragma: no cover - controller invariant
                raise RuntimeError(
                    "configured controller did not retain its configuration"
                )

            self._raise_if_stop_requested()
            self._finalize_recorder(
                previous_recorder,
                reason="configuration replaced",
            )
            with self._lock:
                if self._recorder is previous_recorder:
                    self._recorder = None
                    self._run_id = None
                if self._stop_requested.is_set():
                    raise FileCommandError(
                        "cancelled", "configuration was cancelled by stop"
                    )
                if self._controller is not previous:
                    raise RuntimeError("controller changed during configuration")
                self._controller = candidate
                self._pending_controller = None
                self._device_type = parsed.device_type
                self._configuration = effective_config
                self._recorder = None
                self._run_id = None
            if previous is not None:
                self._dispose_controller(previous)
        except Exception:
            with self._lock:
                if self._pending_controller is candidate:
                    self._pending_controller = None
            self._dispose_controller(candidate)
            raise

    def _load_reference(self, value: np.ndarray) -> None:
        self._raise_if_stop_requested()
        copied = _validate_x(value)
        with self._lock:
            controller = self._controller
            previous_recorder = self._recorder
        self._finalize_recorder(
            previous_recorder,
            reason="reference replaced",
        )
        with self._lock:
            if self._recorder is previous_recorder:
                self._recorder = None
                self._run_id = None
        self._raise_if_stop_requested()
        if controller is not None:
            controller.load_reference(copied)
            self._raise_if_stop_requested()
        copied.setflags(write=False)
        with self._lock:
            self._x = copied

    def _ensure_recorder(self, command_id: str) -> None:
        with self._lock:
            if self._recorder is not None or self._run_store is None:
                return
            config = self._configuration
            reference = self._x
            store = self._run_store
        if config is None or reference is None:
            return
        recorder = store.create_run(config, reference, run_id=command_id)
        with self._lock:
            if self._recorder is None:
                self._recorder = recorder
                self._run_id = command_id

    def _start_run_recorder(
        self,
        command_id: str,
        snapshot: ControllerSnapshot,
    ) -> None:
        """Give every automatic command a self-contained same-ID run."""
        if snapshot.state not in {
            ControllerState.READY,
            ControllerState.POWER_READY,
            ControllerState.CALIBRATED,
        }:
            raise FileCommandError(
                "invalid_state",
                "automatic run requires ready, power_ready, or calibrated state",
            )
        self._raise_if_stop_requested()
        with self._lock:
            store = self._run_store
            previous = self._recorder
            config = self._configuration
            reference = self._x
        if store is None:
            return
        if config is None or reference is None:
            raise FileCommandError(
                "run_inputs_missing", "automatic run requires configuration and x"
            )

        self._finalize_recorder(
            previous,
            reason=f"superseded by automatic run {command_id}",
        )
        with self._lock:
            if self._recorder is previous:
                self._recorder = None
                self._run_id = None
        self._raise_if_stop_requested()

        recorder = store.create_run(config, reference, run_id=command_id)
        try:
            self._raise_if_stop_requested()
            recorder.record_snapshot(snapshot)
            self._raise_if_stop_requested()
        except Exception as exc:
            try:
                recorder.mark_terminal(
                    ControllerState.FAILED,
                    message="automatic run initialization failed",
                    error_code=getattr(exc, "code", "run_initialization_failed"),
                )
            except Exception:  # noqa: BLE001 - preserve the primary failure
                _LOG.debug("failed to terminalize an unclaimed run", exc_info=True)
            recorder.close()
            raise

        with self._lock:
            if self._stop_requested.is_set():
                recorder.mark_terminal(
                    ControllerState.STOPPED,
                    message="automatic run cancelled before execution",
                    error_code="cancelled",
                )
                raise FileCommandError("cancelled", "command was cancelled by stop")
            self._recorder = recorder
            self._run_id = command_id

    def _recorder_context(self) -> Any:
        with self._lock:
            recorder = self._recorder
        return nullcontext() if recorder is None else recorder.active()

    def _export_snapshot(
        self,
        result_path: Path,
        snapshot: ControllerSnapshot,
    ) -> Path:
        from .result_export import export_final_mat, load_final_payload
        from .storage import RunNotFoundError

        with self._lock:
            store = self._run_store
            run_id = self._run_id
        guard = (
            nullcontext()
            if store is None or run_id is None
            else store.export_guard(run_id)
        )
        try:
            with guard as guarded_recorder:
                cached_path = (
                    None
                    if guarded_recorder is None
                    else guarded_recorder.final_result_path
                )
                if cached_path is None:
                    exported = export_final_mat(result_path, snapshot)
                else:
                    load_final_payload(cached_path)
                    _atomic_publish_file(cached_path, result_path)
                    exported = result_path
        except RunNotFoundError:
            exported = export_final_mat(result_path, snapshot)
        return result_path if exported is None else Path(exported)

    def _require_controller(self) -> ClosedLoopController:
        controller = self.controller
        if controller is None:
            raise FileCommandError(
                "not_configured", "configure a device before this action"
            )
        return controller

    def _require_ready_inputs(self) -> None:
        with self._lock:
            configuration = self._configuration
            reference = self._x
        if configuration is None:
            raise FileCommandError("not_configured", "a configuration is required")
        if reference is None:
            raise FileCommandError("reference_missing", "a reference x is required")

    @staticmethod
    def _execution(
        snapshot: ControllerSnapshot | None,
        message: str,
        result_path: Path | None = None,
    ) -> CommandExecution:
        if snapshot is None:
            return CommandExecution(None, "loaded", -1, message, result_path)
        return CommandExecution(
            snapshot,
            snapshot.state.value,
            _snapshot_iteration(snapshot),
            message,
            result_path,
        )

    @staticmethod
    def _dispose_controller(controller: ClosedLoopController) -> None:
        try:
            controller.request_stop()
        except Exception:  # noqa: BLE001 - preserve the primary lifecycle error
            _LOG.debug("failed to stop detached controller", exc_info=True)
        try:
            if controller.snapshot().connected:
                controller.disconnect()
        except Exception:  # noqa: BLE001 - preserve the primary lifecycle error
            _LOG.debug("failed to disconnect detached controller", exc_info=True)


class FileCommandService:
    """Watch, deduplicate, and execute versioned inbox commands."""

    def __init__(
        self,
        exchange_root: str | Path,
        *,
        processor: FileCommandProcessor | None = None,
        controller_factory: ControllerFactory | None = None,
        run_store: RunStore | Any | None = None,
        status_poll_seconds: float = 0.02,
    ) -> None:
        root = Path(exchange_root).expanduser()
        poll_seconds = _finite_positive(status_poll_seconds, "status_poll_seconds")
        if processor is not None and (
            controller_factory is not None or run_store is not None
        ):
            raise ValueError(
                "processor cannot be combined with controller_factory or run_store"
            )

        self.exchange_root = root
        self.inbox = root / "inbox"
        self.outbox = root / "outbox"
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)
        self._inbox_resolved = self.inbox.resolve()
        self._processor = processor or FileCommandProcessor(
            controller_factory=controller_factory,
            run_store=run_store,
        )
        self._status_poll_seconds = poll_seconds
        self._lifecycle_lock = Lock()
        self._dispatch_lock = RLock()
        self._status_write_lock = Lock()
        self._active_command_id: str | None = None
        self._idle = Event()
        self._idle.set()
        self._executor: ThreadPoolExecutor | None = None
        self._observer: Any | None = None
        self._stop_monitors: set[Thread] = set()
        self._active_stop_ids: set[str] = set()
        self._started = False
        self._background_dispatch_stopped = False

    @property
    def processor(self) -> FileCommandProcessor:
        """Return the stateful command processor owned by this service."""

        return self._processor

    @property
    def active_command_id(self) -> str | None:
        """Return the non-stop command currently executing in the worker."""

        with self._dispatch_lock:
            return self._active_command_id

    def process_file(
        self,
        path: str | Path,
        *,
        background: bool = False,
    ) -> CommandStatus:
        """Claim and process one complete formal command file.

        Synchronous processing is the default to make direct integrations and
        tests deterministic. Watchdog events use the single background worker.
        """

        command_path, filename_id = self._validate_command_path(path)
        status_path = self.status_path(filename_id)
        with self._dispatch_lock:
            if background and self._background_dispatch_stopped:
                raise FileCommandError(
                    "service_stopping", "background command dispatch is stopped"
                )
            if status_path.exists():
                try:
                    stored_status = self.read_status(filename_id)
                except FileCommandError as exc:
                    return CommandStatus(
                        command_id=filename_id,
                        accepted=False,
                        state="failed",
                        iteration=-1,
                        message=str(exc),
                        error_code=exc.code,
                        timestamp=_utc_timestamp(),
                    )
                if self._active_command_id == filename_id:
                    return stored_status
                try:
                    stored_command = _parse_command_file(command_path, filename_id)
                except FileCommandError:
                    return stored_status
                terminal_status = self._status_is_terminal(
                    stored_command.action,
                    stored_status,
                )
                if terminal_status:
                    if not self._terminal_result_needs_recovery(
                        stored_command,
                        stored_status,
                    ):
                        return stored_status
                    reconciled = self._recover_interrupted_command(
                        stored_command,
                        stored_status,
                    )
                    if reconciled is not None:
                        self._write_status(reconciled)
                        return reconciled
                    return stored_status
                if (
                    stored_command.action in {"run", "export"}
                    and stored_status.accepted
                ):
                    reconciled = self._recover_interrupted_command(
                        stored_command,
                        stored_status,
                    )
                    if reconciled is not None:
                        self._write_status(reconciled)
                        return reconciled
                if (
                    stored_command.action == "stop"
                    and filename_id in self._active_stop_ids
                ):
                    return stored_status
                interrupted = self._recover_interrupted_command(
                    stored_command,
                    stored_status,
                )
                if interrupted is None:  # pragma: no cover - nonterminal invariant
                    raise RuntimeError(
                        "nonterminal command recovery produced no status"
                    )
                self._write_status(interrupted)
                return interrupted
            try:
                command = _parse_command_file(command_path, filename_id)
            except Exception as exc:  # noqa: BLE001 - invalid input becomes status
                status = self._error_status(filename_id, exc, accepted=False)
                self._write_status(status)
                return status

            recovered = self._recover_interrupted_command(command, None)
            if recovered is not None:
                self._write_status(recovered)
                return recovered

            if self._background_dispatch_stopped and command.action != "stop":
                status = CommandStatus(
                    command_id=command.command_id,
                    accepted=False,
                    state="failed",
                    iteration=self._current_iteration(),
                    message="command dispatch is stopped",
                    error_code="service_stopping",
                    timestamp=_utc_timestamp(),
                )
                self._write_status(status)
                return status

            if command.action == "stop":
                return self._execute_stop(command)
            if self._active_command_id is not None:
                status = CommandStatus(
                    command_id=command.command_id,
                    accepted=False,
                    state="busy",
                    iteration=self._current_iteration(),
                    message=(
                        f"service is busy with command {self._active_command_id!r}"
                    ),
                    error_code="busy",
                    timestamp=_utc_timestamp(),
                )
                self._write_status(status)
                return status
            else:
                self._processor.begin_command()
                initial = CommandStatus(
                    command_id=command.command_id,
                    accepted=True,
                    state="accepted",
                    iteration=self._current_iteration(),
                    message="command accepted",
                    error_code="",
                    timestamp=_utc_timestamp(),
                    run_id=self._initial_run_id(command),
                )
                self._write_status(initial)
                self._active_command_id = command.command_id
                self._idle.clear()

        if background:
            try:
                executor = self._ensure_executor()
                executor.submit(self._execute_claimed, command)
            except (FileCommandError, RuntimeError) as exc:
                status = self._error_status(
                    command.command_id,
                    exc,
                    accepted=True,
                    run_id=self._initial_run_id(command),
                )
                self._write_status(status)
                self._release_claim(command.command_id)
                return status
            return initial
        return self._execute_claimed(command)

    def scan(self, *, background: bool = False) -> tuple[CommandStatus, ...]:
        """Process every complete formal command currently in the inbox."""

        statuses: list[CommandStatus] = []
        for path in sorted(self.inbox.glob("command_*.mat")):
            if COMMAND_FILE_PATTERN.fullmatch(path.name) is None:
                continue
            statuses.append(self.process_file(path, background=background))
        return tuple(statuses)

    def start(self) -> None:
        """Start watchdog observation and scan commands already on disk."""

        with self._lifecycle_lock:
            with self._dispatch_lock:
                if self._started:
                    return
                from watchdog.events import FileSystemEventHandler
                from watchdog.observers import Observer

                service = self

                class _CommandEventHandler(FileSystemEventHandler):
                    def on_created(self, event: Any) -> None:
                        if not event.is_directory:
                            service._process_event_path(Path(event.src_path))

                    def on_moved(self, event: Any) -> None:
                        if not event.is_directory:
                            service._process_event_path(Path(event.dest_path))

                observer = Observer()
                observer.schedule(
                    _CommandEventHandler(), str(self.inbox), recursive=False
                )
                self._background_dispatch_stopped = False
                observer.start()
                self._observer = observer
                self._started = True
            self.scan(background=True)

    def stop(self, *, wait: bool = True) -> None:
        """Stop observation and request cancellation of in-flight work."""

        with self._lifecycle_lock:
            with self._dispatch_lock:
                self._background_dispatch_stopped = True
                observer = self._observer
                self._observer = None
                self._started = False
            if observer is not None:
                observer.stop()
                observer.join(timeout=10.0)
            with self._dispatch_lock:
                executor = self._executor
                self._executor = None
            active_command_id = self.active_command_id
            snapshot = self._processor.snapshot()
            stop_error: Exception | None = None
            if active_command_id is not None or (
                snapshot is not None and snapshot.transmitting
            ):
                try:
                    self._processor.request_stop()
                except Exception as exc:  # noqa: BLE001 - shutdown must continue
                    stop_error = exc
            if executor is not None:
                executor.shutdown(wait=wait, cancel_futures=False)
            if wait and self.active_command_id is not None:
                self._idle.wait(60.0)
            if wait:
                with self._dispatch_lock:
                    monitors = tuple(self._stop_monitors)
                for monitor in monitors:
                    if monitor is not current_thread():
                        monitor.join(timeout=10.0)
            if stop_error is not None:
                raise stop_error

    def close(self) -> None:
        """Stop the watcher and release the processor-owned controller."""

        try:
            self.stop(wait=True)
        finally:
            self._processor.close()

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait until no non-stop command is running."""

        if timeout is not None and timeout < 0.0:
            raise ValueError("timeout must not be negative")
        return self._idle.wait(timeout)

    def status_path(self, command_id: str) -> Path:
        """Return the safe persistent status path for a command identifier."""

        _validate_command_id(command_id)
        return self.outbox / f"status_{command_id}.mat"

    def result_path(self, command_id: str) -> Path:
        """Return the safe formal-result path for a command identifier."""

        _validate_command_id(command_id)
        return self.outbox / f"result_{command_id}.mat"

    def read_status(self, command_id: str) -> CommandStatus:
        """Read an existing status without changing idempotence state."""

        path = self.status_path(command_id)
        try:
            payload = _load_mat_payload(path)
            parsed_id = _strict_string(payload.get("command_id"), "command_id")
            if parsed_id != command_id:
                raise FileCommandError(
                    "invalid_status", "status command_id does not match its filename"
                )
            schema_version = _strict_integer(
                payload.get("schema_version"), "schema_version"
            )
            accepted_value = _strict_integer(payload.get("accepted"), "accepted")
            if schema_version != FILE_COMMAND_SCHEMA_VERSION:
                raise FileCommandError(
                    "invalid_status", "status schema_version is unsupported"
                )
            if accepted_value not in {0, 1}:
                raise FileCommandError(
                    "invalid_status", "status accepted must be zero or one"
                )
            run_id = (
                _strict_string(payload["run_id"], "run_id", allow_empty=True)
                if "run_id" in payload
                else ""
            )
            if run_id:
                _validate_command_id(run_id)
            return CommandStatus(
                schema_version=schema_version,
                command_id=parsed_id,
                accepted=bool(accepted_value),
                state=_strict_string(payload.get("state"), "state"),
                iteration=_strict_integer(payload.get("iteration"), "iteration"),
                message=_strict_string(
                    payload.get("message"), "message", allow_empty=True
                ),
                error_code=_strict_string(
                    payload.get("error_code"), "error_code", allow_empty=True
                ),
                timestamp=_strict_string(payload.get("timestamp"), "timestamp"),
                run_id=run_id,
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise FileCommandError(
                "invalid_status", f"failed to read persistent status: {exc}"
            ) from exc

    def _initial_run_id(self, command: FileCommand) -> str:
        if command.action == "run" and self._processor.run_store is not None:
            return command.command_id
        return self._processor.run_id or ""

    def _terminal_result_needs_recovery(
        self,
        command: FileCommand,
        status: CommandStatus,
    ) -> bool:
        if command.action not in {"run", "export"} or not status.accepted:
            return False
        if status.state == ControllerState.COMPLETED.value and not status.error_code:
            from .result_export import ResultExportError, load_final_payload

            try:
                load_final_payload(self.result_path(command.command_id))
            except (FileNotFoundError, ResultExportError):
                return True
            return False
        if self.result_path(command.command_id).exists():
            return True
        store = self._processor.run_store
        if store is not None:
            run_id = command.command_id if command.action == "run" else status.run_id
            if run_id:
                try:
                    return store.open_run(run_id).final_result_path is not None
                except Exception:  # noqa: BLE001 - recovery will classify corruption
                    return True
        return status.error_code in {
            "recovery_artifact_invalid",
            "recovery_failed",
            "result_publish_failed",
        }

    def _recover_interrupted_command(
        self,
        command: FileCommand,
        stored_status: CommandStatus | None,
    ) -> CommandStatus | None:
        """Reconcile durable evidence without replaying hardware operations."""
        from .result_export import load_final_payload
        from .storage import RunNotFoundError

        result_path = self.result_path(command.command_id)
        outbox_error: Exception | None = None
        if command.action in {"run", "export"} and result_path.exists():
            try:
                payload = load_final_payload(result_path)
            except Exception as exc:  # noqa: BLE001 - alternate cache may recover it
                outbox_error = exc
            else:
                return self._completed_recovery_status(
                    command,
                    payload,
                    run_id=(
                        command.command_id
                        if command.action == "run"
                        and self._processor.run_store is not None
                        else ""
                        if stored_status is None
                        else stored_status.run_id
                    ),
                    message="completed result recovered from outbox",
                )

        store = self._processor.run_store
        if store is not None:
            candidates = [command.command_id]
            if (
                stored_status is not None
                and stored_status.run_id
                and stored_status.run_id not in candidates
            ):
                candidates.append(stored_status.run_id)
            for candidate in candidates:
                try:
                    with store.active_run(candidate) as recorder:
                        return self._recover_from_guarded_run(
                            command,
                            stored_status,
                            recorder,
                            candidate,
                            result_path,
                            outbox_error,
                        )
                except RunNotFoundError:
                    continue

        if outbox_error is not None:
            return self._recovery_failure_status(
                command,
                stored_status,
                run_id="" if stored_status is None else stored_status.run_id,
                code="recovery_artifact_invalid",
                message=f"outbox result is invalid: {outbox_error}",
            )
        if stored_status is None:
            return None
        if self._status_is_terminal(command.action, stored_status):
            return None
        return self._recovery_failure_status(
            command,
            stored_status,
            run_id=stored_status.run_id,
            code="service_restarted",
            message=(
                "previous service stopped before the command reached a terminal status"
            ),
        )

    def _recover_from_guarded_run(
        self,
        command: FileCommand,
        stored_status: CommandStatus | None,
        recorder: RunRecorder | Any,
        run_id: str,
        result_path: Path,
        outbox_error: Exception | None,
    ) -> CommandStatus:
        from .result_export import ResultExportError, load_final_payload
        from .storage import RunStorageError

        if command.action not in {"run", "export"}:
            return self._recover_non_result_command(
                command,
                stored_status,
                recorder,
                run_id,
            )

        cache_error: Exception | None = None
        try:
            cached_path = recorder.final_result_path
        except (RunStorageError, OSError) as exc:
            cached_path = None
            cache_error = exc
        if cached_path is not None:
            try:
                payload = load_final_payload(cached_path)
                recorder.mark_terminal(
                    ControllerState.COMPLETED,
                    message="completed result recovered after service restart",
                )
            except (ResultExportError, RunStorageError, OSError) as exc:
                cache_error = exc
            else:
                try:
                    _atomic_publish_file(cached_path, result_path)
                except FileCommandError as exc:
                    return self._recovery_failure_status(
                        command,
                        stored_status,
                        run_id=run_id,
                        code=exc.code,
                        message=str(exc),
                    )
                return self._completed_recovery_status(
                    command,
                    payload,
                    run_id=run_id,
                    message="completed result recovered from run storage",
                )

        try:
            manifest = recorder.read_manifest()
        except Exception as exc:  # noqa: BLE001 - convert corruption to status
            return self._recovery_failure_status(
                command,
                stored_status,
                run_id=run_id,
                code="recovery_artifact_invalid",
                message=f"run manifest cannot be recovered: {exc}",
            )

        manifest_status = manifest["status"]
        if manifest_status in {
            ControllerState.STOPPED.value,
            ControllerState.FAILED.value,
        }:
            return self._terminal_manifest_status(
                command,
                stored_status,
                recorder,
                manifest,
                run_id,
            )
        if manifest_status == ControllerState.COMPLETED.value:
            cache_error = cache_error or RuntimeError(
                "completed run is missing a valid final_result.mat"
            )

        invalid_artifact = outbox_error is not None or cache_error is not None
        code = "recovery_artifact_invalid" if invalid_artifact else "service_restarted"
        message = (
            "durable result artifacts are invalid after service restart"
            if invalid_artifact
            else "previous service stopped before the command reached a terminal status"
        )
        if manifest_status != ControllerState.COMPLETED.value:
            try:
                recorder.mark_terminal(
                    ControllerState.FAILED,
                    message=message,
                    error_code=code,
                )
            except Exception as exc:  # noqa: BLE001 - report reconciliation failure
                return self._recovery_failure_status(
                    command,
                    stored_status,
                    run_id=run_id,
                    code="recovery_failed",
                    message=f"failed to terminalize interrupted run: {exc}",
                )
        return self._recovery_failure_status(
            command,
            stored_status,
            run_id=run_id,
            code=code,
            message=message,
        )

    @staticmethod
    def _completed_recovery_status(
        command: FileCommand,
        payload: Mapping[str, Any],
        *,
        run_id: str,
        message: str,
    ) -> CommandStatus:
        metrics = payload["metrics"]
        if not isinstance(metrics, Mapping):  # pragma: no cover - validator invariant
            raise FileCommandError("recovery_artifact_invalid", "metrics are invalid")
        return CommandStatus(
            command_id=command.command_id,
            accepted=True,
            state=ControllerState.COMPLETED.value,
            iteration=int(metrics["iteration"]),
            message=message,
            error_code="",
            timestamp=_utc_timestamp(),
            run_id=run_id,
        )

    def _recover_non_result_command(
        self,
        command: FileCommand,
        stored_status: CommandStatus | None,
        recorder: RunRecorder | Any,
        run_id: str,
    ) -> CommandStatus:
        manifest = recorder.read_manifest()
        if command.action in {"load", "configure"} and run_id == command.command_id:
            return CommandStatus(
                command_id=command.command_id,
                accepted=True,
                state=ControllerState.READY.value,
                iteration=-1,
                message="command completion inferred from its persisted run lineage",
                error_code="",
                timestamp=_utc_timestamp(),
                run_id=run_id,
            )

        message = (
            "previous service stopped before the command reached a terminal status"
        )
        if manifest["status"] not in {
            ControllerState.COMPLETED.value,
            ControllerState.STOPPED.value,
            ControllerState.FAILED.value,
        }:
            recorder.mark_terminal(
                ControllerState.FAILED,
                message=message,
                error_code="service_restarted",
            )
        return self._recovery_failure_status(
            command,
            stored_status,
            run_id=run_id,
            code="service_restarted",
            message=message,
        )

    @staticmethod
    def _recovery_failure_status(
        command: FileCommand,
        stored_status: CommandStatus | None,
        *,
        run_id: str,
        code: str,
        message: str,
    ) -> CommandStatus:
        return CommandStatus(
            command_id=command.command_id,
            accepted=stored_status is not None or bool(run_id),
            state=ControllerState.FAILED.value,
            iteration=-1 if stored_status is None else stored_status.iteration,
            message=message,
            error_code=code,
            timestamp=_utc_timestamp(),
            run_id=run_id,
        )

    @staticmethod
    def _terminal_manifest_status(
        command: FileCommand,
        stored_status: CommandStatus | None,
        recorder: RunRecorder | Any,
        manifest: Mapping[str, Any],
        run_id: str,
    ) -> CommandStatus:
        state = str(manifest["status"])
        iteration_entries = manifest.get("iterations", [])
        iteration = (
            int(iteration_entries[-1]["iteration"])
            if iteration_entries
            else (-1 if stored_status is None else stored_status.iteration)
        )
        error_code = ""
        message = f"run recovered in terminal state {state}"
        if state == ControllerState.FAILED.value:
            error_code = "run_failed"
            for event in reversed(recorder.read_events()):
                if event.get("kind") not in {"error", "terminal"}:
                    continue
                details = event.get("details")
                if isinstance(details, Mapping) and details.get("error_code"):
                    error_code = str(details["error_code"])
                if event.get("message"):
                    message = str(event["message"])
                break
        return CommandStatus(
            command_id=command.command_id,
            accepted=True,
            state=state,
            iteration=iteration,
            message=message,
            error_code=error_code,
            timestamp=_utc_timestamp(),
            run_id=run_id,
        )

    def __enter__(self) -> FileCommandService:  # noqa: PYI034 - Python 3.10 support
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _execute_stop(self, command: FileCommand) -> CommandStatus:
        with self._dispatch_lock:
            target_command_id = self._active_command_id
            try:
                snapshot = self._processor.request_stop()
                target_is_active = (
                    target_command_id is not None
                    and self._active_command_id == target_command_id
                )
            except Exception as exc:  # noqa: BLE001 - terminal status is mandatory
                snapshot = None
                target_is_active = False
                request_error: Exception | None = exc
            else:
                request_error = None
        try:
            if request_error is not None:
                raise request_error
            if target_is_active:
                status = CommandStatus(
                    command_id=command.command_id,
                    accepted=True,
                    state=ControllerState.STOPPING.value,
                    iteration=(
                        self._current_iteration()
                        if snapshot is None
                        else _snapshot_iteration(snapshot)
                    ),
                    message=f"stop requested for command {target_command_id!r}",
                    error_code="",
                    timestamp=_utc_timestamp(),
                    run_id=self._processor.run_id or "",
                )
            elif target_command_id is not None:
                status = self._stop_status_from_target(
                    command.command_id,
                    target_command_id,
                )
            else:
                status = self._status_from_snapshot(
                    command.command_id,
                    accepted=True,
                    message=(
                        "stop requested"
                        if snapshot is not None
                        else "no active controller"
                    ),
                    fallback_state="stopped",
                )
        except Exception as exc:  # noqa: BLE001 - terminal status is mandatory
            status = self._error_status(
                command.command_id,
                exc,
                accepted=True,
                run_id=self._initial_run_id(command),
            )
        self._write_status(status)
        if status.state == ControllerState.STOPPING.value and target_command_id:
            monitor = Thread(
                target=self._monitor_stop_command,
                args=(command.command_id, target_command_id, snapshot),
                name=f"file-stop-monitor-{command.command_id}",
                daemon=True,
            )
            with self._dispatch_lock:
                self._stop_monitors.add(monitor)
                self._active_stop_ids.add(command.command_id)
            monitor.start()
        return status

    def _monitor_stop_command(
        self,
        command_id: str,
        target_command_id: str,
        requested_snapshot: ControllerSnapshot | None,
    ) -> None:
        try:
            while self.active_command_id == target_command_id:
                time.sleep(self._status_poll_seconds)
            status = self._stop_status_from_target(
                command_id,
                target_command_id,
                requested_snapshot=requested_snapshot,
            )
            self._write_status(status)
        finally:
            with self._dispatch_lock:
                self._stop_monitors.discard(current_thread())
                self._active_stop_ids.discard(command_id)

    def _stop_status_from_target(
        self,
        stop_command_id: str,
        target_command_id: str,
        *,
        requested_snapshot: ControllerSnapshot | None = None,
    ) -> CommandStatus:
        try:
            target = self.read_status(target_command_id)
        except Exception as exc:  # noqa: BLE001 - report durable stop outcome
            return CommandStatus(
                command_id=stop_command_id,
                accepted=True,
                state=ControllerState.FAILED.value,
                iteration=self._current_iteration(),
                message=f"failed to read stopped command status: {exc}",
                error_code="stop_target_status_invalid",
                timestamp=_utc_timestamp(),
                run_id=self._processor.run_id or "",
            )

        if target.state == ControllerState.STOPPED.value or target.error_code in {
            "cancelled",
            "controller_stopped",
        }:
            state = ControllerState.STOPPED.value
            message = f"command {target_command_id!r} stopped"
            error_code = ""
        elif target.state == ControllerState.COMPLETED.value:
            state = ControllerState.COMPLETED.value
            message = f"command {target_command_id!r} completed before stop"
            error_code = ""
        elif target.state == ControllerState.FAILED.value:
            state = ControllerState.FAILED.value
            message = target.message
            error_code = target.error_code or "stop_failed"
        else:
            requested_state = (
                None if requested_snapshot is None else requested_snapshot.state
            )
            if requested_state in {
                ControllerState.COMPLETED,
                ControllerState.STOPPED,
            }:
                state = requested_state.value
                message = f"command {target_command_id!r} stop request completed"
                error_code = ""
            elif requested_state is ControllerState.FAILED:
                state = ControllerState.FAILED.value
                error = requested_snapshot.last_error
                message = "stop failed" if error is None else error.message
                error_code = "stop_failed" if error is None else error.code
            else:
                state = ControllerState.FAILED.value
                message = f"command {target_command_id!r} ended without terminal status"
                error_code = "stop_target_incomplete"
        return CommandStatus(
            command_id=stop_command_id,
            accepted=True,
            state=state,
            iteration=target.iteration,
            message=message,
            error_code=error_code,
            timestamp=_utc_timestamp(),
            run_id=target.run_id,
        )

    def _execute_claimed(self, command: FileCommand) -> CommandStatus:
        monitor_stop = Event()
        monitor: Thread | None = None
        monitor_errors: list[Exception] = []
        execution_error: Exception | None = None
        if command.action == "run":
            monitor = Thread(
                target=self._monitor_run,
                args=(command.command_id, monitor_stop, monitor_errors),
                name=f"file-command-monitor-{command.command_id}",
                daemon=True,
            )
            monitor.start()
        try:
            self._processor.record_snapshot(self._processor.snapshot())
            execution = self._processor.execute(
                command,
                self.result_path(command.command_id),
            )
            status = CommandStatus(
                command_id=command.command_id,
                accepted=True,
                state=execution.state,
                iteration=execution.iteration,
                message=execution.message,
                error_code="",
                timestamp=_utc_timestamp(),
                run_id=self._processor.run_id or "",
            )
        except Exception as exc:  # noqa: BLE001 - service survives every command
            execution_error = exc
            self._request_stop_after_service_error(
                "command execution or persistence failed"
            )
            try:
                self._processor.record_snapshot(self._processor.snapshot())
            except Exception:  # noqa: BLE001 - retain the command's primary error
                _LOG.debug(
                    "failed to persist the terminal command snapshot",
                    exc_info=True,
                )
            status = self._error_status(
                command.command_id,
                exc,
                accepted=True,
                run_id=self._initial_run_id(command),
            )
        finally:
            monitor_stop.set()
            if monitor is not None:
                monitor.join()
        if execution_error is None and monitor_errors:
            status = self._error_status(
                command.command_id,
                monitor_errors[0],
                accepted=True,
                snapshot=self._processor.snapshot(),
                run_id=self._initial_run_id(command),
            )
        try:
            self._write_status(status)
        except Exception:
            self._request_stop_after_service_error("final command status write failed")
            raise
        finally:
            self._release_claim(command.command_id)
        return status

    def _monitor_run(
        self,
        command_id: str,
        stop_event: Event,
        errors: list[Exception],
    ) -> None:
        previous: tuple[str, int] | None = None
        while not stop_event.is_set():
            store = self._processor.run_store
            current_run_id = self._processor.run_id
            if store is not None and current_run_id != command_id:
                stop_event.wait(self._status_poll_seconds)
                continue
            snapshot = self._processor.snapshot()
            if snapshot is not None:
                signature = (snapshot.state.value, _snapshot_iteration(snapshot))
                if signature != previous and snapshot.state not in {
                    ControllerState.COMPLETED,
                    ControllerState.STOPPED,
                    ControllerState.FAILED,
                }:
                    status = CommandStatus(
                        command_id=command_id,
                        accepted=True,
                        state=signature[0],
                        iteration=signature[1],
                        message="automatic run in progress",
                        error_code="",
                        timestamp=_utc_timestamp(),
                        run_id=command_id if store is not None else "",
                    )
                    try:
                        self._write_status(status)
                    except Exception as exc:  # noqa: BLE001 - report after safe stop
                        errors.append(exc)
                        self._request_stop_after_service_error(
                            "automatic run progress status write failed"
                        )
                        _LOG.exception(
                            "failed to persist automatic run progress; stop requested"
                        )
                        return
                    try:
                        self._processor.record_snapshot(snapshot)
                    except Exception as exc:  # noqa: BLE001 - classify persistence error
                        from .storage import RunConflictError

                        latest = self._processor.snapshot()
                        if isinstance(exc, RunConflictError) and (
                            latest is not None
                            and latest.state
                            in {
                                ControllerState.COMPLETED,
                                ControllerState.STOPPED,
                                ControllerState.FAILED,
                            }
                        ):
                            return
                        errors.append(exc)
                        self._request_stop_after_service_error(
                            "automatic run snapshot persistence failed"
                        )
                        _LOG.exception(
                            "failed to persist automatic run snapshot; stop requested"
                        )
                        return
                    previous = signature
            stop_event.wait(self._status_poll_seconds)

    def _request_stop_after_service_error(self, context: str) -> None:
        snapshot = self._processor.snapshot()
        if snapshot is None or (
            not snapshot.transmitting and snapshot.active_operation is None
        ):
            return
        try:
            self._processor.request_stop()
        except Exception:  # noqa: BLE001 - retain the primary service error
            _LOG.exception("%s; additional stop failure", context)

    def _release_claim(self, command_id: str) -> None:
        with self._dispatch_lock:
            if self._active_command_id == command_id:
                self._active_command_id = None
                self._idle.set()

    def _ensure_executor(self) -> ThreadPoolExecutor:
        with self._dispatch_lock:
            if self._background_dispatch_stopped:
                raise FileCommandError(
                    "service_stopping", "background command dispatch is stopped"
                )
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="file-command",
                )
            return self._executor

    def _process_event_path(self, path: Path) -> None:
        if COMMAND_FILE_PATTERN.fullmatch(path.name) is None:
            return
        try:
            self.process_file(path, background=True)
        except (FileCommandError, OSError):
            # Invalid paths cannot be assigned a safe persistent status filename.
            return

    def _validate_command_path(self, path: str | Path) -> tuple[Path, str]:
        candidate = Path(path)
        if not candidate.is_absolute() and candidate.parent == Path("."):
            candidate = self.inbox / candidate
        if candidate.parent.resolve() != self._inbox_resolved:
            raise FileCommandError(
                "invalid_path", "command must be a direct child of the inbox"
            )
        match = COMMAND_FILE_PATTERN.fullmatch(candidate.name)
        if match is None:
            raise FileCommandError(
                "invalid_filename",
                "command filename must match command_<command_id>.mat",
            )
        try:
            if candidate.is_symlink():
                raise FileCommandError("invalid_path", "command must not be a symlink")
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileCommandError(
                "file_missing", "command file does not exist"
            ) from exc
        if resolved.parent != self._inbox_resolved or not resolved.is_file():
            raise FileCommandError(
                "invalid_path", "command must be a regular inbox file"
            )
        return candidate, match.group(1)

    def _write_status(self, status: CommandStatus) -> None:
        path = self.status_path(status.command_id)
        _atomic_save_mat(path, status.to_mat(), lock=self._status_write_lock)

    def _status_from_snapshot(
        self,
        command_id: str,
        *,
        accepted: bool,
        message: str,
        fallback_state: str,
    ) -> CommandStatus:
        snapshot = self._processor.snapshot()
        return CommandStatus(
            command_id=command_id,
            accepted=accepted,
            state=fallback_state if snapshot is None else snapshot.state.value,
            iteration=-1 if snapshot is None else _snapshot_iteration(snapshot),
            message=message,
            error_code="",
            timestamp=_utc_timestamp(),
            run_id=self._processor.run_id or "",
        )

    def _error_status(
        self,
        command_id: str,
        exc: Exception,
        *,
        accepted: bool,
        snapshot: ControllerSnapshot | None = None,
        run_id: str | None = None,
    ) -> CommandStatus:
        snapshot = snapshot or self._processor.snapshot()
        error_code = getattr(exc, "code", None)
        if (
            error_code is None
            and snapshot is not None
            and snapshot.last_error is not None
        ):
            error_code = snapshot.last_error.code
        if error_code is None:
            error_code = _exception_code(type(exc).__name__)
        state = "failed" if snapshot is None else snapshot.state.value
        if not accepted or state not in {"failed", "stopped"}:
            state = "failed"
        effective_run_id = self._processor.run_id or "" if run_id is None else run_id
        return CommandStatus(
            command_id=command_id,
            accepted=accepted,
            state=state,
            iteration=-1 if snapshot is None else _snapshot_iteration(snapshot),
            message=str(exc) or type(exc).__name__,
            error_code=str(error_code),
            timestamp=_utc_timestamp(),
            run_id=effective_run_id,
        )

    def _current_iteration(self) -> int:
        snapshot = self._processor.snapshot()
        return -1 if snapshot is None else _snapshot_iteration(snapshot)

    @staticmethod
    def _status_is_terminal(action: str, status: CommandStatus) -> bool:
        if not status.accepted or status.error_code:
            return True
        terminal_states = {
            "load": {"loaded", "ready"},
            "configure": {"idle", "ready"},
            "power_tune": {"power_ready"},
            "calibrate": {"calibrated"},
            "step": {"calibrated", "completed"},
            "run": {"completed", "stopped", "failed"},
            "stop": {"completed", "stopped", "failed"},
            "reset": {"idle"},
            "export": {"completed"},
        }
        return status.state in terminal_states[action]


def _atomic_publish_file(source: Path, target: Path) -> None:
    """Publish an already validated artifact without exposing partial bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.mat")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise FileCommandError(
            "result_publish_failed",
            f"failed to publish final MAT result: {exc}",
        ) from exc


def _parse_command_file(path: Path, filename_id: str) -> FileCommand:
    payload = _load_mat_payload(path)
    unknown = set(payload) - _COMMAND_FIELDS
    if unknown:
        raise FileCommandError(
            "unknown_field", f"unsupported command fields: {sorted(unknown)}"
        )
    schema_version = _strict_integer(payload.get("schema_version"), "schema_version")
    if schema_version != FILE_COMMAND_SCHEMA_VERSION:
        raise FileCommandError(
            "unsupported_schema",
            f"schema_version must be {FILE_COMMAND_SCHEMA_VERSION}",
        )
    command_id = _strict_string(payload.get("command_id"), "command_id")
    _validate_command_id(command_id)
    if command_id != filename_id:
        raise FileCommandError(
            "command_id_mismatch", "command_id does not match the command filename"
        )
    action = _strict_string(payload.get("action"), "action")
    if action not in COMMAND_ACTIONS:
        raise FileCommandError("unsupported_action", f"unsupported action {action!r}")

    has_x = "x" in payload
    has_config = "config_json" in payload
    if action == "load" and not has_x:
        raise FileCommandError("reference_missing", "load requires x")
    if action == "configure" and not has_config:
        raise FileCommandError("config_missing", "configure requires config_json")
    if action not in {"load", "run"} and has_x:
        raise FileCommandError("unexpected_field", f"{action} does not accept x")
    if action not in {"configure", "run"} and has_config:
        raise FileCommandError(
            "unexpected_field", f"{action} does not accept config_json"
        )

    x = _validate_x(payload["x"]) if has_x else None
    configuration = (
        _parse_config_json(_strict_string(payload["config_json"], "config_json"))
        if has_config
        else None
    )
    return FileCommand(command_id, action, x, configuration)


def _parse_config_json(value: str) -> ParsedConfiguration:
    try:
        raw = json.loads(
            value,
            parse_constant=lambda token: _raise_json_constant(token),
            object_pairs_hook=_unique_json_object,
        )
    except FileCommandError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FileCommandError("invalid_config", f"invalid config_json: {exc}") from exc
    if not isinstance(raw, dict):
        raise FileCommandError("invalid_config", "config_json must be an object")
    unknown = set(raw) - _CONFIG_FIELDS
    missing = {"device_type", "device_config"} - set(raw)
    if unknown:
        raise FileCommandError(
            "invalid_config", f"unsupported config fields: {sorted(unknown)}"
        )
    if missing:
        raise FileCommandError(
            "invalid_config", f"missing config fields: {sorted(missing)}"
        )

    device_type = _json_string(raw["device_type"], "device_type")
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", device_type):
        raise FileCommandError(
            "invalid_config", "device_type has an invalid registry name"
        )
    device_values = raw["device_config"]
    if not isinstance(device_values, dict):
        raise FileCommandError("invalid_config", "device_config must be an object")
    unknown_device = set(device_values) - _DEVICE_CONFIG_FIELDS
    if unknown_device:
        raise FileCommandError(
            "invalid_config",
            f"unsupported device_config fields: {sorted(unknown_device)}",
        )

    runtime_name = _json_string(raw.get("runtime_name", "basic_ilc"), "runtime_name")
    runtime_config = raw.get("runtime_config", {})
    if not isinstance(runtime_config, dict):
        raise FileCommandError("invalid_config", "runtime_config must be an object")
    decoded_runtime = _decode_typed_json(runtime_config, "runtime_config")
    max_iterations = raw.get("max_iterations", 10)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise FileCommandError("invalid_config", "max_iterations must be an integer")

    try:
        decoded_device = _decode_typed_json(device_values, "device_config")
        device_config = DeviceConfig(**decoded_device)
        closed_loop = ClosedLoopConfig(
            device_config=device_config,
            runtime_name=runtime_name,
            runtime_config=decoded_runtime,
            max_iterations=max_iterations,
        )
    except FileCommandError:
        raise
    except (TypeError, ValueError) as exc:
        raise FileCommandError("invalid_config", str(exc)) from exc
    return ParsedConfiguration(device_type, closed_loop)


def _decode_typed_json(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FileCommandError("invalid_config", f"{path} must be finite")
        return value
    if isinstance(value, list):
        return [
            _decode_typed_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):  # pragma: no cover - JSON type invariant
        raise FileCommandError("invalid_config", f"unsupported value at {path}")
    if "$type" not in value:
        return {
            key: _decode_typed_json(item, f"{path}.{key}")
            for key, item in value.items()
        }

    marker = value["$type"]
    if marker == "complex":
        if set(value) != {"$type", "real", "imag"}:
            raise FileCommandError(
                "invalid_config", f"invalid complex wrapper at {path}"
            )
        real = _finite_json_real(value["real"], f"{path}.real")
        imag = _finite_json_real(value["imag"], f"{path}.imag")
        return complex(real, imag)
    if marker != "ndarray":
        raise FileCommandError(
            "invalid_config", f"unsupported $type {marker!r} at {path}"
        )
    if set(value) != {"$type", "dtype", "shape", "data"}:
        raise FileCommandError("invalid_config", f"invalid ndarray wrapper at {path}")
    try:
        dtype = np.dtype(_json_string(value["dtype"], f"{path}.dtype"))
    except (TypeError, ValueError) as exc:
        raise FileCommandError(
            "invalid_config", f"invalid ndarray dtype at {path}"
        ) from exc
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise FileCommandError("invalid_config", f"unsupported ndarray dtype at {path}")
    supported_width = (
        (dtype.kind == "b" and dtype.itemsize == 1)
        or (dtype.kind in "iu" and dtype.itemsize <= 8)
        or (dtype.kind == "f" and dtype.itemsize <= 8)
        or (dtype.kind == "c" and dtype.itemsize <= 16)
        or (dtype.kind == "U" and dtype.itemsize <= 4096)
    )
    if not supported_width:
        raise FileCommandError("invalid_config", f"unsupported ndarray dtype at {path}")
    shape = value["shape"]
    if not isinstance(shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in shape
    ):
        raise FileCommandError(
            "invalid_config",
            f"ndarray shape must contain non-negative integers at {path}",
        )
    element_count = math.prod(shape)
    if element_count > _MAX_ARRAY_ELEMENTS:
        raise FileCommandError("invalid_config", f"ndarray is too large at {path}")
    decoded_data = _decode_typed_json(value["data"], f"{path}.data")
    try:
        result = np.asarray(decoded_data, dtype=dtype)
        if result.size != element_count:
            raise ValueError("data size does not match shape")
        result = np.array(result.reshape(tuple(shape)), copy=True)
    except (TypeError, ValueError) as exc:
        raise FileCommandError(
            "invalid_config", f"invalid ndarray data at {path}: {exc}"
        ) from exc
    if np.issubdtype(dtype, np.number) and not np.all(np.isfinite(result)):
        raise FileCommandError(
            "invalid_config", f"ndarray samples must be finite at {path}"
        )
    return result


def _load_mat_payload(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat

        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        raise FileCommandError(
            "invalid_mat", f"failed to read MAT command: {exc}"
        ) from exc
    return {key: value for key, value in raw.items() if not key.startswith("__")}


def _atomic_save_mat(path: Path, values: Mapping[str, Any], *, lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.mat")
    with lock:
        try:
            from scipy.io import savemat

            savemat(
                temporary,
                dict(values),
                appendmat=False,
                do_compression=False,
                long_field_names=True,
            )
            os.replace(temporary, path)
        except Exception as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise FileCommandError(
                "status_write_failed", f"failed to write MAT status: {exc}"
            ) from exc


def _strict_integer(value: Any, name: str) -> int:
    if value is None:
        raise FileCommandError("missing_field", f"missing {name}")
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iuf":
        raise FileCommandError("invalid_scalar", f"{name} must be one integer scalar")
    scalar = array.reshape(-1)[0]
    result = float(scalar)
    if not math.isfinite(result) or not result.is_integer():
        raise FileCommandError("invalid_scalar", f"{name} must be one integer scalar")
    return int(result)


def _strict_string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if value is None:
        raise FileCommandError("missing_field", f"missing {name}")
    if isinstance(value, (str, np.str_)):
        result = str(value)
    else:
        array = np.asarray(value)
        if allow_empty and array.size == 0 and array.dtype.kind in "US":
            return ""
        if array.size != 1 or array.dtype.kind not in "US":
            raise FileCommandError(
                "invalid_string", f"{name} must be one string scalar"
            )
        item = array.reshape(-1)[0]
        if isinstance(item, bytes):
            try:
                result = item.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise FileCommandError(
                    "invalid_string", f"{name} must be valid UTF-8"
                ) from exc
        else:
            result = str(item)
    if "\x00" in result or (not allow_empty and not result):
        raise FileCommandError("invalid_string", f"{name} must not be empty")
    return result


def _validate_x(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size == 0:
        raise FileCommandError(
            "invalid_reference", "x must be a non-empty one-dimensional vector"
        )
    if array.dtype.kind not in "iufc":
        raise FileCommandError("invalid_reference", "x must be numeric")
    try:
        result = np.array(array, dtype=np.complex128, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FileCommandError("invalid_reference", "x cannot be converted") from exc
    if not np.all(np.isfinite(result)):
        raise FileCommandError(
            "invalid_reference", "x must contain only finite samples"
        )
    return result


def _validate_command_id(value: str) -> None:
    if not isinstance(value, str) or COMMAND_ID_PATTERN.fullmatch(value) is None:
        raise FileCommandError(
            "invalid_command_id",
            "command_id must contain 1-64 safe ASCII letters, digits, underscores, or hyphens",
        )


def _json_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FileCommandError("invalid_config", f"{name} must be a non-empty string")
    return value


def _finite_json_real(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FileCommandError("invalid_config", f"{path} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise FileCommandError("invalid_config", f"{path} must be finite")
    return result


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FileCommandError(
                "invalid_config", f"duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def _raise_json_constant(token: str) -> Any:
    raise FileCommandError("invalid_config", f"invalid JSON constant {token}")


def _snapshot_iteration(snapshot: ControllerSnapshot) -> int:
    return -1 if snapshot.iteration is None else snapshot.iteration


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _exception_code(name: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return words.removesuffix("_error") or "command_error"


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


__all__ = [
    "COMMAND_ACTIONS",
    "COMMAND_FILE_PATTERN",
    "COMMAND_ID_PATTERN",
    "FILE_COMMAND_SCHEMA_VERSION",
    "CommandExecution",
    "CommandStatus",
    "FileCommand",
    "FileCommandError",
    "FileCommandProcessor",
    "FileCommandService",
    "ParsedConfiguration",
]
