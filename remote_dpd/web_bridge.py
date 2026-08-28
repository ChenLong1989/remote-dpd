"""Transport bridge from strict Web commands to the shared file coordinator."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .controller import ControllerSnapshot, IterationRecord
from .device import (
    DeviceConfig,
    DeviceRegistrationError,
    create_rf_bench,
    list_rf_benches,
)
from .exceptions import MatProtocolError
from .file_interface import (
    COMMAND_ACTIONS,
    FILE_COMMAND_SCHEMA_VERSION,
    CommandStatus,
    FileCommandError,
    FileCommandService,
    parse_configuration_json,
)
from .protocol import load_mat, save_mat
from .storage import RunNotFoundError, RunStore
from .waveforms import MAX_PREVIEW_POINTS, WaveformRepository

WEB_COMMAND_ACTIONS = frozenset(COMMAND_ACTIONS - {"stop"})
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
_LOG = logging.getLogger(__name__)
_SESSION_RECORD_LIMIT = 256
_SESSION_POWER_TRACE_LIMIT = 256
_DIAGNOSTIC_BATCH_LIMIT = 8
_DIAGNOSTIC_SEGMENT_LIMIT = 8
_RUNTIME_DETAIL_MAX_DEPTH = 6
_RUNTIME_DETAIL_MAX_NODES = 128
_RUNTIME_DETAIL_MAX_ITEMS = 16
_RUNTIME_DETAIL_MAX_STRING = 128
_RUNTIME_HISTORY_MAX_DEPTH = 1
_RUNTIME_HISTORY_MAX_NODES = 16
_RUNTIME_HISTORY_MAX_ITEMS = 8
_RUNTIME_HISTORY_MAX_STRING = 64
_RUNTIME_KEY_MAX_STRING = 64
_ERROR_TEXT_MAX_STRING = 512
_ITERATION_METADATA_MAX_DEPTH = 10
_ITERATION_METADATA_MAX_NODES = 2_048
_ITERATION_METADATA_MAX_ITEMS = 64
_ITERATION_METADATA_MAX_STRING = 512
_RUN_DETAIL_ITERATION_LIMIT = 1_000
_RUN_DETAIL_CONFIG_MAX_NODES = 4_096
_RUN_DETAIL_SNAPSHOT_MAX_NODES = 2_048
_RUN_DETAIL_EVENTS_MAX_NODES = 10_000


class WebBridgeError(ValueError):
    """A Web command or query cannot be mapped to the shared coordinator."""

    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class WebCommandBridge:
    """Create durable commands while reusing FileCommandService arbitration."""

    def __init__(
        self,
        command_service: FileCommandService,
        run_store: RunStore,
        waveforms: WaveformRepository,
    ) -> None:
        self._service = command_service
        self._store = run_store
        self._waveforms = waveforms
        self._submit_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._ordering_lock = threading.Lock()
        self._stop_epoch = 0
        self._known_actions: dict[str, str] = {}

    @property
    def command_service(self) -> FileCommandService:
        return self._service

    @property
    def run_store(self) -> RunStore:
        return self._store

    @property
    def waveforms(self) -> WaveformRepository:
        return self._waveforms

    def submit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Persist and submit one ordinary Web command without a second worker."""
        with self._ordering_lock:
            submission_epoch = self._stop_epoch
        normalized = _normalize_command_payload(payload)
        action = normalized["action"]
        request_id = normalized.get("request_id")
        command_id = (
            f"web-command-{request_id}"
            if request_id is not None
            else f"web-command-{uuid.uuid4().hex[:20]}"
        )
        command_path = self._service.inbox / f"command_{command_id}.mat"

        mat_payload: dict[str, Any] = {
            "schema_version": FILE_COMMAND_SCHEMA_VERSION,
            "command_id": command_id,
            "action": action,
        }
        waveform_path = normalized.get("waveform_path")
        if waveform_path is not None:
            mat_payload["x"] = self._waveforms.load_x(waveform_path)
        configuration = normalized.get("configuration")
        if configuration is not None:
            _validate_web_configuration_limits(configuration)
            config_json = _strict_json_dumps(configuration)
            try:
                parse_configuration_json(config_json)
            except FileCommandError as exc:
                raise WebBridgeError(
                    "invalid_configuration",
                    str(exc),
                ) from exc
            mat_payload["config_json"] = config_json

        with self._submit_lock:
            with self._ordering_lock:
                cancelled_by_stop = submission_epoch != self._stop_epoch
            if cancelled_by_stop:
                raise WebBridgeError(
                    "command_cancelled",
                    "command preparation was cancelled by a safety stop",
                    status_code=409,
                )
            if command_path.exists():
                _assert_existing_command_matches(command_path, mat_payload)
            else:
                save_mat(command_path, mat_payload)
            self._known_actions[command_id] = action
            status = self._service.process_file(command_path, background=True)
        return self.status_payload(status, action=action)

    def stop(self, request_id: str | None = None) -> dict[str, Any]:
        """Request cancellation before persisting the formal stop command."""
        normalized_request = _normalize_request_id(request_id)
        command_id = (
            f"web-stop-{normalized_request}"
            if normalized_request is not None
            else f"web-stop-{uuid.uuid4().hex[:19]}"
        )
        command_path = self._service.inbox / f"command_{command_id}.mat"
        mat_payload = {
            "schema_version": FILE_COMMAND_SCHEMA_VERSION,
            "command_id": command_id,
            "action": "stop",
        }
        with self._stop_lock:
            if command_path.exists():
                _assert_existing_command_matches(command_path, mat_payload)
                try:
                    self._service.read_status(command_id)
                except (FileCommandError, FileNotFoundError):
                    pass
                else:
                    self._known_actions[command_id] = "stop"
                    status = self._service.process_file(
                        command_path,
                        background=False,
                    )
                    return self.status_payload(status, action="stop")

            with self._ordering_lock:
                self._stop_epoch += 1
            with self._service.immediate_stop_barrier(), self._submit_lock:
                if command_path.exists():
                    _assert_existing_command_matches(command_path, mat_payload)
                else:
                    save_mat(command_path, mat_payload)
                self._known_actions[command_id] = "stop"
                status = self._service.process_file(
                    command_path,
                    background=False,
                )
        return self.status_payload(status, action="stop")

    def command_status(self, command_id: str) -> dict[str, Any]:
        try:
            status = self._service.read_status(command_id)
        except (FileCommandError, FileNotFoundError) as exc:
            raise WebBridgeError(
                "command_not_found",
                "command status was not found",
                status_code=404,
            ) from exc
        action = self._known_actions.get(command_id) or self._read_action(command_id)
        return self.status_payload(status, action=action)

    def session(self) -> dict[str, Any]:
        snapshot = self._service.processor.snapshot()
        return {
            "schema_version": 1,
            "active_command_id": self._service.active_command_id,
            "run_id": self._service.processor.run_id,
            "controller": None if snapshot is None else snapshot_payload(snapshot),
        }

    def devices(self) -> dict[str, Any]:
        default_common = DeviceConfig().to_dict()
        entries = []
        for device_type in list_rf_benches():
            try:
                bench = create_rf_bench(device_type)
            except DeviceRegistrationError:
                _LOG.warning(
                    "ignoring invalid RF bench registration %r",
                    device_type,
                    exc_info=True,
                )
                continue
            entries.append(
                {
                    "device_type": device_type,
                    "schema": bench.parameter_schema.to_dict(),
                    "default_configuration": {
                        "device_type": device_type,
                        "device_config": default_common,
                        "runtime_name": "basic_ilc",
                        "runtime_config": {"mu": 0.5},
                        "max_iterations": 10,
                    },
                }
            )
        return {"schema_version": 1, "devices": entries}

    def list_runs(self, *, limit: int = 50) -> dict[str, Any]:
        normalized_limit = _bounded_integer(limit, "limit", minimum=1, maximum=200)
        runs = []
        for manifest in self._store.list_runs()[:normalized_limit]:
            iterations = manifest["iterations"]
            runs.append(
                {
                    "run_id": manifest["run_id"],
                    "status": manifest["status"],
                    "device_type": manifest.get("device_type"),
                    "created": manifest["created"],
                    "updated": manifest["updated"],
                    "completed": manifest["completed"],
                    "latest_iteration": (
                        None if not iterations else iterations[-1]["iteration"]
                    ),
                    "result_available": manifest.get("final_result")
                    == "final_result.mat",
                }
            )
        return {"schema_version": 1, "runs": runs}

    def run_detail(self, run_id: str, *, event_limit: int = 500) -> dict[str, Any]:
        normalized_event_limit = _bounded_integer(
            event_limit,
            "event_limit",
            minimum=1,
            maximum=1_000,
        )
        try:
            data = self._store.read_run(run_id)
        except (RunNotFoundError, TypeError, ValueError) as exc:
            raise WebBridgeError(
                "run_not_found",
                "temporary run was not found",
                status_code=404,
            ) from exc
        manifest = data["manifest"]
        stored_iterations = manifest["iterations"]
        config, config_truncated = _bounded_json_document(
            data["config"],
            max_depth=16,
            max_nodes=_RUN_DETAIL_CONFIG_MAX_NODES,
            max_items=512,
            max_string=1_024,
        )
        snapshot, snapshot_truncated = _bounded_json_document(
            data["snapshot"],
            max_depth=12,
            max_nodes=_RUN_DETAIL_SNAPSHOT_MAX_NODES,
            max_items=256,
            max_string=1_024,
        )
        selected_events = data["events"][-normalized_event_limit:]
        events, events_truncated_by_budget = _bounded_json_document(
            selected_events,
            max_depth=12,
            max_nodes=_RUN_DETAIL_EVENTS_MAX_NODES,
            max_items=1_000,
            max_string=512,
        )
        return {
            "schema_version": 1,
            "run": {
                "run_id": manifest["run_id"],
                "status": manifest["status"],
                "device_type": manifest.get("device_type"),
                "created": manifest["created"],
                "updated": manifest["updated"],
                "completed": manifest["completed"],
                "result_available": manifest.get("final_result") == "final_result.mat",
                "iterations": [
                    {"iteration": item["iteration"]}
                    for item in stored_iterations[-_RUN_DETAIL_ITERATION_LIMIT:]
                ],
                "iteration_count": len(stored_iterations),
                "iterations_truncated": max(
                    0,
                    len(stored_iterations) - _RUN_DETAIL_ITERATION_LIMIT,
                ),
                "config": config,
                "config_truncated": config_truncated,
                "snapshot": snapshot,
                "snapshot_truncated": snapshot_truncated,
                "power_trace": data["power_trace"][-1_000:],
                "power_trace_count": len(data["power_trace"]),
                "events": events,
                "event_count": len(data["events"]),
                "events_truncated": (
                    events_truncated_by_budget
                    or len(data["events"]) > len(selected_events)
                ),
            },
        }

    def iteration_preview(
        self,
        run_id: str,
        iteration: int,
        *,
        points: int = 1024,
    ) -> dict[str, Any]:
        normalized_points = _bounded_integer(
            points,
            "points",
            minimum=16,
            maximum=MAX_PREVIEW_POINTS,
        )
        try:
            with self._store.active_run(run_id) as recorder:
                stored = recorder.read_iteration(iteration)
                x = recorder.read_reference()
        except (RunNotFoundError, TypeError, ValueError) as exc:
            raise WebBridgeError(
                "iteration_not_found",
                "run or iteration was not found",
                status_code=404,
            ) from exc
        y = stored["y"]
        z = stored["z"]
        indices = _sample_indices(x.size, normalized_points)
        return {
            "schema_version": 1,
            "run_id": run_id,
            "iteration": int(iteration),
            "sample_count": int(x.size),
            "preview_count": int(indices.size),
            "index": indices.tolist(),
            "x": _complex_preview(x[indices]),
            "y": _complex_preview(y[indices]),
            "z": _complex_preview(z[indices]),
            "error": _complex_preview((z - x)[indices]),
            "metadata": _bounded_iteration_metadata(stored["metadata"]),
        }

    def current_preview(self, *, points: int = 1024) -> dict[str, Any]:
        normalized_points = _bounded_integer(
            points,
            "points",
            minimum=16,
            maximum=MAX_PREVIEW_POINTS,
        )
        snapshot = self._service.processor.snapshot()
        if snapshot is None or snapshot.x is None:
            raise WebBridgeError(
                "reference_missing",
                "no reference waveform is loaded",
                status_code=409,
            )
        x = snapshot.x
        record = snapshot.current_record
        y = x if record is None else record.y
        z = None if record is None else record.z
        indices = _sample_indices(x.size, normalized_points)
        return {
            "schema_version": 1,
            "sample_count": int(x.size),
            "preview_count": int(indices.size),
            "index": indices.tolist(),
            "x": _complex_preview(x[indices]),
            "y": _complex_preview(y[indices]),
            "z": None if z is None else _complex_preview(z[indices]),
            "error": (None if z is None else _complex_preview((z - x)[indices])),
        }

    def status_payload(
        self,
        status: CommandStatus,
        *,
        action: str,
    ) -> dict[str, Any]:
        result_available = self._service.result_path(status.command_id).is_file()
        phase = _command_phase(status, action)
        return {
            "schema_version": 1,
            "command_id": status.command_id,
            "action": action,
            "source": "web",
            "accepted": status.accepted,
            "phase": phase,
            "controller_state": status.state,
            "iteration": status.iteration,
            "run_id": status.run_id or None,
            "message": status.message,
            "error": (
                None
                if not status.error_code
                else {"code": status.error_code, "message": status.message}
            ),
            "updated_at": status.timestamp,
            "result_url": (
                f"/api/v1/results/{status.command_id}.mat" if result_available else None
            ),
        }

    def _read_action(self, command_id: str) -> str:
        path = self._service.inbox / f"command_{command_id}.mat"
        try:
            value = load_mat(path).get("action")
        except (MatProtocolError, OSError, ValueError):
            return "unknown"
        if isinstance(value, str):
            return value
        array = np.asarray(value)
        return str(array.reshape(-1)[0]) if array.size == 1 else "unknown"


def snapshot_payload(snapshot: ControllerSnapshot) -> dict[str, Any]:
    """Serialize controller metadata without copying full waveform history."""
    selected_records = snapshot.records[-_SESSION_RECORD_LIMIT:]
    records = [
        record_payload(
            record,
            include_diagnostics=index == len(selected_records) - 1,
        )
        for index, record in enumerate(selected_records)
    ]
    return {
        "state": snapshot.state.value,
        "device_type": snapshot.device_type,
        "connected": snapshot.connected,
        "configured": snapshot.configured,
        "reference_loaded": snapshot.reference_loaded,
        "transmitting": snapshot.transmitting,
        "stop_requested": snapshot.stop_requested,
        "active_operation": snapshot.active_operation,
        "iteration": snapshot.iteration,
        "max_iterations": snapshot.max_iterations,
        "gain_correction": _finite_or_none(snapshot.gain_correction),
        "locked_attenuation_db": _finite_or_none(snapshot.locked_attenuation_db),
        "latest_power_dbm": _finite_or_none(snapshot.latest_power_dbm),
        "completed_at": snapshot.completed_at,
        "reference_sample_count": None if snapshot.x is None else int(snapshot.x.size),
        "record_count": len(snapshot.records),
        "records": records,
        "power_trace_count": len(snapshot.power_trace),
        "power_trace": [
            {
                "attenuation_db": item.attenuation_db,
                "power_dbm": item.power_dbm,
                "gap_db": item.gap_db,
            }
            for item in snapshot.power_trace[-_SESSION_POWER_TRACE_LIMIT:]
        ],
        "last_error": (
            None
            if snapshot.last_error is None
            else {
                "operation": _bounded_text(snapshot.last_error.operation),
                "code": _bounded_text(snapshot.last_error.code),
                "exception_type": _bounded_text(snapshot.last_error.exception_type),
                "message": _bounded_text(
                    snapshot.last_error.message,
                    maximum=_ERROR_TEXT_MAX_STRING,
                ),
                "shutdown_error": _bounded_text(
                    snapshot.last_error.shutdown_error,
                    maximum=_ERROR_TEXT_MAX_STRING,
                ),
            }
        ),
    }


def record_payload(
    record: IterationRecord,
    *,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Serialize one round with detailed diagnostics only when requested."""
    preprocessing = record.preprocessing
    runtime_metrics, runtime_metrics_truncated = _bounded_runtime_metrics(
        record.runtime_metrics,
        detailed=include_diagnostics,
    )
    first_segment = None
    if preprocessing.batch_diagnostics:
        first_batch = preprocessing.batch_diagnostics[0]
        if first_batch.segments:
            first_segment = first_batch.segments[0]
    candidate_rms = record.digital_safety.candidate_rms
    candidate_peak = record.digital_safety.candidate_peak
    payload = {
        "iteration": record.iteration,
        "power_dbm": _finite_or_none(record.power_dbm),
        "attenuation_db": _finite_or_none(record.attenuation_db),
        "digital_rms": _finite_or_none(
            record.digital_safety.reference_rms
            if candidate_rms is None
            else candidate_rms
        ),
        "digital_peak": _finite_or_none(
            record.digital_safety.reference_peak
            if candidate_peak is None
            else candidate_peak
        ),
        "digital_passed": record.digital_safety.passed,
        "nmse_db": _finite_or_none(preprocessing.nmse_db),
        "aligned_average_nmse_db": _finite_or_none(
            preprocessing.aligned_average_nmse_db
        ),
        "gain_correction": _finite_or_none(preprocessing.gain_correction),
        "gain_correction_db": _finite_or_none(preprocessing.gain_correction_db),
        "segment_count": preprocessing.segment_count,
        "batch_count": len(preprocessing.batch_diagnostics),
        "delay_samples": (
            None
            if first_segment is None
            else _finite_or_none(first_segment.delay_samples)
        ),
        "phase_radians": (
            None
            if first_segment is None
            else _finite_or_none(first_segment.phase_radians)
        ),
        "diagnostics_included": include_diagnostics,
        "runtime_metrics": runtime_metrics,
        "runtime_metrics_truncated": runtime_metrics_truncated,
    }
    if not include_diagnostics:
        return payload

    diagnostics = []
    for batch in preprocessing.batch_diagnostics[:_DIAGNOSTIC_BATCH_LIMIT]:
        diagnostics.append(
            {
                "batch_index": batch.batch_index,
                "segment_count": batch.segment_count,
                "coherent_within_batch": batch.coherent_within_batch,
                "input_rms": _finite_or_none(batch.input_rms),
                "aligned_average_rms": _finite_or_none(batch.aligned_average_rms),
                "aligned_average_nmse_db": _finite_or_none(
                    batch.aligned_average_nmse_db
                ),
                "segments": [
                    {
                        "segment_index": segment.segment_index,
                        "alignment_estimated": segment.alignment_estimated,
                        "delay_samples": _finite_or_none(segment.delay_samples),
                        "phase_radians": _finite_or_none(segment.phase_radians),
                        "phase_correction": {
                            "real": _finite_or_none(segment.phase_correction.real),
                            "imag": _finite_or_none(segment.phase_correction.imag),
                        },
                        "input_rms": _finite_or_none(segment.input_rms),
                        "aligned_rms": _finite_or_none(segment.aligned_rms),
                        "aligned_nmse_db": _finite_or_none(segment.aligned_nmse_db),
                    }
                    for segment in batch.segments[:_DIAGNOSTIC_SEGMENT_LIMIT]
                ],
                "segments_truncated": max(
                    0,
                    batch.segment_count - _DIAGNOSTIC_SEGMENT_LIMIT,
                ),
            }
        )
    payload["batches_truncated"] = max(
        0,
        len(preprocessing.batch_diagnostics) - _DIAGNOSTIC_BATCH_LIMIT,
    )
    payload["batches"] = diagnostics
    return payload


def _normalize_command_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise WebBridgeError("invalid_command", "command body must be an object")
    allowed = {"action", "waveform_path", "configuration", "request_id"}
    unknown = set(payload) - allowed
    if unknown:
        raise WebBridgeError(
            "invalid_command",
            f"unsupported command fields: {sorted(unknown)}",
        )
    action = payload.get("action")
    if not isinstance(action, str) or action not in WEB_COMMAND_ACTIONS:
        raise WebBridgeError("unsupported_action", "unsupported command action")
    request_id = _normalize_request_id(payload.get("request_id"))
    waveform_path = payload.get("waveform_path")
    configuration = payload.get("configuration")

    if action == "load" and not isinstance(waveform_path, str):
        raise WebBridgeError("waveform_missing", "load requires waveform_path")
    if action == "configure" and not isinstance(configuration, Mapping):
        raise WebBridgeError(
            "configuration_missing", "configure requires configuration"
        )
    if action not in {"load", "run"} and waveform_path is not None:
        raise WebBridgeError(
            "unexpected_field",
            f"{action} does not accept waveform_path",
        )
    if action not in {"configure", "run"} and configuration is not None:
        raise WebBridgeError(
            "unexpected_field",
            f"{action} does not accept configuration",
        )
    if waveform_path is not None and not isinstance(waveform_path, str):
        raise WebBridgeError("invalid_waveform_path", "waveform_path must be a string")
    if configuration is not None and not isinstance(configuration, Mapping):
        raise WebBridgeError("invalid_configuration", "configuration must be an object")
    return {
        "action": action,
        "waveform_path": waveform_path,
        "configuration": configuration,
        "request_id": request_id,
    }


def _normalize_request_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _REQUEST_ID_PATTERN.fullmatch(value) is None:
        raise WebBridgeError(
            "invalid_request_id",
            "request_id must contain 1-40 safe ASCII characters",
        )
    return value


def _strict_json_dumps(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise WebBridgeError(
            "invalid_configuration",
            f"configuration is not strict JSON: {exc}",
        ) from exc


def _assert_existing_command_matches(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    try:
        existing = load_mat(path)
    except (MatProtocolError, OSError, ValueError) as exc:
        raise WebBridgeError(
            "idempotency_conflict",
            "existing request command cannot be validated",
            status_code=409,
        ) from exc
    if set(existing) != set(expected):
        raise WebBridgeError(
            "idempotency_conflict",
            "request_id is already bound to a different command",
            status_code=409,
        )
    for name, expected_value in expected.items():
        existing_value = existing[name]
        if name == "x":
            left = np.asarray(existing_value, dtype=np.complex128).reshape(-1)
            right = np.asarray(expected_value, dtype=np.complex128).reshape(-1)
            matches = left.shape == right.shape and np.array_equal(left, right)
        elif name == "schema_version":
            value = np.asarray(existing_value)
            matches = value.size == 1 and int(value.reshape(-1)[0]) == expected_value
        else:
            value = np.asarray(existing_value)
            matches = value.size == 1 and str(value.reshape(-1)[0]) == expected_value
        if not matches:
            raise WebBridgeError(
                "idempotency_conflict",
                "request_id is already bound to a different command",
                status_code=409,
            )


def _validate_web_configuration_limits(configuration: Mapping[str, Any]) -> None:
    device_config = configuration.get("device_config")
    if not isinstance(device_config, Mapping):
        raise WebBridgeError(
            "invalid_configuration",
            "configuration.device_config must be an object",
        )
    limits = {
        "average_segment_count": (1, 10_000),
        "max_adjustments": (1, 10_000),
    }
    for name, (minimum, maximum) in limits.items():
        value = device_config.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise WebBridgeError(
                "configuration_limit_exceeded",
                f"{name} must be between {minimum} and {maximum}",
            )
    duration_limits = {
        "settle_seconds": 60.0,
        "call_timeout_seconds": 300.0,
    }
    for name, maximum in duration_limits.items():
        value = device_config.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= maximum
        ):
            raise WebBridgeError(
                "configuration_limit_exceeded",
                f"{name} must be between 0 and {maximum}",
            )
    iterations = configuration.get("max_iterations")
    if iterations is not None and (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 1 <= iterations <= 1_000
    ):
        raise WebBridgeError(
            "configuration_limit_exceeded",
            "max_iterations must be between 1 and 1000",
        )

    options = device_config.get("device_options")
    if not isinstance(options, Mapping):
        return
    coefficients = options.get("pa_coefficients")
    if coefficients is None:
        return
    if not isinstance(coefficients, list) or not 1 <= len(coefficients) <= 256:
        raise WebBridgeError(
            "configuration_limit_exceeded",
            "pa_coefficients must contain between 1 and 256 rows",
        )
    for index, coefficient in enumerate(coefficients):
        if not isinstance(coefficient, Mapping):
            continue
        order = coefficient.get("p")
        memory = coefficient.get("m")
        if isinstance(order, int) and not isinstance(order, bool) and order > 99:
            raise WebBridgeError(
                "configuration_limit_exceeded",
                f"pa_coefficients[{index}].p must not exceed 99",
            )
        if isinstance(memory, int) and not isinstance(memory, bool) and memory > 4096:
            raise WebBridgeError(
                "configuration_limit_exceeded",
                f"pa_coefficients[{index}].m must not exceed 4096",
            )


def _command_phase(status: CommandStatus, action: str) -> str:
    if not status.accepted:
        return "rejected"
    if status.error_code:
        return "failed"
    terminal_states = {
        "connect": {"idle", "ready"},
        "disconnect": {"idle"},
        "load": {"loaded", "ready"},
        "configure": {"idle", "ready"},
        "start_transmission": {"ready", "power_ready"},
        "stop_transmission": {
            "idle",
            "ready",
            "power_ready",
            "calibrated",
            "completed",
            "stopped",
            "failed",
        },
        "power_tune": {"power_ready"},
        "calibrate": {"calibrated"},
        "step": {"calibrated", "completed"},
        "run": {"completed", "stopped", "failed"},
        "stop": {"completed", "stopped", "failed"},
        "reset": {"idle"},
        "export": {"completed"},
    }
    if status.state in terminal_states.get(action, set()):
        return (
            "completed" if status.state not in {"stopped", "failed"} else status.state
        )
    if status.state == "accepted":
        return "accepted"
    return "running"


def _complex_preview(value: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(value, dtype=np.complex128)
    return {
        "real": array.real.tolist(),
        "imag": array.imag.tolist(),
        "magnitude": np.abs(array).tolist(),
    }


def _sample_indices(sample_count: int, points: int) -> np.ndarray:
    if sample_count <= points:
        return np.arange(sample_count, dtype=np.int64)
    return np.unique(np.linspace(0, sample_count - 1, points, dtype=np.int64))


def _bounded_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WebBridgeError("invalid_parameter", f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise WebBridgeError(
            "invalid_parameter",
            f"{name} must be between {minimum} and {maximum}",
        )
    return value


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _bounded_text(
    value: str | None,
    *,
    maximum: int = _RUNTIME_KEY_MAX_STRING,
) -> str | None:
    if value is None or len(value) <= maximum:
        return value
    return value[:maximum]


class _JsonBudget:
    """Mutable recursive-output budget for one runtime metric mapping."""

    __slots__ = (
        "max_depth",
        "max_items",
        "max_string",
        "remaining_nodes",
        "truncated",
    )

    def __init__(
        self,
        *,
        max_depth: int,
        max_nodes: int,
        max_items: int,
        max_string: int,
    ) -> None:
        self.max_depth = max_depth
        self.remaining_nodes = max_nodes
        self.max_items = max_items
        self.max_string = max_string
        self.truncated = False

    def consume(self) -> bool:
        if self.remaining_nodes <= 0:
            self.truncated = True
            return False
        self.remaining_nodes -= 1
        return True


def _bounded_runtime_metrics(
    value: Mapping[str, Any],
    *,
    detailed: bool,
) -> tuple[dict[str, Any], bool]:
    if detailed:
        budget = _JsonBudget(
            max_depth=_RUNTIME_DETAIL_MAX_DEPTH,
            max_nodes=_RUNTIME_DETAIL_MAX_NODES,
            max_items=_RUNTIME_DETAIL_MAX_ITEMS,
            max_string=_RUNTIME_DETAIL_MAX_STRING,
        )
    else:
        budget = _JsonBudget(
            max_depth=_RUNTIME_HISTORY_MAX_DEPTH,
            max_nodes=_RUNTIME_HISTORY_MAX_NODES,
            max_items=_RUNTIME_HISTORY_MAX_ITEMS,
            max_string=_RUNTIME_HISTORY_MAX_STRING,
        )
    normalized = _bounded_json_value(value, budget=budget, depth=0)
    if not isinstance(normalized, dict):  # pragma: no cover - mapping invariant
        budget.truncated = True
        normalized = {}
    return normalized, budget.truncated


def _bounded_json_document(
    value: Any,
    *,
    max_depth: int,
    max_nodes: int,
    max_items: int,
    max_string: int,
) -> tuple[Any, bool]:
    budget = _JsonBudget(
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_items=max_items,
        max_string=max_string,
    )
    normalized = _bounded_json_value(value, budget=budget, depth=0)
    return normalized, budget.truncated


def _bounded_json_value(
    value: Any,
    *,
    budget: _JsonBudget,
    depth: int,
) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if not budget.consume():
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value.bit_length() > 256:
            budget.truncated = True
            return None
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            budget.truncated = True
            return None
        return value
    if isinstance(value, complex):
        real = _finite_or_none(value.real)
        imag = _finite_or_none(value.imag)
        if real is None or imag is None:
            budget.truncated = True
        return {"real": real, "imag": imag}
    if isinstance(value, str):
        if len(value) > budget.max_string:
            budget.truncated = True
            return value[: budget.max_string]
        return value
    if isinstance(value, np.ndarray):
        if depth >= budget.max_depth:
            budget.truncated = True
            return None
        flat = value.reshape(-1)
        items = []
        for item in flat[: budget.max_items]:
            if budget.remaining_nodes <= 0:
                budget.truncated = True
                break
            items.append(_bounded_json_value(item, budget=budget, depth=depth + 1))
        truncated_items = max(0, int(flat.size) - len(items))
        if truncated_items:
            budget.truncated = True
        return {
            "$type": "ndarray",
            "dtype": str(value.dtype)[:_RUNTIME_KEY_MAX_STRING],
            "shape": [int(item) for item in value.shape],
            "values": items,
            "truncated_items": truncated_items,
        }
    if isinstance(value, Mapping):
        if depth >= budget.max_depth:
            budget.truncated = True
            return None
        normalized: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= budget.max_items or budget.remaining_nodes <= 0:
                budget.truncated = True
                break
            normalized_key = _bounded_mapping_key(
                key,
                index=index,
                existing=normalized,
                budget=budget,
            )
            normalized[normalized_key] = _bounded_json_value(
                item,
                budget=budget,
                depth=depth + 1,
            )
        if len(normalized) < len(value):
            budget.truncated = True
        return normalized
    if isinstance(value, (list, tuple)):
        if depth >= budget.max_depth:
            budget.truncated = True
            return None
        normalized_items = []
        for item in value[: budget.max_items]:
            if budget.remaining_nodes <= 0:
                budget.truncated = True
                break
            normalized_items.append(
                _bounded_json_value(item, budget=budget, depth=depth + 1)
            )
        if len(normalized_items) < len(value):
            budget.truncated = True
        return normalized_items
    budget.truncated = True
    return {"$type": type(value).__name__[:_RUNTIME_KEY_MAX_STRING]}


def _bounded_mapping_key(
    value: object,
    *,
    index: int,
    existing: Mapping[str, Any],
    budget: _JsonBudget,
) -> str:
    raw = value if isinstance(value, str) else type(value).__name__
    if not isinstance(value, str) or len(raw) > _RUNTIME_KEY_MAX_STRING:
        budget.truncated = True
    candidate = raw[:_RUNTIME_KEY_MAX_STRING]
    if candidate not in existing:
        return candidate
    budget.truncated = True
    suffix = f"~{index}"
    candidate = f"{candidate[: _RUNTIME_KEY_MAX_STRING - len(suffix)]}{suffix}"
    while candidate in existing:  # pragma: no cover - defensive duplicate keys
        suffix += "~"
        candidate = f"{raw[: _RUNTIME_KEY_MAX_STRING - len(suffix)]}{suffix}"
    return candidate


def _bounded_iteration_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    bounded = dict(metadata)
    preprocessing = metadata.get("preprocessing")
    if isinstance(preprocessing, Mapping):
        bounded_preprocessing = dict(preprocessing)
        batches = preprocessing.get("batch_diagnostics")
        if isinstance(batches, list):
            bounded_batches = []
            for batch in batches[:_DIAGNOSTIC_BATCH_LIMIT]:
                if not isinstance(batch, Mapping):
                    continue
                bounded_batch = dict(batch)
                segments = batch.get("segments")
                if isinstance(segments, list):
                    bounded_batch["segments"] = segments[:_DIAGNOSTIC_SEGMENT_LIMIT]
                    bounded_batch["segments_truncated"] = max(
                        0,
                        len(segments) - _DIAGNOSTIC_SEGMENT_LIMIT,
                    )
                bounded_batches.append(bounded_batch)
            bounded_preprocessing["batch_diagnostics"] = bounded_batches
            bounded_preprocessing["batches_truncated"] = max(
                0,
                len(batches) - _DIAGNOSTIC_BATCH_LIMIT,
            )
        bounded["preprocessing"] = bounded_preprocessing

    runtime_metrics = metadata.get("runtime_metrics")
    runtime_truncated = False
    if isinstance(runtime_metrics, Mapping):
        bounded_runtime, runtime_truncated = _bounded_runtime_metrics(
            runtime_metrics,
            detailed=True,
        )
        bounded["runtime_metrics"] = bounded_runtime
    elif runtime_metrics is not None:
        bounded["runtime_metrics"] = None
        runtime_truncated = True

    budget = _JsonBudget(
        max_depth=_ITERATION_METADATA_MAX_DEPTH,
        max_nodes=_ITERATION_METADATA_MAX_NODES,
        max_items=_ITERATION_METADATA_MAX_ITEMS,
        max_string=_ITERATION_METADATA_MAX_STRING,
    )
    normalized = _bounded_json_value(bounded, budget=budget, depth=0)
    if not isinstance(normalized, dict):  # pragma: no cover - mapping invariant
        return {"metadata_truncated": True}
    normalized["metadata_truncated"] = budget.truncated or runtime_truncated
    return normalized


__all__ = [
    "WEB_COMMAND_ACTIONS",
    "WebBridgeError",
    "WebCommandBridge",
    "record_payload",
    "snapshot_payload",
]
