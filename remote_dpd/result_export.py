"""Validated final-result export for completed closed-loop runs."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .controller import (
    MAX_REFERENCE_TARGET_RMS_DBFS,
    MIN_REFERENCE_TARGET_RMS_DBFS,
    ControllerSnapshot,
    ControllerState,
    IterationRecord,
)
from .protocol import load_mat, load_mat_file, save_mat

LEGACY_FINAL_RESULT_SCHEMA_VERSION = 1
FINAL_RESULT_SCHEMA_VERSION = 2
_FINAL_FIELDS = frozenset(
    {"schema_version", "x", "y", "z", "metrics", "config", "status", "completed_at"}
)
_LEGACY_FINAL_CONFIG_FIELDS = frozenset(
    {
        "device_type",
        "device_config",
        "runtime_name",
        "runtime_config",
        "max_iterations",
    }
)
_FINAL_CONFIG_FIELDS = _LEGACY_FINAL_CONFIG_FIELDS | {
    "normalize_reference_rms",
    "reference_target_rms_dbfs",
}
_LEGACY_FINAL_METRIC_FIELDS = frozenset(
    {
        "iteration",
        "nmse_db",
        "digital_rms",
        "digital_peak",
        "power_dbm",
        "attenuation_db",
        "gain_correction",
        "capture_segment_count",
        "capture_batch_count",
    }
)
_FINAL_METRIC_FIELDS = _LEGACY_FINAL_METRIC_FIELDS | {
    "source_rms_dbfs",
    "reference_rms_dbfs",
    "reference_scale_db",
}


class ResultExportError(ValueError):
    """Raised when a controller snapshot cannot form a final result."""


def build_final_payload(
    snapshot: ControllerSnapshot,
    *,
    completed_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Build a detached MATLAB payload from a completed controller snapshot.

    The payload deliberately contains only the reference and the final evaluated
    iteration. Full iteration history belongs to temporary run storage.
    """
    if not isinstance(snapshot, ControllerSnapshot):
        raise TypeError("snapshot must be a ControllerSnapshot")
    if snapshot.state is not ControllerState.COMPLETED:
        raise ResultExportError(
            "final result export requires controller state 'completed'; "
            f"got {snapshot.state.value!r}"
        )
    if snapshot.config is None:
        raise ResultExportError("completed snapshot is missing its effective config")
    if snapshot.x is None:
        raise ResultExportError("completed snapshot is missing reference waveform x")
    if len(snapshot.records) < 2:
        raise ResultExportError(
            "completed snapshot must contain calibration record 0 and a final record"
        )

    calibration_record = snapshot.records[0]
    final_record = snapshot.records[-1]
    _validate_record_types(calibration_record, final_record)
    if calibration_record.iteration != 0:
        raise ResultExportError(
            "the first completed record must be calibration record 0"
        )
    if final_record.iteration <= 0:
        raise ResultExportError("the final evaluated record must be an ILC iteration")
    if snapshot.iteration != final_record.iteration:
        raise ResultExportError(
            "snapshot iteration does not identify the final evaluated record"
        )
    if snapshot.max_iterations != snapshot.config.max_iterations:
        raise ResultExportError(
            "snapshot max_iterations does not match the effective config"
        )
    if final_record.iteration != snapshot.config.max_iterations:
        raise ResultExportError(
            "completed snapshot does not contain the configured final iteration"
        )

    x = _column_signal(snapshot.x, "x")
    y = _column_signal(final_record.y, "final record y")
    z = _column_signal(final_record.z, "final record z")
    if x.shape != y.shape or x.shape != z.shape:
        raise ResultExportError("x, final y, and final z must have the same length")

    calibration_y = _column_signal(calibration_record.y, "record 0 y")
    if calibration_y.shape != x.shape or not np.array_equal(calibration_y, x):
        raise ResultExportError("calibration record 0 must have y equal to x")

    metrics = _build_metrics(snapshot, calibration_record, final_record, y)
    config_json = _build_config_json(snapshot)

    completed_timestamp = _snapshot_completed_at(snapshot, completed_at)
    return {
        "schema_version": FINAL_RESULT_SCHEMA_VERSION,
        "x": x,
        "y": y,
        "z": z,
        "metrics": metrics,
        "config": config_json,
        "status": ControllerState.COMPLETED.value,
        "completed_at": completed_timestamp,
    }


def export_final_mat(
    path: str | os.PathLike[str],
    snapshot: ControllerSnapshot,
    *,
    completed_at: datetime | str | None = None,
) -> Path:
    """Atomically write a validated final MAT result and return its path."""
    target = _validate_output_path(path)
    payload = build_final_payload(snapshot, completed_at=completed_at)
    save_mat(target, payload)
    return target


def load_final_payload(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a final MAT result for safe publication or recovery."""
    target = _validate_input_path(path)
    try:
        raw = load_mat(target)
    except Exception as exc:
        raise ResultExportError(f"failed to load final result: {exc}") from exc
    return _validate_final_payload(raw)


def load_final_payload_file(handle: BinaryIO) -> dict[str, Any]:
    """Validate a final result through a caller-owned, already opened file."""
    try:
        raw = load_mat_file(handle)
    except Exception as exc:
        raise ResultExportError(f"failed to load final result: {exc}") from exc
    return _validate_final_payload(raw)


def _validate_final_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one decoded final-result mapping."""
    if set(raw) != _FINAL_FIELDS:
        missing = sorted(_FINAL_FIELDS - set(raw))
        unknown = sorted(set(raw) - _FINAL_FIELDS)
        raise ResultExportError(
            f"final result fields differ from the contract; missing={missing}, "
            f"unknown={unknown}"
        )

    schema_version = _integer_scalar(raw["schema_version"], "schema_version")
    if schema_version not in {
        LEGACY_FINAL_RESULT_SCHEMA_VERSION,
        FINAL_RESULT_SCHEMA_VERSION,
    }:
        raise ResultExportError(
            "schema_version must be "
            f"{LEGACY_FINAL_RESULT_SCHEMA_VERSION} or {FINAL_RESULT_SCHEMA_VERSION}"
        )
    if raw["status"] != ControllerState.COMPLETED.value:
        raise ResultExportError("final result status must be 'completed'")

    completed_at = raw["completed_at"]
    if not isinstance(completed_at, str):
        raise ResultExportError("completed_at must be an ISO 8601 string")
    completed_at = _normalize_completed_at(completed_at)

    x = _loaded_signal(raw["x"], "x")
    y = _loaded_signal(raw["y"], "y")
    z = _loaded_signal(raw["z"], "z")
    if x.size != y.size or x.size != z.size:
        raise ResultExportError("x, y, and z must have the same length")

    config_text, config_values = _validate_loaded_config(
        raw["config"],
        schema_version=schema_version,
    )
    metrics = _validate_loaded_metrics(
        raw["metrics"],
        schema_version=schema_version,
    )
    if metrics["iteration"] != config_values["max_iterations"]:
        raise ResultExportError("metrics.iteration must match config.max_iterations")
    if schema_version == FINAL_RESULT_SCHEMA_VERSION:
        _validate_loaded_normalization(config_values, metrics)

    return {
        "schema_version": schema_version,
        "x": x,
        "y": y,
        "z": z,
        "metrics": metrics,
        "config": config_text,
        "status": ControllerState.COMPLETED.value,
        "completed_at": completed_at,
    }


def _validate_record_types(
    calibration_record: object,
    final_record: object,
) -> None:
    if not isinstance(calibration_record, IterationRecord):
        raise ResultExportError("record 0 must be an IterationRecord")
    if not isinstance(final_record, IterationRecord):
        raise ResultExportError("final record must be an IterationRecord")


def _validate_loaded_config(
    value: object,
    *,
    schema_version: int,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, str):
        raise ResultExportError("config must be a strict JSON string")
    try:
        parsed = json.loads(
            value,
            parse_constant=lambda token: _reject_json_constant(token),
            object_pairs_hook=_unique_json_object,
        )
    except ResultExportError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResultExportError(f"config is not strict JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ResultExportError("config JSON must be an object")
    expected_fields = (
        _LEGACY_FINAL_CONFIG_FIELDS
        if schema_version == LEGACY_FINAL_RESULT_SCHEMA_VERSION
        else _FINAL_CONFIG_FIELDS
    )
    if set(parsed) != expected_fields:
        raise ResultExportError("config JSON fields differ from the final contract")
    device_type = parsed["device_type"]
    if not isinstance(device_type, str) or not re.fullmatch(
        r"[a-z][a-z0-9_-]*", device_type
    ):
        raise ResultExportError("config.device_type is not a valid registry name")
    if not isinstance(parsed["device_config"], dict):
        raise ResultExportError("config.device_config must be an object")
    runtime_name = parsed["runtime_name"]
    if not isinstance(runtime_name, str) or not runtime_name.strip():
        raise ResultExportError("config.runtime_name must be a non-empty string")
    if not isinstance(parsed["runtime_config"], dict):
        raise ResultExportError("config.runtime_config must be an object")
    max_iterations = parsed["max_iterations"]
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ResultExportError("config.max_iterations must be an integer")
    if max_iterations <= 0:
        raise ResultExportError("config.max_iterations must be positive")
    if schema_version == FINAL_RESULT_SCHEMA_VERSION:
        normalize_reference_rms = parsed["normalize_reference_rms"]
        if not isinstance(normalize_reference_rms, bool):
            raise ResultExportError("config.normalize_reference_rms must be a boolean")
        target_rms_dbfs = parsed["reference_target_rms_dbfs"]
        if isinstance(target_rms_dbfs, bool) or not isinstance(
            target_rms_dbfs,
            (int, float),
        ):
            raise ResultExportError("config.reference_target_rms_dbfs must be a number")
        target_rms_dbfs = float(target_rms_dbfs)
        if not math.isfinite(target_rms_dbfs) or not (
            MIN_REFERENCE_TARGET_RMS_DBFS
            <= target_rms_dbfs
            <= MAX_REFERENCE_TARGET_RMS_DBFS
        ):
            raise ResultExportError(
                "config.reference_target_rms_dbfs is outside the supported range"
            )
    return value, parsed


def _validate_loaded_metrics(
    value: object,
    *,
    schema_version: int,
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ResultExportError("metrics must be a MATLAB struct")
    expected_fields = (
        _LEGACY_FINAL_METRIC_FIELDS
        if schema_version == LEGACY_FINAL_RESULT_SCHEMA_VERSION
        else _FINAL_METRIC_FIELDS
    )
    if set(value) != expected_fields:
        raise ResultExportError("metrics fields differ from the final contract")
    iteration = _integer_scalar(value["iteration"], "metrics.iteration")
    segment_count = _integer_scalar(
        value["capture_segment_count"],
        "metrics.capture_segment_count",
    )
    batch_count = _integer_scalar(
        value["capture_batch_count"],
        "metrics.capture_batch_count",
    )
    if iteration <= 0:
        raise ResultExportError("metrics.iteration must be positive")
    if segment_count <= 0 or batch_count <= 0:
        raise ResultExportError("capture counts must be positive")
    metrics: dict[str, int | float] = {
        "iteration": iteration,
        "capture_segment_count": segment_count,
        "capture_batch_count": batch_count,
    }
    for name in expected_fields - {
        "iteration",
        "capture_segment_count",
        "capture_batch_count",
    }:
        metrics[name] = _finite_metric(value[name], f"metrics.{name}")
    if metrics["gain_correction"] <= 0.0:
        raise ResultExportError("metrics.gain_correction must be positive")
    return metrics


def _validate_loaded_normalization(
    config: Mapping[str, Any],
    metrics: Mapping[str, int | float],
) -> None:
    source_rms_dbfs = float(metrics["source_rms_dbfs"])
    reference_rms_dbfs = float(metrics["reference_rms_dbfs"])
    reference_scale_db = float(metrics["reference_scale_db"])
    if not math.isclose(
        reference_rms_dbfs - source_rms_dbfs,
        reference_scale_db,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ResultExportError("reference normalization metrics are inconsistent")
    if config["normalize_reference_rms"]:
        if not math.isclose(
            reference_rms_dbfs,
            float(config["reference_target_rms_dbfs"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ResultExportError(
                "metrics.reference_rms_dbfs does not match the configured target"
            )
    elif not math.isclose(reference_scale_db, 0.0, rel_tol=0.0, abs_tol=1e-10):
        raise ResultExportError(
            "bypassed reference normalization must report zero scale dB"
        )


def _loaded_signal(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.size == 0:
        raise ResultExportError(f"{name} must be a non-empty vector")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ResultExportError(f"{name} must contain numeric IQ samples")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.array(array, dtype=np.complex128, order="C", copy=True)
    if not np.all(np.isfinite(copied)):
        raise ResultExportError(f"{name} must contain only finite IQ samples")
    return np.frombuffer(copied.tobytes(), dtype=np.complex128)


def _integer_scalar(value: object, name: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ResultExportError(f"{name} must be an integer scalar")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(
        scalar, (int, np.integer)
    ):
        raise ResultExportError(f"{name} must be an integer scalar")
    return int(scalar)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResultExportError(f"config contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise ResultExportError(f"config contains non-finite constant {token!r}")


def _build_metrics(
    snapshot: ControllerSnapshot,
    calibration_record: IterationRecord,
    final_record: IterationRecord,
    y: np.ndarray,
) -> dict[str, int | float]:
    preprocessing = final_record.preprocessing
    nmse_db = _finite_metric(preprocessing.nmse_db, "nmse_db")
    digital_rms = _finite_metric(
        final_record.digital_safety.candidate_rms,
        "digital_rms",
    )
    digital_peak = _finite_metric(
        final_record.digital_safety.candidate_peak,
        "digital_peak",
    )
    power_dbm = _finite_metric(final_record.power_dbm, "power_dbm")
    attenuation_db = _finite_metric(
        final_record.attenuation_db,
        "attenuation_db",
    )
    gain_correction = _finite_metric(
        snapshot.gain_correction,
        "gain_correction",
    )
    if gain_correction <= 0.0:
        raise ResultExportError("gain_correction must be greater than zero")

    if not final_record.digital_safety.passed:
        raise ResultExportError("final evaluated y did not pass digital safety checks")
    if final_record.digital_safety.candidate_samples != y.size:
        raise ResultExportError(
            "final digital safety report does not match the final y length"
        )

    actual_peak = float(np.max(np.abs(y)))
    actual_rms = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    if not math.isclose(digital_peak, actual_peak, rel_tol=1e-12, abs_tol=1e-15):
        raise ResultExportError("digital_peak does not match the final evaluated y")
    if not math.isclose(digital_rms, actual_rms, rel_tol=1e-12, abs_tol=1e-15):
        raise ResultExportError("digital_rms does not match the final evaluated y")

    calibration_gain = _finite_metric(
        calibration_record.preprocessing.gain_correction,
        "record 0 gain_correction",
    )
    final_gain = _finite_metric(
        preprocessing.gain_correction,
        "final gain_correction",
    )
    if not math.isclose(gain_correction, calibration_gain, rel_tol=1e-12):
        raise ResultExportError(
            "snapshot gain_correction does not match calibration record 0"
        )
    if not math.isclose(gain_correction, final_gain, rel_tol=1e-12):
        raise ResultExportError(
            "final record did not reuse the fixed calibration gain_correction"
        )

    capture_segment_count = _positive_count(
        preprocessing.segment_count,
        "capture_segment_count",
    )
    capture_batch_count = _positive_count(
        len(preprocessing.batch_diagnostics),
        "capture_batch_count",
    )
    diagnostic_segment_count = sum(
        batch.segment_count for batch in preprocessing.batch_diagnostics
    )
    if diagnostic_segment_count != capture_segment_count:
        raise ResultExportError(
            "capture_segment_count does not match final batch diagnostics"
        )

    normalization = snapshot.reference_normalization
    if normalization is None:
        raise ResultExportError("completed snapshot is missing reference normalization")
    config = snapshot.config
    if config is None:  # pragma: no cover - guarded by build_final_payload
        raise ResultExportError("completed snapshot is missing its effective config")
    if normalization.enabled is not config.normalize_reference_rms:
        raise ResultExportError(
            "reference normalization report does not match the effective config"
        )
    if not math.isclose(
        normalization.target_rms_dbfs,
        config.reference_target_rms_dbfs,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ResultExportError(
            "reference normalization target does not match the effective config"
        )
    source_rms_dbfs = _finite_metric(
        normalization.source_rms_dbfs,
        "source_rms_dbfs",
    )
    reference_rms_dbfs = _finite_metric(
        normalization.effective_rms_dbfs,
        "reference_rms_dbfs",
    )
    reference_scale_db = _finite_metric(
        normalization.scale_db,
        "reference_scale_db",
    )
    reference = snapshot.x
    if reference is None:  # pragma: no cover - guarded by build_final_payload
        raise ResultExportError("completed snapshot is missing reference waveform x")
    actual_reference_rms = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    if not math.isclose(
        normalization.effective_rms,
        actual_reference_rms,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ResultExportError(
            "reference normalization report does not match the effective x"
        )
    if not math.isclose(
        normalization.scale_db,
        normalization.effective_rms_dbfs - normalization.source_rms_dbfs,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ResultExportError("reference normalization scale is inconsistent")
    if normalization.enabled:
        if not math.isclose(
            normalization.effective_rms_dbfs,
            normalization.target_rms_dbfs,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ResultExportError("effective reference RMS does not match its target")
    elif not math.isclose(normalization.scale, 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise ResultExportError("bypassed reference normalization must use unit scale")

    return {
        "iteration": final_record.iteration,
        "nmse_db": nmse_db,
        "digital_rms": digital_rms,
        "digital_peak": digital_peak,
        "power_dbm": power_dbm,
        "attenuation_db": attenuation_db,
        "gain_correction": gain_correction,
        "capture_segment_count": capture_segment_count,
        "capture_batch_count": capture_batch_count,
        "source_rms_dbfs": source_rms_dbfs,
        "reference_rms_dbfs": reference_rms_dbfs,
        "reference_scale_db": reference_scale_db,
    }


def _build_config_json(snapshot: ControllerSnapshot) -> str:
    config = snapshot.config
    if config is None:  # pragma: no cover - guarded by build_final_payload
        raise ResultExportError("completed snapshot is missing its effective config")
    if not isinstance(snapshot.device_type, str) or not snapshot.device_type.strip():
        raise ResultExportError("completed snapshot is missing its device_type")
    values = {
        "device_type": snapshot.device_type,
        **config.to_dict(),
    }

    # Basic ILC accepts an omitted mu and applies 0.5. Persist that effective
    # value so consumers can reproduce a run even when the caller supplied {}.
    if values["runtime_name"] == "basic_ilc":
        values["runtime_config"].setdefault("mu", 0.5)

    try:
        return json.dumps(
            values,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ResultExportError(f"effective config is not strict JSON: {exc}") from exc


def _column_signal(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ResultExportError(f"{name} must be a one-dimensional vector")
    if array.size == 0:
        raise ResultExportError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.bool_,
    ):
        raise ResultExportError(f"{name} must contain numeric IQ samples")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.array(array, dtype=np.complex128, order="C", copy=True)
    if not np.all(np.isfinite(copied)):
        raise ResultExportError(f"{name} must contain only finite IQ samples")
    return copied.reshape(-1, 1)


def _finite_metric(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ResultExportError(f"final metric {name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ResultExportError(f"final metric {name} must be finite")
    return result


def _positive_count(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ResultExportError(f"final metric {name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ResultExportError(f"final metric {name} must be greater than zero")
    return result


def _snapshot_completed_at(
    snapshot: ControllerSnapshot,
    override: datetime | str | None,
) -> str:
    if snapshot.completed_at is None:
        raise ResultExportError("completed snapshot is missing completed_at")
    terminal_timestamp = _normalize_completed_at(snapshot.completed_at)
    if override is None:
        return terminal_timestamp
    requested_timestamp = _normalize_completed_at(override)
    if requested_timestamp != terminal_timestamp:
        raise ResultExportError(
            "completed_at override must match the controller terminal timestamp"
        )
    return terminal_timestamp


def _normalize_completed_at(value: datetime | str) -> str:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ResultExportError("completed_at must not be empty")
        try:
            moment = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ResultExportError(
                "completed_at must be an ISO 8601 timestamp"
            ) from exc
    else:
        raise TypeError("completed_at must be a datetime or ISO 8601 string")

    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ResultExportError("completed_at must include a UTC offset")
    utc = moment.astimezone(timezone.utc)
    return utc.isoformat(timespec="auto").replace("+00:00", "Z")


def _validate_output_path(path: str | os.PathLike[str]) -> Path:
    try:
        target = Path(path)
    except TypeError as exc:
        raise TypeError("path must be a string or path-like value") from exc
    if target.suffix.lower() != ".mat":
        raise ResultExportError("final result path must use the .mat extension")
    if target.exists() and target.is_dir():
        raise ResultExportError(f"final result path is a directory: {target}")
    parent = target.parent
    if parent.exists() and not parent.is_dir():
        raise ResultExportError(f"final result parent is not a directory: {parent}")
    return target


def _validate_input_path(path: str | os.PathLike[str]) -> Path:
    target = _validate_output_path(path)
    if target.is_symlink() or not target.is_file():
        raise ResultExportError("final result path must be a real existing file")
    return target
