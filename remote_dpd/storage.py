"""Atomic temporary storage for one device-driven closed-loop run."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .controller import (
    ClosedLoopConfig,
    ControllerErrorInfo,
    ControllerSnapshot,
    ControllerState,
    IterationRecord,
)
from .power_control import PowerAdjustment
from .preprocessing import PreprocessingResult
from .protocol import replace_with_retry

RUN_SCHEMA_VERSION = "1.0"
DEFAULT_RETENTION_SECONDS = 7 * 24 * 60 * 60
DEFAULT_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVENT_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_TERMINAL_STATES = frozenset(
    {
        ControllerState.COMPLETED.value,
        ControllerState.FAILED.value,
        ControllerState.STOPPED.value,
    }
)


class _SharedStoreState:
    """Process-local synchronization shared by stores for one canonical root."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.active_counts: dict[str, int] = {}
        self.export_counts: dict[str, int] = {}


_SHARED_STATES_LOCK = threading.Lock()
_SHARED_STATES: dict[Path, _SharedStoreState] = {}


class RunStorageError(RuntimeError):
    """Base error for an invalid or failed run-storage operation."""


class RunNotFoundError(RunStorageError):
    """The requested controlled run directory does not exist."""


class RunConflictError(RunStorageError):
    """Immutable stored data conflicts with a repeated synchronization."""


class RunStore:
    """Own controlled run directories, cleanup guards, and a cleanup worker.

    A newly created recorder owns one active guard immediately. Recording a
    terminal controller snapshot releases that guard after the manifest is
    committed. Call :meth:`RunRecorder.close` when abandoning a non-terminal run.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        retention_seconds: float = DEFAULT_RETENTION_SECONDS,
        cleanup_interval_seconds: float = DEFAULT_CLEANUP_INTERVAL_SECONDS,
    ) -> None:
        self._retention_seconds = _non_negative_finite(
            retention_seconds, "retention_seconds"
        )
        self._cleanup_interval_seconds = _positive_finite(
            cleanup_interval_seconds, "cleanup_interval_seconds"
        )

        raw_root = Path(root).expanduser()
        raw_root.mkdir(parents=True, exist_ok=True)
        if raw_root.is_symlink() or not raw_root.is_dir():
            raise RunStorageError("run storage root must be a real directory")
        self._root = raw_root.resolve(strict=True)
        self._runs_root = self._root / "runs"
        if self._runs_root.exists() and self._runs_root.is_symlink():
            raise RunStorageError("runs directory must not be a symbolic link")
        self._runs_root.mkdir(mode=0o700, exist_ok=True)
        if not self._runs_root.is_dir() or self._runs_root.is_symlink():
            raise RunStorageError("runs directory must be a real directory")
        self._runs_root_resolved = self._runs_root.resolve(strict=True)

        shared_state = _shared_store_state(self._runs_root_resolved)
        self._lock = shared_state.lock
        self._active_counts = shared_state.active_counts
        self._export_counts = shared_state.export_counts
        self._cleanup_stop = threading.Event()
        self._cleanup_thread: threading.Thread | None = None
        self._last_cleanup_error: Exception | None = None

    @property
    def root(self) -> Path:
        """Return the canonical storage root."""
        return self._root

    @property
    def runs_root(self) -> Path:
        """Return the canonical controlled-runs directory."""
        return self._runs_root_resolved

    @property
    def retention_seconds(self) -> float:
        return self._retention_seconds

    @property
    def cleanup_interval_seconds(self) -> float:
        return self._cleanup_interval_seconds

    def create_run(
        self,
        config: ClosedLoopConfig,
        x: np.ndarray,
        run_id: str | None = None,
    ) -> RunRecorder:
        """Create an isolated run and return its initially active recorder."""
        if not isinstance(config, ClosedLoopConfig):
            raise TypeError("config must be a ClosedLoopConfig")
        reference = _readonly_signal(x, "x")
        normalized_id = uuid4().hex if run_id is None else _validate_run_id(run_id)
        now = _utc_now()
        manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": normalized_id,
            "status": "created",
            "created": now,
            "updated": now,
            "completed": None,
            "config": "config.json",
            "x": "x.npy",
            "events": "events.json",
            "power_trace": "power_trace.json",
            "snapshot": None,
            "device_type": None,
            "final_result": None,
            "iterations": [],
        }

        with self._lock:
            self._assert_runs_root()
            path = self._runs_root_resolved / normalized_id
            try:
                path.mkdir(mode=0o700)
            except FileExistsError as exc:
                raise RunConflictError(f"run {normalized_id!r} already exists") from exc
            try:
                _atomic_write_json(path / "config.json", config.to_dict())
                _atomic_write_npy(path / "x.npy", reference)
                _atomic_write_json(path / "events.json", [])
                _atomic_write_json(path / "power_trace.json", [])
                _atomic_write_json(path / "manifest.json", manifest)
            except Exception:
                _discard_new_run(path, self._runs_root_resolved)
                raise
            self._increment_guard(self._active_counts, normalized_id)
            return RunRecorder(self, normalized_id, owns_active_guard=True)

    def open_run(self, run_id: str) -> RunRecorder:
        """Open an existing controlled run without implicitly marking it active."""
        normalized_id = _validate_run_id(run_id)
        with self._lock:
            self._require_run_path(normalized_id)
        return RunRecorder(self, normalized_id, owns_active_guard=False)

    get_run = open_run

    def list_runs(self) -> tuple[dict[str, Any], ...]:
        """List valid controlled manifests, newest first."""
        manifests: list[dict[str, Any]] = []
        with self._lock:
            self._assert_runs_root()
            for entry in self._runs_root_resolved.iterdir():
                manifest = self._controlled_manifest(entry)
                if manifest is not None:
                    manifests.append(manifest)
        manifests.sort(key=lambda item: item["created"], reverse=True)
        return tuple(manifests)

    def read_run(self, run_id: str) -> dict[str, Any]:
        """Read run metadata used by later API and console layers."""
        with self.active_run(run_id) as recorder:
            return {
                "manifest": recorder.read_manifest(),
                "config": recorder.read_config(),
                "events": recorder.read_events(),
                "power_trace": recorder.read_power_trace(),
                "snapshot": recorder.read_latest_snapshot(),
            }

    @contextmanager
    def active_run(self, run_id: str) -> Iterator[RunRecorder]:
        """Protect an existing run from cleanup for the context duration."""
        normalized_id = _validate_run_id(run_id)
        with self._lock:
            self._require_run_path(normalized_id)
            self._increment_guard(self._active_counts, normalized_id)
        try:
            yield RunRecorder(self, normalized_id, owns_active_guard=False)
        finally:
            with self._lock:
                self._decrement_guard(self._active_counts, normalized_id)

    @contextmanager
    def export_guard(self, run_id: str) -> Iterator[RunRecorder]:
        """Protect an existing run while a persistent export reads it."""
        normalized_id = _validate_run_id(run_id)
        with self._lock:
            self._require_run_path(normalized_id)
            self._increment_guard(self._export_counts, normalized_id)
        try:
            yield RunRecorder(self, normalized_id, owns_active_guard=False)
        finally:
            with self._lock:
                self._decrement_guard(self._export_counts, normalized_id)

    def cleanup_expired(
        self,
        now: datetime | float | None = None,
    ) -> tuple[str, ...]:
        """Delete expired, unguarded, valid controlled run directories."""
        now_seconds = _normalize_now(now)
        removed: list[str] = []
        with self._lock:
            self._assert_runs_root()
            for entry in tuple(self._runs_root_resolved.iterdir()):
                manifest = self._controlled_manifest(entry)
                if manifest is None:
                    continue
                run_id = manifest["run_id"]
                if self._active_counts.get(run_id, 0) > 0:
                    continue
                if self._export_counts.get(run_id, 0) > 0:
                    continue
                updated = _parse_utc_timestamp(manifest["updated"])
                if now_seconds - updated < self._retention_seconds:
                    continue
                _remove_controlled_run(entry, self._runs_root_resolved, run_id)
                removed.append(run_id)
        return tuple(removed)

    def start_cleanup(self) -> bool:
        """Start one daemon cleanup worker; repeated calls are harmless."""
        with self._lock:
            if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
                return False
            self._cleanup_stop.clear()
            thread = threading.Thread(
                target=self._cleanup_loop,
                name="remote-dpd-run-cleanup",
                daemon=True,
            )
            self._cleanup_thread = thread
            thread.start()
            return True

    def stop_cleanup(self, timeout: float | None = None) -> bool:
        """Stop the cleanup worker; repeated calls are harmless."""
        if timeout is not None:
            timeout = _non_negative_finite(timeout, "timeout")
        with self._lock:
            thread = self._cleanup_thread
            if thread is None:
                return False
            self._cleanup_stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            if thread.is_alive():
                return False
            if self._cleanup_thread is thread:
                self._cleanup_thread = None
            return True

    def close(self) -> None:
        """Stop the optional cleanup worker."""
        self.stop_cleanup()

    def __enter__(self) -> RunStore:  # noqa: PYI034 - Python 3.10 support
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _cleanup_loop(self) -> None:
        while not self._cleanup_stop.wait(self._cleanup_interval_seconds):
            try:
                self.cleanup_expired()
            except Exception as exc:  # noqa: BLE001 - keep the daemon alive
                # A malformed or temporarily inaccessible entry must not kill the
                # periodic worker. The next interval retries the controlled scan.
                self._last_cleanup_error = exc

    def _assert_runs_root(self) -> None:
        if self._runs_root.is_symlink() or not self._runs_root.is_dir():
            raise RunStorageError("controlled runs directory is unavailable or unsafe")
        try:
            resolved = self._runs_root.resolve(strict=True)
        except OSError as exc:
            raise RunStorageError("controlled runs directory is unavailable") from exc
        if resolved != self._runs_root_resolved:
            raise RunStorageError("controlled runs directory escaped its original root")

    def _require_run_path(self, run_id: str) -> Path:
        self._assert_runs_root()
        path = self._runs_root_resolved / run_id
        manifest = self._controlled_manifest(path)
        if manifest is None:
            raise RunNotFoundError(f"controlled run {run_id!r} was not found")
        return path

    def _controlled_manifest(self, path: Path) -> dict[str, Any] | None:
        if path.is_symlink() or not path.is_dir():
            return None
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return None
        if resolved.parent != self._runs_root_resolved:
            return None
        try:
            run_id = _validate_run_id(path.name)
            manifest_path = path / "manifest.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                return None
            manifest = _read_json(manifest_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if not isinstance(manifest, dict):
            return None
        if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
            return None
        if manifest.get("run_id") != run_id:
            return None
        if not _valid_manifest_shape(manifest):
            return None
        return manifest

    @staticmethod
    def _increment_guard(guards: dict[str, int], run_id: str) -> None:
        guards[run_id] = guards.get(run_id, 0) + 1

    @staticmethod
    def _decrement_guard(guards: dict[str, int], run_id: str) -> None:
        count = guards.get(run_id, 0)
        if count <= 1:
            guards.pop(run_id, None)
        else:
            guards[run_id] = count - 1


class RunRecorder:
    """Record immutable iteration artifacts and evolving metadata for one run."""

    def __init__(
        self,
        store: RunStore,
        run_id: str,
        *,
        owns_active_guard: bool,
    ) -> None:
        self._store = store
        self._run_id = _validate_run_id(run_id)
        self._owns_active_guard = owns_active_guard
        self._closed = False

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> Path:
        with self._store._lock:
            return self._store._require_run_path(self._run_id)

    @property
    def owns_active_guard(self) -> bool:
        return self._owns_active_guard and not self._closed

    @property
    def final_result_path(self) -> Path | None:
        """Return the recoverable final MAT cache, including a pre-commit cache."""
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            manifest = _read_json(path / "manifest.json")
            advertised = manifest.get("final_result")
            candidate = path / "final_result.mat"
            if advertised is None and not candidate.exists():
                return None
            if advertised not in (None, "final_result.mat"):
                raise RunStorageError("manifest contains an unsafe final result path")
            if candidate.is_symlink() or not candidate.is_file():
                raise RunStorageError("cached final result is unavailable or unsafe")
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise RunStorageError("cached final result is unavailable") from exc
            if resolved.parent != path:
                raise RunStorageError("cached final result escaped its run directory")
            return resolved

    def mark_terminal(
        self,
        state: ControllerState | str,
        *,
        message: str | None = None,
        error_code: str | None = None,
    ) -> bool:
        """Finish a run manifest without mutating the hardware controller state.

        A completed transition is accepted only when ``final_result.mat`` already
        exists, which makes a cache written before a manifest crash recoverable.
        """
        terminal_state = _normalize_terminal_state(state)
        normalized_message = _optional_string(message, "message")
        normalized_error_code = _optional_string(error_code, "error_code")
        changed = False
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            manifest = _read_json(path / "manifest.json")
            previous_status = manifest["status"]
            if (
                previous_status in _TERMINAL_STATES
                and previous_status != terminal_state
            ):
                raise RunConflictError(
                    f"terminal status {previous_status!r} cannot change to "
                    f"{terminal_state!r}"
                )

            completed_at = manifest["completed"]
            if terminal_state == ControllerState.COMPLETED.value:
                final_result = self.final_result_path
                if final_result is None:
                    raise RunConflictError(
                        "a completed run requires a recoverable final_result.mat"
                    )
                recovered_at, recovered_device_type = _read_cached_metadata(
                    final_result,
                    path,
                    manifest,
                )
                if manifest.get("final_result") != "final_result.mat":
                    manifest["final_result"] = "final_result.mat"
                    changed = True
                if manifest.get("device_type") is None:
                    manifest["device_type"] = recovered_device_type
                    changed = True
                elif manifest["device_type"] != recovered_device_type:
                    raise RunConflictError(
                        "cached final result device_type conflicts with the manifest"
                    )
                if completed_at is None:
                    completed_at = recovered_at
                elif completed_at != recovered_at:
                    raise RunConflictError(
                        "cached final result completed_at conflicts with the manifest"
                    )
            elif completed_at is None:
                completed_at = _utc_now()

            if previous_status != terminal_state:
                manifest["status"] = terminal_state
                changed = True
            if manifest["completed"] != completed_at:
                manifest["completed"] = completed_at
                changed = True

            event_message = normalized_message or f"run marked {terminal_state}"
            fingerprint = {
                "kind": "terminal",
                "message": event_message,
                "details": {
                    "status": terminal_state,
                    "error_code": normalized_error_code,
                },
            }
            events = _read_json(path / "events.json")
            if not isinstance(events, list):
                raise RunStorageError("events.json must contain a JSON array")
            if not events or _event_fingerprint(events[-1]) != fingerprint:
                events.append(
                    {
                        "sequence": len(events),
                        "timestamp": _utc_now(),
                        **fingerprint,
                    }
                )
                _atomic_write_json(path / "events.json", events)
                changed = True

            if changed:
                manifest["updated"] = _utc_now()
                _atomic_write_json(path / "manifest.json", manifest)

        self.close()
        return changed

    def record_snapshot(self, snapshot: ControllerSnapshot) -> bool:
        """Synchronize one controller snapshot without changing committed rounds.

        Repeating an identical snapshot performs no writes. A different payload for
        an already committed iteration raises :class:`RunConflictError`.
        """
        if not isinstance(snapshot, ControllerSnapshot):
            raise TypeError("snapshot must be a ControllerSnapshot")

        terminal = snapshot.state.value in _TERMINAL_STATES
        _validate_snapshot_terminal(snapshot)
        changed = False
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            manifest = _read_json(path / "manifest.json")
            self._validate_snapshot_identity(path, snapshot)
            _validate_snapshot_records(snapshot)

            previous_status = manifest["status"]
            next_status = snapshot.state.value
            if previous_status in _TERMINAL_STATES and previous_status != next_status:
                raise RunConflictError(
                    f"terminal status {previous_status!r} cannot change to {next_status!r}"
                )
            if (
                terminal
                and manifest["completed"] is not None
                and manifest["completed"] != snapshot.completed_at
            ):
                raise RunConflictError(
                    "snapshot completed_at differs from the stored terminal time"
                )
            stored_device_type = manifest.get("device_type")
            if stored_device_type not in (None, snapshot.device_type):
                raise RunConflictError(
                    "snapshot device_type differs from the stored device_type"
                )

            incoming_trace = [
                _power_adjustment_dict(item) for item in snapshot.power_trace
            ]
            stored_trace = _read_json(path / "power_trace.json")
            if not isinstance(stored_trace, list):
                raise RunStorageError("power_trace.json must contain a JSON array")
            _validate_append_only_prefix(stored_trace, incoming_trace, "power trace")

            iteration_entries = list(manifest["iterations"])
            by_iteration = {item["iteration"]: item for item in iteration_entries}
            for record in snapshot.records:
                entry, record_changed = self._record_iteration(path, record)
                changed = changed or record_changed
                existing_entry = by_iteration.get(record.iteration)
                if existing_entry is None:
                    iteration_entries.append(entry)
                    by_iteration[record.iteration] = entry
                    changed = True
                elif existing_entry != entry:
                    raise RunConflictError(
                        f"manifest iteration {record.iteration} conflicts with stored data"
                    )
            iteration_entries.sort(key=lambda item: item["iteration"])

            if len(incoming_trace) > len(stored_trace):
                _atomic_write_json(path / "power_trace.json", incoming_trace)
                changed = True

            snapshot_payload = _snapshot_metadata(snapshot)
            snapshot_changed = _atomic_write_json_if_changed(
                path / "snapshot.json", snapshot_payload
            )
            changed = changed or snapshot_changed

            events = _read_json(path / "events.json")
            if not isinstance(events, list):
                raise RunStorageError("events.json must contain a JSON array")
            events_changed = self._append_snapshot_events(events, snapshot)
            if events_changed:
                _atomic_write_json(path / "events.json", events)
                changed = True

            if manifest["iterations"] != iteration_entries:
                manifest["iterations"] = iteration_entries
                changed = True
            if manifest.get("snapshot") != "snapshot.json":
                manifest["snapshot"] = "snapshot.json"
                changed = True
            if stored_device_type is None:
                manifest["device_type"] = snapshot.device_type
                changed = True
            if snapshot.state is ControllerState.COMPLETED:
                final_result_path = path / "final_result.mat"
                if previous_status != ControllerState.COMPLETED.value:
                    manifest["status"] = "finalizing"
                    manifest["completed"] = None
                    manifest["updated"] = _utc_now()
                    _atomic_write_json(path / "manifest.json", manifest)
                    changed = True
                if manifest.get("final_result") != "final_result.mat":
                    from .result_export import export_final_mat

                    export_final_mat(final_result_path, snapshot)
                    manifest["final_result"] = "final_result.mat"
                    changed = True
                elif not final_result_path.is_file() or final_result_path.is_symlink():
                    from .result_export import export_final_mat

                    export_final_mat(final_result_path, snapshot)
                    changed = True
                if (
                    manifest["status"] != next_status
                    or manifest["completed"] != snapshot.completed_at
                ):
                    changed = True
                manifest["status"] = next_status
                manifest["completed"] = snapshot.completed_at
            else:
                if previous_status != next_status:
                    manifest["status"] = next_status
                    changed = True
                if terminal and manifest["completed"] is None:
                    manifest["completed"] = snapshot.completed_at
                    changed = True

            if changed:
                manifest["updated"] = _utc_now()
                _atomic_write_json(path / "manifest.json", manifest)

        if terminal:
            self.close()
        return changed

    def record_event(
        self,
        kind: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> bool:
        """Append a strict-JSON event, deduplicating an identical latest event."""
        normalized_kind = _validate_event_kind(kind)
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("details must be a mapping or None")
        normalized_details = _json_value({} if details is None else details)
        fingerprint = {
            "kind": normalized_kind,
            "message": message,
            "details": normalized_details,
        }
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            events = _read_json(path / "events.json")
            if not isinstance(events, list):
                raise RunStorageError("events.json must contain a JSON array")
            if events and _event_fingerprint(events[-1]) == fingerprint:
                manifest = _read_json(path / "manifest.json")
                event_timestamp = events[-1].get("timestamp")
                if isinstance(event_timestamp, str) and _parse_utc_timestamp(
                    manifest["updated"]
                ) < _parse_utc_timestamp(event_timestamp):
                    manifest["updated"] = _utc_now()
                    _atomic_write_json(path / "manifest.json", manifest)
                    return True
                return False
            event = {
                "sequence": len(events),
                "timestamp": _utc_now(),
                **fingerprint,
            }
            events.append(event)
            _atomic_write_json(path / "events.json", events)
            manifest = _read_json(path / "manifest.json")
            manifest["updated"] = _utc_now()
            _atomic_write_json(path / "manifest.json", manifest)
            return True

    def record_error(
        self,
        error: BaseException | ControllerErrorInfo,
        *,
        operation: str | None = None,
    ) -> bool:
        """Append a structured error without risking a partial manifest write."""
        if isinstance(error, ControllerErrorInfo):
            details = _controller_error_dict(error)
            message = error.message
        elif isinstance(error, BaseException):
            details = {
                "operation": operation,
                "code": _exception_code(type(error).__name__),
                "exception_type": type(error).__name__,
                "message": str(error),
                "shutdown_error": None,
            }
            message = str(error)
        else:
            raise TypeError("error must be an exception or ControllerErrorInfo")
        return self.record_event("error", message, details)

    @contextmanager
    def active(self) -> Iterator[RunRecorder]:
        """Add a nested active cleanup guard for this recorder."""
        with self._store.active_run(self._run_id):
            yield self

    @contextmanager
    def export_guard(self) -> Iterator[RunRecorder]:
        """Protect this run while an export reads final artifacts."""
        with self._store.export_guard(self._run_id):
            yield self

    def close(self) -> None:
        """Release the active guard owned by a newly created recorder."""
        with self._store._lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_active_guard:
                self._store._decrement_guard(self._store._active_counts, self._run_id)

    def read_manifest(self) -> dict[str, Any]:
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            return _read_json(path / "manifest.json")

    def read_config(self) -> dict[str, Any]:
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            value = _read_json(path / "config.json")
            if not isinstance(value, dict):
                raise RunStorageError("config.json must contain a JSON object")
            return value

    def read_reference(self) -> np.ndarray:
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            return _readonly_loaded_array(path / "x.npy")

    def read_events(self) -> tuple[dict[str, Any], ...]:
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            value = _read_json(path / "events.json")
            if not isinstance(value, list) or not all(
                isinstance(item, dict) for item in value
            ):
                raise RunStorageError("events.json must contain a JSON object array")
            return tuple(value)

    def read_power_trace(self) -> tuple[dict[str, Any], ...]:
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            value = _read_json(path / "power_trace.json")
            if not isinstance(value, list) or not all(
                isinstance(item, dict) for item in value
            ):
                raise RunStorageError(
                    "power_trace.json must contain a JSON object array"
                )
            return tuple(value)

    def read_latest_snapshot(self) -> dict[str, Any] | None:
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            snapshot_path = path / "snapshot.json"
            if not snapshot_path.exists():
                return None
            value = _read_json(snapshot_path)
            if not isinstance(value, dict):
                raise RunStorageError("snapshot.json must contain a JSON object")
            return value

    def read_iteration(self, iteration: int) -> dict[str, Any]:
        """Read one iteration's metadata and three stored waveforms."""
        normalized = _non_negative_integer(iteration, "iteration")
        with self._store._lock:
            path = self._store._require_run_path(self._run_id)
            manifest = _read_json(path / "manifest.json")
            entry = next(
                (
                    item
                    for item in manifest["iterations"]
                    if item["iteration"] == normalized
                ),
                None,
            )
            if entry is None:
                raise RunNotFoundError(
                    f"iteration {normalized} is not stored for run {self._run_id!r}"
                )
            return {
                "metadata": _read_json(path / entry["metadata"]),
                "y": _readonly_loaded_array(path / entry["y"]),
                "z": _readonly_loaded_array(path / entry["z"]),
                "aligned_average": _readonly_loaded_array(
                    path / entry["aligned_average"]
                ),
            }

    def __enter__(self) -> RunRecorder:  # noqa: PYI034 - Python 3.10 support
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _validate_snapshot_identity(
        self,
        path: Path,
        snapshot: ControllerSnapshot,
    ) -> None:
        if snapshot.config is not None:
            stored_config = _read_json(path / "config.json")
            incoming_config = _json_value(snapshot.config.to_dict())
            if stored_config != incoming_config:
                raise RunConflictError("snapshot config differs from the run config")
        if snapshot.x is not None:
            stored_x = _readonly_loaded_array(path / "x.npy")
            incoming_x = _readonly_signal(snapshot.x, "snapshot.x")
            if not np.array_equal(stored_x, incoming_x):
                raise RunConflictError("snapshot x differs from the run reference")

    def _record_iteration(
        self,
        run_path: Path,
        record: IterationRecord,
    ) -> tuple[dict[str, Any], bool]:
        if not isinstance(record, IterationRecord):
            raise TypeError("snapshot.records must contain IterationRecord objects")
        relative_dir = Path("iterations") / f"{record.iteration:06d}"
        iterations_root = _ensure_child_directory(run_path / "iterations", run_path)
        iteration_path = _ensure_child_directory(
            iterations_root / f"{record.iteration:06d}", iterations_root
        )

        entry = {
            "iteration": record.iteration,
            "metadata": (relative_dir / "record.json").as_posix(),
            "y": (relative_dir / "y.npy").as_posix(),
            "z": (relative_dir / "z.npy").as_posix(),
            "aligned_average": (relative_dir / "aligned_average.npy").as_posix(),
        }
        metadata = _iteration_metadata(record, entry)
        changed = False
        changed = _write_immutable_npy(iteration_path / "y.npy", record.y) or changed
        changed = _write_immutable_npy(iteration_path / "z.npy", record.z) or changed
        changed = (
            _write_immutable_npy(
                iteration_path / "aligned_average.npy",
                record.preprocessing.aligned_average,
            )
            or changed
        )
        changed = (
            _write_immutable_json(iteration_path / "record.json", metadata) or changed
        )
        return entry, changed

    @staticmethod
    def _append_snapshot_events(
        events: list[dict[str, Any]],
        snapshot: ControllerSnapshot,
    ) -> bool:
        changed = False
        state_fingerprint = {
            "kind": "state",
            "message": f"controller state changed to {snapshot.state.value}",
            "details": {
                "status": snapshot.state.value,
                "iteration": snapshot.iteration,
                "active_operation": snapshot.active_operation,
            },
        }
        last_state = next(
            (item for item in reversed(events) if item.get("kind") == "state"), None
        )
        if last_state is None or _event_fingerprint(last_state) != state_fingerprint:
            events.append(
                {
                    "sequence": len(events),
                    "timestamp": _utc_now(),
                    **state_fingerprint,
                }
            )
            changed = True

        if snapshot.last_error is not None:
            error_details = _controller_error_dict(snapshot.last_error)
            error_fingerprint = {
                "kind": "error",
                "message": snapshot.last_error.message,
                "details": error_details,
            }
            last_error = next(
                (item for item in reversed(events) if item.get("kind") == "error"),
                None,
            )
            if (
                last_error is None
                or _event_fingerprint(last_error) != error_fingerprint
            ):
                events.append(
                    {
                        "sequence": len(events),
                        "timestamp": _utc_now(),
                        **error_fingerprint,
                    }
                )
                changed = True
        return changed


# Public compatibility name for integrations that prefer "handle" terminology.
RunHandle = RunRecorder


def _snapshot_metadata(snapshot: ControllerSnapshot) -> dict[str, Any]:
    return _json_value(
        {
            "state": snapshot.state.value,
            "connected": snapshot.connected,
            "configured": snapshot.configured,
            "reference_loaded": snapshot.reference_loaded,
            "transmitting": snapshot.transmitting,
            "stop_requested": snapshot.stop_requested,
            "active_operation": snapshot.active_operation,
            "iteration": snapshot.iteration,
            "max_iterations": snapshot.max_iterations,
            "gain_correction": snapshot.gain_correction,
            "locked_attenuation_db": snapshot.locked_attenuation_db,
            "latest_power_dbm": snapshot.latest_power_dbm,
            "device_type": snapshot.device_type,
            "completed_at": snapshot.completed_at,
            "reference_safety": snapshot.reference_safety,
            "reference_normalization": snapshot.reference_normalization,
            "last_error": snapshot.last_error,
        }
    )


def _iteration_metadata(
    record: IterationRecord,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    preprocessing = record.preprocessing
    return _json_value(
        {
            "iteration": record.iteration,
            "power_dbm": record.power_dbm,
            "attenuation_db": record.attenuation_db,
            "digital_safety": record.digital_safety,
            "preprocessing": _preprocessing_metadata(preprocessing),
            "runtime_metrics": record.runtime_metrics,
            "waveforms": {
                "y": entry["y"],
                "z": entry["z"],
                "aligned_average": entry["aligned_average"],
            },
        }
    )


def _preprocessing_metadata(result: PreprocessingResult) -> dict[str, Any]:
    return {
        "gain_correction": result.gain_correction,
        "gain_correction_db": result.gain_correction_db,
        "reference_rms": result.reference_rms,
        "aligned_average_rms": result.aligned_average_rms,
        "z_rms": result.z_rms,
        "aligned_average_nmse_db": result.aligned_average_nmse_db,
        "nmse_db": result.nmse_db,
        "segment_count": result.segment_count,
        "batch_diagnostics": result.batch_diagnostics,
    }


def _power_adjustment_dict(adjustment: PowerAdjustment) -> dict[str, float]:
    if not isinstance(adjustment, PowerAdjustment):
        raise TypeError("power_trace must contain PowerAdjustment objects")
    return {
        "attenuation_db": adjustment.attenuation_db,
        "power_dbm": adjustment.power_dbm,
        "gap_db": adjustment.gap_db,
    }


def _controller_error_dict(error: ControllerErrorInfo) -> dict[str, Any]:
    return {
        "operation": error.operation,
        "code": error.code,
        "exception_type": error.exception_type,
        "message": error.message,
        "shutdown_error": error.shutdown_error,
    }


def _event_fingerprint(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": event.get("kind"),
        "message": event.get("message"),
        "details": event.get("details"),
    }


def _validate_append_only_prefix(
    stored: list[Any], incoming: list[Any], label: str
) -> None:
    if len(incoming) < len(stored) or stored != incoming[: len(stored)]:
        raise RunConflictError(f"{label} conflicts with previously stored data")


def _validate_snapshot_records(snapshot: ControllerSnapshot) -> None:
    iterations: list[int] = []
    for record in snapshot.records:
        if not isinstance(record, IterationRecord):
            raise TypeError("snapshot.records must contain IterationRecord objects")
        iterations.append(record.iteration)
    if iterations != list(range(len(iterations))):
        raise RunConflictError(
            "snapshot records must be ordered, unique, and contiguous from iteration zero"
        )
    expected_iteration = iterations[-1] if iterations else None
    if snapshot.iteration != expected_iteration:
        raise RunConflictError(
            "snapshot iteration must identify the latest committed record"
        )


def _validate_snapshot_terminal(snapshot: ControllerSnapshot) -> None:
    if not isinstance(snapshot.device_type, str) or not snapshot.device_type.strip():
        raise RunConflictError("snapshot device_type must be a non-empty string")
    terminal = snapshot.state.value in _TERMINAL_STATES
    if terminal:
        if snapshot.completed_at is None:
            raise RunConflictError("terminal snapshot must include completed_at")
        _parse_utc_timestamp(snapshot.completed_at)
    elif snapshot.completed_at is not None:
        raise RunConflictError("non-terminal snapshot must not include completed_at")


def _normalize_terminal_state(value: ControllerState | str) -> str:
    normalized = value.value if isinstance(value, ControllerState) else value
    if not isinstance(normalized, str):
        raise TypeError("state must be a ControllerState or string")
    if normalized not in _TERMINAL_STATES:
        raise ValueError("state must be completed, failed, or stopped")
    return normalized


def _optional_string(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return value


def _read_cached_metadata(
    path: Path,
    run_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[str, str]:
    try:
        from .result_export import load_final_payload

        payload = load_final_payload(path)
    except Exception as exc:
        raise RunStorageError("cached final result is invalid") from exc
    config = json.loads(payload["config"])
    stored_x = _readonly_loaded_array(run_path / "x.npy")
    if not np.array_equal(stored_x, payload["x"]):
        raise RunConflictError("cached final result x conflicts with the run reference")
    stored_config = _read_json(run_path / "config.json")
    expected_config = dict(stored_config)
    if expected_config.get("runtime_name") == "basic_ilc":
        runtime_config = dict(expected_config.get("runtime_config", {}))
        runtime_config.setdefault("mu", 0.5)
        expected_config["runtime_config"] = runtime_config
    cached_config = {
        key: value for key, value in config.items() if key != "device_type"
    }
    if cached_config != expected_config:
        raise RunConflictError(
            "cached final result config conflicts with the run config"
        )

    iterations = manifest.get("iterations")
    expected_iterations = list(range(expected_config["max_iterations"] + 1))
    if (
        not isinstance(iterations, list)
        or [
            entry.get("iteration") for entry in iterations if isinstance(entry, Mapping)
        ]
        != expected_iterations
    ):
        raise RunConflictError("completed recovery requires all indexed run iterations")
    for entry in iterations:
        for key in ("metadata", "y", "z", "aligned_average"):
            artifact = run_path / entry[key]
            if artifact.is_symlink() or not artifact.is_file():
                raise RunConflictError(
                    "completed recovery requires every indexed iteration artifact"
                )
    snapshot_path = run_path / "snapshot.json"
    if (
        manifest.get("snapshot") != "snapshot.json"
        or snapshot_path.is_symlink()
        or not snapshot_path.is_file()
    ):
        raise RunConflictError("completed recovery requires final snapshot metadata")
    snapshot = _read_json(snapshot_path)
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("state") != ControllerState.COMPLETED.value
        or snapshot.get("completed_at") != payload["completed_at"]
    ):
        raise RunConflictError("final snapshot metadata conflicts with cached result")
    return payload["completed_at"], config["device_type"]


def _shared_store_state(root: Path) -> _SharedStoreState:
    with _SHARED_STATES_LOCK:
        state = _SHARED_STATES.get(root)
        if state is None:
            state = _SharedStoreState()
            _SHARED_STATES[root] = state
        return state


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, complex):
        return {
            "$type": "complex",
            "real": _json_value(value.real),
            "imag": _json_value(value.imag),
        }
    if isinstance(value, datetime):
        return _datetime_to_utc(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("JSON object keys must be non-empty strings")
            normalized[key] = _json_value(item)
        return normalized
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def _atomic_write_json(path: Path, value: Any) -> None:
    normalized = _json_value(value)
    payload = (
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write_bytes(path, payload)


def _atomic_write_json_if_changed(path: Path, value: Any) -> bool:
    normalized = _json_value(value)
    if path.exists():
        existing = _read_json(path)
        if existing == normalized:
            return False
    _atomic_write_json(path, normalized)
    return True


def _write_immutable_json(path: Path, value: Any) -> bool:
    normalized = _json_value(value)
    if path.exists():
        try:
            existing = _read_json(path)
        except Exception as exc:
            raise RunConflictError(
                f"stored JSON artifact {path.name!r} is invalid"
            ) from exc
        if existing != normalized:
            raise RunConflictError(f"immutable JSON artifact {path.name!r} differs")
        return False
    _atomic_write_json(path, normalized)
    return True


def _atomic_write_npy(path: Path, value: np.ndarray) -> None:
    _require_real_directory(path.parent)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temp_path, path)
        temp_path = None
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _write_immutable_npy(path: Path, value: np.ndarray) -> bool:
    normalized = _readonly_signal(value, path.stem)
    if path.exists():
        try:
            existing = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise RunConflictError(f"stored waveform {path.name!r} is invalid") from exc
        if not np.array_equal(existing, normalized):
            raise RunConflictError(f"immutable waveform {path.name!r} differs")
        return False
    _atomic_write_npy(path, normalized)
    return True


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    _require_real_directory(path.parent)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temp_path, path)
        temp_path = None
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _readonly_loaded_array(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    copied = np.array(value, copy=True, order="C")
    return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)


def _readonly_signal(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must contain numeric samples")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.array(raw, dtype=np.complex128, order="C", copy=True)
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{name} must contain only finite samples")
    return np.frombuffer(copied.tobytes(), dtype=np.complex128)


def _ensure_child_directory(path: Path, parent: Path) -> Path:
    parent_resolved = _require_real_directory(parent)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise RunStorageError(f"artifact directory {path.name!r} is unsafe")
    else:
        path.mkdir(mode=0o700)
    resolved = _require_real_directory(path)
    if resolved.parent != parent_resolved:
        raise RunStorageError(f"artifact directory {path.name!r} escaped its run")
    return resolved


def _require_real_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RunStorageError(f"artifact parent {path!s} must be a real directory")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise RunStorageError(f"artifact parent {path!s} is unavailable") from exc


def _valid_manifest_shape(manifest: Mapping[str, Any]) -> bool:
    try:
        if not isinstance(manifest["status"], str):
            return False
        if manifest.get("config") != "config.json":
            return False
        if manifest.get("x") != "x.npy":
            return False
        if manifest.get("events") != "events.json":
            return False
        if manifest.get("power_trace") != "power_trace.json":
            return False
        if manifest.get("snapshot") not in (None, "snapshot.json"):
            return False
        device_type = manifest.get("device_type")
        if device_type is not None and (
            not isinstance(device_type, str) or not device_type.strip()
        ):
            return False
        if manifest.get("final_result") not in (None, "final_result.mat"):
            return False
        _parse_utc_timestamp(manifest["created"])
        _parse_utc_timestamp(manifest["updated"])
        completed = manifest["completed"]
        if completed is not None:
            _parse_utc_timestamp(completed)
        iterations = manifest["iterations"]
        if not isinstance(iterations, list):
            return False
        seen: set[int] = set()
        for entry in iterations:
            if not isinstance(entry, dict):
                return False
            iteration = _non_negative_integer(entry.get("iteration"), "iteration")
            if iteration in seen:
                return False
            seen.add(iteration)
            for key in ("metadata", "y", "z", "aligned_average"):
                if not _safe_relative_artifact(entry.get(key)):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _safe_relative_artifact(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _validate_run_id(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("run_id must be a string")
    if not _RUN_ID_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "run_id must start with an ASCII letter or digit and contain only "
            "ASCII letters, digits, dots, underscores, or hyphens"
        )
    return value


def _validate_event_kind(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("kind must be a string")
    if not _EVENT_KIND_PATTERN.fullmatch(value):
        raise ValueError(
            "kind must start with a lowercase letter and contain only lowercase "
            "letters, digits, underscores, or hyphens"
        )
    return value


def _normalize_now(value: datetime | float | None) -> float:
    if value is None:
        return time.time()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("now datetime must include timezone information")
        result = value.timestamp()
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("now must be a timestamp, timezone-aware datetime, or None")
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError("now must be finite")
    return result


def _utc_now() -> str:
    return _datetime_to_utc(datetime.now(timezone.utc))


def _datetime_to_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must include timezone information")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc_timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("stored timestamp must be a UTC ISO 8601 string")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    return parsed.timestamp()


def _non_negative_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _non_negative_finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _remove_controlled_run(path: Path, runs_root: Path, run_id: str) -> None:
    if path.is_symlink() or path.resolve(strict=True).parent != runs_root:
        raise RunStorageError("refusing to delete a run outside the controlled root")
    manifest = _read_json(path / "manifest.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RUN_SCHEMA_VERSION
        or manifest.get("run_id") != run_id
    ):
        raise RunStorageError("refusing to delete an uncontrolled run directory")
    shutil.rmtree(path)
    _fsync_directory(runs_root)


def _discard_new_run(path: Path, runs_root: Path) -> None:
    try:
        if not path.is_symlink() and path.resolve(strict=True).parent == runs_root:
            shutil.rmtree(path)
    except OSError:
        return


def _exception_code(name: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return words.removesuffix("_error") or "storage_error"


__all__ = [
    "DEFAULT_CLEANUP_INTERVAL_SECONDS",
    "DEFAULT_RETENTION_SECONDS",
    "RUN_SCHEMA_VERSION",
    "RunConflictError",
    "RunHandle",
    "RunNotFoundError",
    "RunRecorder",
    "RunStorageError",
    "RunStore",
]
