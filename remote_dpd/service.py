"""Resident file-exchange service for the legacy remote DPD protocol."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from .algorithms import DPDEngine, ILCConfig, create_engine
from .config import LegacyConfig, algorithm_config_fingerprint, config_from_mat
from .dsp import align_and_average, nmse_db
from .exceptions import MatProtocolError
from .metrics import symbol_evm
from .protocol import (
    CONFIG_ACK_FILE,
    CONFIG_FILE,
    DPD_IN_ACK_FILE,
    DPD_IN_FILE,
    DPD_OUT_FILE,
    FB_SIGNAL_FILE,
    HEARTBEAT_FILE,
    SYMBOL_EVM_FILE,
    as_vector,
    first_value,
    load_mat,
    resolve_file,
    save_mat,
)
from .state import SessionState, waveform_fingerprint


@dataclass(slots=True)
class ServiceOptions:
    heartbeat_seconds: float = 1800.0
    stable_seconds: float = 0.15
    settle_timeout_seconds: float = 20.0
    poll_seconds: float = 0.5


class RemoteDPDService:
    """Own the transport/session lifecycle while delegating math to an engine."""

    def __init__(
        self,
        directory: str | Path,
        *,
        supplier_name: str = "default",
        engine_name: str = "ilc",
        options: ServiceOptions | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.supplier_name = supplier_name or self.directory.name or "default"
        self.engine_name = engine_name.lower()
        self.options = options or ServiceOptions()
        self.log = logger or logging.getLogger(f"remote_dpd.{self.supplier_name}")
        self.state = SessionState()
        self.config = LegacyConfig(supplier_name=self.supplier_name)
        self.engine: DPDEngine = create_engine(self.engine_name, ILCConfig.from_legacy(self.config))
        self._observer: Any = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_counter = 0
        self._last_event: dict[str, float] = {}

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError as exc:
            raise RuntimeError("watchdog is required to run the resident watcher") from exc

        service = self

        class Handler(FileSystemEventHandler):
            def on_created(self, event: Any) -> None:
                if not event.is_directory:
                    service._on_file_event(Path(event.src_path))

            def on_modified(self, event: Any) -> None:
                if not event.is_directory:
                    service._on_file_event(Path(event.src_path))

            def on_moved(self, event: Any) -> None:
                if not event.is_directory:
                    service._on_file_event(Path(event.dest_path))

        observer = Observer()
        observer.schedule(Handler(), str(self.directory), recursive=True)
        observer.start()
        self._observer = observer
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="remote-dpd-heartbeat", daemon=True)
        self._heartbeat_thread.start()
        self.log.info("watcher started: directory=%s supplier=%s engine=%s", self.directory, self.supplier_name, self.engine_name)

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop_event.wait(self.options.poll_seconds):
                pass
        except KeyboardInterrupt:
            self.log.info("interrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        self.log.info("watcher stopped")

    def process_file(self, path: str | Path) -> None:
        """Process one event synchronously; useful for integration tests."""
        path = Path(path)
        stem = path.stem
        if stem == "safeBack":
            self._safe_back(path.parent)
            return
        if stem == CONFIG_FILE.removesuffix(".mat"):
            self._handle_config(path)
        elif stem == DPD_IN_FILE.removesuffix(".mat"):
            self._handle_dpd_input(path)
        elif stem == FB_SIGNAL_FILE.removesuffix(".mat"):
            self._handle_feedback(path)

    def _on_file_event(self, path: Path) -> None:
        if path.parent != self.directory and self.directory not in path.parents:
            return
        stem = path.stem
        if stem in {HEARTBEAT_FILE.removesuffix(".txt"), CONFIG_ACK_FILE.removesuffix(".mat"),
                    DPD_IN_ACK_FILE.removesuffix(".mat"), DPD_OUT_FILE.removesuffix(".mat"),
                    SYMBOL_EVM_FILE.removesuffix(".mat")}:
            return
        now = time.monotonic()
        previous = self._last_event.get(str(path), 0.0)
        if now - previous < self.options.stable_seconds:
            return
        self._last_event[str(path)] = now
        try:
            self._wait_until_stable(path)
            self.process_file(path)
        except Exception:
            self.log.exception("failed processing event %s", path)

    def _wait_until_stable(self, path: Path) -> None:
        deadline = time.monotonic() + self.options.settle_timeout_seconds
        previous: tuple[int, int] | None = None
        while time.monotonic() < deadline:
            try:
                stat = path.stat()
            except FileNotFoundError:
                return
            current = (stat.st_size, stat.st_mtime_ns)
            if current == previous:
                return
            previous = current
            time.sleep(self.options.stable_seconds)
        self.log.warning("file did not settle before timeout: %s", path)

    def _handle_config(self, event_path: Path) -> None:
        path = resolve_file(event_path.parent, event_path.name)
        payload = load_mat(path)
        config = config_from_mat(payload, supplier_name=self.supplier_name)
        engine = create_engine(self.engine_name, ILCConfig.from_legacy(config))
        config_id = algorithm_config_fingerprint(config)
        with self._lock:
            effective_mode = _effective_mode(self.engine_name, config.ilc_backward_mode)
            config_changed = (
                self.state.last_config_id is not None
                and self.state.last_config_id != config_id
            )
            self.config = config
            self.engine = engine
            if config.reset or (config_changed and effective_mode != "legacy"):
                self.state.reset()
                self.state.last_config_id = config_id
                self._write_ack(1)
                if config.reset:
                    time.sleep(0.05)
                    save_mat(self.directory / DPD_IN_ACK_FILE, {"ACK": np.asarray(0, dtype=np.int8)})
                self.log.info(
                    "configuration reset acknowledged: explicit=%s changed=%s mode=%s",
                    config.reset,
                    config_changed,
                    effective_mode,
                )
                return
            self.state.last_config_id = config_id
            ack = 1 if self.state.reference is None else 0
            self._write_ack(ack)
            self.log.info("configuration loaded: mu=%s starting_sample=%s", config.ilc_mu, config.starting_sample)

    def _handle_dpd_input(self, event_path: Path) -> None:
        payload = load_mat(resolve_file(event_path.parent, event_path.name))
        value = first_value(payload, "DPD_In_cut", "DPD_in", "DPDin")
        reference = as_vector(value, "DPD_In_cut")
        start = max(0, self.config.starting_sample - 1)
        reference = reference[start:]
        if reference.size == 0:
            raise MatProtocolError("DPD_In_cut is empty after StartingSample cropping")
        incoming_session = _optional_binding_text(
            payload,
            "session_id",
            "SessionID",
            "SessionId",
            "IT_ID",
        )
        with self._lock:
            if (
                incoming_session is not None
                and self.state.external_session_id is not None
                and incoming_session != self.state.external_session_id
            ):
                self.state.reset()
            changed = self.state.set_reference(reference)
            if incoming_session is not None:
                self.state.external_session_id = incoming_session
            input_id = self.state.last_input_id
        save_mat(
            self.directory / DPD_IN_ACK_FILE,
            {
                "ACK_DPDin": np.asarray(1, dtype=np.int8),
                "DPDInputID": input_id or "",
                "expectedFeedbackIteration": np.asarray(1, dtype=np.int64),
            },
        )
        self.log.info("DPD input received: samples=%d new_session=%s", reference.size, changed)

    def _handle_feedback(self, event_path: Path) -> None:
        payload = load_mat(resolve_file(event_path.parent, event_path.name))
        feedback = as_vector(first_value(payload, "FB_Signal_cut", "FB_Signal", "feedback"), "FB_Signal_cut")
        feedback_id = waveform_fingerprint(feedback)
        with self._lock:
            reference = self.state.reference
            if reference is None:
                raise MatProtocolError("FB_Signal received before DPD_in")
            binding_verified = self._validate_feedback_binding(payload)
            if feedback_id == self.state.last_feedback_id:
                self.log.info("duplicate FB_Signal ignored")
                return
            result = self.engine.process(reference, self.state.current_dpd, feedback, self.state)
            self.state.current_dpd = result.output
            self.state.last_feedback_id = feedback_id
            self.state.last_output_id = waveform_fingerprint(result.output)
            self.state.feedback_binding_verified = binding_verified
            result.metrics["feedback_binding_verified"] = binding_verified
            self.state.last_metrics = result.metrics
            iteration = self.state.iteration
            self.state.iteration += 1
            output_id = self.state.last_output_id
            external_session = self.state.external_session_id
        output = np.concatenate((np.zeros(max(0, self.config.starting_sample - 1), dtype=np.complex128), result.output))
        output_payload = {
            "DPDout_Nokia": output,
            "iter": np.asarray(iteration, dtype=np.int64),
            "nextFeedbackIteration": np.asarray(iteration + 1, dtype=np.int64),
            "DPDOutputID": output_id or "",
        }
        if external_session is not None:
            output_payload["session_id"] = external_session
        for name in ("ITNum", "IT_ID"):
            if name in payload:
                output_payload[name] = payload[name]
        save_mat(self.directory / DPD_OUT_FILE, output_payload)
        evm = symbol_evm(
            result.aligned_feedback,
            reference,
            bandwidth_hz=self.config.bandwidth_hz,
            sample_rate_hz=self.config.sample_rate_hz,
        )
        save_mat(self.directory / SYMBOL_EVM_FILE, {"symbolEVM": evm})
        metrics = result.metrics | {"symbol_evm_mean_percent": float(np.nanmean(evm)) if evm.size else float("nan")}
        with self._lock:
            self.state.last_metrics = metrics
        self.log.info(
            "iteration=%d samples=%d aligned_nmse=%.3f dB evm=%.3f%% binding_verified=%s",
            iteration,
            output.size,
            metrics["aligned_nmse_db"],
            metrics["symbol_evm_mean_percent"],
            binding_verified,
        )

    def _validate_feedback_binding(self, payload: dict[str, Any]) -> bool:
        """Validate optional modern binding fields without breaking old files."""
        verified = False
        session = _optional_binding_text(
            payload,
            "session_id",
            "SessionID",
            "SessionId",
            "IT_ID",
        )
        if session is not None:
            if self.state.external_session_id is None:
                raise MatProtocolError("feedback has a session identifier but DPD input did not")
            if session != self.state.external_session_id:
                raise MatProtocolError(
                    f"feedback session {session!r} does not match {self.state.external_session_id!r}"
                )
            verified = True

        iteration_value = _optional_binding_integer(
            payload,
            "iteration",
            "Iteration",
            "feedbackIteration",
            "ITNum",
        )
        if iteration_value is not None:
            if iteration_value != self.state.iteration:
                raise MatProtocolError(
                    f"feedback iteration {iteration_value} does not match expected {self.state.iteration}"
                )
            verified = True

        supplied_waveform_id = _optional_binding_text(
            payload,
            "DPDInputID",
            "DPDOutputID",
            "input_id",
            "InputID",
        )
        if supplied_waveform_id is not None:
            expected = self.state.last_output_id or self.state.last_input_id
            if expected is None or supplied_waveform_id != expected:
                raise MatProtocolError(
                    f"feedback waveform identifier {supplied_waveform_id!r} does not match expected input"
                )
            verified = True
        return verified

    def _write_ack(self, value: int) -> None:
        save_mat(self.directory / CONFIG_ACK_FILE, {"ACK": np.asarray(value, dtype=np.int8), "timestamp": datetime.now(timezone.utc).isoformat()})

    def _heartbeat_loop(self) -> None:
        interval = max(0.1, float(self.options.heartbeat_seconds))
        while not self._stop_event.wait(interval):
            self._heartbeat_counter += 1
            path = self.directory / HEARTBEAT_FILE
            path.write_text(f"{self._heartbeat_counter}\n", encoding="ascii")

    def _safe_back(self, directory: Path) -> None:
        if directory.resolve() != self.directory:
            self.log.warning("safeBack outside watched root ignored: %s", directory)
            return
        removed = 0
        for child in self.directory.iterdir():
            if child.is_file() and child.name != "safeBack":
                child.unlink()
                removed += 1
        self.state.reset()
        self.log.info("safeBack cleared %d files and reset state", removed)


def _effective_mode(engine_name: str, configured_mode: str) -> str:
    aliases = {
        "legacy_ilc": "legacy",
        "linear_ilc": "linear",
        "instantaneous_gain_ilc": "instantaneous_gain",
        "model_vjp_ilc": "model_vjp",
        "model_lm_ilc": "model_lm",
    }
    return aliases.get(engine_name.lower(), configured_mode)


def _optional_binding_text(payload: dict[str, Any], *names: str) -> str | None:
    for name in names:
        if name not in payload:
            continue
        array = np.asarray(payload[name])
        if array.size != 1:
            raise MatProtocolError(f"binding field {name} must be scalar")
        item = array.reshape(-1)[0]
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="strict")
        if isinstance(item, np.generic):
            item = item.item()
        value = str(item).strip()
        if not value:
            raise MatProtocolError(f"binding field {name} must not be empty")
        return value
    return None


def _optional_binding_integer(payload: dict[str, Any], *names: str) -> int | None:
    for name in names:
        if name not in payload:
            continue
        array = np.asarray(payload[name])
        if array.size != 1:
            raise MatProtocolError(f"binding field {name} must be scalar")
        try:
            numeric = float(array.reshape(-1)[0])
            value = int(numeric)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MatProtocolError(f"binding field {name} must be an integer") from exc
        if not np.isfinite(numeric) or numeric != value or value < 1:
            raise MatProtocolError(f"binding field {name} must be a positive integer")
        return value
    return None
