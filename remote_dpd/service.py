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
from .config import LegacyConfig, config_from_mat
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
        with self._lock:
            self.config = config
            self.engine = create_engine(self.engine_name, ILCConfig.from_legacy(config))
            if config.reset:
                self.state.reset()
                self._write_ack(1)
                time.sleep(0.05)
                save_mat(self.directory / DPD_IN_ACK_FILE, {"ACK": np.asarray(0, dtype=np.int8)})
                self.log.info("configuration reset acknowledged")
                return
            ack = 1 if self.state.reference is None else 0
            self._write_ack(ack)
            self.log.info("configuration loaded: mu=%s starting_sample=%s", config.ilc_mu, config.starting_sample)

    def _handle_dpd_input(self, event_path: Path) -> None:
        payload = load_mat(resolve_file(event_path.parent, event_path.name))
        value = first_value(payload, "DPD_In_cut", "DPD_in", "DPDin")
        reference = as_vector(value, "DPD_In_cut")
        start = max(0, self.config.starting_sample - 1)
        reference = reference[start:]
        with self._lock:
            changed = self.state.set_reference(reference)
        save_mat(self.directory / DPD_IN_ACK_FILE, {"ACK_DPDin": np.asarray(1, dtype=np.int8)})
        self.log.info("DPD input received: samples=%d new_session=%s", reference.size, changed)

    def _handle_feedback(self, event_path: Path) -> None:
        payload = load_mat(resolve_file(event_path.parent, event_path.name))
        feedback = as_vector(first_value(payload, "FB_Signal_cut", "FB_Signal", "feedback"), "FB_Signal_cut")
        reference = self.state.reference
        if reference is None:
            raise MatProtocolError("FB_Signal received before DPD_in")
        feedback_id = waveform_fingerprint(feedback)
        with self._lock:
            if feedback_id == self.state.last_feedback_id:
                self.log.info("duplicate FB_Signal ignored")
                return
            result = self.engine.process(reference, self.state.current_dpd, feedback, self.state)
            self.state.current_dpd = result.output
            self.state.last_feedback_id = feedback_id
            self.state.last_metrics = result.metrics
            iteration = self.state.iteration
            self.state.iteration += 1
        output = np.concatenate((np.zeros(max(0, self.config.starting_sample - 1), dtype=np.complex128), result.output))
        output_payload = {"DPDout_Nokia": output, "iter": np.asarray(iteration, dtype=np.int64)}
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
        self.log.info("iteration=%d samples=%d aligned_nmse=%.3f dB evm=%.3f%%", iteration, output.size, metrics["aligned_nmse_db"], metrics["symbol_evm_mean_percent"])

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
