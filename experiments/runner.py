"""Resumable, checksum-verified runner for the preregistered simulations."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from experiments.runtime import apply_numeric_thread_limits

# Apply the process policy before importing NumPy, PyTorch, or a BLAS consumer.
apply_numeric_thread_limits()

import numpy as np

from experiments.config import (
    CORE_METHODS,
    DEFAULT_RESOLVED_METHODS,
    ExperimentProtocol,
    PILOT_CANDIDATES,
    PILOT_COMPUTE_COSTS,
    PILOT_LOCKED_STUDIES,
    ResourceLimits,
    TrajectorySpec,
    canonical_json,
    enumerate_study,
    matrix_hash,
    resolved_method_parameters,
    stable_hash,
)
from experiments.statistics import select_pilot_candidates
from experiments.metrics import (
    auec,
    bilateral_aclr_db,
    binned_amam_ampm_error,
    fixed_domain_nmse,
    fixed_domain_nmse_db,
    known_grid_evm,
    papr_db,
)
from experiments.scenarios import (
    PAScenario,
    make_amam_scenario,
    make_ampm_scenario,
    make_gain_rolloff_stress,
    make_hammerstein_pa,
    make_hard_saturation_stress,
    make_wiener_pa,
    scale_to_peak,
    scale_to_rms,
)
from experiments.waveforms import OFDMWaveformConfig, generate_ofdm_waveform, named_seed_sequence
from remote_dpd.algorithms import legacy_ilc_update
from remote_dpd.dsp import legacy_gain_phase_calibration
from remote_dpd.learning import (
    InputSafetyLimits,
    LearningStepResult,
    StopReason,
    instantaneous_gain_ilc_step,
    input_within_safety_limits,
    linear_ilc_step,
    model_lm_ilc_step,
    model_vjp_ilc_step,
    project_input_safety,
    real_inner,
    signal_rms,
    signal_peak,
)
from remote_dpd.pa_model import MemoryPolynomialModel, PAForwardModelConfig, fit_pa_model


REPRESENTATIVE_ITERATIONS = frozenset((0, 1, 2, 5, 10, 20, 30))
ALGORITHM_FAILURE_REASONS = frozenset(
    (
        StopReason.NONFINITE_INPUT,
        StopReason.NONFINITE_UPDATE,
        StopReason.MODEL_FAILURE,
        StopReason.PROJECTION_FAILED,
    )
)


class RunnerError(RuntimeError):
    """Base exception for experiment infrastructure failures."""


class HashMismatchError(RunnerError):
    """Raised when a resume artifact belongs to different scientific code."""


class CapacityGateError(RunnerError):
    """Raised before new work when disk, memory, or time capacity is unsafe."""


class CorruptArtifactError(RunnerError):
    """Raised when an artifact checksum or expected identifier is invalid."""


@dataclass(frozen=True, slots=True)
class RunHashes:
    code_hash: str
    configuration_hash: str
    protocol_hash: str
    matrix_hash: str
    environment_hash: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CapacityReport:
    worker_count: int
    artifact_bytes: int
    free_disk_bytes: int
    current_rss_bytes: int
    allowed: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CapacityProbeReport:
    """Measured one-worker throughput and conservative full-protocol projection."""

    trajectory_id: str
    trajectory_seconds: float
    evaluations_per_second: float
    peak_worker_rss_bytes: int
    numeric_backend_max_threads: int
    projected_protocol_hours: float
    recommended_worker_count: int
    allowed: bool
    reason: str | None
    probe_directory: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunSummary:
    study: str
    expected_count: int
    completed_count: int
    newly_completed_count: int
    resumed_count: int
    algorithm_failure_count: int
    infrastructure_retry_count: int
    elapsed_seconds: float
    effective_worker_count: int
    run_directory: str

    @property
    def complete(self) -> bool:
        return self.completed_count == self.expected_count

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["complete"] = self.complete
        return result


@dataclass(frozen=True, slots=True)
class _WorkerContext:
    run_directory: str
    protocol: ExperimentProtocol
    hashes: RunHashes
    representative_seed_indices: tuple[int, ...]


class _OracleModel:
    """Adapt a differentiable synthetic PA to the learning model protocol."""

    def __init__(self, pa: Any):
        self.pa = pa

    def predict(self, input_signal: np.ndarray) -> np.ndarray:
        return np.asarray(self.pa.forward(input_signal), dtype=np.complex128)

    def jvp(self, input_signal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        return np.asarray(self.pa.jvp(input_signal, tangent), dtype=np.complex128)

    def vjp(self, input_signal: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        return np.asarray(self.pa.vjp(input_signal, cotangent), dtype=np.complex128)


class ExperimentRunner:
    """Prepare, execute, verify, and resume one or more experiment studies."""

    def __init__(
        self,
        output_directory: str | os.PathLike[str],
        *,
        protocol: ExperimentProtocol | None = None,
        resources: ResourceLimits | None = None,
        resolved: Mapping[str, Mapping[str, Any]] | None = None,
        pilot_lock: Mapping[str, Any] | None = None,
        project_root: str | os.PathLike[str] | None = None,
        generation_argv: Sequence[str] | None = None,
    ) -> None:
        apply_numeric_thread_limits()
        self.output_directory = Path(output_directory).resolve()
        self.protocol = ExperimentProtocol() if protocol is None else protocol
        self.resources = ResourceLimits() if resources is None else resources
        self.resolved = resolved_method_parameters(resolved)
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.pilot_lock = _normalize_pilot_lock(pilot_lock)
        if self.pilot_lock is not None and self.pilot_lock["resolved_payload"].get(
            "resolved_methods"
        ) != self.resolved:
            raise ValueError("pilot_lock resolved methods differ from runner methods")
        if self.pilot_lock is not None:
            _validate_pilot_lock_context(
                self.pilot_lock,
                self.protocol,
                self.project_root,
            )
        self.generation_argv = tuple(
            generation_argv
            if generation_argv is not None
            else (sys.executable, "-m", "experiments.run_experiments")
        )

    def capacity_report(self, worker_count: int | None = None) -> CapacityReport:
        """Inspect the hard disk/RSS gates without starting scientific work."""

        workers = self.resources.worker_count if worker_count is None else int(worker_count)
        if not self.resources.minimum_worker_count <= workers <= self.resources.maximum_worker_count:
            raise ValueError("worker_count lies outside the configured bounds")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        artifact_bytes = directory_size(self.output_directory)
        free_bytes = shutil.disk_usage(self.output_directory).free
        rss_bytes = current_rss_bytes()
        reason: str | None = None
        if artifact_bytes >= self.resources.artifact_budget_bytes:
            reason = "artifact_budget_exceeded"
        elif free_bytes < self.resources.minimum_free_disk_bytes:
            reason = "free_disk_below_gate"
        elif rss_bytes > self.resources.per_worker_rss_limit_bytes:
            reason = "controller_rss_exceeded"
        return CapacityReport(
            worker_count=workers,
            artifact_bytes=artifact_bytes,
            free_disk_bytes=free_bytes,
            current_rss_bytes=rss_bytes,
            allowed=reason is None,
            reason=reason,
        )

    def run_capacity_probe(
        self,
        spec: TrajectorySpec | None = None,
    ) -> CapacityProbeReport:
        """Measure one learned-LM trajectory before inspecting method differences."""

        selected = spec
        if selected is None:
            selected = next(
                item
                for item in enumerate_study("smoke", self.protocol, resolved=self.resolved)
                if item.scenario == "amam" and item.algorithm == "model_lm_ilc"
            )
        if selected.study != "smoke":
            raise ValueError("capacity probe must use a smoke trajectory")
        static_report = self.capacity_report(1)
        if not static_report.allowed:
            raise CapacityGateError(static_report.reason or "capacity gate rejected probe")
        cache_key = stable_hash(
            {
                "code_hash": compute_code_hash(self.project_root),
                "protocol_hash": self.protocol.protocol_hash,
                "resolved_methods": self.resolved,
                "trajectory": selected.as_dict(),
                "worker_count": self.resources.worker_count,
                "environment": environment_manifest(),
            }
        )
        probe_base = self.output_directory / "capacity_probe"
        cache_path = probe_base / f"{cache_key}.json"
        if cache_path.exists():
            cached = read_json(cache_path)
            if cached.get("cache_key") != cache_key or not isinstance(cached.get("report"), Mapping):
                raise CorruptArtifactError("capacity-probe cache is malformed")
            report = CapacityProbeReport(**dict(cached["report"]))
            probe_directory = Path(report.probe_directory)
            records = load_completed_records(probe_directory)
            if len(records) != 1 or records[0].get("trajectory_id") != selected.trajectory_id:
                raise CorruptArtifactError("capacity-probe cache does not match its shard")
            return report
        # Keep the nested probe path comfortably below the legacy Windows
        # MAX_PATH boundary; scientific identity remains in the full cache key.
        probe_root = probe_base / "r" / f"{cache_key[:12]}-{time.time_ns():x}"
        probe_resources = ResourceLimits(
            worker_count=1,
            minimum_worker_count=1,
            maximum_worker_count=self.resources.maximum_worker_count,
            per_worker_rss_limit_bytes=self.resources.per_worker_rss_limit_bytes,
            artifact_budget_bytes=self.resources.artifact_budget_bytes,
            minimum_free_disk_bytes=self.resources.minimum_free_disk_bytes,
            infrastructure_retries=self.resources.infrastructure_retries,
            maximum_estimated_hours=self.resources.maximum_estimated_hours,
        )
        probe_runner = ExperimentRunner(
            probe_root,
            protocol=self.protocol,
            resources=probe_resources,
            resolved=self.resolved,
            pilot_lock=self.pilot_lock,
            project_root=self.project_root,
            generation_argv=self.generation_argv,
        )
        probe_started = time.perf_counter()
        probe_runner.run(
            "smoke",
            [selected],
            worker_count=1,
            enforce_estimated_time_gate=False,
        )
        probe_wall_seconds = time.perf_counter() - probe_started
        record = load_completed_records(probe_root / "smoke")[0]
        seconds = float(probe_wall_seconds)
        peak_rss = int(record.get("peak_rss_bytes", 0))
        backend_threads = int(record.get("numeric_backend_max_threads", 0))
        from experiments.config import expected_study_counts

        counts = expected_study_counts(self.protocol)
        scientific_count = sum(count for name, count in counts.items() if name != "smoke")
        projected_hours = seconds * scientific_count / self.resources.worker_count / 3600.0
        recommended_workers = self.resources.worker_count
        reason: str | None = None
        if peak_rss > self.resources.per_worker_rss_limit_bytes:
            recommended_workers = max(
                self.resources.minimum_worker_count,
                self.resources.worker_count - 1,
            )
            if recommended_workers == self.resources.minimum_worker_count and peak_rss > self.resources.per_worker_rss_limit_bytes:
                reason = "single_worker_rss_exceeded"
        if projected_hours > self.resources.maximum_estimated_hours:
            reason = "projected_runtime_exceeded"
        if backend_threads != 1:
            reason = "numeric_backend_thread_limit_not_enforced"
        report = CapacityProbeReport(
            trajectory_id=selected.trajectory_id,
            trajectory_seconds=seconds,
            evaluations_per_second=self.protocol.evaluation_count / max(seconds, 1e-12),
            peak_worker_rss_bytes=peak_rss,
            numeric_backend_max_threads=backend_threads,
            projected_protocol_hours=projected_hours,
            recommended_worker_count=recommended_workers,
            allowed=reason is None,
            reason=reason,
            probe_directory=str(probe_root / "smoke"),
        )
        atomic_write_json(
            cache_path,
            {"cache_key": cache_key, "report": report.as_dict()},
        )
        return report

    def prepare(
        self,
        study: str,
        specs: Sequence[TrajectorySpec] | None = None,
    ) -> tuple[Path, RunHashes, tuple[TrajectorySpec, ...]]:
        """Write or validate the immutable manifest and expected-ID list."""

        expected = tuple(enumerate_study(study, self.protocol, resolved=self.resolved) if specs is None else specs)
        if not expected:
            raise ValueError("an experiment matrix must not be empty")
        if any(spec.study != study for spec in expected):
            raise ValueError("all trajectory specs must belong to the prepared study")
        if study in PILOT_LOCKED_STUDIES and self.pilot_lock is None:
            raise ValueError(
                f"{study} requires verified pilot-lock provenance, not only method parameters"
            )
        if study in PILOT_LOCKED_STUDIES:
            _validate_pilot_lock_context(
                self.pilot_lock,
                self.protocol,
                self.project_root,
            )
            frozen_resolved_lock = _ensure_frozen_resolved_lock(
                self.output_directory,
                self.pilot_lock,
            )
        else:
            frozen_resolved_lock = None
        ids = tuple(spec.trajectory_id for spec in expected)
        if len(set(ids)) != len(ids):
            raise ValueError("expected trajectory IDs contain duplicates")

        run_directory = self.output_directory / study
        for child in ("shards", "work", "waveforms"):
            (run_directory / child).mkdir(parents=True, exist_ok=True)
        environment = environment_manifest()
        hashes = RunHashes(
            code_hash=compute_code_hash(self.project_root),
            configuration_hash=stable_hash(
                {
                    "protocol": self.protocol.as_dict(),
                    "resolved_methods": self.resolved,
                    "resolved_hash": (
                        self.pilot_lock.get("resolved_hash")
                        if self.pilot_lock is not None
                        else None
                    ),
                }
            ),
            protocol_hash=self.protocol.protocol_hash,
            matrix_hash=matrix_hash(expected),
            environment_hash=stable_hash(environment),
        )
        manifest_path = run_directory / "manifest.json"
        manifest = self._manifest_payload(
            study,
            expected,
            hashes,
            environment,
            frozen_resolved_lock,
        )
        if manifest_path.exists():
            existing = read_json(manifest_path)
            _validate_hash_mapping(existing.get("hashes"), hashes, "manifest")
            if int(existing.get("expected_trajectory_count", -1)) != len(expected):
                raise HashMismatchError("manifest expected trajectory count changed")
        else:
            atomic_write_json(manifest_path, manifest)

        expected_payload = {
            "hashes": hashes.as_dict(),
            "trajectory_ids": sorted(ids),
        }
        expected_path = run_directory / "expected_ids.json"
        if expected_path.exists():
            existing = read_json(expected_path)
            _validate_hash_mapping(existing.get("hashes"), hashes, "expected IDs")
            if existing.get("trajectory_ids") != expected_payload["trajectory_ids"]:
                raise HashMismatchError("expected trajectory ID list changed")
        else:
            atomic_write_json(expected_path, expected_payload)

        specs_path = run_directory / "specs.json"
        specs_payload = {
            "hashes": hashes.as_dict(),
            "specs": [spec.as_dict() for spec in sorted(expected, key=lambda item: item.trajectory_id)],
        }
        if specs_path.exists():
            existing = read_json(specs_path)
            _validate_hash_mapping(existing.get("hashes"), hashes, "trajectory specs")
            if existing.get("specs") != specs_payload["specs"]:
                raise HashMismatchError("trajectory specifications changed")
        else:
            atomic_write_json(specs_path, specs_payload)
        self._write_seeds(run_directory, study, expected)
        return run_directory, hashes, expected

    def run(
        self,
        study: str,
        specs: Sequence[TrajectorySpec] | None = None,
        *,
        worker_count: int | None = None,
        enforce_estimated_time_gate: bool = True,
    ) -> RunSummary:
        """Execute missing trajectories and retain every algorithmic failure."""

        run_directory, hashes, expected = self.prepare(study, specs)
        started = time.perf_counter()
        completed_records = self._verified_completed(run_directory, hashes, expected)
        resumed_count = len(completed_records)
        pending = [spec for spec in expected if spec.trajectory_id not in completed_records]
        if not pending:
            summary = self._summary(
                study,
                expected,
                completed_records,
                0,
                resumed_count,
                0,
                0.0,
                worker_count or self.resources.worker_count,
                run_directory,
            )
            atomic_write_json(run_directory / "run_summary.json", summary.as_dict())
            return summary

        workers = self.resources.worker_count if worker_count is None else int(worker_count)
        if not self.resources.minimum_worker_count <= workers <= self.resources.maximum_worker_count:
            raise ValueError("worker_count lies outside the configured bounds")
        context = _WorkerContext(
            run_directory=str(run_directory),
            protocol=self.protocol,
            hashes=hashes,
            representative_seed_indices=(0,),
        )
        retry_count = 0
        newly_completed = 0
        peak_worker_rss = 0
        cursor = 0

        def retain(record: Mapping[str, Any], retries: int) -> None:
            nonlocal retry_count, newly_completed, peak_worker_rss
            completed_records[str(record["trajectory_id"])] = dict(record)
            peak_worker_rss = max(
                peak_worker_rss,
                int(record.get("peak_rss_bytes", 0)),
            )
            retry_count += retries
            newly_completed += 1

        while cursor < len(pending):
            self._validate_live_hashes(hashes, expected)
            report = self.capacity_report(workers)
            if not report.allowed:
                raise CapacityGateError(report.reason or "capacity gate rejected scheduling")
            if workers == 1:
                record, retries = self._execute_with_retries(pending[cursor], context)
                retain(record, retries)
                cursor += 1
            else:
                reduce_workers = False
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    future_to_spec: dict[Any, TrajectorySpec] = {}
                    while cursor < len(pending) and len(future_to_spec) < workers:
                        spec = pending[cursor]
                        future_to_spec[
                            executor.submit(_execute_trajectory_thread_limited, spec, context)
                        ] = spec
                        cursor += 1
                    completions_since_gate = 0
                    while future_to_spec:
                        future = next(as_completed(tuple(future_to_spec)))
                        spec = future_to_spec.pop(future)
                        try:
                            record = future.result()
                            retries = 0
                        except Exception:
                            record, retries = self._execute_with_retries(
                                spec,
                                context,
                                initial_failures=1,
                            )
                        retain(record, retries)
                        completions_since_gate += 1
                        if completions_since_gate >= workers:
                            self._validate_live_hashes(hashes, expected)
                            post_report = self.capacity_report(workers)
                            if not post_report.allowed:
                                raise CapacityGateError(
                                    post_report.reason or "capacity gate rejected scheduling"
                                )
                            if (
                                enforce_estimated_time_gate
                                and newly_completed >= max(3, workers)
                            ):
                                elapsed_so_far = time.perf_counter() - started
                                remaining_total = len(pending) - newly_completed
                                estimate_hours = (
                                    elapsed_so_far
                                    * remaining_total
                                    / newly_completed
                                    / 3600.0
                                )
                                if estimate_hours > self.resources.maximum_estimated_hours:
                                    raise CapacityGateError(
                                        "estimated remaining runtime "
                                        f"{estimate_hours:.2f} h exceeds gate"
                                    )
                            completions_since_gate = 0
                            if peak_worker_rss > self.resources.per_worker_rss_limit_bytes:
                                reduce_workers = True
                        if not reduce_workers and cursor < len(pending):
                            next_spec = pending[cursor]
                            future_to_spec[
                                executor.submit(
                                    _execute_trajectory_thread_limited,
                                    next_spec,
                                    context,
                                )
                            ] = next_spec
                            cursor += 1
                if reduce_workers:
                    workers = max(self.resources.minimum_worker_count, workers - 1)

            if peak_worker_rss > self.resources.per_worker_rss_limit_bytes and workers == 1:
                raise CapacityGateError("single-worker RSS exceeds the 2.5 GiB capacity gate")
            post_batch_report = self.capacity_report(workers)
            if not post_batch_report.allowed:
                raise CapacityGateError(post_batch_report.reason or "capacity gate rejected scheduling")
            if enforce_estimated_time_gate and newly_completed >= max(3, workers):
                remaining = len(pending) - cursor
                elapsed_so_far = time.perf_counter() - started
                estimate_hours = elapsed_so_far * remaining / newly_completed / 3600.0
                if estimate_hours > self.resources.maximum_estimated_hours:
                    raise CapacityGateError(
                        f"estimated remaining runtime {estimate_hours:.2f} h exceeds gate"
                    )

        self._validate_live_hashes(hashes, expected)
        verified = self._verified_completed(run_directory, hashes, expected)
        if set(verified) != {spec.trajectory_id for spec in expected}:
            raise CorruptArtifactError("completed shard IDs do not exactly match expected IDs")
        elapsed = time.perf_counter() - started
        summary = self._summary(
            study,
            expected,
            verified,
            newly_completed,
            resumed_count,
            retry_count,
            elapsed,
            workers,
            run_directory,
        )
        atomic_write_json(run_directory / "run_summary.json", summary.as_dict())
        return summary

    def _validate_live_hashes(
        self,
        expected_hashes: RunHashes,
        specs: Sequence[TrajectorySpec],
    ) -> None:
        """Stop scheduling if scientific source or configuration changed in place."""

        current = RunHashes(
            code_hash=compute_code_hash(self.project_root),
            configuration_hash=stable_hash(
                {
                    "protocol": self.protocol.as_dict(),
                    "resolved_methods": self.resolved,
                    "resolved_hash": (
                        self.pilot_lock.get("resolved_hash")
                        if self.pilot_lock is not None
                        else None
                    ),
                }
            ),
            protocol_hash=self.protocol.protocol_hash,
            matrix_hash=matrix_hash(specs),
            environment_hash=stable_hash(environment_manifest()),
        )
        if current != expected_hashes:
            raise HashMismatchError(
                "scientific code/config/protocol/matrix changed during the run"
            )

    def _execute_with_retries(
        self,
        spec: TrajectorySpec,
        context: _WorkerContext,
        *,
        initial_failures: int = 0,
    ) -> tuple[dict[str, Any], int]:
        retries = initial_failures
        while True:
            try:
                return _execute_trajectory_thread_limited(spec, context), retries
            except (HashMismatchError, CorruptArtifactError, CapacityGateError):
                raise
            except Exception as exc:
                if retries >= self.resources.infrastructure_retries:
                    raise RunnerError(
                        f"infrastructure failed for {spec.trajectory_id} after {retries} retries"
                    ) from exc
                retries += 1

    def _verified_completed(
        self,
        run_directory: Path,
        hashes: RunHashes,
        expected: Sequence[TrajectorySpec],
    ) -> dict[str, dict[str, Any]]:
        expected_ids = {spec.trajectory_id for spec in expected}
        completed: dict[str, dict[str, Any]] = {}
        for path in sorted((run_directory / "shards").glob("*.json")):
            record = read_verified_shard(path)
            trajectory_id = str(record.get("trajectory_id", ""))
            if path.stem != trajectory_id:
                raise CorruptArtifactError(
                    f"shard filename does not match trajectory ID: {path.name}"
                )
            if trajectory_id not in expected_ids:
                raise CorruptArtifactError(f"unexpected shard trajectory ID: {trajectory_id}")
            if trajectory_id in completed:
                raise CorruptArtifactError(f"duplicate shard trajectory ID: {trajectory_id}")
            _validate_hash_mapping(record.get("hashes"), hashes, f"shard {trajectory_id}")
            completed[trajectory_id] = record
        return completed

    def _manifest_payload(
        self,
        study: str,
        specs: Sequence[TrajectorySpec],
        hashes: RunHashes,
        environment: Mapping[str, Any],
        frozen_resolved_lock: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        git = git_state(self.project_root)
        protocol_document = self.project_root / "docs" / "current_plan.md"
        manifest_pilot_lock = None
        if self.pilot_lock is not None:
            manifest_pilot_lock = json.loads(canonical_json(self.pilot_lock))
            try:
                relative_resolved_path = os.path.relpath(
                    Path(self.pilot_lock["resolved_config_path"]),
                    self.output_directory / study,
                )
            except ValueError:
                relative_resolved_path = str(self.pilot_lock["resolved_config_path"])
            manifest_pilot_lock["resolved_config_path"] = Path(relative_resolved_path).as_posix()
        return {
            "schema_version": 1,
            "study": study,
            "created_utc": _utc_timestamp(),
            "hashes": hashes.as_dict(),
            "protocol_document_sha256": file_sha256(protocol_document)
            if protocol_document.exists()
            else None,
            "expected_trajectory_count": len(specs),
            "scientific_protocol": self.protocol.as_dict(),
            "resolved_methods": self.resolved,
            "pilot_lock": manifest_pilot_lock,
            "frozen_resolved_lock": frozen_resolved_lock,
            "runtime_limits": asdict(self.resources),
            "waveform_retention": {
                "representative_seed_indices": [0],
                "iterations": sorted(REPRESENTATIVE_ITERATIONS),
                "ordinary_waveforms_reconstructed_from_seed": True,
            },
            "fixed_calibration": {
                "coefficient_real": 1.0,
                "coefficient_imag": 0.0,
                "domain": "synthetic_pa_native_domain",
            },
            "cell_instances": self._cell_manifest_entries(study, specs),
            "environment": dict(environment),
            "git": git,
            "generation": {
                "argv": list(self.generation_argv),
                "cwd": str(Path.cwd().resolve()),
                "display_command": subprocess.list2cmdline(self.generation_argv),
            },
        }

    def _cell_manifest_entries(
        self,
        study: str,
        specs: Sequence[TrajectorySpec],
    ) -> list[dict[str, Any]]:
        """Materialize reachability and safety values once per paired cell/seed."""

        selected: dict[tuple[str, str, int, int], TrajectorySpec] = {}
        for spec in specs:
            key = (
                spec.scenario,
                spec.severity,
                spec.pa_seed_index,
                spec.waveform_seed_index,
            )
            selected.setdefault(key, spec)
        waveform_config = OFDMWaveformConfig(
            nfft=self.protocol.nfft,
            symbol_count=self.protocol.symbol_count,
            occupied_per_side=self.protocol.occupied_per_side,
            qam_order=self.protocol.qam_order,
            cyclic_prefix_length=self.protocol.cyclic_prefix_length,
        )
        entries: list[dict[str, Any]] = []
        for key, spec in sorted(selected.items()):
            waveform = generate_ofdm_waveform(
                named_seed_sequence(study, "waveform", spec.waveform_seed_index),
                waveform_config,
            )
            scenario = _make_scenario(spec, waveform.samples, self.protocol)
            limits = _safety_limits(scenario, self.protocol)
            entries.append(
                {
                    "scenario": key[0],
                    "severity": key[1],
                    "pa_seed_index": key[2],
                    "waveform_seed_index": key[3],
                    "scenario_metadata": _json_safe(dict(scenario.metadata)),
                    "safety_limits": _json_safe(asdict(limits)),
                    "envelope_bin_edges": _fixed_envelope_bin_edges(
                        limits,
                        self.protocol.envelope_bin_count,
                    ).tolist(),
                }
            )
        return entries

    @staticmethod
    def _write_seeds(
        run_directory: Path,
        study: str,
        specs: Sequence[TrajectorySpec],
    ) -> None:
        path = run_directory / "seeds.csv"
        rows: list[tuple[str, int, str, str]] = []
        pairs = sorted({(spec.pa_seed_index, spec.waveform_seed_index) for spec in specs})
        for pa_index, waveform_index in pairs:
            for role, index in (
                ("pa_pair", pa_index),
                ("waveform", waveform_index),
                ("noise", waveform_index),
            ):
                sequence = named_seed_sequence(study, role, index)
                words = ";".join(str(int(value)) for value in sequence.generate_state(4))
                spawn_key = ";".join(str(int(value)) for value in sequence.spawn_key)
                rows.append((role, index, words, spawn_key))
        if study == "dynamic":
            for family in ("wiener", "hammerstein"):
                for pa_index in sorted({spec.pa_seed_index for spec in specs}):
                    sequence = named_seed_sequence("dynamic_pa", family, pa_index)
                    words = ";".join(str(int(value)) for value in sequence.generate_state(4))
                    spawn_key = ";".join(str(int(value)) for value in sequence.spawn_key)
                    rows.append((f"dynamic_pa_{family}", pa_index, words, spawn_key))
        _atomic_write_csv(path, ("role", "index", "state_words", "spawn_key"), rows)

    @staticmethod
    def _summary(
        study: str,
        expected: Sequence[TrajectorySpec],
        completed: Mapping[str, Mapping[str, Any]],
        newly_completed: int,
        resumed: int,
        retries: int,
        elapsed: float,
        workers: int,
        run_directory: Path,
    ) -> RunSummary:
        failures = sum(record.get("status") == "algorithm_failure" for record in completed.values())
        return RunSummary(
            study=study,
            expected_count=len(expected),
            completed_count=len(completed),
            newly_completed_count=newly_completed,
            resumed_count=resumed,
            algorithm_failure_count=int(failures),
            infrastructure_retry_count=retries,
            elapsed_seconds=float(elapsed),
            effective_worker_count=workers,
            run_directory=str(run_directory),
        )


def _execute_trajectory(spec: TrajectorySpec, context: _WorkerContext) -> dict[str, Any]:
    """Execute one trajectory; algorithm failures become durable results."""

    apply_numeric_thread_limits()
    numeric_backend_max_threads = _numeric_backend_max_threads()
    started = time.perf_counter()
    run_directory = Path(context.run_directory)
    shard_path = run_directory / "shards" / f"{spec.trajectory_id}.json"
    if shard_path.exists():
        record = read_verified_shard(shard_path)
        _validate_hash_mapping(record.get("hashes"), context.hashes, "existing shard")
        return record

    waveform_config = OFDMWaveformConfig(
        nfft=context.protocol.nfft,
        symbol_count=context.protocol.symbol_count,
        occupied_per_side=context.protocol.occupied_per_side,
        qam_order=context.protocol.qam_order,
        cyclic_prefix_length=context.protocol.cyclic_prefix_length,
    )
    waveform_seed = named_seed_sequence(spec.study, "waveform", spec.waveform_seed_index)
    waveform = generate_ofdm_waveform(waveform_seed, waveform_config)
    scenario = _make_scenario(spec, waveform.samples, context.protocol)
    current = np.asarray(scenario.initial_input, dtype=np.complex128).copy()
    desired = np.asarray(scenario.desired, dtype=np.complex128)
    rng = np.random.default_rng(named_seed_sequence(spec.study, "noise", spec.waveform_seed_index))
    safety = _safety_limits(scenario, context.protocol)
    envelope_bin_edges = _fixed_envelope_bin_edges(
        safety,
        context.protocol.envelope_bin_count,
    )
    damping = max(
        float(spec.parameters.get("damping", 1e-3)),
        context.protocol.lm_minimum_damping,
    )
    metrics: list[dict[str, Any]] = []
    start_iteration = 0
    cached_model: MemoryPolynomialModel | None = None
    convergence_iteration: int | None = None
    diverged = False
    constraint_violation = False
    status = "completed"
    terminal_reason = "iteration_limit"
    work_key = hashlib.sha256(spec.trajectory_id.encode("utf-8")).hexdigest()[:20]
    work_directory = run_directory / "work" / work_key
    work_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_directory / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = _read_checkpoint(work_directory, context.hashes, spec)
        current = checkpoint["input"]
        start_iteration = int(checkpoint["next_iteration"])
        metrics = list(checkpoint["metrics"])
        damping = float(checkpoint["damping"])
        rng.bit_generator.state = checkpoint["prng_state"]
        cached_model = checkpoint["model"]
        convergence_iteration = checkpoint["convergence_iteration"]
        diverged = bool(checkpoint["diverged"])
        constraint_violation = bool(checkpoint["constraint_violation"])
        status = str(checkpoint["status"])
        terminal_reason = str(checkpoint["terminal_reason"])

    initial_nmse_db: float | None = None
    if metrics:
        initial_nmse_db = _finite_metric(metrics[0].get("nmse_db"))
    if convergence_iteration is None:
        convergence_iteration = _first_convergence_iteration(
            metrics,
            context.protocol.convergence_nmse_db,
            context.protocol.convergence_hold,
        )
    diverged = diverged or any(bool(item.get("diverged", False)) for item in metrics)
    constraint_violation = constraint_violation or any(
        bool(item.get("constraint_violation", False)) for item in metrics
    )
    model_fallback_count = sum(bool(item.get("model_fallback", False)) for item in metrics)

    for iteration in range(start_iteration, context.protocol.evaluation_count):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            clean_output = np.asarray(scenario.pa.forward(current), dtype=np.complex128).reshape(-1)
        if clean_output.size != current.size:
            raise ValueError("synthetic PA output length differs from its input")
        evaluation_valid = bool(
            np.all(np.isfinite(clean_output.real))
            and np.all(np.isfinite(clean_output.imag))
        )
        measured = np.zeros_like(current)
        if evaluation_valid:
            measured = _captured_measurement(clean_output, desired, spec, rng)
            evaluation_valid = bool(
                np.all(np.isfinite(measured.real))
                and np.all(np.isfinite(measured.imag))
            )
        if evaluation_valid:
            metric = _round_metrics(
                current,
                measured,
                desired,
                waveform,
                scenario,
                context.protocol,
                iteration,
                envelope_bin_edges,
            )
        else:
            status = "algorithm_failure"
            diverged = True
            terminal_reason = "nonfinite_evaluation"
            metric = _numeric_failure_metric(
                current,
                iteration,
                context.protocol.numeric_failure_nmse_db,
            )
        nmse_db_value = float(metric["nmse_db"])
        if initial_nmse_db is None:
            initial_nmse_db = nmse_db_value
        if status == "completed" and not diverged and _consecutive_threshold(
            metrics + [metric],
            "nmse_db",
            lambda value: value <= context.protocol.convergence_nmse_db,
            context.protocol.convergence_hold,
        ):
            if convergence_iteration is None:
                convergence_iteration = iteration - context.protocol.convergence_hold + 1
                terminal_reason = StopReason.CONVERGED
        if status == "completed" and not diverged and _consecutive_threshold(
            metrics + [metric],
            "nmse_db",
            lambda value: value > float(initial_nmse_db) + context.protocol.divergence_margin_db,
            context.protocol.divergence_hold,
        ):
            diverged = True
            metric["diverged"] = True
            terminal_reason = "diverged"

        if evaluation_valid and _is_representative(spec, context, iteration):
            _write_waveform_snapshot(
                run_directory / "waveforms" / spec.trajectory_id / f"k{iteration:03d}.npz",
                current,
                measured,
                desired,
            )

        next_input = current.copy()
        model_for_checkpoint = cached_model
        if (
            iteration < context.protocol.update_count
            and status != "algorithm_failure"
            and not diverged
        ):
            try:
                step, model_for_checkpoint, model_diagnostics = _learning_step(
                    spec,
                    scenario,
                    current,
                    desired,
                    measured,
                    safety,
                    damping,
                    iteration,
                    cached_model,
                    context.protocol,
                )
                metric.update(_json_safe(model_diagnostics))
                metric.update(_json_safe(dict(step.diagnostics)))
                metric.update(
                    {
                        "step_accepted": bool(step.accepted),
                        "step_stop_reason": step.stop_reason,
                        "trust_region_active": bool(step.trust_region_active),
                        "input_projection_active": bool(step.input_projection_active),
                        "saturation_limited": bool(step.saturation_limited),
                        "backtracks": int(step.backtracks),
                    }
                )
                violates_constraints = not _within_limits(step.next_input, safety)
                constraint_violation = constraint_violation or violates_constraints
                metric["constraint_violation"] = violates_constraints
                if bool(model_diagnostics.get("model_fallback", False)):
                    model_fallback_count += 1
                if violates_constraints:
                    diverged = True
                    metric["diverged"] = True
                    next_input = current.copy()
                    terminal_reason = "post_projection_constraint_violation"
                elif step.stop_reason in ALGORITHM_FAILURE_REASONS or step.stop_reason.startswith("cg_"):
                    status = "algorithm_failure"
                    terminal_reason = step.stop_reason
                else:
                    next_input = np.asarray(step.next_input, dtype=np.complex128)
                    terminal_reason = step.stop_reason
                if (
                    spec.algorithm in {"oracle_lm", "model_lm_ilc"}
                    and step.cg_result is not None
                    and "lm_damping" in step.diagnostics
                ):
                    damping = (
                        max(
                            context.protocol.lm_minimum_damping,
                            damping * context.protocol.lm_damping_accept_factor,
                        )
                        if step.accepted
                        else max(
                            context.protocol.lm_minimum_damping,
                            damping * context.protocol.lm_damping_reject_factor,
                        )
                    )
            except (ValueError, TypeError, ArithmeticError, RuntimeError) as exc:
                status = "algorithm_failure"
                terminal_reason = "algorithm_exception"
                metric["algorithm_error"] = f"{type(exc).__name__}: {exc}"

        metrics.append(_json_safe(metric))
        cached_model = model_for_checkpoint
        _write_checkpoint(
            work_directory,
            spec,
            context.hashes,
            next_iteration=iteration + 1,
            input_signal=next_input,
            prng_state=rng.bit_generator.state,
            damping=damping,
            model=cached_model,
            metrics=metrics,
            status=status,
            terminal_reason=terminal_reason,
            convergence_iteration=convergence_iteration,
            diverged=diverged,
            constraint_violation=constraint_violation,
        )
        current = next_input

    nmse_values = np.asarray([_finite_metric(item["nmse_db"]) for item in metrics], dtype=np.float64)
    valid_auec_input = bool(
        nmse_values.size
        and not np.any(np.isnan(nmse_values))
        and not np.any(np.isposinf(nmse_values))
    )
    result = {
        "schema_version": 1,
        "trajectory_id": spec.trajectory_id,
        "spec": spec.as_dict(),
        "hashes": context.hashes.as_dict(),
        "status": status,
        "terminal_reason": terminal_reason,
        "algorithm": spec.algorithm,
        "scenario": spec.scenario,
        "severity": spec.severity,
        "pa_seed_index": spec.pa_seed_index,
        "waveform_seed_index": spec.waveform_seed_index,
        "evaluation_count": len(metrics),
        "auec": float(auec(nmse_values)) if valid_auec_input else None,
        "final_nmse_db": float(nmse_values[-1]) if nmse_values.size else None,
        "convergence_iteration": convergence_iteration,
        "success": (
            convergence_iteration is not None
            and not diverged
            and not constraint_violation
            and status == "completed"
        ),
        "diverged": diverged,
        "constraint_violation": constraint_violation,
        "model_fallback_count": model_fallback_count,
        "scenario_metadata": _json_safe(dict(scenario.metadata)),
        "safety_limits": _json_safe(asdict(safety)),
        "envelope_bin_edges": envelope_bin_edges.tolist(),
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "numeric_backend_max_threads": numeric_backend_max_threads,
        "completed_utc": _utc_timestamp(),
    }
    safe_result = _json_safe(result)
    write_verified_shard(shard_path, safe_result)
    _remove_completed_checkpoint(work_directory)
    return safe_result


def _learning_step(
    spec: TrajectorySpec,
    scenario: PAScenario,
    current: np.ndarray,
    desired: np.ndarray,
    measured: np.ndarray,
    safety: InputSafetyLimits,
    damping: float,
    iteration: int,
    cached_model: MemoryPolynomialModel | None,
    protocol: ExperimentProtocol,
) -> tuple[LearningStepResult, MemoryPolynomialModel | None, dict[str, Any]]:
    parameters = spec.parameters
    learning_measured = measured
    diagnostics: dict[str, Any] = {}
    ablation = str(parameters.get("ablation", ""))
    if spec.algorithm == "legacy_ilc" or ablation == "legacy_dynamic_calibration":
        calibration = legacy_gain_phase_calibration(desired, measured)
        learning_measured = calibration * measured
        diagnostics.update(
            {
                "legacy_dynamic_calibration_real": calibration.real,
                "legacy_dynamic_calibration_imag": calibration.imag,
            }
        )

    if spec.algorithm == "no_dpd":
        return _held_step(current, "no_dpd"), cached_model, diagnostics
    if spec.algorithm == "linear_ilc":
        step = linear_ilc_step(
            current,
            desired,
            learning_measured,
            float(parameters["learning_rate"]),
            safety_limits=safety,
        )
        return step, cached_model, diagnostics
    if spec.algorithm == "legacy_ilc":
        proposed = legacy_ilc_update(
            desired,
            current,
            learning_measured,
            mu=float(parameters["learning_rate"]),
            alpha=0.0,
            gain_db=0.0,
            phase_compensate=bool(parameters.get("phase_precondition", True)),
            phase_threshold=float(parameters["legacy_phase_threshold"]),
            numeric_dtype="complex128",
        )
        projection = project_input_safety(proposed, safety)
        return (
            LearningStepResult(
                next_input=projection.projected_input if projection.feasible else current.copy(),
                update=(projection.projected_input - current)
                if projection.feasible
                else np.zeros_like(current),
                accepted=bool(projection.feasible and signal_rms(projection.projected_input - current) > 0),
                stop_reason=StopReason.ACCEPTED if projection.feasible else StopReason.PROJECTION_FAILED,
                input_projection_active=projection.active,
                diagnostics={"error_rms": signal_rms(learning_measured - desired)},
            ),
            cached_model,
            diagnostics,
        )
    if spec.algorithm == "instantaneous_gain_ilc":
        step = instantaneous_gain_ilc_step(
            current,
            desired,
            learning_measured,
            float(parameters["learning_rate"]),
            damping=float(parameters.get("damping", 1e-2)),
            input_threshold=float(parameters["instantaneous_gain_input_threshold"]),
            safety_limits=safety,
            saturation_tolerance=float(parameters["saturation_tolerance"]),
        )
        return step, cached_model, diagnostics

    numeric_dtype = np.dtype("complex64" if ablation == "complex64" else "complex128")
    working_current = np.asarray(current, dtype=numeric_dtype)
    working_desired = np.asarray(desired, dtype=numeric_dtype)
    working_measured = np.asarray(learning_measured, dtype=numeric_dtype)
    diagnostics["learning_numeric_dtype"] = numeric_dtype.name
    model: Any
    if spec.algorithm == "oracle_lm":
        model = _OracleModel(scenario.pa)
    else:
        should_fit = cached_model is None
        if ablation == "frozen_first_model":
            should_fit = cached_model is None
        elif ablation == "three_iteration_replay":
            should_fit = cached_model is None or iteration % 3 == 0
        else:
            should_fit = True
        fit_input = working_current
        fit_output = working_measured
        fit_failed = False
        if should_fit:
            orders = tuple(int(value) for value in parameters["model_orders"])
            model_config = PAForwardModelConfig(
                orders=orders,
                memory_depth=int(parameters["model_memory_depth"]),
                ridge=0.0 if ablation == "no_ridge" else float(parameters["model_ridge"]),
                envelope_quantile=float(parameters["model_envelope_quantile"]),
                block_size=int(parameters["model_block_size"]),
                validation_every=int(parameters["model_validation_every"]),
                minimum_scale=float(parameters["model_minimum_scale"]),
                column_rms_epsilon=float(parameters["model_column_rms_epsilon"]),
                max_condition_number=float(parameters["model_max_condition_number"]),
                max_validation_nmse_db=float(parameters["model_max_validation_nmse_db"]),
                numeric_dtype=numeric_dtype.name,
            )
            fit = fit_pa_model(fit_input, fit_output, model_config)
            diagnostics.update(fit.diagnostics.as_metrics())
            if fit.succeeded:
                cached_model = fit.model
            else:
                diagnostics["model_fallback"] = True
                fit_failed = True
        reuse_stale_model = ablation in {"frozen_first_model", "three_iteration_replay"}
        if fit_failed and not reuse_stale_model:
            cached_model = None
            fallback = linear_ilc_step(
                working_current,
                working_desired,
                working_measured,
                float(parameters["model_fallback_learning_rate"]),
                safety_limits=safety,
            )
            diagnostics["model_fallback_strategy"] = "linear_ilc"
            return fallback, None, diagnostics
        if cached_model is None:
            fallback = linear_ilc_step(
                working_current,
                working_desired,
                working_measured,
                float(parameters["model_fallback_learning_rate"]),
                safety_limits=safety,
            )
            diagnostics["model_fallback"] = True
            diagnostics["model_fallback_strategy"] = "linear_ilc"
            return fallback, None, diagnostics
        model = cached_model
        diagnostics.setdefault("model_fallback", False)
        diagnostics["learned_gradient_cosine"] = _gradient_cosine(
            model,
            _OracleModel(scenario.pa),
            working_current,
            working_measured - working_desired,
        )

    if spec.algorithm == "oracle_lm":
        diagnostics["learned_gradient_cosine"] = 1.0

    if spec.algorithm == "model_vjp_ilc" or ablation == "raw_vjp":
        step = model_vjp_ilc_step(
            working_current,
            working_desired,
            working_measured,
            model,
            float(parameters.get("learning_rate", 0.1)),
            safety_limits=safety,
            saturation_tolerance=float(parameters["saturation_tolerance"]),
        )
        return step, cached_model, diagnostics

    trust = parameters.get("trust_region_ratio", 0.25)
    trust_ratio = None if ablation == "no_trust_region" else float(trust)
    step = model_lm_ilc_step(
        working_current,
        working_desired,
        working_measured,
        model,
        damping=damping,
        step_size=float(parameters.get("step_size", 0.5)),
        cg_max_iterations=int(parameters["cg_max_iterations"]),
        cg_relative_tolerance=float(parameters["cg_relative_tolerance"]),
        trust_region_ratio=trust_ratio,
        safety_limits=safety,
        max_backtracks=int(parameters["lm_max_backtracks"]),
        backtrack_factor=float(parameters["lm_backtrack_factor"]),
        minimum_relative_decrease=float(parameters["lm_minimum_relative_decrease"]),
        saturation_tolerance=float(parameters["saturation_tolerance"]),
        prediction_mode="unanchored" if ablation == "unanchored_prediction" else "anchored",
    )
    return step, cached_model, diagnostics


def _make_scenario(
    spec: TrajectorySpec,
    waveform: np.ndarray,
    protocol: ExperimentProtocol,
) -> PAScenario:
    severity = float(spec.severity)
    if spec.scenario == "amam":
        return make_amam_scenario(
            waveform,
            severity,
            a_sat=protocol.amam_saturation,
            p=protocol.amam_smoothness,
        )
    if spec.scenario == "ampm":
        return make_ampm_scenario(
            waveform,
            severity,
            target_rms=protocol.ampm_target_rms,
            r0=protocol.ampm_r0,
        )
    if spec.scenario == "hard_saturation":
        return make_hard_saturation_stress(
            waveform,
            target_peak_ratio=float(spec.parameters["target_peak_ratio"]),
            a_sat=protocol.amam_saturation,
        )
    if spec.scenario == "gain_rolloff":
        return make_gain_rolloff_stress(
            waveform,
            target_peak=float(spec.parameters["target_peak"]),
            initial_input_peak=float(spec.parameters["initial_input_peak"]),
            turnover=float(spec.parameters["turnover"]),
        )
    if spec.scenario == "amam_dynamic":
        desired = scale_to_peak(waveform, severity * protocol.amam_saturation)
        pa = make_wiener_pa(spec.pa_seed_index, phase_max_deg=0.0, r0=protocol.ampm_r0)
        return PAScenario(
            name="dynamic_wiener_amam",
            desired=desired,
            initial_input=desired.copy(),
            pa=pa,
            metadata={"mechanism": "dynamic_wiener_amam", "reachable": True},
        )
    if spec.scenario == "ampm_dynamic":
        desired = scale_to_rms(waveform, protocol.ampm_target_rms)
        pa = make_hammerstein_pa(
            spec.pa_seed_index,
            phase_max_deg=severity,
            r0=protocol.ampm_r0,
        )
        return PAScenario(
            name="dynamic_hammerstein_ampm",
            desired=desired,
            initial_input=desired.copy(),
            pa=pa,
            metadata={"mechanism": "dynamic_hammerstein_ampm", "reachable": True},
        )
    raise ValueError(f"unsupported scenario: {spec.scenario}")


def _captured_measurement(
    clean_output: np.ndarray,
    desired: np.ndarray,
    spec: TrajectorySpec,
    rng: np.random.Generator,
) -> np.ndarray:
    capture_count = int(spec.parameters.get("capture_count", 1))
    if capture_count <= 0:
        raise ValueError("capture_count must be positive")
    snr_value = spec.parameters.get("snr_db", "inf")
    if isinstance(snr_value, str) and snr_value.lower() == "inf":
        return clean_output.copy()
    snr_db = float(snr_value)
    if math.isinf(snr_db) and snr_db > 0.0:
        return clean_output.copy()
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite or positive infinity")
    desired_rms = signal_rms(desired)
    noise_rms = desired_rms * 10.0 ** (-snr_db / 20.0)
    accumulated = np.zeros_like(clean_output)
    for _ in range(capture_count):
        noise = (
            rng.standard_normal(clean_output.size) + 1j * rng.standard_normal(clean_output.size)
        ) * (noise_rms / np.sqrt(2.0))
        accumulated += clean_output + noise
    return np.asarray(accumulated / capture_count, dtype=np.complex128)


def _round_metrics(
    current: np.ndarray,
    measured: np.ndarray,
    desired: np.ndarray,
    waveform: Any,
    scenario: PAScenario,
    protocol: ExperimentProtocol,
    iteration: int,
    envelope_bin_edges: np.ndarray,
) -> dict[str, Any]:
    nmse_ratio = fixed_domain_nmse(desired, measured)
    nmse_db_value = fixed_domain_nmse_db(desired, measured)
    aclr = bilateral_aclr_db(
        measured,
        nfft=protocol.nfft,
        occupied_per_side=protocol.occupied_per_side,
    )
    nonzero = np.flatnonzero(np.abs(waveform.samples) > 0.0)
    scale = desired[nonzero[0]] / waveform.samples[nonzero[0]] if nonzero.size else 1.0 + 0.0j
    reference_grid = waveform.grid * scale
    evm = known_grid_evm(measured, reference_grid, waveform.occupied_bins)
    binned = binned_amam_ampm_error(
        current,
        measured,
        desired,
        bin_edges=envelope_bin_edges,
    )
    low_power_mask = binned.bin_centers <= protocol.ampm_r0
    low_power_phase_rmse = _pooled_binned_rmse(
        binned.ampm_rmse_deg[low_power_mask],
        binned.phase_counts[low_power_mask],
    )
    amplitude_rmse = _pooled_binned_rmse(binned.amam_rmse, binned.counts)
    error = measured - desired
    (
        identity_gradient_cosine,
        identity_negative_local_fraction,
        identity_negative_inner_magnitude_fraction,
    ) = _identity_gradient_diagnostics(
        scenario.pa,
        current,
        error,
    )
    return {
        "iteration": iteration,
        "nmse": nmse_ratio,
        "nmse_db": nmse_db_value,
        "input_rms": signal_rms(current),
        "input_peak": signal_peak(current),
        "input_papr_db": papr_db(current),
        "aclr_lower_db": aclr.lower_db,
        "aclr_upper_db": aclr.upper_db,
        "aclr_worst_db": aclr.worst_db,
        "evm_raw_percent": evm.raw_percent,
        "evm_one_tap_percent": evm.one_tap_percent,
        "low_power_amplitude_threshold": protocol.ampm_r0,
        "low_power_phase_rmse_deg": low_power_phase_rmse,
        "binned_amam_rmse": amplitude_rmse,
        "envelope_bin_edges": binned.bin_edges.tolist(),
        "envelope_bin_counts": binned.counts.tolist(),
        "envelope_phase_counts": binned.phase_counts.tolist(),
        "binned_amam_rmse_values": _json_safe(binned.amam_rmse.tolist()),
        "binned_ampm_rmse_deg_values": _json_safe(binned.ampm_rmse_deg.tolist()),
        "identity_gradient_cosine": identity_gradient_cosine,
        "identity_negative_local_fraction": identity_negative_local_fraction,
        "identity_negative_inner_magnitude_fraction": (
            identity_negative_inner_magnitude_fraction
        ),
        "diverged": False,
        "constraint_violation": False,
    }


def _numeric_failure_metric(
    current: np.ndarray,
    iteration: int,
    penalty_nmse_db: float,
) -> dict[str, Any]:
    """Encode a deterministic non-finite PA evaluation with a fixed penalty."""

    penalty = float(penalty_nmse_db)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("numeric failure NMSE penalty must be finite and positive")
    return {
        "iteration": iteration,
        "nmse": float(10.0 ** (penalty / 10.0)),
        "nmse_db": penalty,
        "input_rms": signal_rms(current),
        "input_peak": signal_peak(current),
        "input_papr_db": papr_db(current),
        "aclr_lower_db": None,
        "aclr_upper_db": None,
        "aclr_worst_db": None,
        "evm_raw_percent": None,
        "evm_one_tap_percent": None,
        "low_power_amplitude_threshold": None,
        "low_power_phase_rmse_deg": None,
        "binned_amam_rmse": None,
        "identity_gradient_cosine": None,
        "identity_negative_local_fraction": None,
        "identity_negative_inner_magnitude_fraction": None,
        "evaluation_failure": "nonfinite_pa_or_capture",
        "diverged": True,
        "constraint_violation": False,
    }


def _safety_limits(
    scenario: PAScenario,
    protocol: ExperimentProtocol,
) -> InputSafetyLimits:
    initial = np.asarray(scenario.initial_input, dtype=np.complex128)
    initial_rms = signal_rms(initial)
    initial_peak = signal_peak(initial)
    required_peak = float(scenario.metadata.get("required_input_peak", initial_peak))
    return InputSafetyLimits(
        max_rms=max(protocol.safety_rms_multiplier * initial_rms, initial_rms + 1e-12),
        max_peak=max(
            protocol.safety_required_peak_multiplier * required_peak,
            protocol.safety_peak_multiplier * initial_peak,
        ),
        max_papr_db=papr_db(initial) + protocol.safety_papr_margin_db,
    )


def _fixed_envelope_bin_edges(
    limits: InputSafetyLimits,
    bin_count: int,
) -> np.ndarray:
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")
    if limits.max_peak is None or not math.isfinite(limits.max_peak) or limits.max_peak <= 0.0:
        raise ValueError("fixed envelope bins require a finite positive peak limit")
    upper = np.nextafter(float(limits.max_peak), np.inf)
    return np.linspace(0.0, upper, bin_count + 1, dtype=np.float64)


def _pooled_binned_rmse(values: np.ndarray, counts: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    valid = np.isfinite(values) & (counts > 0)
    total = int(np.sum(counts[valid]))
    if total == 0:
        return None
    return float(np.sqrt(np.sum(counts[valid] * values[valid] ** 2) / total))


def _identity_gradient_diagnostics(
    pa: Any,
    point: np.ndarray,
    error: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    try:
        oracle_gradient = np.asarray(pa.vjp(point, error), dtype=np.complex128)
        cosine = _vector_cosine(error, oracle_gradient)
        local_inner = np.real(np.conj(error) * oracle_gradient)
        local_scale = np.abs(error) * np.abs(oracle_gradient)
        threshold = np.finfo(np.float64).eps * max(
            1.0,
            float(np.max(local_scale)),
        )
        valid = np.isfinite(local_inner) & np.isfinite(local_scale) & (local_scale > threshold)
        if not np.any(valid):
            return cosine, None, None
        selected = local_inner[valid]
        negative_fraction = float(np.mean(selected < 0.0))
        absolute_sum = float(np.sum(np.abs(selected)))
        negative_magnitude_fraction = (
            float(np.sum(np.maximum(-selected, 0.0)) / absolute_sum)
            if absolute_sum > 0.0
            else None
        )
        return cosine, negative_fraction, negative_magnitude_fraction
    except (AttributeError, TypeError, ValueError, ArithmeticError, RuntimeError):
        return None, None, None


def _within_limits(value: np.ndarray, limits: InputSafetyLimits) -> bool:
    return input_within_safety_limits(value, limits)


def _gradient_cosine(model: Any, oracle: Any, point: np.ndarray, error: np.ndarray) -> float | None:
    try:
        learned = np.asarray(model.vjp(point, error), dtype=np.complex128)
        exact = np.asarray(oracle.vjp(point, error), dtype=np.complex128)
        return _vector_cosine(learned, exact)
    except (TypeError, ValueError, ArithmeticError, RuntimeError):
        return None


def _vector_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = math.sqrt(real_inner(left, left) * real_inner(right, right))
    if not math.isfinite(denominator) or denominator <= 0.0:
        return None
    return float(np.clip(real_inner(left, right) / denominator, -1.0, 1.0))


def _held_step(current: np.ndarray, reason: str) -> LearningStepResult:
    return LearningStepResult(
        next_input=current.copy(),
        update=np.zeros_like(current),
        accepted=False,
        stop_reason=reason,
    )


def _consecutive_threshold(
    metrics: Sequence[Mapping[str, Any]],
    key: str,
    predicate: Any,
    count: int,
) -> bool:
    if len(metrics) < count:
        return False
    values = [_finite_metric(item.get(key)) for item in metrics[-count:]]
    return all(not math.isnan(value) and bool(predicate(value)) for value in values)


def _first_convergence_iteration(
    metrics: Sequence[Mapping[str, Any]],
    threshold_db: float,
    hold_count: int,
) -> int | None:
    for end_index in range(hold_count - 1, len(metrics)):
        window = metrics[end_index - hold_count + 1 : end_index + 1]
        if _consecutive_threshold(
            window,
            "nmse_db",
            lambda value: value <= threshold_db,
            hold_count,
        ):
            return int(_finite_metric(window[0].get("iteration")))
    return None


def _is_representative(spec: TrajectorySpec, context: _WorkerContext, iteration: int) -> bool:
    return (
        spec.waveform_seed_index in context.representative_seed_indices
        and iteration in REPRESENTATIVE_ITERATIONS
    )


def _write_waveform_snapshot(
    path: Path,
    input_signal: np.ndarray,
    measured: np.ndarray,
    desired: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(path, input_signal=input_signal, measured=measured, desired=desired)


def _write_checkpoint(
    work_directory: Path,
    spec: TrajectorySpec,
    hashes: RunHashes,
    *,
    next_iteration: int,
    input_signal: np.ndarray,
    prng_state: Mapping[str, Any],
    damping: float,
    model: MemoryPolynomialModel | None,
    metrics: Sequence[Mapping[str, Any]],
    status: str,
    terminal_reason: str,
    convergence_iteration: int | None,
    diverged: bool,
    constraint_violation: bool,
) -> None:
    arrays: dict[str, np.ndarray] = {"input_signal": np.asarray(input_signal, dtype=np.complex128)}
    model_metadata: dict[str, Any] | None = None
    if model is not None:
        arrays["model_coefficients"] = np.asarray(model.coefficients)
        model_metadata = {
            "orders": list(model.orders),
            "envelope_scale": model.envelope_scale,
            "numeric_dtype": model.numeric_dtype,
        }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="checkpoint_arrays.",
        suffix=".tmp",
        dir=work_directory,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        array_checksum = file_sha256(temporary_name)
        # The full checksum is retained and verified in checkpoint metadata;
        # a short filename avoids overflowing Windows path-length limits.
        array_name = f"arrays.{array_checksum[:20]}.npz"
        array_path = work_directory / array_name
        os.replace(temporary_name, array_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    metadata = {
        "schema_version": 1,
        "trajectory_id": spec.trajectory_id,
        "hashes": hashes.as_dict(),
        "next_iteration": next_iteration,
        "prng_state": _json_safe(prng_state),
        "damping": damping,
        "status": status,
        "terminal_reason": terminal_reason,
        "convergence_iteration": convergence_iteration,
        "diverged": diverged,
        "constraint_violation": constraint_violation,
        "model": model_metadata,
        "metrics": _json_safe(list(metrics)),
        "array_file": array_name,
        "array_sha256": array_checksum,
    }
    metadata_path = work_directory / "checkpoint.json"
    atomic_write_json(metadata_path, metadata)
    for obsolete in work_directory.glob("arrays.*.npz"):
        if obsolete.name != array_name:
            try:
                obsolete.unlink()
            except FileNotFoundError:
                pass


def _read_checkpoint(
    work_directory: Path,
    hashes: RunHashes,
    spec: TrajectorySpec,
) -> dict[str, Any]:
    metadata = read_json(work_directory / "checkpoint.json")
    if metadata.get("trajectory_id") != spec.trajectory_id:
        raise HashMismatchError("checkpoint trajectory ID changed")
    _validate_hash_mapping(metadata.get("hashes"), hashes, "checkpoint")
    array_name = metadata.get("array_file")
    if not isinstance(array_name, str) or Path(array_name).name != array_name:
        raise CorruptArtifactError("checkpoint array filename is invalid")
    array_path = work_directory / array_name
    if metadata.get("array_sha256") != file_sha256(array_path):
        raise CorruptArtifactError("checkpoint array checksum mismatch")
    with np.load(array_path, allow_pickle=False) as arrays:
        input_signal = np.asarray(arrays["input_signal"], dtype=np.complex128)
        model_metadata = metadata.get("model")
        model = None
        if model_metadata is not None:
            numeric_dtype = np.dtype(model_metadata.get("numeric_dtype", "complex128"))
            coefficients = np.asarray(arrays["model_coefficients"], dtype=numeric_dtype)
            model = MemoryPolynomialModel(
                tuple(int(value) for value in model_metadata["orders"]),
                coefficients,
                float(model_metadata["envelope_scale"]),
            )
    return {
        "input": input_signal,
        "next_iteration": int(metadata["next_iteration"]),
        "prng_state": metadata["prng_state"],
        "damping": float(metadata["damping"]),
        "status": str(metadata.get("status", "completed")),
        "terminal_reason": str(metadata.get("terminal_reason", "iteration_limit")),
        "convergence_iteration": metadata.get("convergence_iteration"),
        "diverged": bool(metadata.get("diverged", False)),
        "constraint_violation": bool(metadata.get("constraint_violation", False)),
        "model": model,
        "metrics": list(metadata["metrics"]),
    }


def _remove_completed_checkpoint(work_directory: Path) -> None:
    """Remove restart state only after the verified final shard is durable."""

    for path in (
        work_directory / "checkpoint.json",
        *work_directory.glob("arrays.*.npz"),
        *work_directory.glob("checkpoint_arrays.*.npz"),
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    try:
        work_directory.rmdir()
    except OSError:
        # A crash-only temporary file can remain for forensic inspection.  It
        # is never consulted unless checkpoint.json also verifies it.
        pass


def write_verified_shard(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Atomically write a self-checksummed, single-trajectory JSON shard."""

    safe_record = _json_safe(dict(record))
    checksum = stable_hash(safe_record)
    atomic_write_json(Path(path), {"checksum": checksum, "record": safe_record})


def read_verified_shard(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a trajectory shard and verify its canonical payload checksum."""

    wrapper = read_json(Path(path))
    if set(wrapper) != {"checksum", "record"} or not isinstance(wrapper["record"], dict):
        raise CorruptArtifactError(f"invalid shard wrapper: {path}")
    if wrapper["checksum"] != stable_hash(wrapper["record"]):
        raise CorruptArtifactError(f"shard checksum mismatch: {path}")
    return dict(wrapper["record"])


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Write canonical JSON to a sibling temporary file and atomically replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(_json_safe(value)) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_code_hash(project_root: str | os.PathLike[str]) -> str:
    """Hash executable Python sources, excluding tests, environments, and outputs."""

    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    source_paths: list[Path] = []
    for directory_name in ("remote_dpd", "experiments"):
        directory = root / directory_name
        if directory.exists():
            source_paths.extend(directory.rglob("*.py"))
    for path in sorted(source_paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def directory_size(path: str | os.PathLike[str]) -> int:
    """Return file bytes while tolerating entries removed during traversal."""

    total = 0

    def handle_walk_error(error: OSError) -> None:
        if not isinstance(error, FileNotFoundError):
            raise error

    for directory, _, filenames in os.walk(path, onerror=handle_walk_error):
        for filename in filenames:
            entry = Path(directory) / filename
            try:
                metadata = entry.stat()
                if stat.S_ISREG(metadata.st_mode):
                    total += metadata.st_size
            except FileNotFoundError:
                # Workers atomically remove completed checkpoint/work trees.
                # A path observed by scandir may therefore disappear before
                # its type or size is queried.
                continue
    return total


def current_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return 0


def peak_rss_bytes() -> int:
    try:
        import psutil

        memory = psutil.Process().memory_info()
        platform_peak = getattr(memory, "peak_wset", None)
        if platform_peak is not None:
            return max(int(platform_peak), int(memory.rss))
        return int(memory.rss)
    except (ImportError, OSError):
        return current_rss_bytes()


def environment_manifest() -> dict[str, Any]:
    distributions = {}
    for name in (
        "numpy",
        "scipy",
        "torch",
        "watchdog",
        "matplotlib",
        "psutil",
        "threadpoolctl",
    ):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    numeric_backends: list[dict[str, Any]] = []
    try:
        from threadpoolctl import threadpool_info

        for backend in threadpool_info():
            numeric_backends.append(
                {
                    name: backend.get(name)
                    for name in (
                        "user_api",
                        "internal_api",
                        "version",
                        "threading_layer",
                        "architecture",
                    )
                }
            )
        numeric_backends.sort(key=canonical_json)
    except (ImportError, RuntimeError, TypeError, ValueError):
        numeric_backends = []
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": distributions,
        "numeric_backends": numeric_backends,
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def git_state(project_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(project_root)
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "status": [], "error": str(exc)}


def load_completed_records(run_directory: str | os.PathLike[str]) -> tuple[dict[str, Any], ...]:
    """Load every verified result shard in deterministic trajectory-ID order."""

    paths = sorted((Path(run_directory) / "shards").glob("*.json"))
    records = tuple(read_verified_shard(path) for path in paths)
    identifiers = [record.get("trajectory_id") for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise CorruptArtifactError("duplicate trajectory IDs in result shards")
    return records


def _validate_hash_mapping(value: Any, hashes: RunHashes, artifact: str) -> None:
    if not isinstance(value, Mapping):
        raise HashMismatchError(f"{artifact} has no hash mapping")
    if dict(value) != hashes.as_dict():
        raise HashMismatchError(
            f"{artifact} code/config/protocol/matrix/environment hash mismatch"
        )


def _normalize_pilot_lock(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate the immutable pilot chain embedded in locked-study manifests."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("pilot_lock must be a mapping or None")
    required = {
        "resolved_hash",
        "resolved_config_sha256",
        "resolved_config_path",
        "pilot_hashes",
        "pilot_provenance",
        "resolved_payload",
    }
    if set(value) != required:
        raise ValueError("pilot_lock fields do not match the locked provenance schema")
    normalized = json.loads(canonical_json(value))
    for name in ("resolved_hash", "resolved_config_sha256"):
        if not _is_sha256(normalized.get(name)):
            raise ValueError(f"pilot_lock {name} must be a SHA-256 digest")
    if not isinstance(normalized.get("resolved_config_path"), str) or not normalized[
        "resolved_config_path"
    ]:
        raise ValueError("pilot_lock resolved_config_path must be a non-empty string")
    pilot_hashes = normalized.get("pilot_hashes")
    expected_hash_names = {
        "code_hash",
        "configuration_hash",
        "protocol_hash",
        "matrix_hash",
        "environment_hash",
    }
    if not isinstance(pilot_hashes, dict) or set(pilot_hashes) != expected_hash_names:
        raise ValueError("pilot_lock pilot_hashes do not match the run hash schema")
    if any(not _is_sha256(digest) for digest in pilot_hashes.values()):
        raise ValueError("pilot_lock contains an invalid pilot run hash")
    provenance = normalized.get("pilot_provenance")
    expected_provenance_names = {
        "run_directory",
        "manifest_sha256",
        "expected_ids_sha256",
        "records_hash",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance_names:
        raise ValueError("pilot_lock pilot_provenance schema is invalid")
    if not isinstance(provenance.get("run_directory"), str) or not provenance["run_directory"]:
        raise ValueError("pilot_lock pilot run directory must be a non-empty string")
    for name in ("manifest_sha256", "expected_ids_sha256", "records_hash"):
        if not _is_sha256(provenance.get(name)):
            raise ValueError(f"pilot_lock provenance {name} must be a SHA-256 digest")
    resolved_payload = normalized.get("resolved_payload")
    if not isinstance(resolved_payload, dict):
        raise ValueError("pilot_lock resolved_payload must be a mapping")
    if stable_hash(resolved_payload) != normalized["resolved_hash"]:
        raise ValueError("pilot_lock resolved_payload does not match resolved_hash")
    if resolved_payload.get("pilot_hashes") != pilot_hashes:
        raise ValueError("pilot_lock pilot hashes differ from the resolved payload")
    if resolved_payload.get("pilot_provenance") != provenance:
        raise ValueError("pilot_lock pilot provenance differs from the resolved payload")
    return normalized


def _validate_pilot_lock_context(
    lock: Mapping[str, Any],
    protocol: ExperimentProtocol,
    project_root: Path,
) -> None:
    """Reverify pilot artifacts and bind them to the current runtime context."""

    pilot_hashes = dict(lock["pilot_hashes"])
    if pilot_hashes["code_hash"] != compute_code_hash(project_root):
        raise ValueError("pilot lock code hash differs from the current scientific code")
    if pilot_hashes["protocol_hash"] != protocol.protocol_hash:
        raise ValueError("pilot lock protocol hash differs from the current protocol")
    if pilot_hashes["environment_hash"] != stable_hash(environment_manifest()):
        raise ValueError("pilot lock environment hash differs from the current environment")
    _validate_pilot_lock_artifacts(lock, protocol)


def _validate_pilot_lock_artifacts(
    lock: Mapping[str, Any],
    protocol: ExperimentProtocol,
    base_directory: Path | None = None,
) -> None:
    """Reverify the resolved file, complete pilot matrix, and selected methods."""

    pilot_hashes = dict(lock["pilot_hashes"])
    expected_specs = enumerate_study("pilot", protocol)
    if pilot_hashes["matrix_hash"] != matrix_hash(expected_specs):
        raise ValueError("pilot lock matrix hash differs from the frozen pilot matrix")
    expected_configuration_hash = stable_hash(
        {
            "protocol": protocol.as_dict(),
            "resolved_methods": DEFAULT_RESOLVED_METHODS,
            "resolved_hash": None,
        }
    )
    if pilot_hashes["configuration_hash"] != expected_configuration_hash:
        raise ValueError("pilot lock configuration hash differs from the frozen pilot configuration")

    resolved_path = Path(str(lock["resolved_config_path"]))
    if not resolved_path.is_absolute():
        if base_directory is None:
            raise ValueError("relative pilot lock path requires its manifest directory")
        resolved_path = (base_directory / resolved_path).resolve()
    if not resolved_path.is_file():
        raise ValueError("pilot lock resolved config file is unavailable")
    if file_sha256(resolved_path) != lock["resolved_config_sha256"]:
        raise ValueError("pilot lock resolved config file checksum mismatch")
    resolved_document = read_json(resolved_path)
    expected_document = {
        **dict(lock["resolved_payload"]),
        "resolved_hash": lock["resolved_hash"],
    }
    if resolved_document != expected_document:
        raise ValueError("pilot lock resolved config contents differ from the embedded payload")

    provenance = dict(lock["pilot_provenance"])
    pilot_directory = (resolved_path.parent / provenance["run_directory"]).resolve()
    manifest_path = pilot_directory / "manifest.json"
    expected_ids_path = pilot_directory / "expected_ids.json"
    if not manifest_path.is_file() or not expected_ids_path.is_file():
        raise ValueError("pilot lock source manifest or expected IDs are unavailable")
    if file_sha256(manifest_path) != provenance["manifest_sha256"]:
        raise ValueError("pilot lock source manifest checksum mismatch")
    if file_sha256(expected_ids_path) != provenance["expected_ids_sha256"]:
        raise ValueError("pilot lock source expected-ID checksum mismatch")
    manifest = read_json(manifest_path)
    expected_ids = read_json(expected_ids_path)
    if manifest.get("hashes") != pilot_hashes or expected_ids.get("hashes") != pilot_hashes:
        raise ValueError("pilot lock source artifacts contain different run hashes")
    if manifest.get("study") != "pilot" or manifest.get("expected_trajectory_count") != len(
        expected_specs
    ):
        raise ValueError("pilot lock source manifest does not describe the frozen pilot matrix")
    frozen_ids = sorted(spec.trajectory_id for spec in expected_specs)
    if expected_ids.get("trajectory_ids") != frozen_ids:
        raise ValueError("pilot lock source IDs differ from the frozen pilot matrix")

    records = load_completed_records(pilot_directory)
    if stable_hash(records) != provenance["records_hash"]:
        raise ValueError("pilot lock source records checksum mismatch")
    if sorted(str(record.get("trajectory_id")) for record in records) != frozen_ids:
        raise ValueError("pilot lock source records are incomplete or mixed")
    if any(record.get("hashes") != pilot_hashes for record in records):
        raise ValueError("pilot lock source record hashes differ from pilot provenance")
    rows = []
    for record in records:
        specification = record.get("spec")
        parameters = specification.get("parameters") if isinstance(specification, Mapping) else None
        rows.append(
            {
                "algorithm": record.get("algorithm"),
                "candidate_index": (
                    parameters.get("candidate_index") if isinstance(parameters, Mapping) else None
                ),
                "auec": record.get("auec"),
                "safety_failure": bool(record.get("constraint_violation"))
                or bool(record.get("diverged"))
                or record.get("status") != "completed",
            }
        )
    selections = select_pilot_candidates(
        rows,
        candidate_parameters=PILOT_CANDIDATES,
        candidate_costs=PILOT_COMPUTE_COSTS,
    )
    selection_payload = {name: selection.as_dict() for name, selection in selections.items()}
    resolved_payload = dict(lock["resolved_payload"])
    expected_payload_fields = {
        "schema_version",
        "protocol_hash",
        "pilot_hashes",
        "pilot_provenance",
        "resolved_methods",
        "selection",
    }
    if set(resolved_payload) != expected_payload_fields:
        raise ValueError("pilot lock resolved payload schema is invalid")
    if resolved_payload.get("schema_version") != 1:
        raise ValueError("pilot lock resolved payload version is unsupported")
    if resolved_payload.get("protocol_hash") != protocol.protocol_hash:
        raise ValueError("pilot lock resolved payload protocol hash changed")
    if resolved_payload.get("selection") != selection_payload:
        raise ValueError("pilot lock selection differs from verified pilot records")
    resolved_methods = resolved_payload.get("resolved_methods")
    if not isinstance(resolved_methods, dict) or set(resolved_methods) != set(CORE_METHODS):
        raise ValueError("pilot lock resolved methods do not cover the frozen method set")
    for algorithm, selection in selections.items():
        if resolved_methods.get(algorithm) != dict(selection.parameters):
            raise ValueError(f"pilot lock method {algorithm} differs from its verified selection")
    for algorithm in ("no_dpd", "legacy_ilc", "oracle_lm"):
        if resolved_methods.get(algorithm) != dict(DEFAULT_RESOLVED_METHODS[algorithm]):
            raise ValueError(f"pilot lock fixed method {algorithm} changed")


def _ensure_frozen_resolved_lock(
    output_directory: Path,
    pilot_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically bind every locked study under one output root to one pilot."""

    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        resolved_path = os.path.relpath(
            Path(str(pilot_lock["resolved_config_path"])),
            output_directory,
        )
    except ValueError:
        resolved_path = str(pilot_lock["resolved_config_path"])
    payload = {
        "schema_version": 1,
        "resolved_hash": pilot_lock["resolved_hash"],
        "resolved_config_sha256": pilot_lock["resolved_config_sha256"],
        "resolved_config_path": Path(resolved_path).as_posix(),
        "pilot_hashes": pilot_lock["pilot_hashes"],
        "pilot_provenance": pilot_lock["pilot_provenance"],
        "resolved_payload": pilot_lock["resolved_payload"],
    }
    wrapper = {"checksum": stable_hash(payload), "lock": payload}
    path = output_directory / "frozen_resolved_lock.json"
    serialized = canonical_json(wrapper) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        existing = read_json(path)
        if existing != wrapper:
            raise HashMismatchError(
                "output root is already bound to a different pilot resolved hash"
            )
    else:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    return {
        "checksum": wrapper["checksum"],
        "resolved_hash": payload["resolved_hash"],
        "path": "../frozen_resolved_lock.json",
    }


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_csv(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0.0 else "-inf"
        return number
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        return {"real": _json_safe(number.real), "imag": _json_safe(number.imag)}
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _finite_metric(value: Any) -> float:
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if value == "nan" or value is None:
        return float("nan")
    return float(value)


def _numeric_backend_max_threads() -> int:
    """Report the largest active BLAS/OpenMP/PyTorch thread-pool size."""

    counts: list[int] = []
    try:
        from threadpoolctl import threadpool_info

        counts.extend(
            int(item["num_threads"])
            for item in threadpool_info()
            if isinstance(item.get("num_threads"), int)
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            counts.append(int(torch_module.get_num_threads()))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return max(counts, default=1)


def _execute_trajectory_thread_limited(
    spec: TrajectorySpec,
    context: _WorkerContext,
) -> dict[str, Any]:
    """Run one trajectory with runtime-enforced single-thread numeric pools."""

    apply_numeric_thread_limits()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RunnerError("threadpoolctl is required for experiment execution") from exc
    with threadpool_limits(limits=1):
        return _execute_trajectory(spec, context)


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "CapacityGateError",
    "CapacityProbeReport",
    "CapacityReport",
    "CorruptArtifactError",
    "ExperimentRunner",
    "HashMismatchError",
    "RunHashes",
    "RunSummary",
    "RunnerError",
    "atomic_write_json",
    "compute_code_hash",
    "current_rss_bytes",
    "directory_size",
    "environment_manifest",
    "file_sha256",
    "git_state",
    "load_completed_records",
    "peak_rss_bytes",
    "read_verified_shard",
    "write_verified_shard",
]
