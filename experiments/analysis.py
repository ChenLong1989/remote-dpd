"""Verified aggregation and preregistered endpoint analysis.

This module deliberately separates scientific endpoint construction from
plotting.  It verifies result wrappers before reading data, keeps paired seeds
intact, represents non-convergence as right censoring, and makes the handling
of algorithm failures visible in every exported table.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import (
    ExperimentProtocol,
    PILOT_LOCKED_STUDIES,
    TrajectorySpec,
    canonical_json,
    enumerate_study,
    matrix_hash,
    stable_hash,
)
from .statistics import (
    encode_right_censored_convergence,
    evaluate_primary_criterion,
    holm_adjust,
    paired_bootstrap,
    paired_rate_difference_bootstrap,
    paired_relative_reduction_bootstrap,
)


PRIMARY_CELLS = (("amam", "0.97"), ("ampm", "135"))
BASELINE_METHOD = "linear_ilc"
TREATMENT_METHOD = "model_lm_ilc"
FAILURE_ENDPOINT_POLICY = "use_observed_fixed_grid_without_analysis_imputation"
EXACT_ZERO_NMSE_DB_FLOOR = -300.0
EXACT_ZERO_ENDPOINT_POLICY = "retain_zero_linear_nmse_and_floor_db_statistics_at_minus_300_db"
ANALYSIS_SCHEMA_VERSION = 1

PER_ITERATION_METRIC_FIELDS = (
    "nmse",
    "nmse_db",
    "aclr_lower_db",
    "aclr_upper_db",
    "aclr_worst_db",
    "evm_raw_percent",
    "evm_one_tap_percent",
    "low_power_amplitude_threshold",
    "low_power_phase_rmse_deg",
    "binned_amam_rmse",
    "pa_model_train_nmse_db",
    "pa_model_validation_nmse_db",
    "pa_model_rank",
    "pa_model_condition",
    "cg_iterations",
    "cg_relative_residual",
    "input_rms",
    "input_peak",
    "input_papr_db",
    "identity_gradient_cosine",
    "identity_negative_local_fraction",
    "identity_negative_inner_magnitude_fraction",
    "learned_gradient_cosine",
    "envelope_bin_edges",
    "envelope_bin_counts",
    "envelope_phase_counts",
    "binned_amam_rmse_values",
    "binned_ampm_rmse_deg_values",
    "model_fallback",
    "step_accepted",
    "step_stop_reason",
    "saturation_limited",
    "constraint_violation",
    "diverged",
)

SECONDARY_NUMERIC_METRICS = (
    "aclr_lower_db",
    "aclr_upper_db",
    "aclr_worst_db",
    "evm_raw_percent",
    "evm_one_tap_percent",
    "low_power_phase_rmse_deg",
    "binned_amam_rmse",
    "pa_model_train_nmse_db",
    "pa_model_validation_nmse_db",
    "pa_model_rank",
    "pa_model_condition",
    "cg_iterations",
    "cg_relative_residual",
    "input_rms",
    "input_peak",
    "input_papr_db",
    "identity_gradient_cosine",
    "identity_negative_local_fraction",
    "identity_negative_inner_magnitude_fraction",
    "learned_gradient_cosine",
)

ALGORITHM_FAILURE_STEP_REASONS = frozenset(
    {
        "nonfinite_input",
        "nonfinite_update",
        "model_failure",
        "projection_failed",
    }
)


class AnalysisError(RuntimeError):
    """Raised when verified records cannot support a valid analysis."""


class VerificationError(AnalysisError):
    """Raised when a source checksum, identity, or matrix check fails."""


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Verified records and provenance recovered from one result source."""

    records: tuple[dict[str, Any], ...]
    protocol: ExperimentProtocol
    source: str
    dataset_hash: str
    hashes: Mapping[str, str]
    manifest: Mapping[str, Any] | None
    completeness_verified: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "record_count": len(self.records),
            "dataset_hash": self.dataset_hash,
            "hashes": dict(self.hashes),
            "completeness_verified": self.completeness_verified,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryEndpoint:
    """One trajectory encoded on the frozen k=0..K endpoint grid."""

    trajectory_id: str
    study: str
    scenario: str
    severity: str
    algorithm: str
    config_hash: str
    pa_seed_index: int
    waveform_seed_index: int
    parameters: Mapping[str, Any]
    status: str
    terminal_reason: str
    algorithm_failure: bool
    original_evaluation_count: int
    imputed_evaluation_count: int
    endpoint_policy: str
    nmse_db: tuple[float, ...]
    iteration_metrics: tuple[Mapping[str, Any], ...]
    auec: float
    final_nmse_db: float
    final_nmse_db_for_statistics: float
    exact_zero_evaluation_count: int
    convergence_iteration: int
    convergence_event_observed: bool
    success: bool
    diverged: bool
    constraint_violation: bool
    safety_failure: bool
    saturation_limited_round_count: int
    guarded_safe_stop: bool
    learned_gradient_cosine_observation_count: int
    learned_gradient_cosine_negative_count: int
    median_learned_gradient_cosine: float | None
    identity_gradient_cosine_observation_count: int
    identity_gradient_cosine_negative_count: int
    median_identity_gradient_cosine: float | None
    runtime_seconds: float | None
    peak_rss_bytes: int | None

    @property
    def pair_key(self) -> tuple[int, int]:
        return self.pa_seed_index, self.waveform_seed_index

    def as_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "study": self.study,
            "scenario": self.scenario,
            "severity": self.severity,
            "algorithm": self.algorithm,
            "config_hash": self.config_hash,
            "pa_seed_index": self.pa_seed_index,
            "waveform_seed_index": self.waveform_seed_index,
            "parameters": dict(self.parameters),
            "status": self.status,
            "terminal_reason": self.terminal_reason,
            "algorithm_failure": self.algorithm_failure,
            "original_evaluation_count": self.original_evaluation_count,
            "imputed_evaluation_count": self.imputed_evaluation_count,
            "endpoint_policy": self.endpoint_policy,
            "auec": self.auec,
            "final_nmse_db": self.final_nmse_db,
            "final_nmse_db_for_statistics": self.final_nmse_db_for_statistics,
            "exact_zero_evaluation_count": self.exact_zero_evaluation_count,
            "convergence_iteration": self.convergence_iteration,
            "convergence_event_observed": self.convergence_event_observed,
            "success": self.success,
            "diverged": self.diverged,
            "constraint_violation": self.constraint_violation,
            "safety_failure": self.safety_failure,
            "saturation_limited_round_count": self.saturation_limited_round_count,
            "guarded_safe_stop": self.guarded_safe_stop,
            "learned_gradient_cosine_observation_count": (
                self.learned_gradient_cosine_observation_count
            ),
            "learned_gradient_cosine_negative_count": (
                self.learned_gradient_cosine_negative_count
            ),
            "median_learned_gradient_cosine": self.median_learned_gradient_cosine,
            "identity_gradient_cosine_observation_count": (
                self.identity_gradient_cosine_observation_count
            ),
            "identity_gradient_cosine_negative_count": (
                self.identity_gradient_cosine_negative_count
            ),
            "median_identity_gradient_cosine": self.median_identity_gradient_cosine,
            "runtime_seconds": self.runtime_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Complete deterministic analysis used by tables and figures."""

    protocol: ExperimentProtocol
    trajectories: tuple[TrajectoryEndpoint, ...]
    cell_summaries: tuple[dict[str, Any], ...]
    primary_comparisons: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]

    def summary_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "metadata": dict(self.metadata),
            "protocol": self.protocol.as_dict(),
            "cell_summaries": list(self.cell_summaries),
            "primary_comparisons": list(self.primary_comparisons),
            "failure_endpoint_policy": FAILURE_ENDPOINT_POLICY,
            "exact_zero_endpoint_policy": EXACT_ZERO_ENDPOINT_POLICY,
            "exact_zero_nmse_db_floor": EXACT_ZERO_NMSE_DB_FLOOR,
            "per_iteration_metric_fields": list(PER_ITERATION_METRIC_FIELDS),
            "secondary_numeric_metrics": list(SECONDARY_NUMERIC_METRICS),
            "csv_missing_value": "null",
        }


def load_verified_dataset(
    source: str | os.PathLike[str],
    *,
    protocol: ExperimentProtocol | None = None,
    require_complete: bool = True,
) -> LoadedDataset:
    """Load checksum-verified shards or checksum-wrapped JSONL records.

    A JSONL line must have the same ``{"checksum": ..., "record": ...}``
    wrapper as a runner shard.  For a run directory, ``expected_ids.json`` is
    also enforced.  A standalone JSONL remains individually verified but is
    marked as having unknown matrix completeness unless an expected-ID file is
    present beside it.
    """

    path = Path(source).resolve()
    manifest: Mapping[str, Any] | None = None
    expected_path: Path | None = None
    if path.is_dir():
        run_directory = path.parent if path.name == "shards" else path
        shard_directory = path if path.name == "shards" else path / "shards"
        if not shard_directory.is_dir():
            raise VerificationError(f"no shards directory under {path}")
        manifest_path = run_directory / "manifest.json"
        expected_path = run_directory / "expected_ids.json"
        if manifest_path.exists():
            manifest_value = _read_json_object(manifest_path)
            manifest = manifest_value
        records = tuple(
            _read_verified_wrapper(item)
            for item in sorted(shard_directory.glob("*.json"))
        )
    elif path.is_file() and path.suffix.lower() == ".jsonl":
        records = _read_verified_jsonl(path)
        for candidate in (
            path.with_suffix(".expected_ids.json"),
            path.parent / "expected_ids.json",
        ):
            if candidate.exists():
                expected_path = candidate
                break
        manifest_path = path.parent / "manifest.json"
        if manifest_path.exists():
            manifest = _read_json_object(manifest_path)
    else:
        raise VerificationError("source must be a run directory, shards directory, or .jsonl file")

    if not records:
        raise VerificationError("verified result source contains no records")
    reconstructed_specs = _validate_record_identities(records)
    record_hashes = _consistent_record_hashes(records)
    reconstructed_matrix_hash = matrix_hash(reconstructed_specs)
    if record_hashes.get("matrix_hash") != reconstructed_matrix_hash:
        raise VerificationError(
            "record matrix hash does not match reconstructed trajectory specifications"
        )

    completeness_verified = False
    if expected_path is not None and expected_path.exists():
        expected = _read_json_object(expected_path)
        expected_ids = expected.get("trajectory_ids")
        if not isinstance(expected_ids, list) or not all(
            isinstance(value, str) for value in expected_ids
        ):
            raise VerificationError("expected_ids.json has no valid trajectory_ids list")
        if len(expected_ids) != len(set(expected_ids)):
            raise VerificationError("expected_ids.json contains duplicate trajectory IDs")
        actual_ids = sorted(str(record["trajectory_id"]) for record in records)
        if actual_ids != sorted(expected_ids):
            missing = len(set(expected_ids).difference(actual_ids))
            unexpected = len(set(actual_ids).difference(expected_ids))
            raise VerificationError(
                f"result matrix is incomplete or mixed: {missing} missing, {unexpected} unexpected"
            )
        expected_hashes = expected.get("hashes")
        if not isinstance(expected_hashes, Mapping) or dict(expected_hashes) != dict(record_hashes):
            raise VerificationError("expected-ID and record hashes differ")
        completeness_verified = True
    elif require_complete:
        raise VerificationError("matrix completeness cannot be verified without expected_ids.json")

    if manifest is not None:
        manifest_hashes = manifest.get("hashes")
        if not isinstance(manifest_hashes, Mapping) or dict(manifest_hashes) != dict(record_hashes):
            raise VerificationError("manifest and record hashes differ")
        declared_count = manifest.get("expected_trajectory_count")
        if completeness_verified:
            try:
                count = int(declared_count)
            except (TypeError, ValueError) as exc:
                raise VerificationError("manifest trajectory count is invalid") from exc
            if isinstance(declared_count, bool) or count != len(records):
                raise VerificationError("manifest trajectory count differs from verified records")

    loaded_protocol = protocol or _protocol_from_manifest(manifest)
    protocol_hash = str(record_hashes.get("protocol_hash", ""))
    if protocol_hash and protocol_hash != loaded_protocol.protocol_hash:
        raise VerificationError("analysis protocol hash differs from result protocol hash")
    if manifest is not None:
        _validate_runner_matrix_enumeration(
            manifest,
            loaded_protocol,
            reconstructed_specs,
            reconstructed_matrix_hash,
            manifest_path.parent,
        )

    ordered = tuple(
        sorted(
            (dict(record) for record in records),
            key=lambda item: item["trajectory_id"],
        )
    )
    return LoadedDataset(
        records=ordered,
        protocol=loaded_protocol,
        source=str(path),
        dataset_hash=stable_hash(ordered),
        hashes=dict(record_hashes),
        manifest=manifest,
        completeness_verified=completeness_verified,
    )


def normalize_trajectory(
    record: Mapping[str, Any],
    protocol: ExperimentProtocol,
) -> TrajectoryEndpoint:
    """Build one complete endpoint without deleting or shortening failures."""

    trajectory_id = _required_string(record, "trajectory_id")
    status = _required_string(record, "status")
    if status not in {"completed", "algorithm_failure"}:
        raise AnalysisError(f"{trajectory_id}: unsupported status {status!r}")
    spec = record.get("spec")
    if not isinstance(spec, Mapping):
        raise AnalysisError(f"{trajectory_id}: missing trajectory specification")
    metrics = record.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise AnalysisError(f"{trajectory_id}: no per-iteration metrics are available")
    declared_count = _required_int(record, "evaluation_count", minimum=1)
    if declared_count != len(metrics):
        raise AnalysisError(f"{trajectory_id}: evaluation_count does not match metrics")

    values: list[float] = []
    for expected_iteration, metric in enumerate(metrics):
        if not isinstance(metric, Mapping):
            raise AnalysisError(f"{trajectory_id}: invalid metric at k={expected_iteration}")
        if _required_int(metric, "iteration", minimum=0) != expected_iteration:
            raise AnalysisError(f"{trajectory_id}: metric iterations are not contiguous from zero")
        value = _decode_number(metric.get("nmse_db"))
        if math.isnan(value) or value == float("inf"):
            raise AnalysisError(f"{trajectory_id}: unusable NMSE at k={expected_iteration}")
        reported_linear_nmse = metric.get("nmse")
        if reported_linear_nmse is not None:
            expected_linear_nmse = 10.0 ** (value / 10.0)
            _require_matching_number(
                reported_linear_nmse,
                expected_linear_nmse,
                f"{trajectory_id}: metric NMSE at k={expected_iteration}",
            )
        values.append(value)

    expected_count = protocol.evaluation_count
    if len(values) != expected_count or len(metrics) != expected_count:
        raise AnalysisError(f"{trajectory_id}: trajectory does not contain fixed k=0..K metrics")

    imputed_count = 0
    endpoint_policy = (
        FAILURE_ENDPOINT_POLICY
        if status == "algorithm_failure"
        else "observed_k0_through_kK"
    )
    nmse_values = np.asarray(values, dtype=np.float64)
    exact_zero_count = int(np.count_nonzero(np.isneginf(nmse_values)))
    statistical_nmse = np.maximum(nmse_values, EXACT_ZERO_NMSE_DB_FLOOR)
    linear_nmse = np.power(10.0, nmse_values / 10.0)
    if not np.all(np.isfinite(linear_nmse)):
        raise AnalysisError(f"{trajectory_id}: endpoint power ratios overflow")
    computed_auec = float(np.mean(linear_nmse))
    _require_matching_number(
        record.get("auec"),
        computed_auec,
        f"{trajectory_id}: top-level AUEC",
    )
    _require_matching_number(
        record.get("final_nmse_db"),
        float(nmse_values[-1]),
        f"{trajectory_id}: top-level final NMSE",
    )
    (
        saturation_limited_count,
        learned_gradient_count,
        negative_learned_gradient_count,
        median_learned_gradient,
        identity_gradient_count,
        negative_identity_gradient_count,
        median_identity_gradient,
    ) = _metric_diagnostics(metrics, trajectory_id)

    (
        convergence_value,
        diverged,
        constraint,
        metrics_report_algorithm_failure,
    ) = _derive_endpoint_state(metrics, nmse_values, protocol, trajectory_id)
    raw_convergence = record.get("convergence_iteration")
    reported_convergence = (
        None
        if raw_convergence is None
        else _coerce_int(raw_convergence, "convergence_iteration", minimum=0)
    )
    if reported_convergence != convergence_value:
        raise AnalysisError(
            f"{trajectory_id}: top-level convergence_iteration differs from metrics"
        )
    algorithm_failure = status == "algorithm_failure"
    if algorithm_failure != metrics_report_algorithm_failure:
        raise AnalysisError(
            f"{trajectory_id}: status differs from per-iteration failure diagnostics"
        )
    reported_success = _required_bool(record, "success")
    censored = encode_right_censored_convergence(
        [convergence_value],
        final_iteration=protocol.update_count,
    )

    scenario = _consistent_text(record, spec, "scenario", trajectory_id)
    severity = _consistent_text(record, spec, "severity", trajectory_id)
    algorithm = _consistent_text(record, spec, "algorithm", trajectory_id)
    pa_seed = _consistent_integer(record, spec, "pa_seed_index", trajectory_id)
    waveform_seed = _consistent_integer(record, spec, "waveform_seed_index", trajectory_id)
    study = _required_string(spec, "study")
    config_hash = _required_string(spec, "config_hash")
    parameters = spec.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise AnalysisError(f"{trajectory_id}: parameters must be a mapping")
    if _required_bool(record, "diverged") != diverged:
        raise AnalysisError(f"{trajectory_id}: top-level diverged differs from metrics")
    if _required_bool(record, "constraint_violation") != constraint:
        raise AnalysisError(
            f"{trajectory_id}: top-level constraint_violation differs from metrics"
        )
    expected_success = (
        convergence_value is not None
        and not algorithm_failure
        and not diverged
        and not constraint
    )
    if reported_success != expected_success:
        raise AnalysisError(
            f"{trajectory_id}: success is inconsistent with convergence and safety state"
        )
    return TrajectoryEndpoint(
        trajectory_id=trajectory_id,
        study=study,
        scenario=scenario,
        severity=severity,
        algorithm=algorithm,
        config_hash=config_hash,
        pa_seed_index=pa_seed,
        waveform_seed_index=waveform_seed,
        parameters=dict(parameters),
        status=status,
        terminal_reason=str(record.get("terminal_reason", "")),
        algorithm_failure=algorithm_failure,
        original_evaluation_count=declared_count,
        imputed_evaluation_count=imputed_count,
        endpoint_policy=endpoint_policy,
        nmse_db=tuple(float(value) for value in values),
        iteration_metrics=tuple(dict(metric) for metric in metrics),
        auec=computed_auec,
        final_nmse_db=float(values[-1]),
        final_nmse_db_for_statistics=float(statistical_nmse[-1]),
        exact_zero_evaluation_count=exact_zero_count,
        convergence_iteration=int(censored.iterations[0]),
        convergence_event_observed=bool(censored.event_observed[0]),
        success=reported_success,
        diverged=diverged,
        constraint_violation=constraint,
        safety_failure=algorithm_failure or diverged or constraint,
        saturation_limited_round_count=saturation_limited_count,
        guarded_safe_stop=(
            saturation_limited_count > 0 and not (algorithm_failure or diverged or constraint)
        ),
        learned_gradient_cosine_observation_count=learned_gradient_count,
        learned_gradient_cosine_negative_count=negative_learned_gradient_count,
        median_learned_gradient_cosine=median_learned_gradient,
        identity_gradient_cosine_observation_count=identity_gradient_count,
        identity_gradient_cosine_negative_count=negative_identity_gradient_count,
        median_identity_gradient_cosine=median_identity_gradient,
        runtime_seconds=_optional_finite_float(record.get("runtime_seconds")),
        peak_rss_bytes=_optional_nonnegative_int(record.get("peak_rss_bytes")),
    )


def analyze_records(
    records: Iterable[Mapping[str, Any]],
    *,
    protocol: ExperimentProtocol | None = None,
    bootstrap_resamples: int | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Aggregate records and apply the frozen paired primary analysis."""

    cfg = protocol or ExperimentProtocol()
    resamples = cfg.bootstrap_resamples if bootstrap_resamples is None else bootstrap_resamples
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    endpoints = tuple(
        sorted(
            (normalize_trajectory(record, cfg) for record in records),
            key=lambda item: item.trajectory_id,
        )
    )
    if not endpoints:
        raise AnalysisError("analysis requires at least one trajectory")
    identifiers = [item.trajectory_id for item in endpoints]
    if len(set(identifiers)) != len(identifiers):
        raise AnalysisError("analysis records contain duplicate trajectory IDs")
    studies = {item.study for item in endpoints}
    if len(studies) != 1:
        raise AnalysisError("one analysis bundle cannot mix studies")

    cell_summaries = summarize_cells(endpoints)
    primary: tuple[dict[str, Any], ...] = ()
    study = next(iter(studies))
    primary_available = False
    primary_unavailable_reason: str | None = None
    if study == "confirmatory":
        primary = compute_primary_comparisons(endpoints, cfg, resamples=resamples)
        primary_available = True
    elif study == "smoke":
        primary_available, primary_unavailable_reason = _primary_availability(endpoints)
        if primary_available:
            primary = compute_primary_comparisons(endpoints, cfg, resamples=resamples)
    else:
        primary_unavailable_reason = (
            f"study {study!r} is descriptive and has no preregistered primary inference"
        )
    metadata = {
        **dict(source_metadata or {}),
        "study": study,
        "trajectory_count": len(endpoints),
        "algorithm_failure_count": sum(item.algorithm_failure for item in endpoints),
        "imputed_trajectory_count": sum(item.imputed_evaluation_count > 0 for item in endpoints),
        "exact_zero_evaluation_count": sum(item.exact_zero_evaluation_count for item in endpoints),
        "exact_zero_nmse_db_floor": EXACT_ZERO_NMSE_DB_FLOOR,
        "exact_zero_endpoint_policy": EXACT_ZERO_ENDPOINT_POLICY,
        "right_censored_count": sum(not item.convergence_event_observed for item in endpoints),
        "bootstrap_resamples": resamples,
        "bootstrap_confidence": cfg.bootstrap_confidence,
        "holm_alpha": cfg.holm_alpha,
        "pairing_key": ["pa_seed_index", "waveform_seed_index"],
        "bootstrap_cluster": "pa_seed_index",
        "primary_available": primary_available,
        "primary_unavailable_reason": primary_unavailable_reason,
    }
    return AnalysisResult(cfg, endpoints, cell_summaries, primary, metadata)


def compute_primary_comparisons(
    trajectories: Sequence[TrajectoryEndpoint],
    protocol: ExperimentProtocol,
    *,
    resamples: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Compare learned safeguarded LM with linear ILC in both main cells."""

    bootstrap_count = protocol.bootstrap_resamples if resamples is None else resamples
    comparisons: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for scenario, severity in PRIMARY_CELLS:
        baseline = _indexed_cell(trajectories, scenario, severity, BASELINE_METHOD)
        treatment = _indexed_cell(trajectories, scenario, severity, TREATMENT_METHOD)
        if set(baseline) != set(treatment):
            missing_treatment = len(set(baseline).difference(treatment))
            missing_baseline = len(set(treatment).difference(baseline))
            raise AnalysisError(
                f"{scenario}/{severity}: primary pairs are incomplete "
                f"({missing_treatment} treatment missing, {missing_baseline} baseline missing)"
            )
        if not baseline:
            raise AnalysisError(f"{scenario}/{severity}: primary comparison has no seed pairs")
        pair_keys = sorted(baseline)
        linear = [baseline[key] for key in pair_keys]
        model = [treatment[key] for key in pair_keys]
        clusters = [key[0] for key in pair_keys]

        linear_auec = np.asarray([item.auec for item in linear])
        model_auec = np.asarray([item.auec for item in model])
        linear_final = np.asarray([item.final_nmse_db_for_statistics for item in linear])
        model_final = np.asarray([item.final_nmse_db_for_statistics for item in model])
        _require_finite_primary(
            linear_auec,
            model_auec,
            linear_final,
            model_final,
            cell=f"{scenario}/{severity}",
        )
        linear_success = np.asarray([item.success for item in linear], dtype=np.int8)
        model_success = np.asarray([item.success for item in model], dtype=np.int8)
        linear_diverged = np.asarray([item.diverged for item in linear], dtype=np.int8)
        model_diverged = np.asarray([item.diverged for item in model], dtype=np.int8)
        linear_constraint = np.asarray(
            [item.constraint_violation for item in linear],
            dtype=np.int8,
        )
        model_constraint = np.asarray([item.constraint_violation for item in model], dtype=np.int8)
        linear_safe = np.asarray([not item.safety_failure for item in linear], dtype=np.int8)
        model_safe = np.asarray([not item.safety_failure for item in model], dtype=np.int8)
        linear_failure = np.asarray([item.algorithm_failure for item in linear], dtype=np.int8)
        model_failure = np.asarray([item.algorithm_failure for item in model], dtype=np.int8)

        seed = _bootstrap_seed(protocol.root_seed, scenario, severity)
        auec_bootstrap = paired_relative_reduction_bootstrap(
            linear_auec,
            model_auec,
            cluster_ids=clusters,
            statistic="median",
            resamples=bootstrap_count,
            confidence=protocol.bootstrap_confidence,
            seed=seed,
        )
        final_bootstrap = paired_bootstrap(
            linear_final,
            model_final,
            cluster_ids=clusters,
            statistic="median",
            resamples=bootstrap_count,
            confidence=protocol.bootstrap_confidence,
            seed=seed,
        )
        success_bootstrap = paired_rate_difference_bootstrap(
            model_success,
            linear_success,
            cluster_ids=clusters,
            resamples=bootstrap_count,
            confidence=protocol.bootstrap_confidence,
            seed=seed,
        )
        divergence_bootstrap = paired_rate_difference_bootstrap(
            model_diverged,
            linear_diverged,
            cluster_ids=clusters,
            resamples=bootstrap_count,
            confidence=protocol.bootstrap_confidence,
            seed=seed,
        )
        constraint_bootstrap = paired_rate_difference_bootstrap(
            model_constraint,
            linear_constraint,
            cluster_ids=clusters,
            resamples=bootstrap_count,
            confidence=protocol.bootstrap_confidence,
            seed=seed,
        )
        safety_bootstrap = paired_rate_difference_bootstrap(
            model_safe,
            linear_safe,
            cluster_ids=clusters,
            resamples=bootstrap_count,
            confidence=protocol.bootstrap_confidence,
            seed=seed,
        )
        failure_bootstrap = paired_rate_difference_bootstrap(
            model_failure,
            linear_failure,
            cluster_ids=clusters,
            resamples=bootstrap_count,
            confidence=protocol.bootstrap_confidence,
            seed=seed,
        )
        criterion = evaluate_primary_criterion(
            linear_auec=linear_auec,
            model_auec=model_auec,
            linear_final_nmse_db=linear_final,
            model_final_nmse_db=model_final,
            linear_success=linear_success,
            model_success=model_success,
            linear_diverged=linear_diverged,
            model_diverged=model_diverged,
            linear_constraint_violation=linear_constraint,
            model_constraint_violation=model_constraint,
        )
        hypothesis = f"{scenario}_{severity}_auec"
        p_values[hypothesis] = auec_bootstrap.p_value_two_sided
        comparisons.append(
            {
                "hypothesis": hypothesis,
                "scenario": scenario,
                "severity": severity,
                "pair_count": len(pair_keys),
                "cluster_count": len(set(clusters)),
                "pair_keys": [f"pa{pa}:wf{waveform}" for pa, waveform in pair_keys],
                "linear_algorithm_failure_count": int(np.sum(linear_failure)),
                "model_algorithm_failure_count": int(np.sum(model_failure)),
                "linear_safety_rate_percent": 100.0 * float(np.mean(linear_safe)),
                "model_safety_rate_percent": 100.0 * float(np.mean(model_safe)),
                "auec_relative_reduction": auec_bootstrap.as_dict(),
                "final_nmse_improvement_db": final_bootstrap.as_dict(),
                "success_rate_difference_points": success_bootstrap.as_dict(),
                "divergence_rate_difference_points": divergence_bootstrap.as_dict(),
                "constraint_rate_difference_points": constraint_bootstrap.as_dict(),
                "safety_rate_difference_points": safety_bootstrap.as_dict(),
                "algorithm_failure_rate_difference_points": failure_bootstrap.as_dict(),
                "criterion": criterion.as_dict(),
            }
        )

    adjusted = holm_adjust(p_values, alpha=protocol.holm_alpha)
    for comparison in comparisons:
        holm = adjusted[comparison["hypothesis"]]
        comparison["holm"] = {
            "raw_p_value": holm.raw_p_value,
            "adjusted_p_value": holm.adjusted_p_value,
            "rejected": holm.rejected,
            "rank": holm.rank,
            "family": "two_main_cell_auec_comparisons",
            "alpha": protocol.holm_alpha,
        }
    return tuple(comparisons)


def summarize_cells(trajectories: Sequence[TrajectoryEndpoint]) -> tuple[dict[str, Any], ...]:
    """Summarize every algorithm/config cell without pooling variant configs."""

    grouped: dict[tuple[str, str, str, str, str], list[TrajectoryEndpoint]] = {}
    for item in trajectories:
        key = (item.study, item.scenario, item.severity, item.algorithm, item.config_hash)
        grouped.setdefault(key, []).append(item)
    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        observed_convergence = [
            item.convergence_iteration for item in members if item.convergence_event_observed
        ]
        summary = {
                "study": key[0],
                "scenario": key[1],
                "severity": key[2],
                "algorithm": key[3],
                "config_hash": key[4],
                "parameters": dict(members[0].parameters),
                "trajectory_count": len(members),
                "algorithm_failure_count": sum(item.algorithm_failure for item in members),
                "imputed_trajectory_count": sum(
                    item.imputed_evaluation_count > 0 for item in members
                ),
                "right_censored_count": sum(
                    not item.convergence_event_observed for item in members
                ),
                "median_auec": float(np.median([item.auec for item in members])),
                "median_final_nmse_db": float(np.median([item.final_nmse_db for item in members])),
                "median_final_nmse_db_for_statistics": float(
                    np.median([item.final_nmse_db_for_statistics for item in members])
                ),
                "exact_zero_evaluation_count": sum(
                    item.exact_zero_evaluation_count for item in members
                ),
                "success_rate_percent": 100.0 * float(np.mean([item.success for item in members])),
                "divergence_rate_percent": 100.0
                * float(np.mean([item.diverged for item in members])),
                "constraint_violation_rate_percent": 100.0
                * float(np.mean([item.constraint_violation for item in members])),
                "safety_rate_percent": 100.0
                * float(np.mean([not item.safety_failure for item in members])),
                "saturation_limited_trajectory_rate_percent": 100.0
                * float(np.mean([item.saturation_limited_round_count > 0 for item in members])),
                "guarded_safe_stop_rate_percent": 100.0
                * float(np.mean([item.guarded_safe_stop for item in members])),
                "learned_gradient_cosine_observation_count": sum(
                    item.learned_gradient_cosine_observation_count for item in members
                ),
                "negative_learned_gradient_cosine_fraction": (
                    _pooled_negative_learned_gradient_fraction(members)
                ),
                "median_trajectory_learned_gradient_cosine": _median_optional(
                    item.median_learned_gradient_cosine for item in members
                ),
                "identity_gradient_cosine_observation_count": sum(
                    item.identity_gradient_cosine_observation_count for item in members
                ),
                "negative_identity_gradient_cosine_fraction": (
                    _pooled_negative_identity_gradient_fraction(members)
                ),
                "median_trajectory_identity_gradient_cosine": _median_optional(
                    item.median_identity_gradient_cosine for item in members
                ),
                "median_observed_convergence_iteration": (
                    float(np.median(observed_convergence)) if observed_convergence else None
                ),
                "median_runtime_seconds": _median_optional(
                    item.runtime_seconds for item in members
                ),
                "peak_rss_bytes": max(
                    (item.peak_rss_bytes for item in members if item.peak_rss_bytes is not None),
                    default=None,
                ),
            }
        summary.update(_secondary_metric_summary(members))
        summaries.append(summary)
    return tuple(summaries)


def write_analysis_artifacts(
    result: AnalysisResult,
    output_directory: str | os.PathLike[str],
) -> dict[str, Path]:
    """Write deterministic CSV/JSON tables used by the manuscript."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": output / "analysis_summary.json",
        "trajectories_csv": output / "trajectory_endpoints.csv",
        "per_iteration_metrics_csv": output / "per_iteration_metrics.csv",
        "cells_csv": output / "cell_method_summary.csv",
        "primary_csv": output / "primary_comparisons.csv",
        "ampm_fixed_r0_phase_csv": output / "ampm_fixed_r0_phase.csv",
    }
    _atomic_write_json(paths["summary_json"], result.summary_payload())
    trajectory_rows = [item.as_dict() for item in result.trajectories]
    _atomic_write_mapping_csv(paths["trajectories_csv"], trajectory_rows)

    iteration_rows = _per_iteration_metric_rows(result)
    _atomic_write_mapping_csv(paths["per_iteration_metrics_csv"], iteration_rows)
    _atomic_write_mapping_csv(paths["cells_csv"], list(result.cell_summaries))
    _atomic_write_mapping_csv(
        paths["primary_csv"],
        [_flatten_primary(comparison) for comparison in result.primary_comparisons],
        empty_header=("hypothesis", "scenario", "severity"),
    )
    _atomic_write_mapping_csv(
        paths["ampm_fixed_r0_phase_csv"],
        _ampm_fixed_r0_phase_rows(result),
        empty_header=(
            "trajectory_id",
            "scenario",
            "severity",
            "algorithm",
            "iteration",
            "fixed_r0",
            "low_power_phase_rmse_deg",
        ),
    )
    return paths


def _per_iteration_metric_rows(result: AnalysisResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result.trajectories:
        usable_count = len(item.nmse_db) - item.imputed_evaluation_count
        for iteration, nmse_db_value in enumerate(item.nmse_db):
            source_available = (
                iteration < usable_count and iteration < len(item.iteration_metrics)
            )
            raw = item.iteration_metrics[iteration] if source_available else {}
            row: dict[str, Any] = {
                "trajectory_id": item.trajectory_id,
                "study": item.study,
                "scenario": item.scenario,
                "severity": item.severity,
                "algorithm": item.algorithm,
                "config_hash": item.config_hash,
                "pa_seed_index": item.pa_seed_index,
                "waveform_seed_index": item.waveform_seed_index,
                "iteration": iteration,
                "status": item.status,
                "terminal_reason": item.terminal_reason,
                "algorithm_failure": item.algorithm_failure,
                "source_metric_available": source_available,
                "imputed_after_algorithm_failure": not source_available,
                "endpoint_policy": item.endpoint_policy,
            }
            for field in PER_ITERATION_METRIC_FIELDS:
                row[field] = raw.get(field)
            row["nmse_db"] = nmse_db_value
            row["nmse"] = 10.0 ** (nmse_db_value / 10.0)
            missing = [
                field
                for field in PER_ITERATION_METRIC_FIELDS
                if field not in raw or raw.get(field) is None
            ]
            row["missing_preregistered_metric_field_count"] = len(missing)
            row["missing_preregistered_metric_fields"] = missing
            for field in sorted(raw):
                if field != "iteration" and field not in row:
                    row[field] = raw[field]
            rows.append(row)
    return rows


def _ampm_fixed_r0_phase_rows(result: AnalysisResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fixed_r0 = result.protocol.ampm_r0
    for item in result.trajectories:
        if "ampm" not in item.scenario:
            continue
        usable_count = len(item.nmse_db) - item.imputed_evaluation_count
        for iteration in range(len(item.nmse_db)):
            source_available = (
                iteration < usable_count and iteration < len(item.iteration_metrics)
            )
            raw = item.iteration_metrics[iteration] if source_available else {}
            reported_threshold = raw.get("low_power_amplitude_threshold")
            if reported_threshold is not None:
                state, threshold = _numeric_metric_state(reported_threshold)
                if state != "finite" or not math.isclose(
                    threshold,
                    fixed_r0,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise AnalysisError(
                        f"{item.trajectory_id}: low-power phase threshold is not fixed r0"
                    )
            rows.append(
                {
                    "trajectory_id": item.trajectory_id,
                    "study": item.study,
                    "scenario": item.scenario,
                    "severity": item.severity,
                    "algorithm": item.algorithm,
                    "pa_seed_index": item.pa_seed_index,
                    "waveform_seed_index": item.waveform_seed_index,
                    "iteration": iteration,
                    "status": item.status,
                    "algorithm_failure": item.algorithm_failure,
                    "source_metric_available": source_available,
                    "imputed_after_algorithm_failure": not source_available,
                    "fixed_r0": fixed_r0,
                    "reported_low_power_amplitude_threshold": reported_threshold,
                    "low_power_phase_rmse_deg": raw.get(
                        "low_power_phase_rmse_deg"
                    ),
                    "phase_sample_count_below_r0": _phase_sample_count_below_r0(
                        raw,
                        fixed_r0,
                    ),
                    "envelope_bin_edges": raw.get("envelope_bin_edges"),
                    "envelope_phase_counts": raw.get("envelope_phase_counts"),
                    "binned_ampm_rmse_deg_values": raw.get(
                        "binned_ampm_rmse_deg_values"
                    ),
                }
            )
    return rows


def _phase_sample_count_below_r0(
    metric: Mapping[str, Any],
    fixed_r0: float,
) -> int | None:
    edges = metric.get("envelope_bin_edges")
    counts = metric.get("envelope_phase_counts")
    if not isinstance(edges, list) or not isinstance(counts, list):
        return None
    try:
        edge_values = np.asarray(edges, dtype=np.float64)
        count_values = np.asarray(counts, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        edge_values.ndim != 1
        or count_values.ndim != 1
        or edge_values.size != count_values.size + 1
        or not np.all(np.isfinite(edge_values))
        or np.any(count_values < 0)
    ):
        return None
    centers = 0.5 * (edge_values[:-1] + edge_values[1:])
    return int(np.sum(count_values[centers <= fixed_r0]))


def analyze_source(
    source: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    require_complete: bool = True,
) -> tuple[AnalysisResult, dict[str, Path]]:
    """Verify, analyze, and export one runner result source."""

    dataset = load_verified_dataset(source, require_complete=require_complete)
    result = analyze_records(
        dataset.records,
        protocol=dataset.protocol,
        source_metadata=dataset.metadata(),
    )
    return result, write_analysis_artifacts(result, output_directory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and aggregate PA-backpropagation results.")
    parser.add_argument(
        "source",
        type=Path,
        help="Run directory, shards directory, or verified JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for CSV and JSON tables.",
    )
    parser.add_argument(
        "--allow-unlisted-jsonl",
        action="store_true",
        help="Allow JSONL without expected_ids.json; record checksums and pairs remain enforced.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, paths = analyze_source(
        args.source,
        args.output,
        require_complete=not args.allow_unlisted_jsonl,
    )
    print(
        json.dumps(
            {
                "study": result.metadata["study"],
                "trajectory_count": len(result.trajectories),
                "primary_available": result.metadata["primary_available"],
                "primary_unavailable_reason": result.metadata[
                    "primary_unavailable_reason"
                ],
                "primary_comparison_count": len(result.primary_comparisons),
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _read_verified_wrapper(path: Path) -> dict[str, Any]:
    wrapper = _read_json_object(path)
    return _verify_wrapper(wrapper, str(path))


def _read_verified_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(f"{path}:{line_number}: invalid JSON") from exc
            records.append(_verify_wrapper(wrapper, f"{path}:{line_number}"))
    return tuple(records)


def _verify_wrapper(wrapper: Any, source: str) -> dict[str, Any]:
    if not isinstance(wrapper, Mapping) or set(wrapper) != {"checksum", "record"}:
        raise VerificationError(f"{source}: invalid verified-record wrapper")
    record = wrapper.get("record")
    if not isinstance(record, Mapping):
        raise VerificationError(f"{source}: record is not an object")
    try:
        checksum = stable_hash(record)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{source}: record is not canonical finite JSON") from exc
    if wrapper.get("checksum") != checksum:
        raise VerificationError(f"{source}: record checksum mismatch")
    return dict(record)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON object {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{path} does not contain a JSON object")
    return value


def _validate_record_identities(
    records: Sequence[Mapping[str, Any]],
) -> tuple[TrajectorySpec, ...]:
    identifiers: list[str] = []
    specifications: list[TrajectorySpec] = []
    for record in records:
        identifier = _required_string(record, "trajectory_id")
        if int(record.get("schema_version", -1)) != 1:
            raise VerificationError(f"{identifier}: unsupported result schema")
        spec = record.get("spec")
        if not isinstance(spec, Mapping) or spec.get("trajectory_id") != identifier:
            raise VerificationError(f"{identifier}: record/spec trajectory identity mismatch")
        try:
            reconstructed = TrajectorySpec.from_dict(spec)
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError(f"{identifier}: invalid trajectory specification") from exc
        if reconstructed.trajectory_id != identifier:
            raise VerificationError(f"{identifier}: trajectory ID does not match its specification")
        identifiers.append(identifier)
        specifications.append(reconstructed)
    if len(identifiers) != len(set(identifiers)):
        raise VerificationError("verified source contains duplicate trajectory IDs")
    return tuple(specifications)


def _consistent_record_hashes(records: Sequence[Mapping[str, Any]]) -> Mapping[str, str]:
    first = records[0].get("hashes")
    if not isinstance(first, Mapping) or not first:
        raise VerificationError("result records have no hash mapping")
    expected = {str(key): str(value) for key, value in first.items()}
    for record in records[1:]:
        value = record.get("hashes")
        if not isinstance(value, Mapping) or {
            str(key): str(item) for key, item in value.items()
        } != expected:
            raise VerificationError("result source mixes code/config/protocol/matrix hashes")
    return expected


def _validate_runner_matrix_enumeration(
    manifest: Mapping[str, Any],
    protocol: ExperimentProtocol,
    specifications: Sequence[TrajectorySpec],
    reconstructed_matrix_hash: str,
    manifest_directory: Path | None = None,
) -> None:
    study = manifest.get("study")
    if not isinstance(study, str) or not study:
        raise VerificationError("manifest has no valid study")
    if any(specification.study != study for specification in specifications):
        raise VerificationError("manifest study differs from trajectory specifications")

    # Hand-written legacy manifests did not carry runner provenance.  Once any
    # runner-only field is present, require the real generation structure so
    # deleting it cannot silently disable frozen-matrix verification.
    runner_provenance = any(
        key in manifest
        for key in ("generation", "resolved_methods", "runtime_limits", "cell_instances")
    )
    if not runner_provenance:
        return
    generation = manifest.get("generation")
    if not isinstance(generation, Mapping):
        raise VerificationError("runner manifest has no generation mapping")
    generation_argv = generation.get("argv")
    if not isinstance(generation_argv, list) or not generation_argv or not all(
        isinstance(value, str) and value for value in generation_argv
    ):
        raise VerificationError("runner manifest generation argv is invalid")
    if study == "smoke":
        # A smoke invocation may intentionally select a debug-limited prefix.
        return
    resolved = manifest.get("resolved_methods")
    if not isinstance(resolved, Mapping):
        raise VerificationError("runner manifest has no resolved_methods mapping")
    if study in PILOT_LOCKED_STUDIES:
        from .runner import _normalize_pilot_lock, _validate_pilot_lock_artifacts

        try:
            pilot_lock = _normalize_pilot_lock(manifest.get("pilot_lock"))
            if pilot_lock is None:
                raise ValueError("pilot lock is absent")
            _validate_pilot_lock_artifacts(
                pilot_lock,
                protocol,
                base_directory=manifest_directory,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise VerificationError("locked runner manifest has invalid pilot provenance") from exc
        if pilot_lock["resolved_payload"].get("resolved_methods") != dict(resolved):
            raise VerificationError("manifest methods differ from the resolved pilot payload")
        manifest_hashes = manifest.get("hashes")
        if not isinstance(manifest_hashes, Mapping):
            raise VerificationError("locked runner manifest has no run hashes")
        pilot_hashes = pilot_lock["pilot_hashes"]
        for name in ("code_hash", "protocol_hash", "environment_hash"):
            if pilot_hashes.get(name) != manifest_hashes.get(name):
                raise VerificationError(f"pilot and locked-study {name} differ")
        expected_configuration_hash = stable_hash(
            {
                "protocol": protocol.as_dict(),
                "resolved_methods": dict(resolved),
                "resolved_hash": pilot_lock["resolved_hash"],
            }
        )
        if manifest_hashes.get("configuration_hash") != expected_configuration_hash:
            raise VerificationError(
                "locked-study configuration hash does not bind the resolved pilot hash"
            )
        root_reference = manifest.get("frozen_resolved_lock")
        if not isinstance(root_reference, Mapping) or set(root_reference) != {
            "checksum",
            "resolved_hash",
            "path",
        }:
            raise VerificationError("locked runner manifest has no frozen root pilot lock")
        if manifest_directory is None:
            raise VerificationError("frozen root pilot lock needs the manifest directory")
        root_lock_path = (manifest_directory / str(root_reference["path"])).resolve()
        try:
            root_wrapper = _read_json_object(root_lock_path)
        except VerificationError as exc:
            raise VerificationError("frozen root pilot lock is unavailable") from exc
        root_payload = root_wrapper.get("lock")
        if not isinstance(root_payload, Mapping) or root_wrapper.get("checksum") != stable_hash(
            root_payload
        ):
            raise VerificationError("frozen root pilot lock checksum is invalid")
        if root_reference.get("checksum") != root_wrapper.get("checksum"):
            raise VerificationError("manifest and frozen root pilot lock checksums differ")
        if root_reference.get("resolved_hash") != pilot_lock["resolved_hash"]:
            raise VerificationError("manifest and frozen root resolved hashes differ")
        for name in (
            "resolved_hash",
            "resolved_config_sha256",
            "pilot_hashes",
            "pilot_provenance",
            "resolved_payload",
        ):
            if root_payload.get(name) != pilot_lock.get(name):
                raise VerificationError(f"frozen root pilot lock {name} differs from manifest")
        root_resolved_path = (
            root_lock_path.parent / str(root_payload.get("resolved_config_path", ""))
        ).resolve()
        manifest_resolved_path = (
            manifest_directory / str(pilot_lock["resolved_config_path"])
        ).resolve()
        if root_resolved_path != manifest_resolved_path:
            raise VerificationError("frozen root and manifest resolve different pilot files")
    try:
        expected = enumerate_study(study, protocol, resolved=resolved)
    except (TypeError, ValueError) as exc:
        raise VerificationError("runner manifest cannot reconstruct the frozen matrix") from exc
    expected_ids = {specification.trajectory_id for specification in expected}
    actual_ids = {specification.trajectory_id for specification in specifications}
    if actual_ids != expected_ids:
        missing = len(expected_ids.difference(actual_ids))
        unexpected = len(actual_ids.difference(expected_ids))
        raise VerificationError(
            f"runner matrix differs from frozen enumeration: {missing} missing, "
            f"{unexpected} unexpected"
        )
    if matrix_hash(expected) != reconstructed_matrix_hash:
        raise VerificationError("runner matrix hash differs from frozen enumeration")


def _protocol_from_manifest(manifest: Mapping[str, Any] | None) -> ExperimentProtocol:
    if manifest is None:
        return ExperimentProtocol()
    value = manifest.get("scientific_protocol")
    if not isinstance(value, Mapping):
        raise VerificationError("manifest has no scientific_protocol mapping")
    parameters = dict(value)
    for name in (
        "model_orders",
        "amam_severities",
        "ampm_severities_deg",
        "robustness_snr_db",
        "robustness_capture_counts",
    ):
        if name in parameters:
            parameters[name] = tuple(parameters[name])
    try:
        return ExperimentProtocol(**parameters)
    except (TypeError, ValueError) as exc:
        raise VerificationError("manifest scientific protocol is invalid") from exc


def _indexed_cell(
    trajectories: Sequence[TrajectoryEndpoint],
    scenario: str,
    severity: str,
    algorithm: str,
) -> dict[tuple[int, int], TrajectoryEndpoint]:
    selected = [
        item
        for item in trajectories
        if item.scenario == scenario and item.severity == severity and item.algorithm == algorithm
    ]
    indexed: dict[tuple[int, int], TrajectoryEndpoint] = {}
    for item in selected:
        if item.pair_key in indexed:
            raise AnalysisError(
                f"{scenario}/{severity}/{algorithm}: duplicate seed pair {item.pair_key}; "
                "variant configs must not be pooled"
            )
        indexed[item.pair_key] = item
    return indexed


def _primary_availability(
    trajectories: Sequence[TrajectoryEndpoint],
) -> tuple[bool, str | None]:
    """Return whether both preregistered cells have exact method seed pairs."""

    problems: list[str] = []
    for scenario, severity in PRIMARY_CELLS:
        baseline = _indexed_cell(trajectories, scenario, severity, BASELINE_METHOD)
        treatment = _indexed_cell(trajectories, scenario, severity, TREATMENT_METHOD)
        if not baseline or not treatment:
            problems.append(
                f"{scenario}/{severity} lacks one or both primary methods"
            )
            continue
        if set(baseline) != set(treatment):
            missing_treatment = len(set(baseline).difference(treatment))
            missing_baseline = len(set(treatment).difference(baseline))
            problems.append(
                f"{scenario}/{severity} has incomplete seed pairing "
                f"({missing_treatment} treatment missing, "
                f"{missing_baseline} baseline missing)"
            )
    if problems:
        return False, "; ".join(problems)
    return True, None


def _derive_endpoint_state(
    metrics: Sequence[Mapping[str, Any]],
    nmse_values: NDArray[np.float64],
    protocol: ExperimentProtocol,
    trajectory_id: str,
) -> tuple[int | None, bool, bool, bool]:
    """Replay runner stopping semantics from the per-evaluation truth source."""

    if len(metrics) != nmse_values.size or not metrics:
        raise AnalysisError(f"{trajectory_id}: cannot derive state from incomplete metrics")
    convergence: int | None = None
    diverged = False
    constraint_violation = False
    algorithm_failure = False
    initial_nmse = float(nmse_values[0])

    for iteration, metric in enumerate(metrics):
        metric_diverged = _metric_boolean(
            metric,
            "diverged",
            trajectory_id=trajectory_id,
            iteration=iteration,
        )
        metric_constraint = _metric_boolean(
            metric,
            "constraint_violation",
            trajectory_id=trajectory_id,
            iteration=iteration,
        )
        evaluation_failure = metric.get("evaluation_failure")
        numeric_failure = evaluation_failure is not None
        if numeric_failure:
            if not isinstance(evaluation_failure, str) or not evaluation_failure:
                raise AnalysisError(
                    f"{trajectory_id}: invalid evaluation_failure at k={iteration}"
                )
            algorithm_failure = True

        threshold_divergence = False
        if not algorithm_failure and not diverged:
            convergence_start = iteration - protocol.convergence_hold + 1
            if convergence is None and convergence_start >= 0:
                convergence_window = nmse_values[convergence_start : iteration + 1]
                if bool(np.all(convergence_window <= protocol.convergence_nmse_db)):
                    convergence = convergence_start

            divergence_start = iteration - protocol.divergence_hold + 1
            if divergence_start >= 0:
                divergence_window = nmse_values[divergence_start : iteration + 1]
                threshold_divergence = bool(
                    np.all(
                        divergence_window
                        > initial_nmse + protocol.divergence_margin_db
                    )
                )

        expected_metric_diverged = (
            numeric_failure or threshold_divergence or metric_constraint
        )
        if metric_diverged != expected_metric_diverged:
            raise AnalysisError(
                f"{trajectory_id}: per-iteration diverged flag is inconsistent at "
                f"k={iteration}"
            )
        constraint_violation = constraint_violation or metric_constraint
        diverged = diverged or expected_metric_diverged

        step_failure = _metric_reports_algorithm_failure(
            metric,
            trajectory_id=trajectory_id,
            iteration=iteration,
            update_count=protocol.update_count,
        )
        algorithm_failure = algorithm_failure or step_failure

    return convergence, diverged, constraint_violation, algorithm_failure


def _metric_boolean(
    metric: Mapping[str, Any],
    key: str,
    *,
    trajectory_id: str,
    iteration: int,
) -> bool:
    value = metric.get(key)
    if not isinstance(value, (bool, np.bool_)):
        raise AnalysisError(
            f"{trajectory_id}: metric {key} must be boolean at k={iteration}"
        )
    return bool(value)


def _metric_reports_algorithm_failure(
    metric: Mapping[str, Any],
    *,
    trajectory_id: str,
    iteration: int,
    update_count: int,
) -> bool:
    failure = False
    if "step_stop_reason" in metric and metric.get("step_stop_reason") is not None:
        reason = metric.get("step_stop_reason")
        if not isinstance(reason, str) or not reason:
            raise AnalysisError(
                f"{trajectory_id}: invalid step_stop_reason at k={iteration}"
            )
        failure = reason in ALGORITHM_FAILURE_STEP_REASONS or reason.startswith("cg_")
    if "algorithm_error" in metric and metric.get("algorithm_error") is not None:
        error = metric.get("algorithm_error")
        if not isinstance(error, str) or not error:
            raise AnalysisError(f"{trajectory_id}: invalid algorithm_error at k={iteration}")
        failure = True
    if failure and iteration >= update_count:
        raise AnalysisError(
            f"{trajectory_id}: algorithm failure is recorded after the final update"
        )
    return failure


def _require_finite_primary(*arrays: NDArray[np.float64], cell: str) -> None:
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise AnalysisError(
            f"{cell}: primary continuous endpoints must be finite after failure encoding"
        )
    if np.any(arrays[0] <= 0.0):
        raise AnalysisError(f"{cell}: baseline AUEC must be positive")


def _bootstrap_seed(root_seed: int, scenario: str, severity: str) -> int:
    digest = stable_hash({"root_seed": root_seed, "scenario": scenario, "severity": severity})
    return int(digest[:16], 16)


def _metric_diagnostics(
    metrics: Sequence[Mapping[str, Any]],
    trajectory_id: str,
) -> tuple[
    int,
    int,
    int,
    float | None,
    int,
    int,
    float | None,
]:
    saturation_limited_count = 0
    learned_values: list[float] = []
    identity_values: list[float] = []
    for index, metric in enumerate(metrics):
        if not isinstance(metric, Mapping):
            raise AnalysisError(f"{trajectory_id}: invalid metric at k={index}")
        saturation_limited_count += bool(metric.get("saturation_limited", False))
        for key, destination in (
            ("learned_gradient_cosine", learned_values),
            ("identity_gradient_cosine", identity_values),
        ):
            if key not in metric or metric.get(key) is None:
                continue
            value = _decode_number(metric.get(key))
            if not math.isfinite(value) or value < -1.0 - 1e-9 or value > 1.0 + 1e-9:
                raise AnalysisError(
                    f"{trajectory_id}: invalid {key} at k={index}"
                )
            destination.append(float(np.clip(value, -1.0, 1.0)))
    return (
        saturation_limited_count,
        len(learned_values),
        sum(value < 0.0 for value in learned_values),
        float(np.median(learned_values)) if learned_values else None,
        len(identity_values),
        sum(value < 0.0 for value in identity_values),
        float(np.median(identity_values)) if identity_values else None,
    )


def _pooled_negative_learned_gradient_fraction(
    members: Sequence[TrajectoryEndpoint],
) -> float | None:
    total = sum(item.learned_gradient_cosine_observation_count for item in members)
    if total == 0:
        return None
    negative = sum(item.learned_gradient_cosine_negative_count for item in members)
    return float(negative / total)


def _pooled_negative_identity_gradient_fraction(
    members: Sequence[TrajectoryEndpoint],
) -> float | None:
    total = sum(item.identity_gradient_cosine_observation_count for item in members)
    if total == 0:
        return None
    negative = sum(item.identity_gradient_cosine_negative_count for item in members)
    return float(negative / total)


def _secondary_metric_summary(
    members: Sequence[TrajectoryEndpoint],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in SECONDARY_NUMERIC_METRICS:
        finite_observation_count = 0
        missing_observation_count = 0
        nonfinite_observation_count = 0
        trajectory_means: list[float] = []
        trajectory_medians: list[float] = []
        final_values: list[float] = []
        for member in members:
            usable_count = len(member.nmse_db) - member.imputed_evaluation_count
            values: list[float] = []
            for iteration in range(len(member.nmse_db)):
                if iteration >= usable_count or iteration >= len(member.iteration_metrics):
                    missing_observation_count += 1
                    continue
                state, value = _numeric_metric_state(
                    member.iteration_metrics[iteration].get(field)
                )
                if state == "finite":
                    finite_observation_count += 1
                    values.append(value)
                elif state == "nonfinite":
                    nonfinite_observation_count += 1
                else:
                    missing_observation_count += 1
            if values:
                trajectory_means.append(float(np.mean(values)))
                trajectory_medians.append(float(np.median(values)))
            if usable_count == len(member.nmse_db) and member.iteration_metrics:
                state, value = _numeric_metric_state(
                    member.iteration_metrics[-1].get(field)
                )
                if state == "finite":
                    final_values.append(value)

        summary.update(
            {
                f"{field}_finite_observation_count": finite_observation_count,
                f"{field}_missing_observation_count": missing_observation_count,
                f"{field}_nonfinite_observation_count": nonfinite_observation_count,
                f"{field}_trajectory_coverage_count": len(trajectory_means),
                f"median_trajectory_mean_{field}": (
                    float(np.median(trajectory_means)) if trajectory_means else None
                ),
                f"median_trajectory_median_{field}": (
                    float(np.median(trajectory_medians)) if trajectory_medians else None
                ),
                f"{field}_final_coverage_count": len(final_values),
                f"median_final_{field}": (
                    float(np.median(final_values)) if final_values else None
                ),
            }
        )
    return summary


def _numeric_metric_state(value: Any) -> tuple[str, float]:
    if value is None or value == "nan":
        return "missing", float("nan")
    try:
        number = _decode_number(value)
    except AnalysisError:
        return "missing", float("nan")
    if not math.isfinite(number):
        return "nonfinite", number
    return "finite", number


def _flatten_primary(value: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        key: item
        for key, item in value.items()
        if key not in {
            "pair_keys",
            "auec_relative_reduction",
            "final_nmse_improvement_db",
            "success_rate_difference_points",
            "divergence_rate_difference_points",
            "constraint_rate_difference_points",
            "safety_rate_difference_points",
            "algorithm_failure_rate_difference_points",
            "criterion",
            "holm",
        }
    }
    for prefix in (
        "auec_relative_reduction",
        "final_nmse_improvement_db",
        "success_rate_difference_points",
        "divergence_rate_difference_points",
        "constraint_rate_difference_points",
        "safety_rate_difference_points",
        "algorithm_failure_rate_difference_points",
    ):
        for key, item in value[prefix].items():
            row[f"{prefix}_{key}"] = item
    for key, item in value["criterion"].items():
        row[f"criterion_{key}"] = item
    for key, item in value["holm"].items():
        row[f"holm_{key}"] = item
    return row


def _atomic_write_json(path: Path, value: Any) -> None:
    safe = _json_safe(value)
    _atomic_write_text(path, canonical_json(safe) + "\n")


def _atomic_write_mapping_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_header: Sequence[str] = (),
) -> None:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    if not headers:
        headers = list(empty_header)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in headers})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
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
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0.0 else "-inf"
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _csv_value(value: Any) -> Any:
    safe = _json_safe(value)
    if safe is None:
        return "null"
    if isinstance(safe, (dict, list)):
        return canonical_json(safe)
    return safe


def _decode_number(value: Any) -> float:
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if value == "nan" or value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"metric is not numeric: {value!r}") from exc


def _require_matching_number(reported: Any, expected: float, context: str) -> None:
    """Require a serialized derived scalar to match its metric-derived value."""

    if isinstance(reported, (bool, np.bool_)):
        raise AnalysisError(f"{context} is not numeric")
    actual = _decode_number(reported)
    if math.isnan(actual):
        raise AnalysisError(f"{context} is missing or NaN")
    if math.isinf(expected):
        matches = actual == expected
    elif expected == 0.0:
        matches = actual == 0.0
    else:
        matches = math.isfinite(actual) and math.isclose(
            actual,
            expected,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    if not matches:
        raise AnalysisError(f"{context} differs from per-iteration metrics")


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"missing non-empty string field {key}")
    return value


def _required_int(mapping: Mapping[str, Any], key: str, *, minimum: int) -> int:
    return _coerce_int(mapping.get(key), key, minimum=minimum)


def _required_bool(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, (bool, np.bool_)):
        raise AnalysisError(f"{key} must be boolean")
    return bool(value)


def _coerce_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AnalysisError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise AnalysisError(f"{name} must be at least {minimum}")
    return result


def _consistent_text(
    record: Mapping[str, Any],
    spec: Mapping[str, Any],
    key: str,
    trajectory_id: str,
) -> str:
    left = _required_string(record, key)
    right = _required_string(spec, key)
    if left != right:
        raise AnalysisError(f"{trajectory_id}: record/spec {key} mismatch")
    return left


def _consistent_integer(
    record: Mapping[str, Any],
    spec: Mapping[str, Any],
    key: str,
    trajectory_id: str,
) -> int:
    left = _required_int(record, key, minimum=0)
    right = _required_int(spec, key, minimum=0)
    if left != right:
        raise AnalysisError(f"{trajectory_id}: record/spec {key} mismatch")
    return left


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("optional numeric field is invalid") from exc
    if not math.isfinite(result) or result < 0.0:
        raise AnalysisError("optional numeric field must be finite and non-negative")
    return result


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, "peak_rss_bytes", minimum=0)


def _median_optional(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(np.median(finite)) if finite else None


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "AnalysisError",
    "AnalysisResult",
    "BASELINE_METHOD",
    "EXACT_ZERO_ENDPOINT_POLICY",
    "EXACT_ZERO_NMSE_DB_FLOOR",
    "FAILURE_ENDPOINT_POLICY",
    "LoadedDataset",
    "PER_ITERATION_METRIC_FIELDS",
    "PRIMARY_CELLS",
    "SECONDARY_NUMERIC_METRICS",
    "TREATMENT_METHOD",
    "TrajectoryEndpoint",
    "VerificationError",
    "analyze_records",
    "analyze_source",
    "compute_primary_comparisons",
    "load_verified_dataset",
    "normalize_trajectory",
    "summarize_cells",
    "write_analysis_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
