"""Frozen configuration and deterministic matrix enumeration for experiments.

The scientific protocol lives separately from runtime scheduling controls.  A
change to worker count may therefore improve throughput, while every
``TrajectorySpec`` and its identifier remain unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .waveforms import (
    DEFAULT_NFFT,
    DEFAULT_OCCUPIED_PER_SIDE,
    DEFAULT_SYMBOL_COUNT,
    ROOT_SEED,
)


PROTOCOL_NAME = "pa-model-backprop-ilc"
PROTOCOL_REVISION = "2026-08-22"

STUDIES = (
    "smoke",
    "pilot",
    "confirmatory",
    "robustness",
    "mismatch",
    "ablation",
    "dynamic",
    "stress",
)

PILOT_LOCKED_STUDIES = frozenset(
    ("confirmatory", "robustness", "mismatch", "ablation", "dynamic", "stress")
)

CORE_METHODS = (
    "no_dpd",
    "linear_ilc",
    "legacy_ilc",
    "instantaneous_gain_ilc",
    "oracle_lm",
    "model_vjp_ilc",
    "model_lm_ilc",
)

ROBUSTNESS_METHODS = (
    "linear_ilc",
    "instantaneous_gain_ilc",
    "model_lm_ilc",
)

PILOT_CANDIDATES: Mapping[str, tuple[Mapping[str, float], ...]] = {
    "linear_ilc": tuple({"learning_rate": value} for value in (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)),
    "instantaneous_gain_ilc": (
        {"learning_rate": 0.1, "damping": 1e-2},
        {"learning_rate": 0.2, "damping": 1e-2},
        {"learning_rate": 0.4, "damping": 1e-2},
        {"learning_rate": 0.8, "damping": 1e-2},
        {"learning_rate": 0.2, "damping": 1e-3},
        {"learning_rate": 0.4, "damping": 1e-3},
    ),
    "model_vjp_ilc": tuple(
        {"learning_rate": value} for value in (0.01, 0.03, 0.1, 0.3, 0.6, 1.0)
    ),
    "model_lm_ilc": (
        {"step_size": 0.25, "damping": 1e-2, "trust_region_ratio": 0.1},
        {"step_size": 0.5, "damping": 1e-2, "trust_region_ratio": 0.25},
        {"step_size": 1.0, "damping": 1e-2, "trust_region_ratio": 0.25},
        {"step_size": 0.25, "damping": 1e-3, "trust_region_ratio": 0.1},
        {"step_size": 0.5, "damping": 1e-3, "trust_region_ratio": 0.25},
        {"step_size": 1.0, "damping": 1e-3, "trust_region_ratio": 0.25},
    ),
}

# Every candidate within one method has the same preregistered worst-case
# operation budget (same model structure, CG cap, and backtracking cap).  The
# explicit table keeps the cost-before-table-order rule auditable.
PILOT_COMPUTE_COSTS: Mapping[str, tuple[float, ...]] = {
    algorithm: tuple(1.0 for _ in candidates)
    for algorithm, candidates in PILOT_CANDIDATES.items()
}

DEFAULT_RESOLVED_METHODS: Mapping[str, Mapping[str, float | bool]] = {
    "no_dpd": {},
    "linear_ilc": {"learning_rate": 0.2},
    "legacy_ilc": {"learning_rate": 0.2, "phase_precondition": True},
    "instantaneous_gain_ilc": {"learning_rate": 0.4, "damping": 1e-2},
    "oracle_lm": {"step_size": 0.5, "damping": 1e-3, "trust_region_ratio": 0.25},
    "model_vjp_ilc": {"learning_rate": 0.1},
    "model_lm_ilc": {"step_size": 0.5, "damping": 1e-3, "trust_region_ratio": 0.25},
}

ABLATIONS = (
    "raw_vjp",
    "no_ridge",
    "frozen_first_model",
    "three_iteration_replay",
    "unanchored_prediction",
    "no_trust_region",
    "complex64",
    "legacy_dynamic_calibration",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON data identically across platforms and process order."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any) -> str:
    """Return a full SHA-256 hash of canonical JSON data."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _finite_positive(value: float, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    """Scientific constants frozen before method-comparison results exist."""

    root_seed: int = ROOT_SEED
    nfft: int = DEFAULT_NFFT
    symbol_count: int = DEFAULT_SYMBOL_COUNT
    occupied_per_side: int = DEFAULT_OCCUPIED_PER_SIDE
    qam_order: int = 256
    cyclic_prefix_length: int = 0
    update_count: int = 30
    convergence_nmse_db: float = -35.0
    convergence_hold: int = 3
    divergence_margin_db: float = 3.0
    divergence_hold: int = 3
    amam_severities: tuple[float, ...] = (0.55, 0.75, 0.90, 0.97)
    ampm_severities_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    amam_saturation: float = 1.0
    amam_smoothness: float = 4.0
    ampm_target_rms: float = 0.35
    ampm_r0: float = 0.21
    model_orders: tuple[int, ...] = (1, 3, 5, 7, 9)
    model_memory_depth: int = 3
    model_ridge: float = 1e-6
    model_envelope_quantile: float = 0.999
    model_block_size: int = 256
    model_validation_every: int = 5
    envelope_bin_count: int = 20
    model_minimum_scale: float = 1e-12
    model_column_rms_epsilon: float = 1e-14
    model_max_condition_number: float = 1e12
    model_max_validation_nmse_db: float = 0.0
    model_fallback_learning_rate: float = 0.05
    legacy_phase_threshold: float = 0.15
    lm_cg_max_iterations: int = 8
    lm_cg_relative_tolerance: float = 1e-3
    lm_minimum_damping: float = 1e-8
    lm_max_backtracks: int = 8
    lm_backtrack_factor: float = 0.5
    lm_minimum_relative_decrease: float = 0.0
    lm_damping_accept_factor: float = 0.5
    lm_damping_reject_factor: float = 10.0
    instantaneous_gain_input_threshold: float = 1e-8
    saturation_tolerance: float = 1e-10
    numeric_failure_nmse_db: float = 300.0
    safety_rms_multiplier: float = 2.0
    safety_peak_multiplier: float = 2.0
    safety_required_peak_multiplier: float = 1.15
    safety_papr_margin_db: float = 4.0
    confirmatory_seed_count: int = 40
    pilot_seed_count: int = 6
    robustness_seed_count: int = 20
    mismatch_seed_count: int = 12
    ablation_seed_count: int = 12
    dynamic_seed_count: int = 20
    stress_seed_count: int = 12
    robustness_snr_db: tuple[float | str, ...] = ("inf", 50.0, 40.0, 30.0)
    robustness_capture_counts: tuple[int, ...] = (1, 10)
    bootstrap_resamples: int = 10_000
    bootstrap_confidence: float = 0.95
    holm_alpha: float = 0.05

    def __post_init__(self) -> None:
        integer_fields = (
            "root_seed",
            "nfft",
            "symbol_count",
            "occupied_per_side",
            "qam_order",
            "update_count",
            "convergence_hold",
            "divergence_hold",
            "model_memory_depth",
            "model_block_size",
            "model_validation_every",
            "envelope_bin_count",
            "lm_cg_max_iterations",
            "lm_max_backtracks",
            "confirmatory_seed_count",
            "pilot_seed_count",
            "robustness_seed_count",
            "mismatch_seed_count",
            "ablation_seed_count",
            "dynamic_seed_count",
            "stress_seed_count",
            "bootstrap_resamples",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.root_seed != ROOT_SEED:
            raise ValueError(f"the frozen experiment root seed is {ROOT_SEED}")
        if isinstance(self.cyclic_prefix_length, bool) or not isinstance(
            self.cyclic_prefix_length,
            int,
        ):
            raise ValueError("cyclic_prefix_length must be an integer")
        if self.cyclic_prefix_length != 0:
            raise ValueError("the experiment protocol requires cyclic_prefix_length=0")
        if self.qam_order != 256:
            raise ValueError("the experiment protocol requires 256-QAM")
        if 2 * self.occupied_per_side + 1 > self.nfft:
            raise ValueError("occupied subcarriers and DC do not fit in nfft")
        if any(isinstance(order, bool) or not isinstance(order, int) for order in self.model_orders):
            raise ValueError("model_orders must contain integers")
        if tuple(sorted(set(self.model_orders))) != self.model_orders:
            raise ValueError("model_orders must be unique and increasing")
        if any(order <= 0 or order % 2 == 0 for order in self.model_orders):
            raise ValueError("model_orders must contain positive odd integers")
        for name in (
            "convergence_hold",
            "divergence_hold",
        ):
            if getattr(self, name) > self.evaluation_count:
                raise ValueError(f"{name} cannot exceed evaluation_count")
        for name in (
            "convergence_nmse_db",
            "divergence_margin_db",
            "amam_saturation",
            "amam_smoothness",
            "ampm_target_rms",
            "ampm_r0",
            "model_envelope_quantile",
            "model_ridge",
            "model_block_size",
            "model_validation_every",
            "lm_cg_relative_tolerance",
            "lm_minimum_damping",
            "model_minimum_scale",
            "model_column_rms_epsilon",
            "model_max_condition_number",
            "model_max_validation_nmse_db",
            "model_fallback_learning_rate",
            "legacy_phase_threshold",
            "lm_backtrack_factor",
            "lm_minimum_relative_decrease",
            "lm_damping_accept_factor",
            "lm_damping_reject_factor",
            "instantaneous_gain_input_threshold",
            "saturation_tolerance",
            "numeric_failure_nmse_db",
            "safety_rms_multiplier",
            "safety_peak_multiplier",
            "safety_required_peak_multiplier",
            "safety_papr_margin_db",
            "bootstrap_confidence",
            "holm_alpha",
        ):
            _finite_real(getattr(self, name), name)
        for name in (
            "divergence_margin_db",
            "amam_saturation",
            "amam_smoothness",
            "ampm_target_rms",
            "ampm_r0",
            "model_minimum_scale",
            "model_column_rms_epsilon",
            "model_max_condition_number",
            "model_fallback_learning_rate",
            "legacy_phase_threshold",
            "lm_minimum_damping",
            "lm_damping_accept_factor",
            "lm_damping_reject_factor",
            "saturation_tolerance",
            "numeric_failure_nmse_db",
            "safety_rms_multiplier",
            "safety_peak_multiplier",
            "safety_required_peak_multiplier",
        ):
            if _finite_real(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.model_ridge < 0.0:
            raise ValueError("model_ridge must be non-negative")
        if self.lm_minimum_damping < 1e-8:
            raise ValueError("lm_minimum_damping must be at least 1e-8")
        if self.instantaneous_gain_input_threshold < 0.0:
            raise ValueError("instantaneous_gain_input_threshold must be non-negative")
        if self.safety_papr_margin_db < 0.0:
            raise ValueError("safety_papr_margin_db must be non-negative")
        if any(
            not 0.0 < _finite_real(value, "AM/AM severity") < 1.0
            for value in self.amam_severities
        ):
            raise ValueError("AM/AM severities must be finite and lie in (0, 1)")
        for value in self.ampm_severities_deg:
            _finite_real(value, "AM/PM severity")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.robustness_capture_counts
        ):
            raise ValueError("robustness capture counts must be positive integers")
        for value in self.robustness_snr_db:
            if isinstance(value, str):
                if value.lower() != "inf":
                    raise ValueError("string SNR values must be 'inf'")
            else:
                _finite_real(value, "numeric SNR")
        if not 0.0 < self.model_envelope_quantile <= 1.0:
            raise ValueError("model_envelope_quantile must be in (0, 1]")
        if not 0.0 < self.lm_cg_relative_tolerance < 1.0:
            raise ValueError("lm_cg_relative_tolerance must be in (0, 1)")
        if not 0.0 < self.lm_backtrack_factor < 1.0:
            raise ValueError("lm_backtrack_factor must be in (0, 1)")
        if not 0.0 < self.lm_damping_accept_factor <= 1.0:
            raise ValueError("lm_damping_accept_factor must be in (0, 1]")
        if self.lm_damping_reject_factor <= 1.0:
            raise ValueError("lm_damping_reject_factor must exceed one")
        if self.model_max_validation_nmse_db < -300.0:
            raise ValueError("model_max_validation_nmse_db is implausibly low")
        if self.lm_minimum_relative_decrease < 0.0 or self.lm_minimum_relative_decrease >= 1.0:
            raise ValueError("lm_minimum_relative_decrease must be in [0, 1)")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must be in (0, 1)")
        if not 0.0 < self.holm_alpha < 1.0:
            raise ValueError("holm_alpha must be in (0, 1)")

    @property
    def sample_count(self) -> int:
        return self.nfft * self.symbol_count

    @property
    def evaluation_count(self) -> int:
        return self.update_count + 1

    def as_dict(self) -> dict[str, Any]:
        return _json_copy(asdict(self))

    @property
    def protocol_hash(self) -> str:
        return stable_hash(
            {
                "protocol_name": PROTOCOL_NAME,
                "protocol_revision": PROTOCOL_REVISION,
                "scientific_parameters": self.as_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Runtime-only capacity controls that never alter the scientific cells."""

    worker_count: int = 6
    minimum_worker_count: int = 1
    maximum_worker_count: int = 8
    per_worker_rss_limit_bytes: int = 2684354560
    artifact_budget_bytes: int = 5368709120
    minimum_free_disk_bytes: int = 53687091200
    infrastructure_retries: int = 2
    maximum_estimated_hours: float = 72.0

    def __post_init__(self) -> None:
        for name in (
            "worker_count",
            "minimum_worker_count",
            "maximum_worker_count",
            "per_worker_rss_limit_bytes",
            "artifact_budget_bytes",
            "minimum_free_disk_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.infrastructure_retries, bool) or not isinstance(
            self.infrastructure_retries,
            int,
        ):
            raise ValueError("infrastructure_retries must be a non-negative integer")
        if not self.minimum_worker_count <= self.worker_count <= self.maximum_worker_count:
            raise ValueError("worker_count must lie inside the worker bounds")
        if self.infrastructure_retries < 0:
            raise ValueError("infrastructure_retries must be non-negative")
        _finite_positive(self.maximum_estimated_hours, "maximum_estimated_hours")


@dataclass(frozen=True, slots=True)
class TrajectorySpec:
    """One deterministic trajectory in an experiment matrix."""

    study: str
    scenario: str
    severity: str
    pa_seed_index: int
    waveform_seed_index: int
    algorithm: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.study not in STUDIES:
            raise ValueError(f"unknown study: {self.study}")
        if not self.scenario or not self.severity:
            raise ValueError("scenario and severity must not be empty")
        if self.algorithm not in CORE_METHODS:
            raise ValueError(f"unknown algorithm: {self.algorithm}")
        for name in ("pa_seed_index", "waveform_seed_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        parameters = _json_copy(dict(self.parameters))
        object.__setattr__(self, "parameters", parameters)
        expected_hash = stable_hash(
            {
                "study": self.study,
                "scenario": self.scenario,
                "severity": self.severity,
                "algorithm": self.algorithm,
                "parameters": parameters,
            }
        )
        if self.config_hash:
            if not re.fullmatch(r"[0-9a-f]{64}", self.config_hash):
                raise ValueError("config_hash must be a lowercase SHA-256 digest")
            if self.config_hash != expected_hash:
                raise ValueError("config_hash does not match trajectory parameters")
        else:
            object.__setattr__(self, "config_hash", expected_hash)

    @property
    def trajectory_id(self) -> str:
        pieces = (
            self.study,
            self.scenario,
            self.severity,
            f"pa{self.pa_seed_index:04d}",
            f"wf{self.waveform_seed_index:04d}",
            self.algorithm,
            self.config_hash[:16],
        )
        return "--".join(_slug(piece) for piece in pieces)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "study": self.study,
            "scenario": self.scenario,
            "severity": self.severity,
            "pa_seed_index": self.pa_seed_index,
            "waveform_seed_index": self.waveform_seed_index,
            "algorithm": self.algorithm,
            "parameters": _json_copy(self.parameters),
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectorySpec":
        return cls(
            study=str(value["study"]),
            scenario=str(value["scenario"]),
            severity=str(value["severity"]),
            pa_seed_index=int(value["pa_seed_index"]),
            waveform_seed_index=int(value["waveform_seed_index"]),
            algorithm=str(value["algorithm"]),
            parameters=dict(value.get("parameters", {})),
            config_hash=str(value["config_hash"]),
        )


def resolved_method_parameters(
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a validated, complete method-parameter mapping.

    The built-in values make matrix inspection and smoke tests executable.  A
    confirmatory run should pass the hash-locked mapping selected by the pilot.
    """

    result = {name: _json_copy(values) for name, values in DEFAULT_RESOLVED_METHODS.items()}
    if overrides is not None:
        unknown = set(overrides).difference(CORE_METHODS)
        if unknown:
            raise ValueError(f"unknown resolved methods: {sorted(unknown)}")
        for name, values in overrides.items():
            result[name] = _json_copy(dict(values))
    missing = set(CORE_METHODS).difference(result)
    if missing:
        raise ValueError(f"missing resolved methods: {sorted(missing)}")
    return result


def enumerate_study(
    study: str,
    protocol: ExperimentProtocol | None = None,
    *,
    resolved: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[TrajectorySpec, ...]:
    """Enumerate a frozen matrix without depending on scheduling order."""

    if study not in STUDIES:
        raise ValueError(f"study must be one of {STUDIES}")
    cfg = ExperimentProtocol() if protocol is None else protocol
    methods = resolved_method_parameters(resolved)
    common = {
        "update_count": cfg.update_count,
        "model_orders": list(cfg.model_orders),
        "model_memory_depth": cfg.model_memory_depth,
        "model_ridge": cfg.model_ridge,
        "model_envelope_quantile": cfg.model_envelope_quantile,
        "model_block_size": cfg.model_block_size,
        "model_validation_every": cfg.model_validation_every,
        "model_minimum_scale": cfg.model_minimum_scale,
        "model_column_rms_epsilon": cfg.model_column_rms_epsilon,
        "model_max_condition_number": cfg.model_max_condition_number,
        "model_max_validation_nmse_db": cfg.model_max_validation_nmse_db,
        "model_fallback_learning_rate": cfg.model_fallback_learning_rate,
        "legacy_phase_threshold": cfg.legacy_phase_threshold,
        "cg_max_iterations": cfg.lm_cg_max_iterations,
        "cg_relative_tolerance": cfg.lm_cg_relative_tolerance,
        "lm_max_backtracks": cfg.lm_max_backtracks,
        "lm_backtrack_factor": cfg.lm_backtrack_factor,
        "lm_minimum_relative_decrease": cfg.lm_minimum_relative_decrease,
        "instantaneous_gain_input_threshold": cfg.instantaneous_gain_input_threshold,
        "saturation_tolerance": cfg.saturation_tolerance,
    }
    specs: list[TrajectorySpec] = []

    def add(
        scenario: str,
        severity: float | str,
        seed_index: int,
        algorithm: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        parameters = dict(common)
        parameters.update(methods[algorithm])
        if extra:
            parameters.update(_json_copy(dict(extra)))
        specs.append(
            TrajectorySpec(
                study=study,
                scenario=scenario,
                severity=_severity_string(severity),
                pa_seed_index=seed_index,
                waveform_seed_index=seed_index,
                algorithm=algorithm,
                parameters=parameters,
            )
        )

    main_cells = (("amam", 0.97), ("ampm", 135.0))
    if study == "pilot":
        for scenario, severity in main_cells:
            for algorithm, candidates in PILOT_CANDIDATES.items():
                for candidate_index, candidate in enumerate(candidates):
                    for seed_index in range(cfg.pilot_seed_count):
                        add(
                            scenario,
                            severity,
                            seed_index,
                            algorithm,
                            {"candidate_index": candidate_index, **candidate},
                        )
    elif study in {"confirmatory", "smoke"}:
        cells: Sequence[tuple[str, float]]
        seed_count: int
        if study == "smoke":
            cells = main_cells
            seed_count = 1
        else:
            cells = tuple(("amam", value) for value in cfg.amam_severities) + tuple(
                ("ampm", value) for value in cfg.ampm_severities_deg
            )
            seed_count = cfg.confirmatory_seed_count
        for scenario, severity in cells:
            for algorithm in CORE_METHODS:
                for seed_index in range(seed_count):
                    add(scenario, severity, seed_index, algorithm)
    elif study == "robustness":
        for scenario, severity in main_cells:
            for snr_db in cfg.robustness_snr_db:
                for capture_count in cfg.robustness_capture_counts:
                    for algorithm in ROBUSTNESS_METHODS:
                        for seed_index in range(cfg.robustness_seed_count):
                            add(
                                scenario,
                                severity,
                                seed_index,
                                algorithm,
                                {"snr_db": snr_db, "capture_count": capture_count},
                            )
    elif study == "mismatch":
        for scenario, severity in main_cells:
            for order in (3, 5, 7, 9):
                for memory_depth in (1, 3, 5):
                    for seed_index in range(cfg.mismatch_seed_count):
                        add(
                            scenario,
                            severity,
                            seed_index,
                            "model_lm_ilc",
                            {
                                "model_orders": list(range(1, order + 1, 2)),
                                "model_memory_depth": memory_depth,
                            },
                        )
    elif study == "ablation":
        for scenario, severity in main_cells:
            for ablation in ABLATIONS:
                for seed_index in range(cfg.ablation_seed_count):
                    extra: dict[str, Any] = {"ablation": ablation}
                    if ablation == "raw_vjp":
                        extra["learning_rate"] = float(
                            methods["model_vjp_ilc"]["learning_rate"]
                        )
                    add(
                        scenario,
                        severity,
                        seed_index,
                        "model_lm_ilc",
                        extra,
                    )
    elif study == "dynamic":
        for dynamic_family, scenario, severity in (
            ("wiener", "amam_dynamic", 0.97),
            ("hammerstein", "ampm_dynamic", 135.0),
        ):
            for algorithm in (
                "linear_ilc",
                "instantaneous_gain_ilc",
                "oracle_lm",
                "model_lm_ilc",
            ):
                for seed_index in range(cfg.dynamic_seed_count):
                    add(
                        scenario,
                        severity,
                        seed_index,
                        algorithm,
                        {"dynamic_family": dynamic_family},
                    )
    elif study == "stress":
        for scenario, severity, stress_parameters in (
            (
                "hard_saturation",
                2.00,
                {"target_peak_ratio": 2.00, "reachable": False},
            ),
            (
                "gain_rolloff",
                0.40,
                {
                    "target_peak": 0.40,
                    "initial_input_peak": 2.50,
                    "turnover": 0.70,
                    "reachable": True,
                },
            ),
        ):
            for algorithm in CORE_METHODS:
                for seed_index in range(cfg.stress_seed_count):
                    add(
                        scenario,
                        severity,
                        seed_index,
                        algorithm,
                        stress_parameters,
                    )

    identifiers = [spec.trajectory_id for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"{study} enumeration produced duplicate trajectory identifiers")
    return tuple(specs)


def expected_study_counts(protocol: ExperimentProtocol | None = None) -> dict[str, int]:
    """Return the preregistered trajectory count for each study."""

    cfg = ExperimentProtocol() if protocol is None else protocol
    return {
        "smoke": 2 * len(CORE_METHODS),
        "pilot": 2 * sum(len(values) for values in PILOT_CANDIDATES.values()) * cfg.pilot_seed_count,
        "confirmatory": (len(cfg.amam_severities) + len(cfg.ampm_severities_deg))
        * len(CORE_METHODS)
        * cfg.confirmatory_seed_count,
        "robustness": 2
        * len(cfg.robustness_snr_db)
        * len(cfg.robustness_capture_counts)
        * len(ROBUSTNESS_METHODS)
        * cfg.robustness_seed_count,
        "mismatch": 2 * 4 * 3 * cfg.mismatch_seed_count,
        "ablation": 2 * len(ABLATIONS) * cfg.ablation_seed_count,
        "dynamic": 2 * 4 * cfg.dynamic_seed_count,
        "stress": 2 * len(CORE_METHODS) * cfg.stress_seed_count,
    }


def matrix_hash(specs: Iterable[TrajectorySpec]) -> str:
    """Hash an expected matrix in stable identifier order."""

    values = sorted((spec.as_dict() for spec in specs), key=lambda item: item["trajectory_id"])
    return stable_hash(values)


def _severity_string(value: float | str) -> str:
    if isinstance(value, str):
        if not value:
            raise ValueError("severity string must not be empty")
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("severity must be finite")
    return format(number, ".12g")


def _slug(value: object) -> str:
    result = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_").lower()
    if not result:
        raise ValueError("trajectory identifier component has no safe characters")
    return result


__all__ = [
    "ABLATIONS",
    "CORE_METHODS",
    "DEFAULT_RESOLVED_METHODS",
    "ExperimentProtocol",
    "PILOT_CANDIDATES",
    "PILOT_COMPUTE_COSTS",
    "PILOT_LOCKED_STUDIES",
    "PROTOCOL_NAME",
    "PROTOCOL_REVISION",
    "ROBUSTNESS_METHODS",
    "ResourceLimits",
    "STUDIES",
    "TrajectorySpec",
    "canonical_json",
    "enumerate_study",
    "expected_study_counts",
    "matrix_hash",
    "resolved_method_parameters",
    "stable_hash",
]
