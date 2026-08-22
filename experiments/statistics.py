"""Preregistered paired statistics for the PA-backpropagation experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .waveforms import named_seed_sequence


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """A paired percentile-bootstrap estimate and confidence interval."""

    estimate: float
    confidence_low: float
    confidence_high: float
    standard_error: float
    p_value_two_sided: float
    confidence: float
    resamples: int
    pair_count: int
    cluster_count: int
    statistic: str

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "estimate": self.estimate,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "standard_error": self.standard_error,
            "p_value_two_sided": self.p_value_two_sided,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "pair_count": self.pair_count,
            "cluster_count": self.cluster_count,
            "statistic": self.statistic,
        }


@dataclass(frozen=True, slots=True)
class HolmResult:
    """One entry in a Holm step-down family-wise correction."""

    hypothesis: str
    raw_p_value: float
    adjusted_p_value: float
    rejected: bool
    rank: int


@dataclass(frozen=True, slots=True)
class PilotSelection:
    """Deterministic pilot selection for one algorithm."""

    algorithm: str
    candidate_index: int
    parameters: Mapping[str, Any]
    median_auec: float
    safe: bool
    tie_set: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "candidate_index": self.candidate_index,
            "parameters": dict(self.parameters),
            "median_auec": self.median_auec,
            "safe": self.safe,
            "tie_set": list(self.tie_set),
        }


@dataclass(frozen=True, slots=True)
class PrimaryCriterion:
    """Outcome of the preregistered model-LM versus linear-ILC criteria."""

    auec_reduction_fraction: float
    final_nmse_improvement_db: float
    success_rate_improvement: float
    divergence_rate_difference: float
    constraint_rate_difference: float
    passed: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "auec_reduction_fraction": self.auec_reduction_fraction,
            "final_nmse_improvement_db": self.final_nmse_improvement_db,
            "success_rate_improvement": self.success_rate_improvement,
            "divergence_rate_difference": self.divergence_rate_difference,
            "constraint_rate_difference": self.constraint_rate_difference,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True, eq=False)
class RightCensoredConvergence:
    """Convergence times with an explicit event-observed mask."""

    iterations: NDArray[np.int64]
    event_observed: NDArray[np.bool_]
    final_iteration: int

    @property
    def count(self) -> int:
        return int(self.iterations.size)


def encode_right_censored_convergence(
    convergence_iterations: Sequence[int | None],
    *,
    final_iteration: int = 30,
) -> RightCensoredConvergence:
    """Encode non-converged trajectories as censored at ``final_iteration``."""

    if isinstance(final_iteration, bool) or not isinstance(final_iteration, int):
        raise ValueError("final_iteration must be a non-negative integer")
    if final_iteration < 0:
        raise ValueError("final_iteration must be a non-negative integer")
    if not convergence_iterations:
        raise ValueError("convergence_iterations must not be empty")
    iterations = np.empty(len(convergence_iterations), dtype=np.int64)
    observed = np.empty(len(convergence_iterations), dtype=np.bool_)
    for index, value in enumerate(convergence_iterations):
        if value is None:
            iterations[index] = final_iteration
            observed[index] = False
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("convergence iterations must be integers or None")
        converted = int(value)
        if converted < 0 or converted > final_iteration:
            raise ValueError("observed convergence iteration lies outside the study")
        iterations[index] = converted
        observed[index] = True
    return RightCensoredConvergence(iterations, observed, final_iteration)


def paired_bootstrap(
    left: ArrayLike,
    right: ArrayLike,
    *,
    cluster_ids: Sequence[object] | None = None,
    statistic: str = "median",
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | np.random.SeedSequence | None = None,
) -> BootstrapResult:
    """Bootstrap the paired effect ``left - right``.

    Resampling is hierarchical when ``cluster_ids`` are supplied: clusters are
    sampled first, then observations are sampled within every selected
    cluster.  This covers both the one-pair-per-seed design and future designs
    with multiple waveform observations nested within a PA seed.  Non-finite
    observations are rejected rather than silently deleting failed runs.
    """

    left_values, right_values, clusters = _paired_inputs(left, right, cluster_ids)
    statistic_function = _statistic_function(statistic)
    _validate_bootstrap_settings(resamples, confidence)
    differences = left_values - right_values
    estimate = float(statistic_function(differences))
    samples = _hierarchical_bootstrap_values(
        differences,
        clusters,
        statistic_function,
        resamples,
        _bootstrap_rng(seed, "paired_difference", statistic),
    )
    return _bootstrap_result(estimate, samples, confidence, differences.size, len(clusters), statistic)


def paired_relative_reduction_bootstrap(
    baseline: ArrayLike,
    treatment: ArrayLike,
    *,
    cluster_ids: Sequence[object] | None = None,
    statistic: str = "median",
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | np.random.SeedSequence | None = None,
) -> BootstrapResult:
    """Bootstrap paired fractional reduction ``(baseline-treatment)/baseline``."""

    baseline_values, treatment_values, clusters = _paired_inputs(
        baseline,
        treatment,
        cluster_ids,
    )
    if np.any(baseline_values <= 0.0):
        raise ValueError("baseline values must be positive for relative reduction")
    statistic_function = _statistic_function(statistic)
    _validate_bootstrap_settings(resamples, confidence)
    reductions = (baseline_values - treatment_values) / baseline_values
    estimate = float(statistic_function(reductions))
    samples = _hierarchical_bootstrap_values(
        reductions,
        clusters,
        statistic_function,
        resamples,
        _bootstrap_rng(seed, "paired_relative_reduction", statistic),
    )
    return _bootstrap_result(
        estimate,
        samples,
        confidence,
        reductions.size,
        len(clusters),
        f"relative_{statistic}",
    )


def paired_rate_difference_bootstrap(
    treatment: ArrayLike,
    baseline: ArrayLike,
    *,
    cluster_ids: Sequence[object] | None = None,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | np.random.SeedSequence | None = None,
) -> BootstrapResult:
    """Bootstrap the paired percentage-point effect ``treatment - baseline``."""

    treatment_values = np.asarray(treatment)
    baseline_values = np.asarray(baseline)
    if not np.all(np.isin(treatment_values, (0, 1, False, True))):
        raise ValueError("treatment must contain binary indicators")
    if not np.all(np.isin(baseline_values, (0, 1, False, True))):
        raise ValueError("baseline must contain binary indicators")
    result = paired_bootstrap(
        treatment_values.astype(np.float64),
        baseline_values.astype(np.float64),
        cluster_ids=cluster_ids,
        statistic="mean",
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    return BootstrapResult(
        estimate=100.0 * result.estimate,
        confidence_low=100.0 * result.confidence_low,
        confidence_high=100.0 * result.confidence_high,
        standard_error=100.0 * result.standard_error,
        p_value_two_sided=result.p_value_two_sided,
        confidence=result.confidence,
        resamples=result.resamples,
        pair_count=result.pair_count,
        cluster_count=result.cluster_count,
        statistic="paired_rate_difference_percentage_points",
    )


def holm_adjust(
    p_values: Mapping[str, float] | Sequence[float],
    *,
    alpha: float = 0.05,
) -> dict[str, HolmResult] | tuple[HolmResult, ...]:
    """Apply Holm's step-down correction with monotone adjusted p-values."""

    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    is_mapping = isinstance(p_values, Mapping)
    if is_mapping:
        names = [str(name) for name in p_values]
        values = np.asarray([p_values[name] for name in p_values], dtype=np.float64)
    else:
        values = np.asarray(tuple(p_values), dtype=np.float64)
        names = [str(index) for index in range(values.size)]
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional collection")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and lie in [0, 1]")

    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running_adjusted = 0.0
    still_rejecting = True
    rejected = np.zeros(values.size, dtype=np.bool_)
    ranks = np.empty(values.size, dtype=np.int64)
    count = values.size
    for position, original_index in enumerate(order):
        multiplier = count - position
        running_adjusted = max(running_adjusted, multiplier * float(values[original_index]))
        adjusted[original_index] = min(1.0, running_adjusted)
        threshold = alpha / multiplier
        if still_rejecting and values[original_index] <= threshold:
            rejected[original_index] = True
        else:
            still_rejecting = False
        ranks[original_index] = position + 1

    results = tuple(
        HolmResult(
            hypothesis=names[index],
            raw_p_value=float(values[index]),
            adjusted_p_value=float(adjusted[index]),
            rejected=bool(rejected[index]),
            rank=int(ranks[index]),
        )
        for index in range(count)
    )
    if is_mapping:
        return {result.hypothesis: result for result in results}
    return results


def select_pilot_candidates(
    records: Iterable[Mapping[str, Any]],
    *,
    candidate_parameters: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_costs: Mapping[str, Sequence[float]] | None = None,
    tie_fraction: float = 0.02,
) -> dict[str, PilotSelection]:
    """Apply the locked safe/median-AUEC/2%-tie/table-order pilot rule."""

    if not math.isfinite(tie_fraction) or tie_fraction < 0.0:
        raise ValueError("tie_fraction must be finite and non-negative")
    rows = tuple(records)
    selections: dict[str, PilotSelection] = {}
    for algorithm, candidates in candidate_parameters.items():
        costs = (
            tuple(1.0 for _ in candidates)
            if candidate_costs is None
            else tuple(candidate_costs.get(algorithm, ()))
        )
        if len(costs) != len(candidates) or any(
            not math.isfinite(float(cost)) or float(cost) < 0.0 for cost in costs
        ):
            raise ValueError(f"candidate costs are invalid for {algorithm}")
        summaries: list[tuple[int, bool, float]] = []
        for index in range(len(candidates)):
            matching = [
                row
                for row in rows
                if row.get("algorithm") == algorithm and int(row.get("candidate_index", -1)) == index
            ]
            if not matching:
                raise ValueError(f"missing pilot records for {algorithm} candidate {index}")
            safe = not any(bool(row.get("safety_failure", False)) for row in matching)
            auec_values = np.asarray([row.get("auec") for row in matching], dtype=np.float64)
            if not np.all(np.isfinite(auec_values)):
                safe = False
                median_auec = float("inf")
            else:
                median_auec = float(np.median(auec_values))
            summaries.append((index, safe, median_auec))

        safe_summaries = [summary for summary in summaries if summary[1]]
        if not safe_summaries:
            raise ValueError(f"all pilot candidates failed safety for {algorithm}")
        best_value = min(summary[2] for summary in safe_summaries)
        tolerance = tie_fraction * abs(best_value)
        tied = tuple(
            summary[0]
            for summary in safe_summaries
            if summary[2] <= best_value + tolerance + np.finfo(float).eps
        )
        selected_index = min(tied, key=lambda index: (float(costs[index]), index))
        selected_summary = summaries[selected_index]
        selections[algorithm] = PilotSelection(
            algorithm=algorithm,
            candidate_index=selected_index,
            parameters=dict(candidate_parameters[algorithm][selected_index]),
            median_auec=selected_summary[2],
            safe=True,
            tie_set=tied,
        )
    return selections


def evaluate_primary_criterion(
    *,
    linear_auec: ArrayLike,
    model_auec: ArrayLike,
    linear_final_nmse_db: ArrayLike,
    model_final_nmse_db: ArrayLike,
    linear_success: ArrayLike,
    model_success: ArrayLike,
    linear_diverged: ArrayLike,
    model_diverged: ArrayLike,
    linear_constraint_violation: ArrayLike,
    model_constraint_violation: ArrayLike,
    divergence_noninferiority_margin_points: float = 10.0,
) -> PrimaryCriterion:
    """Evaluate the five locked effect-size and safety thresholds."""

    linear_auec_values, model_auec_values, _ = _paired_inputs(linear_auec, model_auec, None)
    linear_nmse, model_nmse, _ = _paired_inputs(
        linear_final_nmse_db,
        model_final_nmse_db,
        None,
    )
    if np.any(linear_auec_values <= 0.0):
        raise ValueError("linear_auec must be positive")
    if (
        not math.isfinite(divergence_noninferiority_margin_points)
        or divergence_noninferiority_margin_points < 0.0
    ):
        raise ValueError("divergence noninferiority margin must be finite and non-negative")
    auec_reduction = float(np.median((linear_auec_values - model_auec_values) / linear_auec_values))
    nmse_improvement = float(np.median(linear_nmse - model_nmse))
    success_improvement = _binary_rate_difference(model_success, linear_success)
    divergence_difference = _binary_rate_difference(model_diverged, linear_diverged)
    constraint_difference = _binary_rate_difference(
        model_constraint_violation,
        linear_constraint_violation,
    )
    passed = (
        auec_reduction >= 0.25
        and nmse_improvement >= 3.0
        and success_improvement >= 20.0
        and divergence_difference <= divergence_noninferiority_margin_points
        and constraint_difference <= 0.0
    )
    return PrimaryCriterion(
        auec_reduction_fraction=auec_reduction,
        final_nmse_improvement_db=nmse_improvement,
        success_rate_improvement=success_improvement,
        divergence_rate_difference=divergence_difference,
        constraint_rate_difference=constraint_difference,
        passed=passed,
    )


def _binary_rate_difference(left: ArrayLike, right: ArrayLike) -> float:
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape or left_values.size == 0:
        raise ValueError("paired binary inputs must have equal non-empty shapes")
    if not np.all(np.isin(left_values, (0, 1, False, True))) or not np.all(
        np.isin(right_values, (0, 1, False, True))
    ):
        raise ValueError("rate inputs must be binary")
    return float(100.0 * np.mean(left_values.astype(float) - right_values.astype(float)))


def _paired_inputs(
    left: ArrayLike,
    right: ArrayLike,
    cluster_ids: Sequence[object] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[NDArray[np.int64], ...]]:
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_values.size == 0 or left_values.shape != right_values.shape:
        raise ValueError("paired inputs must have equal, non-empty shapes")
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        raise ValueError("paired inputs must be finite; encode failures before analysis")
    if cluster_ids is None:
        cluster_values: Sequence[object] = tuple(range(left_values.size))
    else:
        cluster_values = tuple(cluster_ids)
        if len(cluster_values) != left_values.size:
            raise ValueError("cluster_ids must have one entry per pair")
    grouped: dict[str, list[int]] = {}
    for index, cluster in enumerate(cluster_values):
        key = repr(cluster)
        grouped.setdefault(key, []).append(index)
    clusters = tuple(np.asarray(indices, dtype=np.int64) for indices in grouped.values())
    return left_values, right_values, clusters


def _statistic_function(name: str) -> Callable[[NDArray[np.float64]], float]:
    if name == "median":
        return lambda values: float(np.median(values))
    if name == "mean":
        return lambda values: float(np.mean(values))
    raise ValueError("statistic must be 'median' or 'mean'")


def _validate_bootstrap_settings(resamples: int, confidence: float) -> None:
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")


def _bootstrap_rng(
    seed: int | np.random.SeedSequence | None,
    *name: object,
) -> np.random.Generator:
    if isinstance(seed, np.random.SeedSequence):
        sequence = seed
    elif seed is None:
        sequence = named_seed_sequence("statistics", *name)
    else:
        sequence = np.random.SeedSequence(int(seed))
    return np.random.default_rng(sequence)


def _hierarchical_bootstrap_values(
    values: NDArray[np.float64],
    clusters: tuple[NDArray[np.int64], ...],
    statistic: Callable[[NDArray[np.float64]], float],
    resamples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    samples = np.empty(resamples, dtype=np.float64)
    cluster_count = len(clusters)
    for resample_index in range(resamples):
        selected_clusters = rng.integers(0, cluster_count, size=cluster_count)
        selected_parts: list[NDArray[np.float64]] = []
        for cluster_index in selected_clusters:
            indices = clusters[int(cluster_index)]
            within = rng.choice(indices, size=indices.size, replace=True)
            selected_parts.append(values[within])
        samples[resample_index] = statistic(np.concatenate(selected_parts))
    return samples


def _bootstrap_result(
    estimate: float,
    samples: NDArray[np.float64],
    confidence: float,
    pair_count: int,
    cluster_count: int,
    statistic: str,
) -> BootstrapResult:
    tail = 0.5 * (1.0 - confidence)
    low, high = np.quantile(samples, (tail, 1.0 - tail))
    less_or_equal = (np.count_nonzero(samples <= 0.0) + 1.0) / (samples.size + 1.0)
    greater_or_equal = (np.count_nonzero(samples >= 0.0) + 1.0) / (samples.size + 1.0)
    p_value = min(1.0, 2.0 * min(less_or_equal, greater_or_equal))
    standard_error = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    return BootstrapResult(
        estimate=float(estimate),
        confidence_low=float(low),
        confidence_high=float(high),
        standard_error=standard_error,
        p_value_two_sided=float(p_value),
        confidence=float(confidence),
        resamples=int(samples.size),
        pair_count=pair_count,
        cluster_count=cluster_count,
        statistic=statistic,
    )


__all__ = [
    "BootstrapResult",
    "HolmResult",
    "PilotSelection",
    "PrimaryCriterion",
    "RightCensoredConvergence",
    "encode_right_censored_convergence",
    "evaluate_primary_criterion",
    "holm_adjust",
    "paired_bootstrap",
    "paired_rate_difference_bootstrap",
    "paired_relative_reduction_bootstrap",
    "select_pilot_candidates",
]
