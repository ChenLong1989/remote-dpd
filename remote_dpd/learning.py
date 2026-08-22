"""Pure NumPy learning rules for waveform-level ILC.

The complex vectors in this module represent real vector spaces.  Consequently,
all adjoints and Krylov-space scalars use ``real(vdot(left, right))``.  This is
important for PA models whose differential contains both a tangent and its
complex conjugate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


MIN_LM_DAMPING = 1e-8


class StopReason:
    """Stable string values used in learning and experiment diagnostics."""

    ACCEPTED = "accepted"
    CONVERGED = "converged"
    SATURATION_LIMITED = "saturation_limited"
    SAFETY_LIMITED = "safety_limited"
    PREDICTION_REJECTED = "prediction_rejected"
    NONFINITE_INPUT = "nonfinite_input"
    NONFINITE_UPDATE = "nonfinite_update"
    MODEL_FAILURE = "model_failure"
    PROJECTION_FAILED = "projection_failed"
    ZERO_RHS = "zero_rhs"
    CG_CONVERGED = "converged"
    CG_MAX_ITERATIONS = "max_iterations"
    CG_NON_POSITIVE_CURVATURE = "non_positive_curvature"
    CG_NONFINITE_OPERATOR = "nonfinite_operator"
    CG_NONFINITE_RESIDUAL = "nonfinite_residual"
    CG_MODEL_FAILURE = "model_failure"


@runtime_checkable
class PAModelLinearization(Protocol):
    """Frozen real-linear derivative returned by a PA forward model."""

    def jvp(self, tangent: np.ndarray) -> np.ndarray:
        """Apply the frozen real Jacobian to ``tangent``."""

    def vjp(self, cotangent: np.ndarray) -> np.ndarray:
        """Apply the frozen real-Jacobian transpose to ``cotangent``."""


@runtime_checkable
class PAForwardModel(Protocol):
    """Minimum forward-model interface accepted by the learning rules.

    The production memory-polynomial model additionally provides
    ``linearize(input_signal)``.  Solvers detect that method and freeze its
    result once per solve.  Direct ``jvp(input, tangent)`` and
    ``vjp(input, cotangent)`` remain a lightweight fallback for analytic or
    oracle models used by experiments.
    """

    def predict(self, input_signal: np.ndarray) -> np.ndarray:
        """Predict the PA output at ``input_signal``."""

    def jvp(self, input_signal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        """Apply the real Jacobian at ``input_signal`` to ``tangent``."""

    def vjp(self, input_signal: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        """Apply the real-Jacobian transpose at ``input_signal``."""


@dataclass(frozen=True, slots=True)
class InputSafetyLimits:
    """Hard limits applied to a candidate PA input waveform."""

    max_rms: float | None = None
    max_peak: float | None = None
    max_papr_db: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_rms", "max_peak"):
            value = getattr(self, name)
            if value is not None:
                value = _positive_finite(value, name)
                object.__setattr__(self, name, value)
        if self.max_papr_db is not None:
            value = _finite_float(self.max_papr_db, "max_papr_db")
            if value < 0.0:
                raise ValueError("max_papr_db must be greater than or equal to zero")
            object.__setattr__(self, "max_papr_db", value)


@dataclass(slots=True)
class InputProjectionResult:
    """Result of projecting a candidate onto the configured input limits."""

    projected_input: np.ndarray
    feasible: bool
    active: bool
    rms_active: bool = False
    peak_active: bool = False
    papr_active: bool = False
    stop_reason: str = StopReason.ACCEPTED

    @property
    def value(self) -> np.ndarray:
        return self.projected_input


@dataclass(slots=True)
class TrustRegionResult:
    """RMS trust-region projection for a proposed update."""

    update: np.ndarray
    active: bool
    radius_rms: float
    original_update_rms: float


@dataclass(slots=True)
class CGResult:
    """Structured outcome of the real-space matrix-free CG solve."""

    solution: np.ndarray
    converged: bool
    iterations: int
    relative_residual: float
    stop_reason: str
    initial_residual_norm: float
    residual_norm: float
    message: str | None = None

    @property
    def delta(self) -> np.ndarray:
        return self.solution


@dataclass(slots=True)
class LearningStepResult:
    """Common result returned by all learning-step functions."""

    next_input: np.ndarray
    update: np.ndarray
    accepted: bool
    stop_reason: str
    predicted_output: np.ndarray | None = None
    cg_result: CGResult | None = None
    trust_region_active: bool = False
    input_projection_active: bool = False
    saturation_limited: bool = False
    backtracks: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def output(self) -> np.ndarray:
        return self.next_input

    @property
    def delta(self) -> np.ndarray:
        return self.update


def real_inner(left: np.ndarray, right: np.ndarray) -> float:
    """Return the Euclidean inner product of two complex-vector encodings."""

    dtype = _common_complex_dtype(left, right)
    left_vector = np.asarray(left, dtype=dtype).reshape(-1)
    right_vector = np.asarray(right, dtype=dtype).reshape(-1)
    if left_vector.size != right_vector.size:
        raise ValueError("real_inner operands must have equal lengths")
    return float(np.real(np.vdot(left_vector, right_vector)))


def signal_rms(value: np.ndarray) -> float:
    """Compute waveform RMS with scaling that avoids avoidable overflow."""

    vector = _vector(value, "value")
    scale, _, normalized_rms = _scaled_signal_statistics(vector)
    return float(scale * normalized_rms)


def signal_peak(value: np.ndarray) -> float:
    """Compute the maximum complex magnitude without float32 overflow."""

    vector = _vector(value, "value")
    scale, normalized_peak, _ = _scaled_signal_statistics(vector)
    return float(scale * normalized_peak)


def signal_papr_db(value: np.ndarray) -> float:
    """Return amplitude PAPR in dB; the all-zero waveform has ``-inf`` PAPR."""

    vector = _vector(value, "value")
    _, normalized_peak, normalized_rms = _scaled_signal_statistics(vector)
    if normalized_rms == 0.0:
        return float("-inf")
    return float(20.0 * np.log10(normalized_peak / normalized_rms))


def input_within_safety_limits(
    value: np.ndarray,
    limits: InputSafetyLimits | None,
) -> bool:
    """Return whether ``value`` satisfies the shared numeric safety contract.

    The comparison tolerances follow the waveform dtype.  In particular, a
    feasible ``complex64`` projection must not be rejected later merely due to
    the final float32 rounding of a boundary value.
    """

    vector = _vector(value, "value")
    if not _is_finite(vector):
        return False
    if limits is None:
        return True
    if not isinstance(limits, InputSafetyLimits):
        raise TypeError("limits must be InputSafetyLimits or None")

    tolerance = max(2e-10, _relative_numeric_tolerance(vector))
    if limits.max_rms is not None:
        if signal_rms(vector) > limits.max_rms * (1.0 + tolerance):
            return False
    if limits.max_peak is not None:
        if signal_peak(vector) > limits.max_peak * (1.0 + tolerance):
            return False
    if limits.max_papr_db is not None:
        if signal_papr_db(vector) > limits.max_papr_db + _papr_numeric_tolerance_db(vector):
            return False
    return True


def _real_dtype_epsilon(value: np.ndarray) -> float:
    vector = np.asarray(value)
    dtype = np.float32 if vector.dtype == np.dtype(np.complex64) else np.float64
    return float(np.finfo(dtype).eps)


def _relative_numeric_tolerance(value: np.ndarray) -> float:
    return max(2e-12, 16.0 * _real_dtype_epsilon(value))


def _papr_numeric_tolerance_db(value: np.ndarray) -> float:
    return max(2e-9, 16.0 * _real_dtype_epsilon(value))


def apply_rms_trust_region(
    update: np.ndarray,
    current_input: np.ndarray,
    ratio: float,
) -> TrustRegionResult:
    """Limit update RMS to ``ratio * RMS(current_input)``.

    A zero current waveform has a zero relative trust radius.  A caller that
    needs to leave the origin must initialize the waveform or use no trust
    region for that step.
    """

    ratio = _positive_finite(ratio, "ratio")
    dtype = _common_complex_dtype(update, current_input)
    update_vector = _vector(update, "update", dtype=dtype)
    current_vector = _vector(current_input, "current_input", dtype=dtype)
    if update_vector.size != current_vector.size:
        raise ValueError("update and current_input must have equal lengths")
    if not _is_finite(update_vector) or not _is_finite(current_vector):
        raise ValueError("trust-region inputs must be finite")

    original_rms = signal_rms(update_vector)
    radius = ratio * signal_rms(current_vector)
    tolerance = _relative_numeric_tolerance(update_vector)
    if original_rms <= radius * (1.0 + tolerance):
        return TrustRegionResult(update_vector.copy(), False, radius, original_rms)
    if radius == 0.0:
        limited = np.zeros_like(update_vector)
    else:
        limited = update_vector * (radius / original_rms)
        limited = np.asarray(limited, dtype=update_vector.dtype)
        limited_rms = signal_rms(limited)
        if limited_rms > radius:
            limited *= (radius / limited_rms) * (1.0 - 4.0 * _real_dtype_epsilon(limited))
    return TrustRegionResult(limited, True, radius, original_rms)


def project_input_safety(
    candidate: np.ndarray,
    limits: InputSafetyLimits | None,
) -> InputProjectionResult:
    """Project a candidate onto peak, RMS, and PAPR safety limits.

    PAPR is reduced by phase-preserving magnitude clipping.  RMS and peak
    limits are then enforced by a common scale, which preserves the achieved
    PAPR.  If attenuation-only PAPR clipping is infeasible (for example, too
    many exact zeros for the requested ratio), the safe all-zero waveform is
    returned instead of a constraint-violating candidate.
    """

    vector = _vector(candidate, "candidate")
    if not _is_finite(vector):
        return InputProjectionResult(
            projected_input=np.zeros_like(vector),
            feasible=False,
            active=True,
            stop_reason=StopReason.NONFINITE_UPDATE,
        )
    if limits is None:
        return InputProjectionResult(vector.copy(), True, False)
    if not isinstance(limits, InputSafetyLimits):
        raise TypeError("limits must be InputSafetyLimits or None")

    projected = vector.copy()
    papr_active = False
    rms_active = False
    peak_active = False

    papr_tolerance_db = _papr_numeric_tolerance_db(projected)
    if (
        limits.max_papr_db is not None
        and signal_papr_db(projected) > limits.max_papr_db + papr_tolerance_db
    ):
        projected = _clip_papr(projected, limits.max_papr_db)
        papr_active = True

    rms_value = signal_rms(projected)
    peak_value = signal_peak(projected)
    scale = 1.0
    if limits.max_rms is not None and rms_value > limits.max_rms:
        scale = min(scale, limits.max_rms / rms_value)
        rms_active = True
    if limits.max_peak is not None and peak_value > limits.max_peak:
        scale = min(scale, limits.max_peak / peak_value)
        peak_active = True
    if scale < 1.0:
        projected *= scale

    feasible = input_within_safety_limits(projected, limits)
    active = papr_active or rms_active or peak_active
    return InputProjectionResult(
        projected_input=np.asarray(projected, dtype=vector.dtype),
        feasible=bool(feasible),
        active=active,
        rms_active=rms_active,
        peak_active=peak_active,
        papr_active=papr_active,
        stop_reason=StopReason.ACCEPTED if feasible else StopReason.PROJECTION_FAILED,
    )


def linear_ilc_step(
    current_input: np.ndarray,
    desired_output: np.ndarray,
    measured_output: np.ndarray,
    learning_rate: float,
    *,
    safety_limits: InputSafetyLimits | None = None,
) -> LearningStepResult:
    """Apply the public scalar linear ILC update ``u + mu * (d - y)``."""

    learning_rate = _positive_finite(learning_rate, "learning_rate")
    current, desired, measured = _step_vectors(current_input, desired_output, measured_output)
    if not _all_finite(current, desired, measured):
        return _held_result(current, StopReason.NONFINITE_INPUT)
    error = measured - desired
    error_rms = signal_rms(error)
    if error_rms == 0.0:
        return _held_result(current, StopReason.CONVERGED, diagnostics={"error_rms": 0.0})

    proposed = current - learning_rate * error
    projection = project_input_safety(proposed, safety_limits)
    if not projection.feasible:
        return _held_result(
            current,
            StopReason.PROJECTION_FAILED,
            projection_active=projection.active,
            diagnostics={"error_rms": error_rms},
        )
    update = projection.projected_input - current
    predicted = measured + update
    accepted = signal_rms(update) > 0.0
    return LearningStepResult(
        next_input=projection.projected_input,
        update=update,
        accepted=accepted,
        stop_reason=StopReason.ACCEPTED if accepted else StopReason.SAFETY_LIMITED,
        predicted_output=predicted,
        input_projection_active=projection.active,
        diagnostics={
            "error_rms": error_rms,
            "update_rms": signal_rms(update),
            "predicted_error_rms": signal_rms(predicted - desired),
        },
    )


def instantaneous_gain_ilc_step(
    current_input: np.ndarray,
    desired_output: np.ndarray,
    measured_output: np.ndarray,
    learning_rate: float,
    *,
    damping: float = 1e-2,
    input_threshold: float = 1e-8,
    safety_limits: InputSafetyLimits | None = None,
    saturation_tolerance: float = 1e-10,
) -> LearningStepResult:
    """Apply a thresholded, Tikhonov-damped instantaneous-gain ILC step.

    At valid samples, ``gain = measured / current_input`` and the correction is
    ``mu * conj(gain) * (desired - measured) / (abs(gain)**2 + damping)``.
    Samples below ``input_threshold`` are held, because their gain estimate is
    undefined.  A nonzero residual with no usable local gain is reported as
    ``saturation_limited``.
    """

    learning_rate = _positive_finite(learning_rate, "learning_rate")
    damping = _positive_finite(damping, "damping")
    input_threshold = _nonnegative_finite(input_threshold, "input_threshold")
    saturation_tolerance = _positive_finite(saturation_tolerance, "saturation_tolerance")
    current, desired, measured = _step_vectors(current_input, desired_output, measured_output)
    if not _all_finite(current, desired, measured):
        return _held_result(current, StopReason.NONFINITE_INPUT)
    error = measured - desired
    error_rms = signal_rms(error)
    if error_rms == 0.0:
        return _held_result(current, StopReason.CONVERGED, diagnostics={"error_rms": 0.0})

    gain = np.zeros_like(current)
    valid = np.abs(current) > input_threshold
    gain[valid] = measured[valid] / current[valid]
    valid &= np.isfinite(gain.real) & np.isfinite(gain.imag)
    correction = np.zeros_like(current)
    denominator = np.abs(gain[valid]) ** 2 + damping
    correction[valid] = -learning_rate * np.conj(gain[valid]) * error[valid] / denominator
    if not _is_finite(correction):
        return _held_result(current, StopReason.NONFINITE_UPDATE, diagnostics={"error_rms": error_rms})
    if _locally_unresponsive(error, correction, saturation_tolerance):
        return _held_result(
            current,
            StopReason.SATURATION_LIMITED,
            saturation_limited=True,
            diagnostics={
                "error_rms": error_rms,
                "gain_valid_fraction": float(np.mean(valid)),
                "update_rms": signal_rms(correction),
            },
        )

    projection = project_input_safety(current + correction, safety_limits)
    if not projection.feasible:
        return _held_result(
            current,
            StopReason.PROJECTION_FAILED,
            projection_active=projection.active,
            diagnostics={"error_rms": error_rms},
        )
    update = projection.projected_input - current
    predicted = measured.copy()
    predicted[valid] += gain[valid] * update[valid]
    accepted = signal_rms(update) > 0.0
    return LearningStepResult(
        next_input=projection.projected_input,
        update=update,
        accepted=accepted,
        stop_reason=StopReason.ACCEPTED if accepted else StopReason.SAFETY_LIMITED,
        predicted_output=predicted,
        input_projection_active=projection.active,
        diagnostics={
            "error_rms": error_rms,
            "gain_valid_fraction": float(np.mean(valid)),
            "update_rms": signal_rms(update),
            "predicted_error_rms": signal_rms(predicted - desired),
        },
    )


def model_vjp_ilc_step(
    current_input: np.ndarray,
    desired_output: np.ndarray,
    measured_output: np.ndarray,
    model: PAForwardModel,
    learning_rate: float,
    *,
    safety_limits: InputSafetyLimits | None = None,
    saturation_tolerance: float = 1e-10,
) -> LearningStepResult:
    """Apply the raw PA-model VJP update used as a mechanism ablation."""

    learning_rate = _positive_finite(learning_rate, "learning_rate")
    saturation_tolerance = _positive_finite(saturation_tolerance, "saturation_tolerance")
    current, desired, measured = _step_vectors(
        current_input,
        desired_output,
        measured_output,
    )
    if not _all_finite(current, desired, measured):
        return _held_result(current, StopReason.NONFINITE_INPUT)
    error = measured - desired
    error_rms = signal_rms(error)
    if error_rms == 0.0:
        return _held_result(current, StopReason.CONVERGED, diagnostics={"error_rms": 0.0})

    try:
        linearization = _freeze_linearization(model, current)
        gradient = _linearization_vjp(linearization, error, current.size)
    except (TypeError, ValueError, ArithmeticError, RuntimeError) as exc:
        return _held_result(
            current,
            StopReason.MODEL_FAILURE,
            diagnostics={"error_rms": error_rms, "model_error": str(exc)},
        )
    if not _is_finite(gradient):
        return _held_result(current, StopReason.MODEL_FAILURE, diagnostics={"error_rms": error_rms})
    update = -learning_rate * gradient
    if _locally_unresponsive(error, gradient, saturation_tolerance):
        return _held_result(
            current,
            StopReason.SATURATION_LIMITED,
            saturation_limited=True,
            diagnostics={"error_rms": error_rms, "gradient_rms": signal_rms(gradient)},
        )
    projection = project_input_safety(current + update, safety_limits)
    if not projection.feasible:
        return _held_result(current, StopReason.PROJECTION_FAILED, projection_active=projection.active)
    effective_update = projection.projected_input - current
    predicted: np.ndarray | None = None
    prediction_error: str | None = None
    try:
        predicted = anchored_prediction(model, current, measured, effective_update)
    except (TypeError, ValueError, ArithmeticError, RuntimeError) as exc:
        prediction_error = str(exc)
    diagnostics: dict[str, Any] = {
        "error_rms": error_rms,
        "gradient_rms": signal_rms(gradient),
        "update_rms": signal_rms(effective_update),
    }
    if predicted is not None:
        diagnostics["predicted_error_rms"] = signal_rms(predicted - desired)
    if prediction_error is not None:
        diagnostics["prediction_error"] = prediction_error
    accepted = signal_rms(effective_update) > 0.0
    return LearningStepResult(
        next_input=projection.projected_input,
        update=effective_update,
        accepted=accepted,
        stop_reason=StopReason.ACCEPTED if accepted else StopReason.SAFETY_LIMITED,
        predicted_output=predicted,
        input_projection_active=projection.active,
        diagnostics=diagnostics,
    )


def damped_normal_matvec(
    model: PAForwardModel,
    linearization_input: np.ndarray,
    vector: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Apply ``J.T @ J + damping * I`` without materializing ``J``."""

    damping = _lm_damping(damping)
    dtype = _common_complex_dtype(linearization_input, vector)
    point = _vector(linearization_input, "linearization_input", dtype=dtype)
    tangent = _vector(vector, "vector", dtype=dtype)
    if point.size != tangent.size:
        raise ValueError("linearization_input and vector must have equal lengths")
    if not _all_finite(point, tangent):
        raise ValueError("normal-operator inputs must be finite")
    linearization = _freeze_linearization(model, point)
    return _frozen_normal_matvec(linearization, tangent, damping, point.size)


def damped_lm_cg(
    model: PAForwardModel,
    linearization_input: np.ndarray,
    error: np.ndarray,
    *,
    damping: float,
    max_iterations: int = 8,
    relative_tolerance: float = 1e-3,
    linearization: PAModelLinearization | None = None,
) -> CGResult:
    """Solve the damped real Gauss--Newton system with matrix-free CG.

    The right-hand side is ``-J.T @ error`` and the normal operator is exactly
    ``J.T @ J + damping * I``.  CG ``alpha`` and ``beta`` are always real.
    Non-positive curvature and every non-finite intermediate terminate safely
    with the last finite iterate.
    """

    damping = _lm_damping(damping)
    max_iterations = _positive_integer(max_iterations, "max_iterations")
    relative_tolerance = _relative_tolerance(relative_tolerance)
    dtype = _common_complex_dtype(linearization_input, error)
    point = _vector(linearization_input, "linearization_input", dtype=dtype)
    residual_error = _vector(error, "error", dtype=dtype)
    zero = np.zeros_like(point)
    if not _all_finite(point, residual_error):
        return CGResult(zero, False, 0, float("inf"), StopReason.NONFINITE_INPUT, 0.0, float("inf"))

    try:
        frozen = _freeze_linearization(model, point) if linearization is None else linearization
        rhs = -_linearization_vjp(frozen, residual_error, point.size)
    except (TypeError, ValueError, ArithmeticError, RuntimeError) as exc:
        return CGResult(
            zero,
            False,
            0,
            float("inf"),
            StopReason.CG_MODEL_FAILURE,
            0.0,
            float("inf"),
            str(exc),
        )
    if not _is_finite(rhs):
        return CGResult(
            zero,
            False,
            0,
            float("inf"),
            StopReason.CG_NONFINITE_RESIDUAL,
            float("inf"),
            float("inf"),
        )

    solution = zero.copy()
    residual = rhs.copy()
    direction = residual.copy()
    residual_squared = real_inner(residual, residual)
    if not np.isfinite(residual_squared) or residual_squared < 0.0:
        return CGResult(
            zero,
            False,
            0,
            float("inf"),
            StopReason.CG_NONFINITE_RESIDUAL,
            float("inf"),
            float("inf"),
        )
    initial_norm = float(np.sqrt(residual_squared))
    if initial_norm == 0.0:
        return CGResult(zero, True, 0, 0.0, StopReason.ZERO_RHS, 0.0, 0.0)

    for iteration in range(1, max_iterations + 1):
        try:
            normal_direction = _frozen_normal_matvec(frozen, direction, damping, point.size)
        except (TypeError, ValueError, ArithmeticError, RuntimeError) as exc:
            return CGResult(
                solution,
                False,
                iteration - 1,
                float(np.sqrt(residual_squared) / initial_norm),
                StopReason.CG_NONFINITE_OPERATOR,
                initial_norm,
                float(np.sqrt(residual_squared)),
                str(exc),
            )
        denominator = real_inner(direction, normal_direction)
        if not np.isfinite(denominator):
            return CGResult(
                solution,
                False,
                iteration - 1,
                float(np.sqrt(residual_squared) / initial_norm),
                StopReason.CG_NONFINITE_OPERATOR,
                initial_norm,
                float(np.sqrt(residual_squared)),
            )
        if denominator <= 0.0:
            return CGResult(
                solution,
                False,
                iteration - 1,
                float(np.sqrt(residual_squared) / initial_norm),
                StopReason.CG_NON_POSITIVE_CURVATURE,
                initial_norm,
                float(np.sqrt(residual_squared)),
            )

        alpha = float(residual_squared / denominator)
        if not np.isfinite(alpha):
            return CGResult(
                solution,
                False,
                iteration - 1,
                float(np.sqrt(residual_squared) / initial_norm),
                StopReason.CG_NONFINITE_RESIDUAL,
                initial_norm,
                float(np.sqrt(residual_squared)),
            )
        candidate_solution = solution + alpha * direction
        candidate_residual = residual - alpha * normal_direction
        if not _all_finite(candidate_solution, candidate_residual):
            return CGResult(
                solution,
                False,
                iteration - 1,
                float(np.sqrt(residual_squared) / initial_norm),
                StopReason.CG_NONFINITE_RESIDUAL,
                initial_norm,
                float(np.sqrt(residual_squared)),
            )
        new_residual_squared = real_inner(candidate_residual, candidate_residual)
        if not np.isfinite(new_residual_squared) or new_residual_squared < 0.0:
            return CGResult(
                solution,
                False,
                iteration - 1,
                float(np.sqrt(residual_squared) / initial_norm),
                StopReason.CG_NONFINITE_RESIDUAL,
                initial_norm,
                float(np.sqrt(residual_squared)),
            )

        solution = np.asarray(candidate_solution, dtype=point.dtype)
        residual = np.asarray(candidate_residual, dtype=point.dtype)
        residual_norm = float(np.sqrt(new_residual_squared))
        relative_residual = residual_norm / initial_norm
        if relative_residual <= relative_tolerance:
            return CGResult(
                solution,
                True,
                iteration,
                relative_residual,
                StopReason.CG_CONVERGED,
                initial_norm,
                residual_norm,
            )

        beta = float(new_residual_squared / residual_squared)
        if not np.isfinite(beta) or beta < 0.0:
            return CGResult(
                solution,
                False,
                iteration,
                relative_residual,
                StopReason.CG_NONFINITE_RESIDUAL,
                initial_norm,
                residual_norm,
            )
        direction = residual + beta * direction
        if not _is_finite(direction):
            return CGResult(
                solution,
                False,
                iteration,
                relative_residual,
                StopReason.CG_NONFINITE_RESIDUAL,
                initial_norm,
                residual_norm,
            )
        residual_squared = new_residual_squared

    residual_norm = float(np.sqrt(residual_squared))
    return CGResult(
        solution,
        False,
        max_iterations,
        residual_norm / initial_norm,
        StopReason.CG_MAX_ITERATIONS,
        initial_norm,
        residual_norm,
    )


def anchored_prediction(
    model: PAForwardModel,
    current_input: np.ndarray,
    measured_output: np.ndarray,
    update: np.ndarray,
) -> np.ndarray:
    """Predict from the measurement anchor, cancelling static model bias."""

    dtype = _common_complex_dtype(current_input, measured_output, update)
    current = _vector(current_input, "current_input", dtype=dtype)
    measured = _vector(measured_output, "measured_output", dtype=dtype)
    delta = _vector(update, "update", dtype=dtype)
    if current.size != delta.size:
        raise ValueError("current_input and update must have equal lengths")
    if not _all_finite(current, measured, delta):
        raise ValueError("anchored-prediction inputs must be finite")
    prediction_at_current = _model_predict(model, current)
    prediction_at_candidate = _model_predict(model, current + delta)
    if prediction_at_current.size != measured.size or prediction_at_candidate.size != measured.size:
        raise ValueError("model prediction and measured_output must have equal lengths")
    prediction = measured + prediction_at_candidate - prediction_at_current
    if not _is_finite(prediction):
        raise ArithmeticError("anchored prediction returned non-finite values")
    return np.asarray(prediction, dtype=current.dtype)


def model_lm_ilc_step(
    current_input: np.ndarray,
    desired_output: np.ndarray,
    measured_output: np.ndarray,
    model: PAForwardModel,
    *,
    damping: float,
    step_size: float = 1.0,
    cg_max_iterations: int = 8,
    cg_relative_tolerance: float = 1e-3,
    trust_region_ratio: float | None = 0.25,
    safety_limits: InputSafetyLimits | None = None,
    max_backtracks: int = 8,
    backtrack_factor: float = 0.5,
    minimum_relative_decrease: float = 0.0,
    saturation_tolerance: float = 1e-10,
    prediction_mode: str = "anchored",
) -> LearningStepResult:
    """Apply a safeguarded damped Gauss--Newton/LM ILC step."""

    damping = _lm_damping(damping)
    step_size = _positive_finite(step_size, "step_size")
    cg_max_iterations = _positive_integer(cg_max_iterations, "cg_max_iterations")
    cg_relative_tolerance = _relative_tolerance(cg_relative_tolerance)
    if trust_region_ratio is not None:
        trust_region_ratio = _positive_finite(trust_region_ratio, "trust_region_ratio")
    if isinstance(max_backtracks, (bool, np.bool_)) or not isinstance(max_backtracks, (int, np.integer)):
        raise ValueError("max_backtracks must be a nonnegative integer")
    max_backtracks = int(max_backtracks)
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer")
    backtrack_factor = _finite_float(backtrack_factor, "backtrack_factor")
    if not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must be strictly between zero and one")
    minimum_relative_decrease = _finite_float(minimum_relative_decrease, "minimum_relative_decrease")
    if not 0.0 <= minimum_relative_decrease < 1.0:
        raise ValueError("minimum_relative_decrease must be in [0, 1)")
    saturation_tolerance = _positive_finite(saturation_tolerance, "saturation_tolerance")
    if prediction_mode not in {"anchored", "unanchored"}:
        raise ValueError("prediction_mode must be 'anchored' or 'unanchored'")

    current, desired, measured = _step_vectors(
        current_input,
        desired_output,
        measured_output,
    )
    if not _all_finite(current, desired, measured):
        return _held_result(current, StopReason.NONFINITE_INPUT)
    error = measured - desired
    error_rms = signal_rms(error)
    if error_rms == 0.0:
        return _held_result(current, StopReason.CONVERGED, diagnostics={"error_rms": 0.0})

    try:
        linearization = _freeze_linearization(model, current)
        gradient = _linearization_vjp(linearization, error, current.size)
    except (TypeError, ValueError, ArithmeticError, RuntimeError) as exc:
        return _held_result(
            current,
            StopReason.MODEL_FAILURE,
            diagnostics={"error_rms": error_rms, "model_error": str(exc)},
        )
    if not _is_finite(gradient):
        return _held_result(current, StopReason.MODEL_FAILURE, diagnostics={"error_rms": error_rms})
    gradient_rms = signal_rms(gradient)
    if _locally_unresponsive(error, gradient, saturation_tolerance):
        return _held_result(
            current,
            StopReason.SATURATION_LIMITED,
            saturation_limited=True,
            diagnostics={"error_rms": error_rms, "gradient_rms": gradient_rms},
        )

    cg_result = damped_lm_cg(
        model,
        current,
        error,
        damping=damping,
        max_iterations=cg_max_iterations,
        relative_tolerance=cg_relative_tolerance,
        linearization=linearization,
    )
    if cg_result.stop_reason in {
        StopReason.CG_NON_POSITIVE_CURVATURE,
        StopReason.CG_NONFINITE_OPERATOR,
        StopReason.CG_NONFINITE_RESIDUAL,
        StopReason.CG_MODEL_FAILURE,
    }:
        return _held_result(
            current,
            f"cg_{cg_result.stop_reason}",
            cg_result=cg_result,
            diagnostics={"error_rms": error_rms, "gradient_rms": gradient_rms},
        )
    if not _is_finite(cg_result.solution):
        return _held_result(
            current,
            StopReason.NONFINITE_UPDATE,
            cg_result=cg_result,
            diagnostics={"error_rms": error_rms, "gradient_rms": gradient_rms},
        )

    current_loss = 0.5 * real_inner(error, error)
    required_loss = current_loss * (1.0 - minimum_relative_decrease)
    last_prediction: np.ndarray | None = None
    any_trust_active = False
    any_projection_active = False
    last_prediction_rms = float("nan")
    last_model_error: str | None = None
    post_projection_trust_rejections = 0
    base_update = step_size * cg_result.solution
    trust_radius: float | None = None
    if trust_region_ratio is not None:
        trust_result = apply_rms_trust_region(base_update, current, trust_region_ratio)
        base_update = trust_result.update
        trust_radius = trust_result.radius_rms
        any_trust_active = trust_result.active

    for backtracks in range(max_backtracks + 1):
        update = (backtrack_factor**backtracks) * base_update
        projection = project_input_safety(current + update, safety_limits)
        any_projection_active = any_projection_active or projection.active
        if not projection.feasible:
            continue
        effective_update = projection.projected_input - current
        if not _is_finite(effective_update) or signal_rms(effective_update) == 0.0:
            continue
        if (
            trust_radius is not None
            and signal_rms(effective_update)
            > trust_radius * (1.0 + _relative_numeric_tolerance(effective_update))
        ):
            # Peak/PAPR projection is not a projection in update space and can
            # move the candidate outside the RMS trust ball. Reject it and let
            # the existing line search try a smaller jointly feasible step.
            any_trust_active = True
            post_projection_trust_rejections += 1
            continue
        try:
            if prediction_mode == "anchored":
                prediction = anchored_prediction(model, current, measured, effective_update)
            else:
                prediction = _model_predict(model, current + effective_update)
                if prediction.size != measured.size:
                    raise ValueError("model prediction and measured_output must have equal lengths")
                prediction = np.asarray(prediction, dtype=current.dtype)
        except (TypeError, ValueError, ArithmeticError, RuntimeError) as exc:
            last_model_error = str(exc)
            continue
        last_prediction = prediction
        predicted_error = prediction - desired
        predicted_loss = 0.5 * real_inner(predicted_error, predicted_error)
        last_prediction_rms = signal_rms(predicted_error)
        if np.isfinite(predicted_loss) and predicted_loss < required_loss:
            return LearningStepResult(
                next_input=projection.projected_input,
                update=effective_update,
                accepted=True,
                stop_reason=StopReason.ACCEPTED,
                predicted_output=prediction,
                cg_result=cg_result,
                trust_region_active=any_trust_active,
                input_projection_active=any_projection_active,
                backtracks=backtracks,
                diagnostics={
                    "error_rms": error_rms,
                    "gradient_rms": gradient_rms,
                    "update_rms": signal_rms(effective_update),
                    "predicted_error_rms": last_prediction_rms,
                    "predicted_relative_reduction": 1.0 - predicted_loss / current_loss,
                    "lm_damping": damping,
                    "cg_iterations": cg_result.iterations,
                    "cg_relative_residual": cg_result.relative_residual,
                    "post_projection_trust_rejections": post_projection_trust_rejections,
                    "prediction_mode": prediction_mode,
                },
            )

    diagnostics: dict[str, Any] = {
        "error_rms": error_rms,
        "gradient_rms": gradient_rms,
        "predicted_error_rms": last_prediction_rms,
        "lm_damping": damping,
        "cg_iterations": cg_result.iterations,
        "cg_relative_residual": cg_result.relative_residual,
        "post_projection_trust_rejections": post_projection_trust_rejections,
        "prediction_mode": prediction_mode,
    }
    if last_model_error is not None:
        diagnostics["model_error"] = last_model_error
    return _held_result(
        current,
        StopReason.PREDICTION_REJECTED,
        cg_result=cg_result,
        trust_active=any_trust_active,
        projection_active=any_projection_active,
        predicted_output=last_prediction,
        backtracks=max_backtracks,
        diagnostics=diagnostics,
    )


# Descriptive aliases kept for callers that name the solve or step by method.
solve_damped_lm_cg = damped_lm_cg
damped_lm_ilc_step = model_lm_ilc_step


def _held_result(
    current: np.ndarray,
    reason: str,
    *,
    cg_result: CGResult | None = None,
    trust_active: bool = False,
    projection_active: bool = False,
    saturation_limited: bool = False,
    predicted_output: np.ndarray | None = None,
    backtracks: int = 0,
    diagnostics: Mapping[str, Any] | None = None,
) -> LearningStepResult:
    current_vector = _vector(current, "current").copy()
    if not _is_finite(current_vector):
        current_vector = np.zeros_like(current_vector)
    return LearningStepResult(
        next_input=current_vector,
        update=np.zeros_like(current_vector),
        accepted=False,
        stop_reason=reason,
        predicted_output=predicted_output,
        cg_result=cg_result,
        trust_region_active=trust_active,
        input_projection_active=projection_active,
        saturation_limited=saturation_limited,
        backtracks=backtracks,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def _step_vectors(
    current_input: np.ndarray,
    desired_output: np.ndarray,
    measured_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = _common_complex_dtype(current_input, desired_output, measured_output)
    current = _vector(current_input, "current_input", dtype=dtype)
    desired = _vector(desired_output, "desired_output", dtype=dtype)
    measured = _vector(measured_output, "measured_output", dtype=dtype)
    if current.size != desired.size or desired.size != measured.size:
        raise ValueError("current_input, desired_output, and measured_output must have equal lengths")
    return current, desired, measured


@dataclass(slots=True)
class _PointLinearization:
    """Adapter for models exposing only point-wise two-argument derivatives."""

    model: PAForwardModel
    point: np.ndarray

    def jvp(self, tangent: np.ndarray) -> np.ndarray:
        return _model_jvp(self.model, self.point, tangent)

    def vjp(self, cotangent: np.ndarray) -> np.ndarray:
        return _model_vjp(self.model, self.point, cotangent)


def _freeze_linearization(model: PAForwardModel, point: np.ndarray) -> PAModelLinearization:
    linearize = getattr(model, "linearize", None)
    if callable(linearize):
        frozen = linearize(point)
        if not callable(getattr(frozen, "jvp", None)) or not callable(getattr(frozen, "vjp", None)):
            raise TypeError("model.linearize() must return an object with jvp() and vjp()")
        return frozen
    if not callable(getattr(model, "jvp", None)) or not callable(getattr(model, "vjp", None)):
        raise TypeError("model must provide linearize() or point-wise jvp() and vjp()")
    return _PointLinearization(model, point.copy())


def _linearization_jvp(linearization: PAModelLinearization, tangent: np.ndarray) -> np.ndarray:
    result = _vector(linearization.jvp(tangent), "model JVP")
    if not _is_finite(result):
        raise ArithmeticError("model JVP returned non-finite values")
    return result


def _linearization_vjp(
    linearization: PAModelLinearization,
    cotangent: np.ndarray,
    input_size: int,
) -> np.ndarray:
    result = _vector(linearization.vjp(cotangent), "model VJP")
    if result.size != input_size:
        raise ValueError("model VJP must have the same length as the linearization input")
    if not _is_finite(result):
        raise ArithmeticError("model VJP returned non-finite values")
    return result


def _frozen_normal_matvec(
    linearization: PAModelLinearization,
    tangent: np.ndarray,
    damping: float,
    input_size: int,
) -> np.ndarray:
    j_tangent = _linearization_jvp(linearization, tangent)
    result = _linearization_vjp(linearization, j_tangent, input_size) + damping * tangent
    if not _is_finite(result):
        raise ArithmeticError("normal operator returned non-finite values")
    return np.asarray(result, dtype=tangent.dtype)


def _model_predict(model: PAForwardModel, point: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict"):
        raise TypeError("model must provide predict(input_signal)")
    result = _vector(model.predict(point), "model prediction")
    if not _is_finite(result):
        raise ArithmeticError("model prediction returned non-finite values")
    return result


def _model_jvp(model: PAForwardModel, point: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    if not hasattr(model, "jvp"):
        raise TypeError("model must provide jvp(input_signal, tangent)")
    result = _vector(model.jvp(point, tangent), "model JVP")
    if not _is_finite(result):
        raise ArithmeticError("model JVP returned non-finite values")
    return result


def _model_vjp(model: PAForwardModel, point: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
    if not hasattr(model, "vjp"):
        raise TypeError("model must provide vjp(input_signal, cotangent)")
    result = _vector(model.vjp(point, cotangent), "model VJP")
    if result.size != point.size:
        raise ValueError("model VJP must have the same length as the linearization input")
    if not _is_finite(result):
        raise ArithmeticError("model VJP returned non-finite values")
    return result


def _clip_papr(vector: np.ndarray, max_papr_db: float) -> np.ndarray:
    component_scale = _component_scale(vector)
    if component_scale == 0.0:
        return vector.copy()
    normalized = _divide_by_real_scale(vector, component_scale)
    magnitudes = np.hypot(normalized.real, normalized.imag)
    peak = float(np.max(magnitudes))
    target_ratio = float(10.0 ** (max_papr_db / 20.0))
    epsilon = _real_dtype_epsilon(vector)
    low = 0.0
    high = peak
    for _ in range(100):
        cap = 0.5 * (low + high)
        clipped_rms = _stable_l2_norm(np.minimum(magnitudes, cap)) / np.sqrt(vector.size)
        if cap <= target_ratio * clipped_rms:
            low = cap
        else:
            high = cap
    cap = low
    if cap <= peak * epsilon:
        return np.zeros_like(vector)
    # Move inside the boundary before casting back to the waveform dtype.
    cap *= max(0.0, 1.0 - 32.0 * epsilon)
    tiny = np.finfo(np.float32 if vector.dtype == np.dtype(np.complex64) else np.float64).tiny
    scale = np.minimum(1.0, cap / np.maximum(magnitudes, tiny))
    clipped = vector * scale
    if signal_papr_db(clipped) > max_papr_db + _papr_numeric_tolerance_db(clipped):
        cap *= max(0.0, 1.0 - 64.0 * epsilon)
        scale = np.minimum(1.0, cap / np.maximum(magnitudes, tiny))
        clipped = vector * scale
    return np.asarray(clipped, dtype=vector.dtype)


def _locally_unresponsive(error: np.ndarray, response: np.ndarray, tolerance: float) -> bool:
    joint_scale = max(_component_scale(error), _component_scale(response))
    if joint_scale == 0.0:
        return False
    error_norm = _stable_l2_norm(_divide_by_real_scale(error, joint_scale))
    if error_norm == 0.0:
        return False
    response_norm = _stable_l2_norm(_divide_by_real_scale(response, joint_scale))
    return response_norm <= tolerance * error_norm


def _component_scale(value: np.ndarray) -> float:
    """Return the largest real or imaginary component magnitude."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        return max(float(np.max(np.abs(array.real))), float(np.max(np.abs(array.imag))))
    return float(np.max(np.abs(array)))


def _stable_l2_norm(value: np.ndarray) -> float:
    """Compute an L2 norm after scaling, avoiding spurious square overflow."""

    array = np.asarray(value)
    scale = _component_scale(array)
    if scale == 0.0:
        return 0.0
    if np.iscomplexobj(array):
        normalized = _divide_by_real_scale(array, scale)
        squared_sum = np.sum(normalized.real**2 + normalized.imag**2)
    else:
        normalized = np.asarray(array, dtype=np.float64) / scale
        squared_sum = np.sum(normalized**2)
    return float(scale * np.sqrt(squared_sum))


def _scaled_signal_statistics(value: np.ndarray) -> tuple[float, float, float]:
    """Return component scale plus peak and RMS in normalized units."""

    vector = np.asarray(value)
    scale = _component_scale(vector)
    if scale == 0.0:
        return 0.0, 0.0, 0.0
    normalized = _divide_by_real_scale(vector, scale)
    magnitudes = np.hypot(normalized.real, normalized.imag)
    peak = float(np.max(magnitudes))
    rms = _stable_l2_norm(magnitudes) / np.sqrt(vector.size)
    return scale, peak, rms


def _divide_by_real_scale(value: np.ndarray, scale: float) -> np.ndarray:
    """Divide components separately so complex division handles subnormals."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        normalized = np.empty(array.shape, dtype=np.complex128)
        normalized.real = np.asarray(array.real, dtype=np.float64) / scale
        normalized.imag = np.asarray(array.imag, dtype=np.float64) / scale
        return normalized
    return np.asarray(array, dtype=np.float64) / scale


def _complex_dtype(value: Any) -> np.dtype[Any]:
    return (
        np.dtype(np.complex64)
        if np.asarray(value).dtype == np.dtype(np.complex64)
        else np.dtype(np.complex128)
    )


def _common_complex_dtype(*values: Any) -> np.dtype[Any]:
    return (
        np.dtype(np.complex64)
        if values and all(_complex_dtype(value) == np.dtype(np.complex64) for value in values)
        else np.dtype(np.complex128)
    )


def _vector(
    value: np.ndarray,
    name: str,
    *,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    try:
        vector = np.asarray(
            value,
            dtype=_complex_dtype(value) if dtype is None else dtype,
        ).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a complex vector") from exc
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    return vector


def _is_finite(value: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(value.real)) and np.all(np.isfinite(value.imag)))


def _all_finite(*values: np.ndarray) -> bool:
    return all(_is_finite(value) for value in values)


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_finite(value: float, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return converted


def _nonnegative_finite(value: float, name: str) -> float:
    converted = _finite_float(value, name)
    if converted < 0.0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return converted


def _lm_damping(value: float) -> float:
    converted = _positive_finite(value, "damping")
    if converted < MIN_LM_DAMPING:
        raise ValueError(f"damping must be at least {MIN_LM_DAMPING:g}")
    return converted


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _relative_tolerance(value: float) -> float:
    converted = _positive_finite(value, "relative_tolerance")
    if converted >= 1.0:
        raise ValueError("relative_tolerance must be less than one")
    return converted


__all__ = [
    "CGResult",
    "InputProjectionResult",
    "InputSafetyLimits",
    "LearningStepResult",
    "MIN_LM_DAMPING",
    "PAForwardModel",
    "PAModelLinearization",
    "StopReason",
    "TrustRegionResult",
    "anchored_prediction",
    "apply_rms_trust_region",
    "damped_lm_cg",
    "damped_lm_ilc_step",
    "damped_normal_matvec",
    "instantaneous_gain_ilc_step",
    "input_within_safety_limits",
    "linear_ilc_step",
    "model_lm_ilc_step",
    "model_vjp_ilc_step",
    "project_input_safety",
    "real_inner",
    "signal_papr_db",
    "signal_peak",
    "signal_rms",
    "solve_damped_lm_cg",
]
