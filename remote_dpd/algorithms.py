"""Extensible DPD algorithm boundary and the supported ILC implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import numpy as np

try:  # PyTorch is the execution backend; keep import errors actionable.
    import torch
except ImportError:  # pragma: no cover - exercised only in incomplete installs
    torch = None  # type: ignore

from .config import LegacyConfig
from .dsp import align_and_average, align_and_average_detailed, nmse_db, resample_signal, rms
from .learning import (
    InputSafetyLimits,
    LearningStepResult,
    StopReason,
    instantaneous_gain_ilc_step,
    linear_ilc_step,
    model_lm_ilc_step,
    model_vjp_ilc_step,
    signal_rms,
)
from .pa_model import PAForwardModelConfig, PAForwardModelFit, fit_pa_model
from .state import SessionState


@dataclass(slots=True)
class ILCConfig:
    """Numerical settings independent of the MAT/file transport layer."""

    mu: float = 0.5
    alpha: float = 0.0
    gain_db: float = 0.0
    phase_compensate: bool = False
    phase_threshold: float = 0.15
    input_sample_rate_hz: float = 983.04e6
    feedback_sample_rate_hz: float = 983.04e6
    output_sample_rate_hz: float = 983.04e6
    tx_fir: np.ndarray | None = None
    error_fir: np.ndarray | None = None
    backward_mode: str = "legacy"
    calibration_mode: str = "auto"
    calibration_coefficient: complex | None = None
    pa_model_order: int = 9
    pa_model_memory_depth: int = 3
    pa_model_ridge: float = 1e-6
    pa_model_min_validation_nmse_db: float = -20.0
    lm_damping: float = 1e-2
    cg_max_iterations: int = 8
    cg_tolerance: float = 1e-3
    trust_region_ratio: float = 0.25
    max_input_rms: float | None = None
    max_input_peak: float | None = None
    max_input_papr_db: float | None = None
    pa_model_fallback: str = "linear"
    device: str = "auto"
    dtype: str = "complex64"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        finite_fields = {
            "mu": self.mu,
            "alpha": self.alpha,
            "gain_db": self.gain_db,
            "phase_threshold": self.phase_threshold,
            "input_sample_rate_hz": self.input_sample_rate_hz,
            "feedback_sample_rate_hz": self.feedback_sample_rate_hz,
            "output_sample_rate_hz": self.output_sample_rate_hz,
            "pa_model_ridge": self.pa_model_ridge,
            "pa_model_min_validation_nmse_db": self.pa_model_min_validation_nmse_db,
            "lm_damping": self.lm_damping,
            "cg_tolerance": self.cg_tolerance,
            "trust_region_ratio": self.trust_region_ratio,
        }
        for name, value in finite_fields.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.phase_threshold <= 0.0:
            raise ValueError("phase_threshold must be positive")
        for name in (
            "input_sample_rate_hz",
            "feedback_sample_rate_hz",
            "output_sample_rate_hz",
            "lm_damping",
            "cg_tolerance",
            "trust_region_ratio",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.pa_model_ridge < 0.0:
            raise ValueError("pa_model_ridge must be non-negative")
        if self.pa_model_min_validation_nmse_db > 0.0:
            raise ValueError("pa_model_min_validation_nmse_db must be at most 0 dB")
        if self.cg_tolerance >= 1.0:
            raise ValueError("cg_tolerance must be less than one")
        if self.lm_damping < 1e-8:
            raise ValueError("lm_damping must be at least 1e-8")
        if self.trust_region_ratio > 1.0:
            raise ValueError("trust_region_ratio must be at most one")
        if self.backward_mode not in {
            "legacy",
            "linear",
            "instantaneous_gain",
            "model_vjp",
            "model_lm",
        }:
            raise ValueError(f"unsupported backward_mode {self.backward_mode!r}")
        if self.calibration_mode not in {"auto", "legacy_dynamic", "frozen_first", "explicit"}:
            raise ValueError(f"unsupported calibration_mode {self.calibration_mode!r}")
        if self.calibration_coefficient is not None:
            coefficient = complex(self.calibration_coefficient)
            if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag) or coefficient == 0.0:
                raise ValueError("calibration_coefficient must be finite and non-zero")
            self.calibration_coefficient = coefficient
        if self.calibration_mode == "explicit" and self.calibration_coefficient is None:
            raise ValueError("explicit calibration requires calibration_coefficient")
        integer_fields = {
            "pa_model_order": self.pa_model_order,
            "pa_model_memory_depth": self.pa_model_memory_depth,
            "cg_max_iterations": self.cg_max_iterations,
        }
        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.pa_model_order < 1 or self.pa_model_order > 21 or self.pa_model_order % 2 == 0:
            raise ValueError("pa_model_order must be odd and in [1, 21]")
        if self.pa_model_memory_depth < 1 or self.pa_model_memory_depth > 16:
            raise ValueError("pa_model_memory_depth must be in [1, 16]")
        if self.cg_max_iterations < 1:
            raise ValueError("cg_max_iterations must be positive")
        if self.cg_max_iterations > 128:
            raise ValueError("cg_max_iterations must be at most 128")
        if self.pa_model_fallback not in {"linear", "hold"}:
            raise ValueError("pa_model_fallback must be 'linear' or 'hold'")
        for name in ("max_input_rms", "max_input_peak", "max_input_papr_db"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when provided")
        if self.dtype not in {"complex64", "complex128"}:
            raise ValueError("dtype must be 'complex64' or 'complex128'")
        for name in ("tx_fir", "error_fir"):
            taps = getattr(self, name)
            if taps is None:
                continue
            taps = np.asarray(taps, dtype=np.complex128).reshape(-1)
            if not np.all(np.isfinite(taps.real)) or not np.all(np.isfinite(taps.imag)):
                raise ValueError(f"{name} must contain finite taps")
            setattr(self, name, taps)
        if self.backward_mode in {"model_vjp", "model_lm"}:
            _validate_model_mode_compatibility(self)

    @classmethod
    def from_legacy(cls, config: LegacyConfig) -> "ILCConfig":
        return cls(
            mu=config.ilc_mu,
            alpha=config.alpha,
            gain_db=config.dpd_gain_db,
            phase_compensate=config.phase_compensate,
            phase_threshold=config.phase_compensation_threshold,
            input_sample_rate_hz=config.sample_rate_hz,
            feedback_sample_rate_hz=config.feedback_sample_rate_hz,
            output_sample_rate_hz=config.output_sample_rate_hz,
            tx_fir=config.tx_fir,
            error_fir=config.error_fir,
            backward_mode=config.ilc_backward_mode,
            calibration_mode=config.calibration_mode,
            calibration_coefficient=config.calibration_coefficient,
            pa_model_order=config.pa_model_order,
            pa_model_memory_depth=config.pa_model_memory_depth,
            pa_model_ridge=config.pa_model_ridge,
            pa_model_min_validation_nmse_db=config.pa_model_min_validation_nmse_db,
            lm_damping=config.ilc_lm_damping,
            cg_max_iterations=config.ilc_cg_max_iterations,
            cg_tolerance=config.ilc_cg_tolerance,
            trust_region_ratio=config.ilc_trust_region_ratio,
            max_input_rms=config.ilc_max_input_rms,
            max_input_peak=config.ilc_max_input_peak,
            max_input_papr_db=config.ilc_max_input_papr_db,
            pa_model_fallback=config.pa_model_fallback,
            extra=config.extra,
        )


@dataclass(slots=True)
class ILCResult:
    output: np.ndarray
    aligned_feedback: np.ndarray
    metrics: dict[str, Any]


def legacy_ilc_update(
    reference: np.ndarray,
    current_output: np.ndarray,
    aligned_feedback: np.ndarray,
    *,
    mu: float,
    alpha: float = 0.0,
    gain_db: float = 0.0,
    phase_compensate: bool = False,
    phase_threshold: float = 0.15,
    error_fir: np.ndarray | None = None,
    tx_fir: np.ndarray | None = None,
    numeric_dtype: str | np.dtype[Any] | type = "complex128",
) -> np.ndarray:
    """Apply one legacy ILC update using only NumPy.

    ``reference`` is the unscaled reference waveform and ``aligned_feedback``
    is the time-aligned, legacy gain/phase-calibrated capture.  The seemingly
    duplicated gain in the alpha term is intentional: it preserves the
    historical formula in which ``reference_dpd = gain * reference`` and the
    alpha term is ``gain * alpha * reference_dpd``.
    """
    dtype = np.dtype(numeric_dtype)
    if dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise ValueError("numeric_dtype must be complex64 or complex128")

    reference = np.asarray(reference).reshape(-1)
    current_output = np.asarray(current_output, dtype=dtype).reshape(-1)
    aligned_feedback = np.asarray(aligned_feedback, dtype=dtype).reshape(-1)
    if not (reference.size == current_output.size == aligned_feedback.size):
        raise ValueError("legacy ILC vectors must have equal lengths")

    gain = 10.0 ** (float(gain_db) / 20.0)
    # Preserve the production path's historical rounding order: scale the
    # complex128 transport reference before converting to the compute dtype.
    reference_dpd = np.asarray(gain * reference, dtype=dtype)
    error = aligned_feedback - reference_dpd

    if phase_compensate and reference.size:
        real_dtype = reference_dpd.real.dtype
        epsilon = float(np.finfo(real_dtype).eps)
        threshold = max(float(phase_threshold) * rms(reference_dpd), epsilon)
        magnitude_ref = np.abs(current_output)
        magnitude_feedback = np.abs(aligned_feedback)
        magnitude_ref_squared = np.square(magnitude_ref)
        magnitude_feedback_squared = np.square(magnitude_feedback)
        weight = magnitude_ref_squared / (magnitude_ref_squared + threshold**2)
        weight *= magnitude_feedback_squared / (
            magnitude_feedback_squared + threshold**2
        )
        phase = np.ones_like(error)
        safe = magnitude_ref > max(threshold * 0.1, epsilon)
        ratio = aligned_feedback[safe] / current_output[safe]
        ratio_magnitude = np.abs(ratio)
        normalized = np.divide(
            ratio,
            ratio_magnitude,
            out=np.zeros_like(ratio),
            where=ratio_magnitude > 0.0,
        )
        phase[safe] = np.conj(normalized)
        error *= weight * phase + (1.0 - weight)

    error = _apply_fir_numpy(error, error_fir, dtype)
    next_output = (
        gain * float(alpha) * reference_dpd
        + (1.0 - float(alpha)) * current_output
        - gain * float(mu) * error
    )
    return _apply_fir_numpy(next_output, tx_fir, dtype)


class DPDEngine(ABC):
    """Stable extension point for future ILC variants and model-based DPD."""

    name: ClassVar[str]

    @abstractmethod
    def process(
        self,
        reference: np.ndarray,
        current_output: np.ndarray | None,
        feedback: np.ndarray,
        state: SessionState,
    ) -> ILCResult:
        raise NotImplementedError


class ILCAlgorithm(DPDEngine):
    name = "ilc"
    mode_override: ClassVar[str | None] = None

    def __init__(self, config: ILCConfig) -> None:
        self.config = config
        mode = self.mode_override or self.config.backward_mode
        if mode in {"model_vjp", "model_lm"}:
            _validate_model_mode_compatibility(self.config)

    def process(
        self,
        reference: np.ndarray,
        current_output: np.ndarray | None,
        feedback: np.ndarray,
        state: SessionState,
    ) -> ILCResult:
        mode = self.mode_override or self.config.backward_mode
        if mode == "legacy":
            return self._process_legacy(reference, current_output, feedback, state)
        return self._process_strategy(mode, reference, current_output, feedback, state)

    def _process_legacy(
        self,
        reference: np.ndarray,
        current_output: np.ndarray | None,
        feedback: np.ndarray,
        state: SessionState,
    ) -> ILCResult:
        reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
        current_output = reference if current_output is None else np.asarray(current_output, dtype=np.complex128).reshape(-1)
        # The legacy transport packs ten 32768-sample captures into one
        # 327680-sample vector. MATLAB trains on the first capture and repeats
        # the resulting correction for the ten output blocks.
        packed_capture = feedback.size == 327680 and reference.size >= 32768
        work_length = 32768 if packed_capture else reference.size
        reference_work = reference[:work_length]
        current_work = current_output[:work_length]
        if current_work.size != reference_work.size:
            current_work = _resize_vector(current_work, reference_work.size)

        feedback = resample_signal(
            np.asarray(feedback, dtype=np.complex128).reshape(-1),
            self.config.input_sample_rate_hz / self.config.feedback_sample_rate_hz,
        )
        feedback, delays, gains = align_and_average(reference_work, feedback)
        feedback = _resize_vector(feedback, reference_work.size)
        reference_dpd = reference_work * (10.0 ** (self.config.gain_db / 20.0))
        result = np.asarray(
            legacy_ilc_update(
                reference_work,
                current_work,
                feedback,
                mu=self.config.mu,
                alpha=self.config.alpha,
                gain_db=self.config.gain_db,
                phase_compensate=self.config.phase_compensate,
                phase_threshold=self.config.phase_threshold,
                error_fir=self.config.error_fir,
                tx_fir=self.config.tx_fir,
                numeric_dtype=self.config.dtype,
            ),
            dtype=np.complex128,
        )
        if packed_capture:
            result = np.tile(result, 10)
        metrics = {
            "iteration": state.iteration,
            "aligned_nmse_db": nmse_db(reference_dpd, feedback),
            "feedback_gain_correction_db": 20.0 * np.log10(max(float(np.mean(gains)), np.finfo(float).tiny)),
            "alignment_delays": delays,
            "capture_count": len(delays),
            "feedback_rms": rms(feedback),
            "output_rms": rms(result),
        }
        return ILCResult(output=result, aligned_feedback=feedback, metrics=metrics)

    def _process_strategy(
        self,
        mode: str,
        reference: np.ndarray,
        current_output: np.ndarray | None,
        feedback: np.ndarray,
        state: SessionState,
    ) -> ILCResult:
        reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
        if reference.size == 0:
            raise ValueError("reference must not be empty")
        gain = 10.0 ** (self.config.gain_db / 20.0)
        desired_full = gain * reference
        current = desired_full if current_output is None else np.asarray(current_output, dtype=np.complex128).reshape(-1)
        packed_capture = np.asarray(feedback).size == 327680 and reference.size >= 32768
        work_length = 32768 if packed_capture else reference.size
        reference_work = reference[:work_length]
        desired = desired_full[:work_length]
        current_work = _resize_vector(current[:work_length], work_length)
        measured_raw = resample_signal(
            np.asarray(feedback, dtype=np.complex128).reshape(-1),
            self.config.input_sample_rate_hz / self.config.feedback_sample_rate_hz,
        )
        alignment, calibration_mode = self._align_strategy_feedback(
            desired,
            measured_raw,
            state,
        )
        measured = _resize_vector(alignment.signal, work_length)
        safety_limits = InputSafetyLimits(
            max_rms=self.config.max_input_rms,
            max_peak=self.config.max_input_peak,
            max_papr_db=self.config.max_input_papr_db,
        )

        model_fit: PAForwardModelFit | None = None
        if mode == "linear":
            step = linear_ilc_step(
                current_work,
                desired,
                measured,
                self.config.mu,
                safety_limits=safety_limits,
            )
        elif mode == "instantaneous_gain":
            step = instantaneous_gain_ilc_step(
                current_work,
                desired,
                measured,
                self.config.mu,
                damping=self.config.lm_damping,
                safety_limits=safety_limits,
            )
        elif mode in {"model_vjp", "model_lm"}:
            model_fit = fit_pa_model(
                current_work,
                measured,
                PAForwardModelConfig(
                    orders=tuple(range(1, self.config.pa_model_order + 1, 2)),
                    memory_depth=self.config.pa_model_memory_depth,
                    ridge=self.config.pa_model_ridge,
                    max_validation_nmse_db=self.config.pa_model_min_validation_nmse_db,
                ),
            )
            if not model_fit.succeeded:
                step = self._model_fallback(current_work, desired, measured, safety_limits)
            else:
                assert model_fit.model is not None
                if mode == "model_vjp":
                    step = model_vjp_ilc_step(
                        current_work,
                        desired,
                        measured,
                        model_fit.model,
                        self.config.mu,
                        safety_limits=safety_limits,
                    )
                else:
                    step = model_lm_ilc_step(
                        current_work,
                        desired,
                        measured,
                        model_fit.model,
                        damping=max(self.config.lm_damping, 1e-8),
                        step_size=self.config.mu,
                        cg_max_iterations=self.config.cg_max_iterations,
                        cg_relative_tolerance=self.config.cg_tolerance,
                        trust_region_ratio=self.config.trust_region_ratio,
                        safety_limits=safety_limits,
                    )
        else:  # Defensive: config validation and engine classes make this unreachable.
            raise ValueError(f"unsupported ILC mode {mode!r}")

        result = np.asarray(step.next_input, dtype=np.complex128).reshape(-1)
        if packed_capture:
            result = np.tile(result, 10)
        coefficient = _mean_complex(alignment.coefficients)
        metrics: dict[str, Any] = {
            "iteration": state.iteration,
            "algorithm": _mode_engine_name(mode),
            "backward_mode": mode,
            "calibration_mode": calibration_mode,
            "calibration_coefficient_real": float(coefficient.real),
            "calibration_coefficient_imag": float(coefficient.imag),
            "aligned_nmse_db": nmse_db(desired, measured),
            "feedback_gain_correction_db": 20.0
            * np.log10(max(abs(coefficient), np.finfo(float).tiny)),
            "alignment_delays": alignment.delays,
            "capture_count": len(alignment.delays),
            "feedback_rms": rms(measured),
            "output_rms": rms(result),
            "gradient_rms": float(step.diagnostics.get("gradient_rms", float("nan"))),
            "update_rms": float(step.diagnostics.get("update_rms", signal_rms(step.update))),
            "gradient_cosine_oracle": float("nan"),
            "lm_damping": self.config.lm_damping if mode == "model_lm" else float("nan"),
            "cg_iterations": step.cg_result.iterations if step.cg_result is not None else 0,
            "cg_relative_residual": (
                step.cg_result.relative_residual if step.cg_result is not None else float("nan")
            ),
            "trust_region_active": step.trust_region_active,
            "input_projection_active": step.input_projection_active,
            "saturation_limited": step.saturation_limited,
            "stop_reason": step.stop_reason,
            "update_accepted": step.accepted,
            "backtracks": step.backtracks,
        }
        metrics.update(dict(step.diagnostics))
        if model_fit is not None:
            metrics.update(model_fit.diagnostics.as_metrics())
            if not model_fit.succeeded:
                metrics["stop_reason"] = f"model_fallback_{step.stop_reason}"
        else:
            metrics.update(_empty_model_metrics())
        return ILCResult(output=result, aligned_feedback=measured, metrics=metrics)

    def _align_strategy_feedback(
        self,
        desired: np.ndarray,
        feedback: np.ndarray,
        state: SessionState,
    ):
        calibration_mode = self.config.calibration_mode
        if calibration_mode == "auto":
            calibration_mode = "frozen_first"
        if calibration_mode == "legacy_dynamic":
            return align_and_average_detailed(desired, feedback), calibration_mode
        if calibration_mode == "explicit":
            assert self.config.calibration_coefficient is not None
            state.feedback_calibration = self.config.calibration_coefficient
            return (
                align_and_average_detailed(
                    desired,
                    feedback,
                    calibration=self.config.calibration_coefficient,
                ),
                calibration_mode,
            )
        if state.feedback_calibration is None:
            initial = align_and_average_detailed(desired, feedback)
            state.feedback_calibration = _mean_complex(initial.coefficients)
            return initial, calibration_mode
        return (
            align_and_average_detailed(
                desired,
                feedback,
                calibration=state.feedback_calibration,
            ),
            calibration_mode,
        )

    def _model_fallback(
        self,
        current: np.ndarray,
        desired: np.ndarray,
        measured: np.ndarray,
        safety_limits: InputSafetyLimits,
    ) -> LearningStepResult:
        if self.config.pa_model_fallback == "linear":
            return linear_ilc_step(
                current,
                desired,
                measured,
                self.config.mu,
                safety_limits=safety_limits,
            )
        return LearningStepResult(
            next_input=current.copy(),
            update=np.zeros_like(current),
            accepted=False,
            stop_reason=StopReason.MODEL_FAILURE,
            diagnostics={"update_rms": 0.0},
        )


class LegacyILCAlgorithm(ILCAlgorithm):
    name = "legacy_ilc"
    mode_override = "legacy"


class LinearILCAlgorithm(ILCAlgorithm):
    name = "linear_ilc"
    mode_override = "linear"


class InstantaneousGainILCAlgorithm(ILCAlgorithm):
    name = "instantaneous_gain_ilc"
    mode_override = "instantaneous_gain"


class ModelVJPILCAlgorithm(ILCAlgorithm):
    name = "model_vjp_ilc"
    mode_override = "model_vjp"


class ModelLMILCAlgorithm(ILCAlgorithm):
    name = "model_lm_ilc"
    mode_override = "model_lm"


def _to_torch(array: np.ndarray, config: ILCConfig):
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.complex64 if config.dtype == "complex64" else torch.complex128
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.complex128).reshape(-1)


def _torch_rms(value: Any) -> Any:
    return torch.sqrt(torch.mean(torch.abs(value) ** 2))


def _apply_fir_torch(value: Any, taps: np.ndarray | None) -> Any:
    if taps is None or np.asarray(taps).size <= 1:
        return value
    # The legacy FIR is centered/circular. Roll-based convolution avoids an
    # implicit zero-padding edge that would alter the iterative correction.
    taps = np.asarray(taps, dtype=np.complex128).reshape(-1)
    result = torch.zeros_like(value)
    center = (len(taps) - 1) // 2
    for index, tap in enumerate(taps):
        result = result + complex(tap) * torch.roll(value, index - center)
    return result


def _apply_fir_numpy(
    value: np.ndarray,
    taps: np.ndarray | None,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Apply the legacy centered/circular FIR while preserving numeric dtype."""
    value = np.asarray(value, dtype=dtype).reshape(-1)
    if taps is None or np.asarray(taps).size <= 1:
        return value.copy()
    taps = np.asarray(taps, dtype=dtype).reshape(-1)
    result = np.zeros_like(value)
    center = (len(taps) - 1) // 2
    for index, tap in enumerate(taps):
        result += tap * np.roll(value, index - center)
    return result


def _resize_vector(value: np.ndarray, length: int) -> np.ndarray:
    if value.size == length:
        return value
    if value.size > length:
        return value[:length]
    output = np.zeros(length, dtype=np.complex128)
    output[:value.size] = value
    return output


def _validate_model_mode_compatibility(config: ILCConfig) -> None:
    conflicts: list[str] = []
    if config.phase_compensate:
        conflicts.append("phase_compensate")
    if config.alpha != 0.0:
        conflicts.append("alpha")
    if config.tx_fir is not None:
        conflicts.append("tx_fir")
    if config.error_fir is not None:
        conflicts.append("error_fir")
    if conflicts:
        raise ValueError(
            "model-based ILC cannot be combined with legacy preconditioners: "
            + ", ".join(conflicts)
        )


def _mean_complex(values: list[complex]) -> complex:
    if not values:
        return 1.0 + 0.0j
    value = complex(np.mean(np.asarray(values, dtype=np.complex128)))
    if not np.isfinite(value.real) or not np.isfinite(value.imag) or abs(value) <= np.finfo(float).eps:
        return 1.0 + 0.0j
    return value


def _mode_engine_name(mode: str) -> str:
    return {
        "legacy": "legacy_ilc",
        "linear": "linear_ilc",
        "instantaneous_gain": "instantaneous_gain_ilc",
        "model_vjp": "model_vjp_ilc",
        "model_lm": "model_lm_ilc",
    }[mode]


def _empty_model_metrics() -> dict[str, Any]:
    return {
        "pa_model_train_nmse_db": float("nan"),
        "pa_model_validation_nmse_db": float("nan"),
        "pa_model_rank": 0,
        "pa_model_condition": float("nan"),
        "pa_model_fallback_reason": None,
    }


_ENGINE_TYPES = (
    ILCAlgorithm,
    LegacyILCAlgorithm,
    LinearILCAlgorithm,
    InstantaneousGainILCAlgorithm,
    ModelVJPILCAlgorithm,
    ModelLMILCAlgorithm,
)
_ENGINES: dict[str, type[DPDEngine]] = {engine.name: engine for engine in _ENGINE_TYPES}


def register_engine(name: str, engine_type: type[DPDEngine]) -> None:
    if not name or not issubclass(engine_type, DPDEngine):
        raise TypeError("engine must be a DPDEngine subclass with a non-empty name")
    _ENGINES[name] = engine_type


def create_engine(name: str, config: ILCConfig) -> DPDEngine:
    try:
        engine_type = _ENGINES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported DPD engine {name!r}; available: {sorted(_ENGINES)}") from exc
    return engine_type(config)
