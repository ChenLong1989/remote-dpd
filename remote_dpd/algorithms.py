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
from .dsp import align_and_average, circular_fir, nmse_db, resample_signal, rms
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
    device: str = "auto"
    dtype: str = "complex64"
    extra: dict[str, Any] = field(default_factory=dict)

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
            extra=config.extra,
        )


@dataclass(slots=True)
class ILCResult:
    output: np.ndarray
    aligned_feedback: np.ndarray
    metrics: dict[str, Any]


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

    def __init__(self, config: ILCConfig) -> None:
        self.config = config

    def process(
        self,
        reference: np.ndarray,
        current_output: np.ndarray | None,
        feedback: np.ndarray,
        state: SessionState,
    ) -> ILCResult:
        if torch is None:
            raise RuntimeError("PyTorch is required for the ILC engine; install project dependencies")
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
        output = _to_torch(current_work, self.config).clone()
        ref_tensor = _to_torch(reference_dpd, self.config)
        fb_tensor = _to_torch(feedback, self.config)
        error = fb_tensor - ref_tensor

        if self.config.phase_compensate:
            phase_reference = output
            threshold = max(float(self.config.phase_threshold) * _torch_rms(ref_tensor), torch.finfo(ref_tensor.real.dtype).eps)
            magnitude_ref = torch.abs(phase_reference)
            magnitude_fb = torch.abs(fb_tensor)
            weight = (magnitude_ref.square() / (magnitude_ref.square() + threshold**2))
            weight = weight * (magnitude_fb.square() / (magnitude_fb.square() + threshold**2))
            phase = torch.ones_like(error)
            safe = (magnitude_ref > max(float(threshold) * 0.1, torch.finfo(ref_tensor.real.dtype).eps))
            phase[safe] = torch.conj(torch.sign(fb_tensor[safe] / phase_reference[safe]))
            error = error * (weight * phase + (1.0 - weight))

        error = _apply_fir_torch(error, self.config.error_fir)
        next_output = (
            (10.0 ** (self.config.gain_db / 20.0)) * self.config.alpha * ref_tensor
            + (1.0 - self.config.alpha) * output
            - (10.0 ** (self.config.gain_db / 20.0)) * self.config.mu * error
        )
        next_output = _apply_fir_torch(next_output, self.config.tx_fir)
        result = _to_numpy(next_output)
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
    center = len(taps) // 2
    for index, tap in enumerate(taps):
        result = result + complex(tap) * torch.roll(value, center - index)
    return result


def _resize_vector(value: np.ndarray, length: int) -> np.ndarray:
    if value.size == length:
        return value
    if value.size > length:
        return value[:length]
    output = np.zeros(length, dtype=np.complex128)
    output[:value.size] = value
    return output


_ENGINES: dict[str, type[DPDEngine]] = {ILCAlgorithm.name: ILCAlgorithm}


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
