"""Feedback preprocessing independent of devices and DPD algorithms."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .dsp import align_signal, fractional_shift, rms
from .dsp import nmse_db as calculate_nmse_db


@dataclass(frozen=True, slots=True)
class CaptureBatch:
    """A packed sequence of complete, contiguous periodic IQ segments."""

    iq: np.ndarray = field(repr=False)
    segment_length: int
    segment_count: int
    sample_rate_hz: float
    coherent_within_batch: bool = True

    def __post_init__(self) -> None:
        segment_length = _positive_integer(self.segment_length, "segment_length")
        segment_count = _positive_integer(self.segment_count, "segment_count")
        sample_rate_hz = _positive_real(self.sample_rate_hz, "sample_rate_hz")
        if not isinstance(self.coherent_within_batch, (bool, np.bool_)):
            raise TypeError("coherent_within_batch must be a boolean")

        iq = _finite_complex_vector(self.iq, "iq")
        expected_length = segment_length * segment_count
        if iq.size != expected_length:
            raise ValueError(
                "iq length must equal segment_length * segment_count "
                f"({expected_length}), got {iq.size}"
            )

        object.__setattr__(self, "iq", iq)
        object.__setattr__(self, "segment_length", segment_length)
        object.__setattr__(self, "segment_count", segment_count)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(
            self, "coherent_within_batch", bool(self.coherent_within_batch)
        )

    @property
    def segments(self) -> np.ndarray:
        """Return a read-only `(segment_count, segment_length)` packed view."""
        return self.iq.reshape(self.segment_count, self.segment_length)


@dataclass(frozen=True, slots=True)
class SegmentDiagnostic:
    """Alignment values applied to one feedback segment."""

    segment_index: int
    alignment_estimated: bool
    delay_samples: float
    phase_correction: complex
    phase_radians: float
    input_rms: float
    aligned_rms: float
    aligned_nmse_db: float


@dataclass(frozen=True, slots=True)
class BatchDiagnostic:
    """Structured diagnostics for one capture batch."""

    batch_index: int
    coherent_within_batch: bool
    input_rms: float
    aligned_average_rms: float
    aligned_average_nmse_db: float
    segments: tuple[SegmentDiagnostic, ...]

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def alignment_estimate_count(self) -> int:
        return sum(segment.alignment_estimated for segment in self.segments)

    @property
    def delays_samples(self) -> tuple[float, ...]:
        return tuple(segment.delay_samples for segment in self.segments)

    @property
    def phase_corrections(self) -> tuple[complex, ...]:
        return tuple(segment.phase_correction for segment in self.segments)

    @property
    def phase_radians(self) -> tuple[float, ...]:
        return tuple(segment.phase_radians for segment in self.segments)


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Aligned feedback and diagnostics consumed by a DPD runtime."""

    z: np.ndarray = field(repr=False)
    aligned_average: np.ndarray = field(repr=False)
    gain_correction: float
    gain_correction_db: float
    reference_rms: float
    aligned_average_rms: float
    z_rms: float
    aligned_average_nmse_db: float
    nmse_db: float
    segment_count: int
    batch_diagnostics: tuple[BatchDiagnostic, ...]

    @property
    def delays_samples(self) -> tuple[float, ...]:
        return tuple(
            delay for batch in self.batch_diagnostics for delay in batch.delays_samples
        )

    @property
    def phase_corrections(self) -> tuple[complex, ...]:
        return tuple(
            phase
            for batch in self.batch_diagnostics
            for phase in batch.phase_corrections
        )

    @property
    def phase_radians(self) -> tuple[float, ...]:
        return tuple(
            phase for batch in self.batch_diagnostics for phase in batch.phase_radians
        )

    @property
    def z_nmse_db(self) -> float:
        return self.nmse_db


class FeedbackPreprocessor:
    """Align and coherently average feedback against a fixed reference."""

    _SAMPLE_RATE_REL_TOLERANCE = 1e-12

    def __init__(self, reference: np.ndarray, sample_rate_hz: float) -> None:
        self.reference = _finite_complex_vector(reference, "reference")
        self.sample_rate_hz = _positive_real(sample_rate_hz, "sample_rate_hz")
        self.reference_rms = rms(self.reference)
        if not math.isfinite(self.reference_rms) or self.reference_rms <= 0.0:
            raise ValueError("reference must have non-zero finite RMS")

    def process(
        self,
        batches: Iterable[CaptureBatch],
        gain_correction: float | None = None,
    ) -> PreprocessingResult:
        """Preprocess one round of captures and optionally reuse a fixed gain."""
        try:
            capture_batches = tuple(batches)
        except TypeError as exc:
            raise TypeError(
                "batches must be an iterable of CaptureBatch objects"
            ) from exc
        if not capture_batches:
            raise ValueError("at least one capture batch is required")

        fixed_gain = (
            None
            if gain_correction is None
            else _positive_real(gain_correction, "gain_correction")
        )
        aligned_segments: list[np.ndarray] = []
        diagnostics: list[BatchDiagnostic] = []

        for batch_index, batch in enumerate(capture_batches):
            if not isinstance(batch, CaptureBatch):
                raise TypeError(
                    f"batches[{batch_index}] must be a CaptureBatch, got {type(batch).__name__}"
                )
            self._validate_batch(batch, batch_index)
            batch_aligned, batch_diagnostic = self._process_batch(batch, batch_index)
            aligned_segments.extend(batch_aligned)
            diagnostics.append(batch_diagnostic)

        aligned_average = _readonly_complex_copy(
            np.mean(np.stack(aligned_segments, axis=0), axis=0)
        )
        aligned_average_rms = rms(aligned_average)
        if not math.isfinite(aligned_average_rms):
            raise ValueError("aligned feedback average has non-finite RMS")

        if fixed_gain is None:
            if aligned_average_rms <= 0.0:
                raise ValueError(
                    "cannot calculate gain correction from zero-RMS feedback"
                )
            fixed_gain = self.reference_rms / aligned_average_rms
            if not math.isfinite(fixed_gain) or fixed_gain <= 0.0:
                raise ValueError(
                    "calculated gain correction must be positive and finite"
                )

        with np.errstate(over="ignore", invalid="ignore"):
            z = _readonly_complex_copy(aligned_average * fixed_gain)
        if not np.all(np.isfinite(z)):
            raise ValueError("gain correction produced non-finite feedback")

        return PreprocessingResult(
            z=z,
            aligned_average=aligned_average,
            gain_correction=fixed_gain,
            gain_correction_db=float(20.0 * math.log10(fixed_gain)),
            reference_rms=self.reference_rms,
            aligned_average_rms=aligned_average_rms,
            z_rms=rms(z),
            aligned_average_nmse_db=calculate_nmse_db(self.reference, aligned_average),
            nmse_db=calculate_nmse_db(self.reference, z),
            segment_count=len(aligned_segments),
            batch_diagnostics=tuple(diagnostics),
        )

    def _validate_batch(self, batch: CaptureBatch, batch_index: int) -> None:
        if batch.segment_length != self.reference.size:
            raise ValueError(
                f"batches[{batch_index}].segment_length must equal reference length "
                f"({self.reference.size}), got {batch.segment_length}"
            )
        if not math.isclose(
            batch.sample_rate_hz,
            self.sample_rate_hz,
            rel_tol=self._SAMPLE_RATE_REL_TOLERANCE,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"batches[{batch_index}].sample_rate_hz must equal reference sample rate "
                f"({self.sample_rate_hz}), got {batch.sample_rate_hz}"
            )

    def _process_batch(
        self,
        batch: CaptureBatch,
        batch_index: int,
    ) -> tuple[list[np.ndarray], BatchDiagnostic]:
        aligned_segments: list[np.ndarray] = []
        segment_diagnostics: list[SegmentDiagnostic] = []
        reused_alignment: tuple[float, complex] | None = None

        for segment_index, segment in enumerate(batch.segments):
            should_estimate = not batch.coherent_within_batch or segment_index == 0
            if should_estimate:
                aligned, delay_samples, phase_correction = self._estimate_alignment(
                    segment
                )
                if batch.coherent_within_batch:
                    reused_alignment = delay_samples, phase_correction
            else:
                if (
                    reused_alignment is None
                ):  # pragma: no cover - guarded by iteration order
                    raise RuntimeError("coherent batch alignment was not initialized")
                delay_samples, phase_correction = reused_alignment
                aligned = fractional_shift(segment, delay_samples) * phase_correction

            aligned = _readonly_complex_copy(aligned)
            if not np.all(np.isfinite(aligned)):
                raise ValueError(
                    f"alignment produced non-finite values for batch {batch_index}, "
                    f"segment {segment_index}"
                )
            aligned_segments.append(aligned)
            segment_diagnostics.append(
                SegmentDiagnostic(
                    segment_index=segment_index,
                    alignment_estimated=should_estimate,
                    delay_samples=float(delay_samples),
                    phase_correction=complex(phase_correction),
                    phase_radians=float(np.angle(phase_correction)),
                    input_rms=rms(segment),
                    aligned_rms=rms(aligned),
                    aligned_nmse_db=calculate_nmse_db(self.reference, aligned),
                )
            )

        batch_average = np.mean(np.stack(aligned_segments, axis=0), axis=0)
        diagnostic = BatchDiagnostic(
            batch_index=batch_index,
            coherent_within_batch=batch.coherent_within_batch,
            input_rms=rms(batch.iq),
            aligned_average_rms=rms(batch_average),
            aligned_average_nmse_db=calculate_nmse_db(self.reference, batch_average),
            segments=tuple(segment_diagnostics),
        )
        return aligned_segments, diagnostic

    def _estimate_alignment(
        self, segment: np.ndarray
    ) -> tuple[np.ndarray, float, complex]:
        delay_aligned, delay_samples, _ = align_signal(
            self.reference,
            segment,
            gain_phase=False,
        )
        cross_correlation = np.vdot(delay_aligned, self.reference)
        if not np.isfinite(cross_correlation):
            raise ValueError("phase estimation produced a non-finite correlation")
        if abs(cross_correlation) > 0.0:
            phase_correction = complex(cross_correlation / abs(cross_correlation))
        else:
            phase_correction = 1.0 + 0.0j
        return delay_aligned * phase_correction, float(delay_samples), phase_correction


def _finite_complex_vector(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain numeric IQ samples")
    result = _readonly_complex_copy(array)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite samples")
    return result


def _readonly_complex_copy(value: object) -> np.ndarray:
    copied = np.array(value, dtype=np.complex128, order="C", copy=True).reshape(-1)
    return np.frombuffer(copied.tobytes(), dtype=np.complex128)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result
