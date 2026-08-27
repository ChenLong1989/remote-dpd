"""Non-mutating digital waveform safety checks for the TX boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

PEAK_LIMIT = 1.0
RMS_GROWTH_LIMIT_DB = 2.0
RMS_GROWTH_FACTOR = 10.0 ** (RMS_GROWTH_LIMIT_DB / 20.0)


@dataclass(frozen=True, slots=True)
class DigitalSafetyReport:
    """Serializable measurements and violations from one safety decision."""

    signal_role: str
    passed: bool
    reference_samples: int
    reference_peak: float | None
    reference_rms: float | None
    candidate_samples: int | None
    candidate_peak: float | None
    candidate_rms: float | None
    peak_limit: float
    candidate_rms_limit: float | None
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible representation."""
        return asdict(self)


class DigitalSafetyError(ValueError):
    """Raised when a waveform must not be sent to the transmitter."""

    def __init__(self, report: DigitalSafetyReport) -> None:
        self.report = report
        details = ", ".join(report.violations) or "unknown violation"
        super().__init__(
            f"{report.signal_role} failed digital safety checks: {details}"
        )


@dataclass(frozen=True, slots=True)
class _SignalStats:
    samples: int
    peak: float | None
    rms: float | None
    array: np.ndarray | None
    violations: tuple[str, ...]


def check_reference(reference: Any) -> DigitalSafetyReport:
    """Evaluate a reference waveform without scaling or clipping it."""
    stats = _inspect_signal(reference, "reference")
    violations = list(stats.violations)
    if not stats.violations and stats.peak is not None and stats.peak > PEAK_LIMIT:
        violations.append("reference_peak_exceeded")
    return DigitalSafetyReport(
        signal_role="reference",
        passed=not violations,
        reference_samples=stats.samples,
        reference_peak=stats.peak,
        reference_rms=stats.rms,
        candidate_samples=None,
        candidate_peak=None,
        candidate_rms=None,
        peak_limit=PEAK_LIMIT,
        candidate_rms_limit=None,
        violations=tuple(violations),
    )


def validate_reference(reference: Any) -> DigitalSafetyReport:
    """Return a safe reference report or raise ``DigitalSafetyError``."""
    report = check_reference(reference)
    if not report.passed:
        raise DigitalSafetyError(report)
    return report


def check_candidate(reference: Any, candidate: Any) -> DigitalSafetyReport:
    """Evaluate a candidate against fixed peak and reference-relative limits."""
    reference_stats = _inspect_signal(reference, "reference")
    candidate_stats = _inspect_signal(candidate, "candidate")
    violations = list(reference_stats.violations)
    violations.extend(candidate_stats.violations)

    if (
        not reference_stats.violations
        and reference_stats.peak is not None
        and reference_stats.peak > PEAK_LIMIT
    ):
        violations.append("reference_peak_exceeded")
    if (
        not candidate_stats.violations
        and candidate_stats.peak is not None
        and candidate_stats.peak > PEAK_LIMIT
    ):
        violations.append("candidate_peak_exceeded")

    if (
        reference_stats.array is not None
        and candidate_stats.array is not None
        and reference_stats.array.size != candidate_stats.array.size
    ):
        violations.append("length_mismatch")

    rms_limit: float | None = None
    if not reference_stats.violations and reference_stats.rms is not None:
        rms_limit = reference_stats.rms * RMS_GROWTH_FACTOR
        if (
            not candidate_stats.violations
            and candidate_stats.rms is not None
            and candidate_stats.rms > rms_limit
        ):
            violations.append("candidate_rms_exceeded")

    return DigitalSafetyReport(
        signal_role="candidate",
        passed=not violations,
        reference_samples=reference_stats.samples,
        reference_peak=reference_stats.peak,
        reference_rms=reference_stats.rms,
        candidate_samples=candidate_stats.samples,
        candidate_peak=candidate_stats.peak,
        candidate_rms=candidate_stats.rms,
        peak_limit=PEAK_LIMIT,
        candidate_rms_limit=rms_limit,
        violations=tuple(violations),
    )


def validate_candidate(reference: Any, candidate: Any) -> DigitalSafetyReport:
    """Return a safe candidate report or raise ``DigitalSafetyError``."""
    report = check_candidate(reference, candidate)
    if not report.passed:
        raise DigitalSafetyError(report)
    return report


def _inspect_signal(value: Any, role: str) -> _SignalStats:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return _invalid_stats(f"{role}_non_numeric")

    samples = int(raw.size)
    if raw.ndim != 1:
        return _invalid_stats(f"{role}_not_one_dimensional", samples=samples)
    if raw.size == 0:
        return _invalid_stats(f"{role}_empty")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        return _invalid_stats(f"{role}_non_numeric", samples=samples)

    with np.errstate(over="ignore", invalid="ignore"):
        signal = np.asarray(raw, dtype=np.complex128)
        magnitude = np.abs(signal)
        peak = float(np.max(magnitude))
        rms = _stable_rms(magnitude)
    violations: tuple[str, ...] = ()
    if not np.all(np.isfinite(signal)):
        violations = (f"{role}_non_finite",)
        peak = None
        rms = None
    return _SignalStats(
        samples=samples,
        peak=peak,
        rms=rms,
        array=signal,
        violations=violations,
    )


def _invalid_stats(violation: str, *, samples: int = 0) -> _SignalStats:
    return _SignalStats(
        samples=samples,
        peak=None,
        rms=None,
        array=None,
        violations=(violation,),
    )


def _stable_rms(magnitude: np.ndarray) -> float:
    scale = float(np.max(magnitude))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return scale
    return float(scale * np.sqrt(np.mean((magnitude / scale) ** 2)))
