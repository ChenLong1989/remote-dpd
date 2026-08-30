"""Small periodic-signal DSP primitives used by preprocessing and simulation."""

from __future__ import annotations

import numpy as np

_ALIGNMENT_FACTOR = 32


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal)
    value = np.sqrt(np.mean(np.abs(signal) ** 2))
    return float(value)


def nmse_db(reference: np.ndarray, measured: np.ndarray) -> float:
    reference, measured = _same_length(reference, measured)
    denominator = float(np.vdot(reference, reference).real)
    if denominator <= np.finfo(float).eps:
        return float("nan")
    error = measured - reference
    return float(
        10.0
        * np.log10(
            max(float(np.vdot(error, error).real) / denominator, np.finfo(float).tiny)
        )
    )


def fractional_shift(signal: np.ndarray, shift_samples: float) -> np.ndarray:
    """Circularly shift a complex waveform by a fractional number of samples."""
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if not signal.size or abs(shift_samples) < 1e-12:
        return signal.copy()
    bins = np.fft.fftfreq(signal.size)
    spectrum = np.fft.fft(signal)
    return np.fft.ifft(spectrum * np.exp(-2j * np.pi * bins * shift_samples))


def align_signal(
    reference: np.ndarray, measured: np.ndarray, *, gain_phase: bool = True
) -> tuple[np.ndarray, float, complex]:
    """Align measured to reference with 1/32-sample delay resolution.

    The implementation evaluates the complete interpolated periodic
    correlation as 32 original-length IFFTs. It returns `(aligned, delay,
    complex_gain)` where delay is the signed shift applied to measured before
    gain/phase correction.
    """
    reference, measured = _same_length(reference, measured)
    if reference.size == 0:
        return measured, 0.0, 1.0 + 0j
    delay = _periodic_delay(reference, measured, factor=_ALIGNMENT_FACTOR)
    aligned = fractional_shift(measured, delay)
    coefficient = 1.0 + 0j
    if gain_phase:
        cross = np.vdot(aligned, reference)
        if abs(cross) > np.finfo(float).eps:
            phase = cross / abs(cross)
        else:
            phase = 1.0 + 0j
        input_rms = rms(aligned)
        reference_rms = rms(reference)
        if input_rms > np.finfo(float).eps:
            coefficient = phase * reference_rms / input_rms
            aligned = aligned * coefficient
    return np.asarray(aligned, dtype=np.complex128), delay, complex(coefficient)


def _periodic_delay(
    reference: np.ndarray,
    measured: np.ndarray,
    *,
    factor: int,
) -> float:
    """Return the global periodic-correlation peak on a fractional grid."""
    sample_count = reference.size
    cross_spectrum = np.fft.fft(reference) * np.conj(np.fft.fft(measured))
    fractional_phase = np.ones(sample_count, dtype=np.complex128)
    phase_step = np.exp(2j * np.pi * np.fft.fftfreq(sample_count) / factor)
    best_magnitude = -1.0
    best_high_resolution_index = 0

    for fractional_offset in range(factor):
        correlation = np.fft.ifft(cross_spectrum * fractional_phase)
        integer_index = int(np.argmax(np.abs(correlation)))
        magnitude = float(abs(correlation[integer_index]))
        high_resolution_index = factor * integer_index + fractional_offset
        if magnitude > best_magnitude or (
            magnitude == best_magnitude
            and high_resolution_index < best_high_resolution_index
        ):
            best_magnitude = magnitude
            best_high_resolution_index = high_resolution_index
        fractional_phase *= phase_step

    high_resolution_size = sample_count * factor
    signed_index = (
        best_high_resolution_index
        if best_high_resolution_index < high_resolution_size / 2
        else best_high_resolution_index - high_resolution_size
    )
    return float(signed_index) / factor


def _same_length(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first, dtype=np.complex128).reshape(-1)
    second = np.asarray(second, dtype=np.complex128).reshape(-1)
    length = min(first.size, second.size)
    return first[:length], second[:length]
