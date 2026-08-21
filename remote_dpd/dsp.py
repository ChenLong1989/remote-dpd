"""Small, testable DSP primitives used by the ILC engine."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable

import numpy as np


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
    return float(10.0 * np.log10(max(float(np.vdot(error, error).real) / denominator, np.finfo(float).tiny)))


def circular_fir(signal: np.ndarray, taps: np.ndarray | None) -> np.ndarray:
    """Apply the centered, circular FIR convention used by MATLAB `filterC`."""
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if taps is None or np.asarray(taps).size <= 1:
        return signal.copy()
    taps = np.asarray(taps, dtype=np.complex128).reshape(-1)
    length = signal.size
    if length == 0:
        return signal.copy()
    padded = np.concatenate((signal[-(len(taps) // 2 + 1):], signal, signal[: int(np.ceil(len(taps) / 2))]))
    convolved = np.convolve(padded, taps, mode="full")
    return np.asarray(convolved[len(taps):len(taps) + length], dtype=np.complex128)


def resample_signal(signal: np.ndarray, ratio: float, *, taps: int = 23) -> np.ndarray:
    """Resample with a polyphase filter; ratio 1 is an exact no-op."""
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"sample-rate ratio must be positive, got {ratio}")
    if abs(ratio - 1.0) < 1e-12:
        return signal.copy()
    from scipy.signal import resample_poly

    fraction = Fraction(float(ratio)).limit_denominator(4096)
    return np.asarray(resample_poly(signal, fraction.numerator, fraction.denominator), dtype=np.complex128)


def _fft_resample(signal: np.ndarray, size: int) -> np.ndarray:
    """Periodic FFT interpolation, equivalent to zero-padding the spectrum."""
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if size == signal.size:
        return signal.copy()
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    output = np.zeros(size, dtype=np.complex128)
    if size >= spectrum.size:
        start = (size - spectrum.size) // 2
        output[start:start + spectrum.size] = spectrum
    else:
        start = (spectrum.size - size) // 2
        output[:] = spectrum[start:start + size]
    return np.fft.ifft(np.fft.ifftshift(output)) * (size / signal.size)


def fractional_shift(signal: np.ndarray, shift_samples: float) -> np.ndarray:
    """Circularly shift a complex waveform by a fractional number of samples."""
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if not signal.size or abs(shift_samples) < 1e-12:
        return signal.copy()
    bins = np.fft.fftfreq(signal.size)
    spectrum = np.fft.fft(signal)
    return np.fft.ifft(spectrum * np.exp(-2j * np.pi * bins * shift_samples))


def align_signal(reference: np.ndarray, measured: np.ndarray, *, gain_phase: bool = True) -> tuple[np.ndarray, float, complex]:
    """Align measured to reference with 1/32-sample delay resolution.

    The implementation uses a zero-padded FFT correlation rather than the old
    MATLAB nested loop. It returns `(aligned, delay, complex_gain)` where delay
    is the signed shift applied to measured before gain/phase correction.
    """
    reference, measured = _same_length(reference, measured)
    if reference.size == 0:
        return measured, 0.0, 1.0 + 0j
    n = reference.size
    factor = 32
    ref_up = _fft_resample(reference, n * factor)
    measured_up = _fft_resample(measured, n * factor)
    correlation = np.fft.ifft(np.fft.fft(ref_up) * np.conj(np.fft.fft(measured_up)))
    peak = int(np.argmax(np.abs(correlation)))
    signed_peak = peak if peak < n * factor / 2 else peak - n * factor
    delay = float(signed_peak) / factor
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


def align_and_average(reference: np.ndarray, feedback: np.ndarray) -> tuple[np.ndarray, list[float], list[float]]:
    """Align one capture or the legacy ten-capture packed feedback."""
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    feedback = np.asarray(feedback, dtype=np.complex128).reshape(-1)
    if feedback.size == 10 * reference.size and reference.size > 0:
        captures = feedback.reshape((reference.size, 10), order="F").T
    else:
        captures = feedback.reshape(1, -1)
    aligned = []
    delays = []
    gains = []
    for capture in captures:
        current_ref, current_capture = _same_length(reference, capture)
        result, delay, gain = align_signal(current_ref, current_capture, gain_phase=True)
        aligned.append(result)
        delays.append(delay)
        gains.append(abs(gain))
    return np.mean(np.stack(aligned, axis=0), axis=0), delays, gains


def _same_length(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first, dtype=np.complex128).reshape(-1)
    second = np.asarray(second, dtype=np.complex128).reshape(-1)
    length = min(first.size, second.size)
    return first[:length], second[:length]
