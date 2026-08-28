"""Small periodic-signal DSP primitives used by preprocessing and simulation."""

from __future__ import annotations

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
    return float(
        10.0
        * np.log10(
            max(float(np.vdot(error, error).real) / denominator, np.finfo(float).tiny)
        )
    )


def _fft_resample(signal: np.ndarray, size: int) -> np.ndarray:
    """Periodic FFT interpolation, equivalent to zero-padding the spectrum."""
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if size == signal.size:
        return signal.copy()
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    output = np.zeros(size, dtype=np.complex128)
    if size >= spectrum.size:
        start = (size - spectrum.size) // 2
        output[start : start + spectrum.size] = spectrum
    else:
        start = (spectrum.size - size) // 2
        output[:] = spectrum[start : start + size]
    return np.fft.ifft(np.fft.ifftshift(output)) * (size / signal.size)


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

    The implementation uses a zero-padded FFT correlation. It returns
    `(aligned, delay, complex_gain)` where delay
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


def _same_length(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first, dtype=np.complex128).reshape(-1)
    second = np.asarray(second, dtype=np.complex128).reshape(-1)
    length = min(first.size, second.size)
    return first[:length], second[:length]
