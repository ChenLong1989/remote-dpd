"""MAT-compatible EVM/NMSE metrics without NR MATLAB toolbox code."""

from __future__ import annotations

import numpy as np

from .dsp import nmse_db


_RB_TABLE = {
    15_000: {5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 40: 216, 50: 270},
    30_000: {5: 11, 10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 40: 106, 50: 133, 60: 162, 70: 189, 80: 217, 90: 245, 100: 273},
    60_000: {10: 11, 15: 18, 20: 24, 25: 31, 30: 38, 40: 51, 50: 65, 60: 79, 70: 93, 80: 107, 90: 121, 100: 135},
}


def nr_parameters(bandwidth_hz: float, sample_rate_hz: float) -> dict[str, float]:
    bandwidth_mhz = bandwidth_hz / 1e6
    scs = 15_000.0 if bandwidth_hz == 20e6 else 30_000.0
    table = _RB_TABLE.get(int(scs), _RB_TABLE[30_000])
    bandwidth_key = min(table, key=lambda item: abs(item - bandwidth_mhz))
    rb = table[bandwidth_key]
    mu = scs / 15_000.0
    nfft = (2048.0 / mu) * (2 ** np.ceil(np.log2(rb * 12) - np.log2(2048.0 / mu)))
    nfft = max(float(nfft), 1.0)
    osf = sample_rate_hz / (2048.0 * mu * scs)
    return {"scs": scs, "mu": mu, "nfft": nfft, "osf": osf, "ncp0": (144.0 / mu + 16.0), "ncp": 144.0 / mu}


def symbol_evm(
    measured: np.ndarray,
    reference: np.ndarray,
    *,
    bandwidth_hz: float,
    sample_rate_hz: float,
    equalizer: np.ndarray | None = None,
) -> np.ndarray:
    """Return one EVM percentage per complete OFDM symbol.

    This mirrors the legacy function's signal-domain NMSE conversion. It is
    intentionally tolerant: short or non-NR captures return one global value.
    """
    measured = np.asarray(measured, dtype=np.complex128).reshape(-1)
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    length = min(measured.size, reference.size)
    measured, reference = measured[:length], reference[:length]
    if length == 0:
        return np.asarray([], dtype=np.float64)
    measured = measured / max(np.sqrt(np.mean(np.abs(measured) ** 2)), np.finfo(float).eps)
    reference = reference / max(np.sqrt(np.mean(np.abs(reference) ** 2)), np.finfo(float).eps)
    params = nr_parameters(bandwidth_hz, sample_rate_hz)
    nfft = int(round(params["nfft"] * params["osf"]))
    nfft = max(nfft, 16)
    cp0 = max(int(round(params["ncp0"] * params["osf"])), 0)
    cp = max(int(round(params["ncp"] * params["osf"])), 0)
    symbol_lengths = [nfft + cp0] + [nfft + cp] * 13
    if length < min(symbol_lengths):
        return np.asarray([10.0 ** (nmse_db(reference, measured) / 20.0) * 100.0])
    values: list[float] = []
    position = 0
    symbol = 0
    while position < length:
        current_cp = cp0 if symbol % 14 == 0 else cp
        start = position + current_cp
        stop = start + nfft
        if stop > length:
            break
        measured_symbol = measured[start:stop]
        reference_symbol = reference[start:stop]
        spectrum = np.fft.fft(measured_symbol)
        if equalizer is not None:
            eq = np.asarray(equalizer).reshape(-1)
            if eq.size == spectrum.size:
                measured_symbol = np.fft.ifft(spectrum * eq)
        error = np.linalg.norm(measured_symbol - reference_symbol)
        denominator = max(np.linalg.norm(measured_symbol), np.finfo(float).eps)
        values.append(float(10.0 ** (20.0 * np.log10(error / denominator) / 20.0) * 100.0))
        position = stop
        symbol += 1
    return np.asarray(values or [10.0 ** (nmse_db(reference, measured) / 20.0) * 100.0], dtype=np.float64)
