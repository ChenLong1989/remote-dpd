"""Fixed-domain metrics for the reproducible PA/ILC experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _complex_vector(values: ArrayLike, *, name: str) -> NDArray[np.complex128]:
    vector = np.asarray(values, dtype=np.complex128).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def fixed_domain_nmse(reference: ArrayLike, measured: ArrayLike) -> float:
    """Return tracking NMSE as a linear power ratio without re-alignment.

    This metric intentionally performs no RMS normalization, gain fitting,
    phase rotation, or delay search.  Both inputs must already be in the frozen
    physical calibration domain selected by the experiment runner.
    """

    desired = _complex_vector(reference, name="reference")
    observed = _complex_vector(measured, name="measured")
    if observed.shape != desired.shape:
        raise ValueError("reference and measured must have the same size")
    denominator = float(np.vdot(desired, desired).real)
    if denominator <= 0.0:
        raise ValueError("reference must have positive energy")
    error = observed - desired
    numerator = float(np.vdot(error, error).real)
    return numerator / denominator


def fixed_domain_nmse_db(reference: ArrayLike, measured: ArrayLike) -> float:
    """Return fixed-domain tracking NMSE in dB."""

    ratio = fixed_domain_nmse(reference, measured)
    if ratio == 0.0:
        return -np.inf
    return float(10.0 * np.log10(ratio))


# Short names are convenient in analysis scripts while retaining the explicit
# fixed-domain names at publication/reporting boundaries.
nmse = fixed_domain_nmse
nmse_db = fixed_domain_nmse_db


def auec(nmse_db_values: ArrayLike) -> float:
    """Return the preregistered area under the error convergence curve.

    For evaluations ``k=0..K``, AUEC is exactly the arithmetic mean of linear
    NMSE: ``sum(10**(NMSE_k/10)) / (K+1)``.  No logarithmic averaging or
    post-hoc best-iteration selection is performed.
    """

    values = np.asarray(nmse_db_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("nmse_db_values must not be empty")
    if np.any(np.isnan(values)):
        return np.nan
    with np.errstate(over="ignore"):
        return float(np.mean(np.power(10.0, values / 10.0)))


def papr_db(samples: ArrayLike) -> float:
    """Return peak-to-average power ratio in dB."""

    values = _complex_vector(samples, name="samples")
    power = np.abs(values) ** 2
    mean_power = float(np.mean(power))
    if mean_power <= 0.0:
        raise ValueError("samples must have positive mean power")
    return float(10.0 * np.log10(float(np.max(power)) / mean_power))


@dataclass(frozen=True)
class BinnedAMAMPMError:
    """Per-envelope-bin amplitude and wrapped phase tracking errors."""

    bin_edges: NDArray[np.float64]
    bin_centers: NDArray[np.float64]
    counts: NDArray[np.int64]
    phase_counts: NDArray[np.int64]
    amam_bias: NDArray[np.float64]
    amam_rmse: NDArray[np.float64]
    ampm_bias_deg: NDArray[np.float64]
    ampm_rmse_deg: NDArray[np.float64]


def binned_amam_ampm_error(
    input_signal: ArrayLike,
    measured: ArrayLike,
    reference_output: ArrayLike | None = None,
    *,
    bin_edges: ArrayLike | None = None,
    bin_count: int = 20,
    phase_magnitude_floor: float | None = None,
) -> BinnedAMAMPMError:
    """Measure AM/AM and AM/PM tracking error in input-envelope bins.

    AM/AM error is ``abs(measured) - abs(reference_output)``.  AM/PM error is
    the wrapped phase of ``measured * conj(reference_output)`` in degrees.  If
    ``reference_output`` is omitted, the PA input is the ideal unity response.
    Empty bins are represented by NaN rather than silently imputed values.
    """

    pa_input = _complex_vector(input_signal, name="input_signal")
    observed = _complex_vector(measured, name="measured")
    ideal = pa_input if reference_output is None else _complex_vector(
        reference_output,
        name="reference_output",
    )
    if observed.shape != pa_input.shape or ideal.shape != pa_input.shape:
        raise ValueError("input_signal, measured, and reference_output must have the same size")

    envelope = np.abs(pa_input)
    if bin_edges is None:
        if bin_count <= 0:
            raise ValueError("bin_count must be positive")
        upper = float(np.max(envelope))
        if upper <= 0.0:
            upper = 1.0
        # nextafter guarantees the largest observed sample belongs to the last
        # half-open bin without changing any scientifically meaningful edge.
        edges = np.linspace(0.0, np.nextafter(upper, np.inf), bin_count + 1)
    else:
        edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
        if edges.size < 2 or not np.all(np.isfinite(edges)):
            raise ValueError("bin_edges must contain at least two finite values")
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("bin_edges must be strictly increasing")
        bin_count = edges.size - 1

    amplitude_error = np.abs(observed) - np.abs(ideal)
    phase_error_deg = np.rad2deg(np.angle(observed * np.conj(ideal)))
    if phase_magnitude_floor is None:
        phase_floor = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(ideal))))
    else:
        phase_floor = float(phase_magnitude_floor)
        if not np.isfinite(phase_floor) or phase_floor < 0.0:
            raise ValueError("phase_magnitude_floor must be finite and non-negative")
    phase_valid = (np.abs(ideal) > phase_floor) & (np.abs(observed) > phase_floor)

    assignment = np.searchsorted(edges, envelope, side="right") - 1
    in_range = (assignment >= 0) & (assignment < bin_count)
    counts = np.zeros(bin_count, dtype=np.int64)
    phase_counts = np.zeros(bin_count, dtype=np.int64)
    amam_bias = np.full(bin_count, np.nan, dtype=np.float64)
    amam_rmse = np.full(bin_count, np.nan, dtype=np.float64)
    ampm_bias = np.full(bin_count, np.nan, dtype=np.float64)
    ampm_rmse = np.full(bin_count, np.nan, dtype=np.float64)

    for bin_index in range(bin_count):
        selected = in_range & (assignment == bin_index)
        counts[bin_index] = int(np.count_nonzero(selected))
        if counts[bin_index] > 0:
            errors = amplitude_error[selected]
            amam_bias[bin_index] = float(np.mean(errors))
            amam_rmse[bin_index] = float(np.sqrt(np.mean(np.square(errors))))

        phase_selected = selected & phase_valid
        phase_counts[bin_index] = int(np.count_nonzero(phase_selected))
        if phase_counts[bin_index] > 0:
            errors_deg = phase_error_deg[phase_selected]
            # Bias is a circular mean; RMSE uses the already wrapped errors.
            mean_phasor = np.mean(np.exp(1j * np.deg2rad(errors_deg)))
            ampm_bias[bin_index] = float(np.rad2deg(np.angle(mean_phasor)))
            ampm_rmse[bin_index] = float(np.sqrt(np.mean(np.square(errors_deg))))

    return BinnedAMAMPMError(
        bin_edges=edges,
        bin_centers=0.5 * (edges[:-1] + edges[1:]),
        counts=counts,
        phase_counts=phase_counts,
        amam_bias=amam_bias,
        amam_rmse=amam_rmse,
        ampm_bias_deg=ampm_bias,
        ampm_rmse_deg=ampm_rmse,
    )


@dataclass(frozen=True)
class BilateralACLR:
    """Lower- and upper-adjacent channel leakage ratios in positive dB."""

    lower_db: float
    upper_db: float
    main_power: float
    lower_power: float
    upper_power: float
    adjacent_bins_per_side: int

    @property
    def worst_db(self) -> float:
        return min(self.lower_db, self.upper_db)


def _power_ratio_db(numerator: float, denominator: float) -> float:
    if numerator <= 0.0:
        raise ValueError("main-channel power must be positive")
    if denominator < 0.0:
        raise ValueError("channel power cannot be negative")
    if denominator == 0.0:
        return np.inf
    return float(10.0 * np.log10(numerator / denominator))


def bilateral_aclr_db(
    samples: ArrayLike,
    *,
    nfft: int = 2048,
    occupied_per_side: int = 600,
    adjacent_bins_per_side: int | None = None,
) -> BilateralACLR:
    """Return sampled-band ACLR on both sides of the occupied channel.

    Each OFDM-length block is transformed independently and periodograms are
    averaged.  The main-channel mask contains DC and bins ``+-1..+-K``.  Each
    adjacent mask starts at the corresponding main-channel edge.  With the
    frozen 2048/600 waveform a full equal-bandwidth adjacent channel does not
    fit below Nyquist, so the default uses all 423 symmetric guard bins per
    side and records that width in the result.  Callers may request a narrower
    common measurement bandwidth explicitly.
    """

    values = _complex_vector(samples, name="samples")
    if nfft <= 0 or values.size % nfft != 0:
        raise ValueError("sample count must be a positive multiple of nfft")
    if occupied_per_side <= 0 or 2 * occupied_per_side + 1 >= nfft:
        raise ValueError("occupied_per_side leaves no symmetric adjacent bands")
    available = (nfft - (2 * occupied_per_side + 1)) // 2
    width = available if adjacent_bins_per_side is None else int(adjacent_bins_per_side)
    if width <= 0 or width > available:
        raise ValueError("adjacent_bins_per_side exceeds the symmetric sampled guard")

    blocks = values.reshape(-1, nfft)
    spectra = np.fft.fft(blocks, axis=1, norm="ortho")
    power = np.mean(np.abs(spectra) ** 2, axis=0)

    main_power = float(
        np.sum(power[: occupied_per_side + 1])
        + np.sum(power[nfft - occupied_per_side :])
    )
    upper_power = float(
        np.sum(power[occupied_per_side + 1 : occupied_per_side + 1 + width])
    )
    lower_start = nfft - occupied_per_side - width
    lower_power = float(np.sum(power[lower_start : nfft - occupied_per_side]))
    return BilateralACLR(
        lower_db=_power_ratio_db(main_power, lower_power),
        upper_db=_power_ratio_db(main_power, upper_power),
        main_power=main_power,
        lower_power=lower_power,
        upper_power=upper_power,
        adjacent_bins_per_side=width,
    )


@dataclass(frozen=True)
class KnownGridEVM:
    """Raw and least-squares one-tap-equalized EVM on known occupied REs."""

    raw_percent: float
    one_tap_percent: float
    raw_db: float
    one_tap_db: float
    fitted_gain: complex
    resource_element_count: int


def _evm_db(evm_ratio: float) -> float:
    if evm_ratio == 0.0:
        return -np.inf
    return float(20.0 * np.log10(evm_ratio))


def known_grid_evm(
    measured_samples: ArrayLike,
    reference_grid: ArrayLike,
    occupied_bins: ArrayLike | None = None,
) -> KnownGridEVM:
    """Compute raw and one-complex-tap EVM from an exactly known OFDM grid.

    No timing, phase, RMS, or frequency-offset estimator is hidden in this
    metric.  ``measured_samples`` must already have the same symbol boundaries
    and frozen calibration as ``reference_grid``.
    """

    grid = np.asarray(reference_grid, dtype=np.complex128)
    if grid.ndim != 2 or grid.shape[0] <= 0 or grid.shape[1] <= 0:
        raise ValueError("reference_grid must have shape (symbols, nfft)")
    if not np.all(np.isfinite(grid)):
        raise ValueError("reference_grid must be finite")
    symbol_count, nfft = grid.shape
    measured = _complex_vector(measured_samples, name="measured_samples")
    if measured.size != symbol_count * nfft:
        raise ValueError("measured sample count does not match reference_grid")

    if occupied_bins is None:
        bin_mask = np.any(np.abs(grid) > 0.0, axis=0)
        bins = np.flatnonzero(bin_mask)
    else:
        candidate = np.asarray(occupied_bins)
        if candidate.dtype == np.bool_:
            if candidate.ndim != 1 or candidate.size != nfft:
                raise ValueError("boolean occupied_bins mask must have length nfft")
            bins = np.flatnonzero(candidate)
        else:
            bins = np.asarray(candidate, dtype=np.int64).reshape(-1)
            if np.any((bins < 0) | (bins >= nfft)):
                raise ValueError("occupied_bins contains an out-of-range index")
    if bins.size == 0:
        raise ValueError("occupied_bins must select at least one subcarrier")

    measured_grid = np.fft.fft(measured.reshape(symbol_count, nfft), axis=1, norm="ortho")
    reference_re = grid[:, bins].reshape(-1)
    measured_re = measured_grid[:, bins].reshape(-1)
    reference_energy = float(np.vdot(reference_re, reference_re).real)
    if reference_energy <= 0.0:
        raise ValueError("selected reference resource elements have zero energy")

    raw_error = measured_re - reference_re
    raw_ratio = float(np.sqrt(np.vdot(raw_error, raw_error).real / reference_energy))
    fitted_gain = complex(np.vdot(reference_re, measured_re) / reference_energy)
    if abs(fitted_gain) <= np.finfo(np.float64).eps:
        one_tap_ratio = np.inf
    else:
        equalized = measured_re / fitted_gain
        equalized_error = equalized - reference_re
        one_tap_ratio = float(
            np.sqrt(np.vdot(equalized_error, equalized_error).real / reference_energy)
        )

    return KnownGridEVM(
        raw_percent=100.0 * raw_ratio,
        one_tap_percent=100.0 * one_tap_ratio,
        raw_db=_evm_db(raw_ratio),
        one_tap_db=_evm_db(one_tap_ratio),
        fitted_gain=fitted_gain,
        resource_element_count=int(reference_re.size),
    )


grid_evm = known_grid_evm


__all__ = [
    "BilateralACLR",
    "BinnedAMAMPMError",
    "KnownGridEVM",
    "auec",
    "bilateral_aclr_db",
    "binned_amam_ampm_error",
    "fixed_domain_nmse",
    "fixed_domain_nmse_db",
    "grid_evm",
    "known_grid_evm",
    "nmse",
    "nmse_db",
    "papr_db",
]
