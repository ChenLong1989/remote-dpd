"""Synthetic power-amplifier families used by the experiment protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .waveforms import named_rng


WIENER_FIR_BASE = np.array(
    [0.94 + 0.0j, 0.18 - 0.08j, -0.05 + 0.03j],
    dtype=np.complex128,
)
HAMMERSTEIN_FIR_BASE = np.array(
    [0.92 + 0.0j, 0.12 + 0.06j, -0.04 + 0.02j],
    dtype=np.complex128,
)


def _complex_array(values: ArrayLike, *, name: str) -> NDArray[np.complex128]:
    result = np.asarray(values, dtype=np.complex128)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _real_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(value: float, *, name: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


@runtime_checkable
class PA(Protocol):
    """Minimal interface shared by all simulated power amplifiers."""

    def forward(self, input_signal: ArrayLike) -> NDArray[np.complex128]:
        ...


@runtime_checkable
class DifferentiablePA(PA, Protocol):
    """A PA exposing real-linear Jacobian products for oracle experiments."""

    def jvp(self, input_signal: ArrayLike, tangent: ArrayLike) -> NDArray[np.complex128]:
        ...

    def vjp(self, input_signal: ArrayLike, cotangent: ArrayLike) -> NDArray[np.complex128]:
        ...


def rapp_amam(
    input_amplitude: ArrayLike,
    a_sat: float = 1.0,
    p: float = 4.0,
) -> NDArray[np.float64]:
    """Evaluate the smooth Rapp AM/AM response."""

    amplitude = _real_array(input_amplitude, name="input_amplitude")
    saturation = _positive_finite(a_sat, name="a_sat")
    smoothness = _positive_finite(p, name="p")
    if np.any(amplitude < 0.0):
        raise ValueError("input_amplitude must be non-negative")
    ratio_power = np.power(amplitude / saturation, 2.0 * smoothness)
    return np.asarray(
        amplitude / np.power(1.0 + ratio_power, 1.0 / (2.0 * smoothness)),
        dtype=np.float64,
    )


def rapp_amam_derivative(
    input_amplitude: ArrayLike,
    a_sat: float = 1.0,
    p: float = 4.0,
) -> NDArray[np.float64]:
    """Evaluate the analytic slope of the Rapp AM/AM response."""

    amplitude = _real_array(input_amplitude, name="input_amplitude")
    saturation = _positive_finite(a_sat, name="a_sat")
    smoothness = _positive_finite(p, name="p")
    if np.any(amplitude < 0.0):
        raise ValueError("input_amplitude must be non-negative")
    ratio_power = np.power(amplitude / saturation, 2.0 * smoothness)
    exponent = -(1.0 + 1.0 / (2.0 * smoothness))
    return np.asarray(np.power(1.0 + ratio_power, exponent), dtype=np.float64)


def rapp_reachable(output_amplitude: ArrayLike, a_sat: float = 1.0) -> NDArray[np.bool_]:
    """Return the analytic finite-input reachability mask for a Rapp PA."""

    amplitude = _real_array(output_amplitude, name="output_amplitude")
    saturation = _positive_finite(a_sat, name="a_sat")
    return np.asarray((amplitude >= 0.0) & (amplitude < saturation), dtype=np.bool_)


def rapp_inverse_amplitude(
    output_amplitude: ArrayLike,
    a_sat: float = 1.0,
    p: float = 4.0,
) -> NDArray[np.float64]:
    """Return the exact finite-input inverse of the Rapp AM/AM response.

    The Rapp curve approaches ``a_sat`` only asymptotically.  Consequently an
    output amplitude equal to or above ``a_sat`` is rejected as unreachable.
    """

    amplitude = _real_array(output_amplitude, name="output_amplitude")
    saturation = _positive_finite(a_sat, name="a_sat")
    smoothness = _positive_finite(p, name="p")
    reachable = rapp_reachable(amplitude, saturation)
    if not np.all(reachable):
        raise ValueError("Rapp output amplitude must satisfy 0 <= amplitude < a_sat")
    normalized_power = np.power(amplitude / saturation, 2.0 * smoothness)
    denominator = np.power(1.0 - normalized_power, 1.0 / (2.0 * smoothness))
    return np.asarray(amplitude / denominator, dtype=np.float64)


def exponential_ampm(
    input_amplitude: ArrayLike,
    phase_max_rad: float,
    r0: float = 0.21,
) -> NDArray[np.float64]:
    """Evaluate the low-power exponential AM/PM phase law in radians."""

    amplitude = _real_array(input_amplitude, name="input_amplitude")
    if np.any(amplitude < 0.0):
        raise ValueError("input_amplitude must be non-negative")
    phase_max = float(phase_max_rad)
    radius = _positive_finite(r0, name="r0")
    if not np.isfinite(phase_max):
        raise ValueError("phase_max_rad must be finite")
    return np.asarray(phase_max * np.exp(-np.square(amplitude / radius)), dtype=np.float64)


def exponential_ampm_derivative(
    input_amplitude: ArrayLike,
    phase_max_rad: float,
    r0: float = 0.21,
) -> NDArray[np.float64]:
    """Evaluate the analytic radial derivative of exponential AM/PM."""

    amplitude = _real_array(input_amplitude, name="input_amplitude")
    phase = exponential_ampm(amplitude, phase_max_rad, r0)
    radius = _positive_finite(r0, name="r0")
    return np.asarray(-2.0 * amplitude * phase / (radius * radius), dtype=np.float64)


class RadialMemorylessPA:
    """Base class for circularly symmetric memoryless PA maps.

    Subclasses provide output amplitude, its slope, phase, and phase slope as
    functions of input amplitude.  ``jvp`` and ``vjp`` use the corresponding
    two-dimensional real Jacobian, so they remain correct for non-holomorphic
    complex maps.
    """

    def amplitude(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        raise NotImplementedError

    def amplitude_derivative(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        raise NotImplementedError

    def phase(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        amplitude = _real_array(input_amplitude, name="input_amplitude")
        return np.zeros_like(amplitude)

    def phase_derivative(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        amplitude = _real_array(input_amplitude, name="input_amplitude")
        return np.zeros_like(amplitude)

    def forward(self, input_signal: ArrayLike) -> NDArray[np.complex128]:
        values = _complex_array(input_signal, name="input_signal")
        radius = np.abs(values)
        output_radius = self.amplitude(radius)
        rotation = np.exp(1j * self.phase(radius))
        radial_unit = np.divide(
            values,
            radius,
            out=np.ones_like(values),
            where=radius > 0.0,
        )
        output = output_radius * radial_unit * rotation
        output = np.where(radius > 0.0, output, 0.0 + 0.0j)
        return np.asarray(output, dtype=np.complex128)

    __call__ = forward

    def jvp(self, input_signal: ArrayLike, tangent: ArrayLike) -> NDArray[np.complex128]:
        values = _complex_array(input_signal, name="input_signal")
        direction = _complex_array(tangent, name="tangent")
        if direction.shape != values.shape:
            raise ValueError("tangent must have the same shape as input_signal")

        radius = np.abs(values)
        nonzero = radius > 0.0
        input_unit = np.divide(
            values,
            radius,
            out=np.ones_like(values),
            where=nonzero,
        )
        local_direction = np.conj(input_unit) * direction
        radial_direction = np.real(local_direction)
        tangential_direction = np.imag(local_direction)

        output_radius = self.amplitude(radius)
        radial_slope = self.amplitude_derivative(radius)
        phase = self.phase(radius)
        phase_slope = self.phase_derivative(radius)
        tangential_gain = np.divide(
            output_radius,
            radius,
            out=np.asarray(radial_slope, dtype=np.float64).copy(),
            where=nonzero,
        )
        output_unit = input_unit * np.exp(1j * phase)
        local_output = (
            (radial_slope + 1j * output_radius * phase_slope) * radial_direction
            + 1j * tangential_gain * tangential_direction
        )
        result = output_unit * local_output

        if np.any(~nonzero):
            origin_gain = radial_slope * np.exp(1j * phase)
            result = np.where(nonzero, result, origin_gain * direction)
        return np.asarray(result, dtype=np.complex128)

    def vjp(self, input_signal: ArrayLike, cotangent: ArrayLike) -> NDArray[np.complex128]:
        values = _complex_array(input_signal, name="input_signal")
        vector = _complex_array(cotangent, name="cotangent")
        if vector.shape != values.shape:
            raise ValueError("cotangent must have the same shape as input_signal")

        radius = np.abs(values)
        nonzero = radius > 0.0
        input_unit = np.divide(
            values,
            radius,
            out=np.ones_like(values),
            where=nonzero,
        )
        output_radius = self.amplitude(radius)
        radial_slope = self.amplitude_derivative(radius)
        phase = self.phase(radius)
        phase_slope = self.phase_derivative(radius)
        output_unit = input_unit * np.exp(1j * phase)

        local_vector = np.conj(output_unit) * vector
        output_radial = np.real(local_vector)
        output_tangential = np.imag(local_vector)
        tangential_gain = np.divide(
            output_radius,
            radius,
            out=np.asarray(radial_slope, dtype=np.float64).copy(),
            where=nonzero,
        )
        input_radial = radial_slope * output_radial + output_radius * phase_slope * output_tangential
        input_tangential = tangential_gain * output_tangential
        result = input_unit * (input_radial + 1j * input_tangential)

        if np.any(~nonzero):
            origin_adjoint = radial_slope * np.exp(-1j * phase) * vector
            result = np.where(nonzero, result, origin_adjoint)
        return np.asarray(result, dtype=np.complex128)


@dataclass(frozen=True)
class RappPA(RadialMemorylessPA):
    """Rapp AM/AM with optional exponential low-power AM/PM."""

    a_sat: float = 1.0
    p: float = 4.0
    phase_max_deg: float = 0.0
    r0: float = 0.21

    def __post_init__(self) -> None:
        _positive_finite(self.a_sat, name="a_sat")
        _positive_finite(self.p, name="p")
        _positive_finite(self.r0, name="r0")
        if not np.isfinite(self.phase_max_deg):
            raise ValueError("phase_max_deg must be finite")

    @property
    def phase_max_rad(self) -> float:
        return float(np.deg2rad(self.phase_max_deg))

    def amplitude(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        return rapp_amam(input_amplitude, self.a_sat, self.p)

    def amplitude_derivative(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        return rapp_amam_derivative(input_amplitude, self.a_sat, self.p)

    def phase(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        return exponential_ampm(input_amplitude, self.phase_max_rad, self.r0)

    def phase_derivative(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        return exponential_ampm_derivative(input_amplitude, self.phase_max_rad, self.r0)

    def reachable(self, output_amplitude: ArrayLike) -> NDArray[np.bool_]:
        return rapp_reachable(output_amplitude, self.a_sat)

    def inverse_amplitude(self, output_amplitude: ArrayLike) -> NDArray[np.float64]:
        return rapp_inverse_amplitude(output_amplitude, self.a_sat, self.p)

    def inverse(self, output_signal: ArrayLike) -> NDArray[np.complex128]:
        desired = _complex_array(output_signal, name="output_signal")
        output_radius = np.abs(desired)
        input_radius = self.inverse_amplitude(output_radius)
        input_phase = np.angle(desired) - self.phase(input_radius)
        result = input_radius * np.exp(1j * input_phase)
        return np.where(output_radius > 0.0, result, 0.0 + 0.0j).astype(np.complex128)


@dataclass(frozen=True)
class ExponentialAMPMPA(RadialMemorylessPA):
    """Unity AM/AM with severe phase distortion concentrated at low power."""

    phase_max_deg: float
    r0: float = 0.21

    def __post_init__(self) -> None:
        _positive_finite(self.r0, name="r0")
        if not np.isfinite(self.phase_max_deg):
            raise ValueError("phase_max_deg must be finite")

    @property
    def phase_max_rad(self) -> float:
        return float(np.deg2rad(self.phase_max_deg))

    def amplitude(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        return _real_array(input_amplitude, name="input_amplitude")

    def amplitude_derivative(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        amplitude = _real_array(input_amplitude, name="input_amplitude")
        return np.ones_like(amplitude)

    def phase(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        return exponential_ampm(input_amplitude, self.phase_max_rad, self.r0)

    def phase_derivative(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        return exponential_ampm_derivative(input_amplitude, self.phase_max_rad, self.r0)

    def inverse(self, output_signal: ArrayLike) -> NDArray[np.complex128]:
        desired = _complex_array(output_signal, name="output_signal")
        radius = np.abs(desired)
        phase = np.angle(desired) - self.phase(radius)
        result = radius * np.exp(1j * phase)
        return np.where(radius > 0.0, result, 0.0 + 0.0j).astype(np.complex128)


@dataclass(frozen=True)
class HardSaturationPA(RadialMemorylessPA):
    """Ideal hard amplitude limiter used only as an unreachable stress case."""

    a_sat: float = 1.0

    def __post_init__(self) -> None:
        _positive_finite(self.a_sat, name="a_sat")

    def amplitude(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        amplitude = _real_array(input_amplitude, name="input_amplitude")
        return np.minimum(amplitude, self.a_sat)

    def amplitude_derivative(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        amplitude = _real_array(input_amplitude, name="input_amplitude")
        # The derivative is undefined at the knee.  Zero is the conservative
        # oracle convention at that measure-zero boundary.
        return np.asarray(amplitude < self.a_sat, dtype=np.float64)

    def reachable(self, output_amplitude: ArrayLike) -> NDArray[np.bool_]:
        amplitude = _real_array(output_amplitude, name="output_amplitude")
        return np.asarray((amplitude >= 0.0) & (amplitude <= self.a_sat), dtype=np.bool_)


@dataclass(frozen=True)
class GainRolloffPA(RadialMemorylessPA):
    """Smooth AM/AM whose local slope becomes negative above ``turnover``."""

    turnover: float = 0.7

    def __post_init__(self) -> None:
        _positive_finite(self.turnover, name="turnover")

    def amplitude(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        amplitude = _real_array(input_amplitude, name="input_amplitude")
        return np.asarray(
            amplitude * np.exp(-0.5 * np.square(amplitude / self.turnover)),
            dtype=np.float64,
        )

    def amplitude_derivative(self, input_amplitude: ArrayLike) -> NDArray[np.float64]:
        amplitude = _real_array(input_amplitude, name="input_amplitude")
        normalized_square = np.square(amplitude / self.turnover)
        return np.asarray(
            np.exp(-0.5 * normalized_square) * (1.0 - normalized_square),
            dtype=np.float64,
        )


def circular_fir(input_signal: ArrayLike, taps: ArrayLike) -> NDArray[np.complex128]:
    """Apply causal-tap circular FIR convolution ``sum h[m] x[n-m]``."""

    values = _complex_array(input_signal, name="input_signal")
    coefficients = _complex_array(taps, name="taps").reshape(-1)
    if values.ndim != 1:
        raise ValueError("input_signal must be one-dimensional")
    output = np.zeros_like(values)
    for delay, coefficient in enumerate(coefficients):
        output += coefficient * np.roll(values, delay)
    return output


def circular_fir_adjoint(cotangent: ArrayLike, taps: ArrayLike) -> NDArray[np.complex128]:
    """Apply the real/complex inner-product adjoint of ``circular_fir``."""

    vector = _complex_array(cotangent, name="cotangent")
    coefficients = _complex_array(taps, name="taps").reshape(-1)
    if vector.ndim != 1:
        raise ValueError("cotangent must be one-dimensional")
    result = np.zeros_like(vector)
    for delay, coefficient in enumerate(coefficients):
        result += np.conj(coefficient) * np.roll(vector, -delay)
    return result


def perturbed_unit_dc_taps(
    base_taps: ArrayLike,
    *,
    family: str,
    seed_index: int,
    relative_perturbation: float = 0.05,
) -> NDArray[np.complex128]:
    """Perturb non-leading tap components and normalize complex DC gain to one."""

    taps = _complex_array(base_taps, name="base_taps").reshape(-1).copy()
    if taps.size < 2:
        raise ValueError("dynamic PA FIR requires at least two taps")
    if not family:
        raise ValueError("family must not be empty")
    if seed_index < 0:
        raise ValueError("seed_index must be non-negative")
    perturbation = float(relative_perturbation)
    if not np.isfinite(perturbation) or perturbation < 0.0:
        raise ValueError("relative_perturbation must be finite and non-negative")

    rng = named_rng("dynamic_pa", family, seed_index)
    real_scale = 1.0 + rng.uniform(-perturbation, perturbation, size=taps.size - 1)
    imag_scale = 1.0 + rng.uniform(-perturbation, perturbation, size=taps.size - 1)
    taps[1:] = taps[1:].real * real_scale + 1j * taps[1:].imag * imag_scale
    dc_gain = np.sum(taps)
    if not np.isfinite(dc_gain) or abs(dc_gain) <= np.finfo(np.float64).eps:
        raise ValueError("perturbed FIR has zero or invalid DC gain")
    taps /= dc_gain
    return taps


@dataclass(frozen=True)
class WienerPA:
    """Complex input FIR followed by a memoryless nonlinear PA."""

    taps: NDArray[np.complex128]
    nonlinearity: DifferentiablePA

    def __post_init__(self) -> None:
        taps = _complex_array(self.taps, name="taps").reshape(-1).copy()
        object.__setattr__(self, "taps", taps)

    def forward(self, input_signal: ArrayLike) -> NDArray[np.complex128]:
        return self.nonlinearity.forward(circular_fir(input_signal, self.taps))

    __call__ = forward

    def jvp(self, input_signal: ArrayLike, tangent: ArrayLike) -> NDArray[np.complex128]:
        filtered_input = circular_fir(input_signal, self.taps)
        filtered_tangent = circular_fir(tangent, self.taps)
        return self.nonlinearity.jvp(filtered_input, filtered_tangent)

    def vjp(self, input_signal: ArrayLike, cotangent: ArrayLike) -> NDArray[np.complex128]:
        filtered_input = circular_fir(input_signal, self.taps)
        nonlinear_adjoint = self.nonlinearity.vjp(filtered_input, cotangent)
        return circular_fir_adjoint(nonlinear_adjoint, self.taps)


@dataclass(frozen=True)
class HammersteinPA:
    """Memoryless nonlinear PA followed by a complex output FIR."""

    taps: NDArray[np.complex128]
    nonlinearity: DifferentiablePA

    def __post_init__(self) -> None:
        taps = _complex_array(self.taps, name="taps").reshape(-1).copy()
        object.__setattr__(self, "taps", taps)

    def forward(self, input_signal: ArrayLike) -> NDArray[np.complex128]:
        return circular_fir(self.nonlinearity.forward(input_signal), self.taps)

    __call__ = forward

    def jvp(self, input_signal: ArrayLike, tangent: ArrayLike) -> NDArray[np.complex128]:
        return circular_fir(self.nonlinearity.jvp(input_signal, tangent), self.taps)

    def vjp(self, input_signal: ArrayLike, cotangent: ArrayLike) -> NDArray[np.complex128]:
        filtered_adjoint = circular_fir_adjoint(cotangent, self.taps)
        return self.nonlinearity.vjp(input_signal, filtered_adjoint)


def make_wiener_pa(
    seed_index: int,
    *,
    phase_max_deg: float = 135.0,
    r0: float = 0.21,
) -> WienerPA:
    """Build the frozen out-of-family Wiener ground-truth PA."""

    taps = perturbed_unit_dc_taps(
        WIENER_FIR_BASE,
        family="wiener",
        seed_index=seed_index,
    )
    return WienerPA(taps=taps, nonlinearity=RappPA(phase_max_deg=phase_max_deg, r0=r0))


def make_hammerstein_pa(
    seed_index: int,
    *,
    phase_max_deg: float = 135.0,
    r0: float = 0.21,
) -> HammersteinPA:
    """Build the frozen out-of-family Hammerstein ground-truth PA."""

    taps = perturbed_unit_dc_taps(
        HAMMERSTEIN_FIR_BASE,
        family="hammerstein",
        seed_index=seed_index,
    )
    return HammersteinPA(taps=taps, nonlinearity=RappPA(phase_max_deg=phase_max_deg, r0=r0))


def scale_to_peak(samples: ArrayLike, target_peak: float) -> NDArray[np.complex128]:
    """Scale a finite nonzero waveform to an exact peak magnitude."""

    values = _complex_array(samples, name="samples")
    peak = float(np.max(np.abs(values)))
    requested = _positive_finite(target_peak, name="target_peak")
    if peak <= 0.0:
        raise ValueError("zero waveform cannot be peak-scaled")
    return np.asarray(values * (requested / peak), dtype=np.complex128)


def scale_to_rms(samples: ArrayLike, target_rms: float) -> NDArray[np.complex128]:
    """Scale a finite nonzero waveform to an exact RMS magnitude."""

    values = _complex_array(samples, name="samples")
    rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
    requested = _positive_finite(target_rms, name="target_rms")
    if rms <= 0.0:
        raise ValueError("zero waveform cannot be RMS-scaled")
    return np.asarray(values * (requested / rms), dtype=np.complex128)


@dataclass(frozen=True)
class PAScenario:
    """A fixed-domain target, initial PA input, ground-truth PA, and metadata."""

    name: str
    desired: NDArray[np.complex128]
    initial_input: NDArray[np.complex128]
    pa: PA
    metadata: Mapping[str, Any]


def make_amam_scenario(
    waveform: ArrayLike,
    target_peak_ratio: float,
    *,
    a_sat: float = 1.0,
    p: float = 4.0,
) -> PAScenario:
    """Create a reachable smooth-Rapp AM/AM experiment cell."""

    pa = RappPA(a_sat=a_sat, p=p)
    desired_peak = float(target_peak_ratio) * pa.a_sat
    if not 0.0 < target_peak_ratio < 1.0:
        raise ValueError("reachable Rapp target_peak_ratio must lie strictly between zero and one")
    desired = scale_to_peak(waveform, desired_peak)
    required_peak = float(pa.inverse_amplitude(np.asarray(desired_peak)))
    return PAScenario(
        name=f"amam_rapp_peak_{target_peak_ratio:.3f}",
        desired=desired,
        initial_input=desired.copy(),
        pa=pa,
        metadata={
            "mechanism": "smooth_rapp_amam",
            "a_sat": pa.a_sat,
            "p": pa.p,
            "target_peak_ratio": float(target_peak_ratio),
            "target_peak": desired_peak,
            "reachable": True,
            "required_input_peak": required_peak,
        },
    )


def make_ampm_scenario(
    waveform: ArrayLike,
    phase_max_deg: float,
    *,
    target_rms: float = 0.35,
    r0: float = 0.21,
) -> PAScenario:
    """Create the low-power exponential AM/PM experiment cell."""

    pa = ExponentialAMPMPA(phase_max_deg=phase_max_deg, r0=r0)
    desired = scale_to_rms(waveform, target_rms)
    return PAScenario(
        name=f"ampm_exp_{phase_max_deg:g}_deg",
        desired=desired,
        initial_input=desired.copy(),
        pa=pa,
        metadata={
            "mechanism": "low_power_exponential_ampm",
            "phase_max_deg": float(phase_max_deg),
            "r0": pa.r0,
            "target_rms": float(target_rms),
            "reachable": True,
        },
    )


def make_hard_saturation_stress(
    waveform: ArrayLike,
    *,
    target_peak_ratio: float = 2.00,
    a_sat: float = 1.0,
) -> PAScenario:
    """Create the explicitly unreachable hard-saturation safety stress cell."""

    pa = HardSaturationPA(a_sat=a_sat)
    if target_peak_ratio <= 1.0:
        raise ValueError("hard-saturation stress target must exceed a_sat")
    desired = scale_to_peak(waveform, target_peak_ratio * a_sat)
    return PAScenario(
        name=f"hard_saturation_unreachable_{target_peak_ratio:.3f}",
        desired=desired,
        initial_input=desired.copy(),
        pa=pa,
        metadata={
            "mechanism": "unreachable_hard_saturation",
            "a_sat": float(a_sat),
            "target_peak_ratio": float(target_peak_ratio),
            "reachable": False,
        },
    )


def make_gain_rolloff_stress(
    waveform: ArrayLike,
    *,
    target_peak: float = 0.40,
    initial_input_peak: float = 2.50,
    turnover: float = 0.70,
) -> PAScenario:
    """Create a reachable target initialized beyond the AM/AM slope reversal."""

    pa = GainRolloffPA(turnover=turnover)
    maximum_output = turnover * np.exp(-0.5)
    if not 0.0 < target_peak < maximum_output:
        raise ValueError("gain-rolloff target_peak must lie below the AM/AM maximum")
    if initial_input_peak <= turnover:
        raise ValueError("initial_input_peak must exceed the slope turnover")
    desired = scale_to_peak(waveform, target_peak)
    initial_input = scale_to_peak(waveform, initial_input_peak)
    low = 0.0
    high = turnover
    for _ in range(100):
        midpoint = 0.5 * (low + high)
        if float(pa.amplitude(np.asarray(midpoint))) < target_peak:
            low = midpoint
        else:
            high = midpoint
    required_input_peak = 0.5 * (low + high)
    return PAScenario(
        name=f"gain_rolloff_target_{target_peak:.3f}_initial_{initial_input_peak:.3f}",
        desired=desired,
        initial_input=initial_input,
        pa=pa,
        metadata={
            "mechanism": "smooth_gain_rolloff",
            "turnover": float(turnover),
            "target_peak": float(target_peak),
            "initial_input_peak": float(initial_input_peak),
            "contains_negative_slope": True,
            "reachable": True,
            "required_input_peak": required_input_peak,
        },
    )


__all__ = [
    "DifferentiablePA",
    "ExponentialAMPMPA",
    "GainRolloffPA",
    "HAMMERSTEIN_FIR_BASE",
    "HammersteinPA",
    "HardSaturationPA",
    "PA",
    "PAScenario",
    "RappPA",
    "RadialMemorylessPA",
    "WIENER_FIR_BASE",
    "WienerPA",
    "circular_fir",
    "circular_fir_adjoint",
    "exponential_ampm",
    "exponential_ampm_derivative",
    "make_amam_scenario",
    "make_ampm_scenario",
    "make_gain_rolloff_stress",
    "make_hammerstein_pa",
    "make_hard_saturation_stress",
    "make_wiener_pa",
    "perturbed_unit_dc_taps",
    "rapp_amam",
    "rapp_amam_derivative",
    "rapp_inverse_amplitude",
    "rapp_reachable",
    "scale_to_peak",
    "scale_to_rms",
]
