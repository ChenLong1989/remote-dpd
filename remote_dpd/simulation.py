"""Deterministic simulated RF bench for hardware-independent closed-loop tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from .device import (
    CaptureRequest,
    DeviceConfig,
    DeviceParameterField,
    DeviceParameterSchema,
    DeviceParameterType,
    PowerSensor,
    Receiver,
    RFBench,
    Transmitter,
)
from .dsp import fractional_shift, rms
from .preprocessing import CaptureBatch

_DEFAULT_PA_COEFFICIENTS = [
    {"p": 1, "m": 0, "real": 1.0, "imag": 0.0},
    {"p": 1, "m": 1, "real": 0.04, "imag": 0.015},
    {"p": 3, "m": 0, "real": -0.12, "imag": 0.025},
    {"p": 3, "m": 1, "real": -0.02, "imag": 0.01},
]

_PA_COEFFICIENT_FIELD = DeviceParameterField(
    name="coefficient",
    value_type=DeviceParameterType.OBJECT,
    properties=(
        DeviceParameterField(
            name="p",
            value_type=DeviceParameterType.INTEGER,
            minimum=1,
            step=2,
            required=True,
            description="Positive odd nonlinearity order.",
        ),
        DeviceParameterField(
            name="m",
            value_type=DeviceParameterType.INTEGER,
            minimum=0,
            required=True,
            description="Non-negative memory delay in samples.",
        ),
        DeviceParameterField(
            name="real",
            value_type=DeviceParameterType.NUMBER,
            required=True,
            description="Real part of the complex coefficient.",
        ),
        DeviceParameterField(
            name="imag",
            value_type=DeviceParameterType.NUMBER,
            required=True,
            description="Imaginary part of the complex coefficient.",
        ),
    ),
    additional_properties=False,
)

SIMULATED_DEVICE_SCHEMA = DeviceParameterSchema(
    device_type="simulated",
    schema_version=1,
    fields=(
        DeviceParameterField(
            name="pa_coefficients",
            value_type=DeviceParameterType.ARRAY,
            default=_DEFAULT_PA_COEFFICIENTS,
            items=_PA_COEFFICIENT_FIELD,
            description="Complex memory-polynomial PA coefficients.",
        ),
        DeviceParameterField(
            name="system_gain_db",
            value_type=DeviceParameterType.NUMBER,
            minimum=-120.0,
            maximum=120.0,
            default=-6.0,
            unit="dB",
            description="Fixed feedback-path amplitude gain.",
        ),
        DeviceParameterField(
            name="system_phase_rad",
            value_type=DeviceParameterType.NUMBER,
            default=0.35,
            unit="rad",
            description="Fixed feedback-path phase rotation.",
        ),
        DeviceParameterField(
            name="delay_samples",
            value_type=DeviceParameterType.NUMBER,
            default=2.25,
            unit="sample",
            description="Fixed periodic fractional delay.",
        ),
        DeviceParameterField(
            name="noise_dbfs",
            value_type=DeviceParameterType.NUMBER,
            minimum=-300.0,
            maximum=0.0,
            default=-80.0,
            unit="dBFS",
            description="Complex Gaussian-noise RMS relative to full scale.",
        ),
        DeviceParameterField(
            name="random_seed",
            value_type=DeviceParameterType.INTEGER,
            minimum=0,
            maximum=2**32 - 1,
            default=42,
            description="Seed used for deterministic capture noise.",
        ),
        DeviceParameterField(
            name="power_reference_dbm",
            value_type=DeviceParameterType.NUMBER,
            minimum=-300.0,
            maximum=300.0,
            default=20.0,
            unit="dBm",
            description="Measured power for a unit-RMS noiseless PA output.",
        ),
        DeviceParameterField(
            name="max_capture_samples",
            value_type=DeviceParameterType.INTEGER,
            minimum=1,
            maximum=100_000_000,
            default=1_000_000,
            unit="sample",
            description="Largest sample count accepted by one capture call.",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class _PACoefficient:
    order: int
    memory: int
    value: complex


@dataclass(frozen=True, slots=True)
class _SimulationSettings:
    pa_coefficients: tuple[_PACoefficient, ...]
    system_gain: float
    system_phase_rad: float
    delay_samples: float
    noise_rms: float
    random_seed: int
    power_reference_dbm: float
    max_capture_samples: int


class SimulatedRFBench(RFBench, Transmitter, Receiver, PowerSensor):
    """Integrated transmitter, receiver, and power sensor backed by a PA model."""

    def __init__(self) -> None:
        self._connected = False
        self._config: DeviceConfig | None = None
        self._settings: _SimulationSettings | None = None
        self._waveform: np.ndarray | None = None
        self._transmitting = False
        self._attenuation_db = 0.0
        self._rng = np.random.default_rng(42)
        self._max_capture_samples = 1_000_000

    @property
    def transmitter(self) -> Transmitter:
        """Return this integrated bench as its transmitter capability."""

        return self

    @property
    def receiver(self) -> Receiver:
        """Return this integrated bench as its receiver capability."""

        return self

    @property
    def power_sensor(self) -> PowerSensor:
        """Return this integrated bench as its power-sensor capability."""

        return self

    @property
    def parameter_schema(self) -> DeviceParameterSchema:
        """Return the versioned simulated-device parameter schema."""

        return SIMULATED_DEVICE_SCHEMA

    @property
    def max_capture_samples(self) -> int:
        """Return the configured local capture-size limit without device I/O."""

        return self._max_capture_samples

    def connect(self, timeout_seconds: float) -> None:
        """Open a fresh simulated-device session."""

        _validate_timeout(timeout_seconds)
        if self._connected:
            raise RuntimeError("simulated RF bench is already connected")
        self._connected = True

    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        """Validate, deep-copy, and apply a complete simulation configuration."""

        _validate_timeout(timeout_seconds)
        self._require_connected()
        if self._transmitting:
            raise RuntimeError("cannot configure while transmission is running")
        if not isinstance(config, DeviceConfig):
            raise TypeError("config must be a DeviceConfig")

        options = SIMULATED_DEVICE_SCHEMA.validate_options(config.device_options)
        coefficient_rows = options["pa_coefficients"]
        if not isinstance(
            coefficient_rows, list
        ):  # pragma: no cover - schema invariant
            raise TypeError("pa_coefficients must be an array")
        if not coefficient_rows:
            raise ValueError("pa_coefficients must contain at least one coefficient")

        effective_values = config.to_dict()
        effective_values["device_options"] = options
        effective_config = DeviceConfig(**effective_values)
        settings = _settings_from_options(options)

        self._config = effective_config
        self._settings = settings
        self._waveform = None
        self._transmitting = False
        self._attenuation_db = effective_config.initial_attenuation_db
        self._rng = np.random.default_rng(settings.random_seed)
        self._max_capture_samples = settings.max_capture_samples

    def upload_waveform(self, waveform: np.ndarray, timeout_seconds: float) -> None:
        """Store an exact private copy of one cyclic finite IQ waveform."""

        _validate_timeout(timeout_seconds)
        self._require_configured()
        if self._transmitting:
            raise RuntimeError("cannot upload a waveform while transmission is running")
        self._waveform = _copy_waveform(waveform)

    def start_transmission(self, timeout_seconds: float) -> None:
        """Start cyclic transmission of the currently uploaded waveform."""

        _validate_timeout(timeout_seconds)
        self._require_configured()
        if self._waveform is None:
            raise RuntimeError("a waveform must be uploaded before transmission starts")
        if self._transmitting:
            raise RuntimeError("transmission is already running")
        self._transmitting = True

    def stop_transmission(self, timeout_seconds: float) -> None:
        """Stop transmission; repeated calls in a connected session are harmless."""

        _validate_timeout(timeout_seconds)
        self._require_connected()
        self._transmitting = False

    def get_attenuation_db(self, timeout_seconds: float) -> float:
        """Return the simulated adjustable TX attenuation."""

        _validate_timeout(timeout_seconds)
        self._require_configured()
        return self._attenuation_db

    def set_attenuation_db(
        self,
        attenuation_db: float,
        timeout_seconds: float,
    ) -> None:
        """Set TX attenuation, including while cyclic transmission is active."""

        _validate_timeout(timeout_seconds)
        config = self._require_configured()
        attenuation = _finite_real("attenuation_db", attenuation_db)
        if not config.min_attenuation_db <= attenuation <= config.max_attenuation_db:
            raise ValueError(
                "attenuation_db must be within the configured attenuation range"
            )
        self._attenuation_db = attenuation

    def capture(self, request: CaptureRequest, timeout_seconds: float) -> CaptureBatch:
        """Return one coherent batch of complete contiguous waveform periods."""

        _validate_timeout(timeout_seconds)
        config, settings, waveform = self._require_transmitting()
        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be a CaptureRequest")
        if request.segment_length != waveform.size:
            raise ValueError(
                "request segment_length must equal the uploaded waveform length"
            )
        if request.sample_count > settings.max_capture_samples:
            raise ValueError(
                "requested sample count exceeds max_capture_samples "
                f"({request.sample_count} > {settings.max_capture_samples})"
            )

        feedback_period = self._feedback_period(waveform, settings)
        feedback = np.tile(feedback_period, request.segment_count)
        if settings.noise_rms > 0.0:
            noise = (
                self._rng.normal(size=request.sample_count)
                + 1j * self._rng.normal(size=request.sample_count)
            ) * (settings.noise_rms / math.sqrt(2.0))
            feedback = feedback + noise
        if not np.all(np.isfinite(feedback)):
            raise RuntimeError("simulated feedback contains non-finite samples")

        return CaptureBatch(
            iq=feedback,
            segment_length=request.segment_length,
            segment_count=request.segment_count,
            sample_rate_hz=config.sample_rate_hz,
            coherent_within_batch=True,
        )

    def measure_power_dbm(self, timeout_seconds: float) -> float:
        """Measure noiseless PA-output RMS using the configured dBm reference."""

        _validate_timeout(timeout_seconds)
        _, settings, waveform = self._require_transmitting()
        output_rms = rms(self._pa_output(waveform, settings))
        if output_rms == 0.0:
            return float("-inf")
        return settings.power_reference_dbm + 20.0 * math.log10(output_rms)

    def safe_shutdown(self, timeout_seconds: float) -> None:
        """Stop RF output while preserving the connected configured session."""

        _validate_timeout(timeout_seconds)
        if self._connected:
            self._transmitting = False

    def disconnect(self, timeout_seconds: float) -> None:
        """Safely stop output and release all simulated session state."""

        _validate_timeout(timeout_seconds)
        if not self._connected:
            return
        self._transmitting = False
        self._waveform = None
        self._settings = None
        self._config = None
        self._attenuation_db = 0.0
        self._max_capture_samples = 1_000_000
        self._connected = False

    def _pa_output(
        self,
        waveform: np.ndarray,
        settings: _SimulationSettings,
    ) -> np.ndarray:
        attenuation_scale = 10.0 ** (-self._attenuation_db / 20.0)
        pa_input = waveform * attenuation_scale
        output = np.zeros(pa_input.size, dtype=np.complex128)
        with np.errstate(over="ignore", invalid="ignore"):
            magnitude = np.abs(pa_input)
            for coefficient in settings.pa_coefficients:
                delayed = np.roll(pa_input, coefficient.memory)
                delayed_magnitude = np.roll(magnitude, coefficient.memory)
                output += (
                    coefficient.value
                    * delayed
                    * delayed_magnitude ** (coefficient.order - 1)
                )
        if not np.all(np.isfinite(output)):
            raise RuntimeError("simulated PA produced non-finite samples")
        return output

    def _feedback_period(
        self,
        waveform: np.ndarray,
        settings: _SimulationSettings,
    ) -> np.ndarray:
        pa_output = self._pa_output(waveform, settings)
        system_rotation = settings.system_gain * np.exp(1j * settings.system_phase_rad)
        feedback = pa_output * system_rotation
        return fractional_shift(feedback, settings.delay_samples)

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("simulated RF bench is not connected")

    def _require_configured(self) -> DeviceConfig:
        self._require_connected()
        if self._config is None or self._settings is None:
            raise RuntimeError("simulated RF bench is not configured")
        return self._config

    def _require_transmitting(
        self,
    ) -> tuple[DeviceConfig, _SimulationSettings, np.ndarray]:
        config = self._require_configured()
        if not self._transmitting or self._waveform is None:
            raise RuntimeError("simulated RF bench is not transmitting")
        settings = self._settings
        if settings is None:  # pragma: no cover - lifecycle invariant
            raise RuntimeError("simulated RF bench is not configured")
        return config, settings, self._waveform


def _settings_from_options(options: dict[str, Any]) -> _SimulationSettings:
    rows = options["pa_coefficients"]
    coefficients = tuple(
        _PACoefficient(
            order=int(row["p"]),
            memory=int(row["m"]),
            value=complex(float(row["real"]), float(row["imag"])),
        )
        for row in rows
    )
    return _SimulationSettings(
        pa_coefficients=coefficients,
        system_gain=10.0 ** (float(options["system_gain_db"]) / 20.0),
        system_phase_rad=float(options["system_phase_rad"]),
        delay_samples=float(options["delay_samples"]),
        noise_rms=10.0 ** (float(options["noise_dbfs"]) / 20.0),
        random_seed=int(options["random_seed"]),
        power_reference_dbm=float(options["power_reference_dbm"]),
        max_capture_samples=int(options["max_capture_samples"]),
    )


def _validate_timeout(timeout_seconds: float) -> float:
    timeout = _finite_real("timeout_seconds", timeout_seconds)
    if timeout <= 0.0:
        raise ValueError("timeout_seconds must be greater than zero")
    return timeout


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _copy_waveform(waveform: np.ndarray) -> np.ndarray:
    raw = np.asarray(waveform)
    if raw.ndim != 1:
        raise ValueError("waveform must be a one-dimensional array")
    if raw.size == 0:
        raise ValueError("waveform must not be empty")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("waveform must contain numeric samples")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.complex128, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("waveform must contain only finite samples")
    result.setflags(write=False)
    return result
