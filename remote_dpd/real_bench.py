"""Real RF bench adapter for the local NI RFIC test station.

The bench combines one NI PXIe-5842 VST (transmitter and receiver through the
NI-RFSG / NI-RFSA driver sessions), one Agilent N1912A power meter, and one
Agilent N5767A drain supply guard, all exposed through the standard capability
contracts in ``remote_dpd.device`` and registered under ``vst5842``.

Hard safety constraints for the GaN PA under test:

- The E3648A auxiliary supplies (8 V / 12 V biases) must never be written by
  this project. This module only issues read-only queries against them for
  interlock checks before transmission starts.
- The N5767A drain supply is the only writable supply and only ``OUTP OFF``
  is ever sent. Turning the PA on is a manual operator action; this module
  deliberately provides no code path that enables the 44 V output.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .device import (
    CaptureRequest,
    DeviceCapability,
    DeviceConfig,
    DeviceParameterField,
    DeviceParameterSchema,
    DeviceParameterType,
    PowerSensor,
    Receiver,
    RFBench,
    Transmitter,
    register_rf_bench,
)
from .preprocessing import CaptureBatch

_WAVEFORM_NAME = "rdpdWave"
_SCRIPT_NAME = "rdpdCyclic"
_CYCLE_SCRIPT = (
    f"script {_SCRIPT_NAME}\n"
    "repeat forever\n"
    f"generate {_WAVEFORM_NAME}\n"
    "end repeat\n"
    "end script\n"
)
_DEFAULT_AUX_SUPPLY_RESOURCES = ("GPIB1::7::INSTR", "GPIB1::8::INSTR")
_AUX_VOLTAGE_TOLERANCE_V = 0.5

VST5842_DEVICE_SCHEMA = DeviceParameterSchema(
    device_type="vst5842",
    schema_version=1,
    fields=(
        DeviceParameterField(
            name="vst_resource",
            value_type=DeviceParameterType.STRING,
            default="PXI2Slot2",
            description="NI-RFSG/NI-RFSA device name of the PXIe-5842 VST.",
        ),
        DeviceParameterField(
            name="reference_power_dbm",
            value_type=DeviceParameterType.NUMBER,
            minimum=-60.0,
            maximum=10.0,
            default=-17.0,
            unit="dBm",
            description=(
                "RFSG output level that corresponds to 0 dB of TX attenuation."
            ),
        ),
        DeviceParameterField(
            name="reference_level_dbm",
            value_type=DeviceParameterType.NUMBER,
            minimum=-60.0,
            maximum=99.9,
            default=55.0,
            unit="dBm",
            description=(
                "RFSA reference level; matches the station GUI and implicitly "
                "accounts for the RX front-end external attenuation."
            ),
        ),
        DeviceParameterField(
            name="power_meter_resource",
            value_type=DeviceParameterType.STRING,
            default="TCPIP0::192.168.255.40::inst0::INSTR",
            description="VISA resource of the Agilent N1912A power meter.",
        ),
        DeviceParameterField(
            name="power_meter_average",
            value_type=DeviceParameterType.INTEGER,
            minimum=1,
            maximum=1024,
            default=64,
            description="Power-meter averaging count for one reading.",
        ),
        DeviceParameterField(
            name="supply_resource",
            value_type=DeviceParameterType.STRING,
            default="GPIB1::5::INSTR",
            description="VISA resource of the N5767A PA drain supply.",
        ),
        DeviceParameterField(
            name="aux_supply_resources",
            value_type=DeviceParameterType.ARRAY,
            default=list(_DEFAULT_AUX_SUPPLY_RESOURCES),
            items=DeviceParameterField(
                name="resource",
                value_type=DeviceParameterType.STRING,
                required=True,
                description="VISA resource of one read-only bias supply.",
            ),
            description=(
                "E3648A bias supplies verified (read-only) before transmission."
            ),
        ),
        DeviceParameterField(
            name="enable_supply_shutdown",
            value_type=DeviceParameterType.BOOLEAN,
            default=True,
            description="Turn the 44 V drain output off during safe shutdown.",
        ),
        DeviceParameterField(
            name="enable_supply_interlock",
            value_type=DeviceParameterType.BOOLEAN,
            default=True,
            description=(
                "Verify both bias supplies are powered before transmission; "
                "checked with read-only queries only."
            ),
        ),
        DeviceParameterField(
            name="max_capture_samples",
            value_type=DeviceParameterType.INTEGER,
            minimum=1,
            maximum=250_000_000,
            default=64_000_000,
            unit="sample",
            description="Largest sample count accepted by one capture call.",
        ),
    ),
)

#: Common configuration defaults matching the station's 5G NR 100 MHz
#: operating point (2026-09-01 baseline: 1.84 GHz, 491.52 MS/s, PA output
#: around +38 dBm, safety limit 39 dBm).
VST5842_RECOMMENDED_CONFIG = DeviceConfig(
    center_frequency_hz=1.84e9,
    sample_rate_hz=491.52e6,
    average_segment_count=16,
    target_power_dbm=38.0,
    safety_power_limit_dbm=39.0,
    initial_attenuation_db=20.0,
    min_attenuation_db=0.0,
    max_attenuation_db=40.0,
    settle_seconds=0.5,
)


def _validate_timeout(timeout_seconds: float) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real):
        raise TypeError("timeout_seconds must be a real number")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout):
        raise ValueError("timeout_seconds must be finite")
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


@dataclass(frozen=True, slots=True)
class _BenchSettings:
    vst_resource: str
    reference_power_dbm: float
    reference_level_dbm: float
    power_meter_resource: str
    power_meter_average: int
    supply_resource: str
    aux_supply_resources: tuple[str, ...]
    enable_supply_shutdown: bool
    enable_supply_interlock: bool
    max_capture_samples: int


def _settings_from_options(options: dict[str, object]) -> _BenchSettings:
    aux = options["aux_supply_resources"]
    if not isinstance(aux, list) or not all(isinstance(item, str) for item in aux):
        raise TypeError("aux_supply_resources must be an array of strings")
    if not aux:
        raise ValueError("aux_supply_resources must not be empty")
    return _BenchSettings(
        vst_resource=str(options["vst_resource"]),
        reference_power_dbm=_finite_real(
            "reference_power_dbm", options["reference_power_dbm"]
        ),
        reference_level_dbm=_finite_real(
            "reference_level_dbm", options["reference_level_dbm"]
        ),
        power_meter_resource=str(options["power_meter_resource"]),
        power_meter_average=int(options["power_meter_average"]),
        supply_resource=str(options["supply_resource"]),
        aux_supply_resources=tuple(aux),
        enable_supply_shutdown=bool(options["enable_supply_shutdown"]),
        enable_supply_interlock=bool(options["enable_supply_interlock"]),
        max_capture_samples=int(options["max_capture_samples"]),
    )


class _Vst5842Instrument(Transmitter, Receiver):
    """Integrated transmitter and receiver backed by one PXIe-5842 VST."""

    def __init__(self, settings: _BenchSettings) -> None:
        self._settings = settings
        self._rfsg: object | None = None
        self._rfsa: object | None = None
        self._connected = False
        self._config: DeviceConfig | None = None
        self._waveform: np.ndarray | None = None
        self._transmitting = False
        self._pretransmit_check: Callable[[float], None] | None = None

    def update_settings(self, settings: _BenchSettings) -> None:
        """Replace adapter settings after a fresh schema validation."""

        self._settings = settings

    def set_pretransmit_check(
        self, check: Callable[[float], None] | None
    ) -> None:
        """Install or clear the read-only interlock run before RF starts."""

        self._pretransmit_check = check

    # -- shared lifecycle -------------------------------------------------

    def connect(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if self._connected:
            raise RuntimeError("VST instrument is already connected")
        import nirfsg  # deferred: driver packages are optional dependencies
        import nirfsa

        self._rfsg = nirfsg.Session(self._settings.vst_resource, reset_device=False)
        try:
            self._rfsa = nirfsa.Session(self._settings.vst_resource, reset_device=False)
        except Exception:
            self._rfsg.close()
            self._rfsg = None
            raise
        self._connected = True

    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        self._require_connected()
        if self._transmitting:
            raise RuntimeError("cannot configure while transmission is running")
        if not isinstance(config, DeviceConfig):
            raise TypeError("config must be a DeviceConfig")

        rfsg = self._rfsg
        rfsa = self._rfsa
        import nirfsg
        import nirfsa

        rfsg.generation_mode = nirfsg.GenerationMode.SCRIPT
        rfsg.frequency = config.center_frequency_hz
        rfsg.iq_rate = config.sample_rate_hz
        rfsg.power_level = (
            self._settings.reference_power_dbm - config.initial_attenuation_db
        )
        rfsg.output_enabled = False

        rfsa.acquisition_type = nirfsa.AcquisitionType.IQ
        rfsa.center_frequency = config.center_frequency_hz
        rfsa.iq_rate = config.sample_rate_hz
        rfsa.reference_level = self._settings.reference_level_dbm
        rfsa.start_trigger_type = nirfsa.StartTriggerType.NONE

        self._config = config
        self._waveform = None
        self._transmitting = False

    def disconnect(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if not self._connected:
            return
        rfsg, rfsa = self._rfsg, self._rfsa
        self._rfsg = None
        self._rfsa = None
        self._connected = False
        self._config = None
        self._waveform = None
        self._transmitting = False
        rfsg.close()
        rfsa.close()

    # -- transmitter ------------------------------------------------------

    def upload_waveform(self, waveform: np.ndarray, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        config = self._require_configured()
        if self._transmitting:
            raise RuntimeError("cannot upload a waveform while transmission is running")
        data = _copy_waveform(waveform)
        self._rfsg.write_arb_waveform(_WAVEFORM_NAME, data)
        self._rfsg.write_script(_CYCLE_SCRIPT)
        self._waveform = data

    def start_transmission(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        config = self._require_configured()
        if self._waveform is None:
            raise RuntimeError("a waveform must be uploaded before transmission starts")
        if self._transmitting:
            raise RuntimeError("transmission is already running")
        if self._pretransmit_check is not None:
            self._pretransmit_check(timeout_seconds)
        self._transmitting = True
        self._rfsg.output_enabled = True
        self._rfsg.initiate()

    def stop_transmission(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        self._require_connected()
        if not self._transmitting:
            return
        self._transmitting = False
        self._rfsg.output_enabled = False
        self._rfsg.abort()

    def get_attenuation_db(self, timeout_seconds: float) -> float:
        _validate_timeout(timeout_seconds)
        self._require_configured()
        return self._settings.reference_power_dbm - self._rfsg.power_level

    def set_attenuation_db(
        self,
        attenuation_db: float,
        timeout_seconds: float,
    ) -> None:
        _validate_timeout(timeout_seconds)
        config = self._require_configured()
        attenuation = _finite_real("attenuation_db", attenuation_db)
        if not config.min_attenuation_db <= attenuation <= config.max_attenuation_db:
            raise ValueError(
                "attenuation_db must be within the configured attenuation range"
            )
        self._rfsg.power_level = self._settings.reference_power_dbm - attenuation

    # -- receiver ---------------------------------------------------------

    @property
    def max_capture_samples(self) -> int:
        return self._settings.max_capture_samples

    def capture(self, request: CaptureRequest, timeout_seconds: float) -> CaptureBatch:
        _validate_timeout(timeout_seconds)
        config, waveform = self._require_transmitting()
        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be a CaptureRequest")
        if request.segment_length != waveform.size:
            raise ValueError(
                "request segment_length must equal the uploaded waveform length"
            )
        if request.sample_count > self._settings.max_capture_samples:
            raise ValueError(
                "requested sample count exceeds max_capture_samples "
                f"({request.sample_count} > {self._settings.max_capture_samples})"
            )
        import hightime

        rfsa = self._rfsa
        rfsa.number_of_samples = request.sample_count
        buffer = np.empty(request.sample_count, dtype=np.complex64)
        rfsa.read_iq_single_record_into(
            buffer, timeout=hightime.timedelta(seconds=timeout_seconds)
        )
        iq = np.asarray(buffer, dtype=np.complex128)
        if not np.all(np.isfinite(iq)):
            raise RuntimeError("captured IQ contains non-finite samples")
        return CaptureBatch(
            iq=iq,
            segment_length=request.segment_length,
            segment_count=request.segment_count,
            sample_rate_hz=config.sample_rate_hz,
            coherent_within_batch=True,
        )

    # -- helpers ----------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("VST instrument is not connected")

    def _require_configured(self) -> DeviceConfig:
        self._require_connected()
        if self._config is None:
            raise RuntimeError("VST instrument is not configured")
        return self._config

    def _require_transmitting(self) -> tuple[DeviceConfig, np.ndarray]:
        config = self._require_configured()
        if self._waveform is None or not self._transmitting:
            raise RuntimeError("feedback capture requires an active transmission")
        return config, self._waveform


class _N1912APowerSensor(PowerSensor):
    """Calibrated PA output power readings from an Agilent N1912A meter."""

    def __init__(self, settings: _BenchSettings) -> None:
        self._settings = settings
        self._resource: object | None = None
        self._connected = False
        self._center_frequency_hz: float | None = None

    def update_settings(self, settings: _BenchSettings) -> None:
        """Replace adapter settings after a fresh schema validation."""

        self._settings = settings

    def connect(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if self._connected:
            raise RuntimeError("power meter is already connected")
        import pyvisa

        manager = pyvisa.ResourceManager()
        self._resource = manager.open_resource(
            self._settings.power_meter_resource,
            timeout=int(_validate_timeout(timeout_seconds) * 1000),
        )
        self._resource.read_termination = "\n"
        self._connected = True

    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if not self._connected:
            raise RuntimeError("power meter is not connected")
        self._center_frequency_hz = config.center_frequency_hz

    def disconnect(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if not self._connected:
            return
        resource, self._resource = self._resource, None
        self._connected = False
        self._center_frequency_hz = None
        resource.close()

    def measure_power_dbm(self, timeout_seconds: float) -> float:
        timeout = _validate_timeout(timeout_seconds)
        if not self._connected:
            raise RuntimeError("power meter is not connected")
        if self._center_frequency_hz is None:
            raise RuntimeError("power meter is not configured")
        average = self._settings.power_meter_average
        # Averaged READ? blocks until all averages complete; keep the sensor
        # sensor-calibration frequency and averaging consistent with the run.
        read_timeout = max(timeout, average * 0.25 + 5.0)
        self._resource.timeout = int(read_timeout * 1000)
        self._resource.write(f"SENS1:FREQ {self._center_frequency_hz:.6E}")
        self._resource.write(f"SENS1:AVER:COUN {average}")
        response = self._resource.query("READ1?")
        value = float(response)
        if not math.isfinite(value):
            raise RuntimeError(f"power meter returned a non-finite value {response!r}")
        return value


class _N5767ASupplyGuard(DeviceCapability):
    """Drain-supply guard enforcing the GaN PA power-safety red lines.

    The guard only ever sends ``OUTP OFF`` to the N5767A drain supply and only
    read-only queries to the E3648A bias supplies. Enabling the 44 V output is
    intentionally impossible from code and remains a manual operator action.
    """

    def __init__(self, settings: _BenchSettings) -> None:
        self._settings = settings
        self._resource: object | None = None
        self._connected = False
        self._opened = False

    def update_settings(self, settings: _BenchSettings) -> None:
        """Replace adapter settings after a fresh schema validation."""

        if self._opened and self._settings.supply_resource != settings.supply_resource:
            raise RuntimeError(
                "cannot change supply_resource while the drain-supply "
                "connection is open"
            )
        self._settings = settings

    def connect(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if self._connected:
            raise RuntimeError("supply guard is already connected")
        import pyvisa

        manager = pyvisa.ResourceManager()
        self._resource = manager.open_resource(
            self._settings.supply_resource,
            timeout=int(_validate_timeout(timeout_seconds) * 1000),
        )
        self._resource.read_termination = "\n"
        self._opened = True
        self._connected = True

    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if not self._connected:
            raise RuntimeError("supply guard is not connected")

    def disconnect(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if not self._connected:
            return
        resource, self._resource = self._resource, None
        self._connected = False
        if resource is not None:
            resource.close()
            self._opened = False

    def check_aux_supplies(self, timeout_seconds: float) -> None:
        """Fail closed unless every bias supply reports output on and stable."""

        timeout_ms = int(_validate_timeout(timeout_seconds) * 1000)
        import pyvisa

        manager = pyvisa.ResourceManager()
        for resource_name in self._settings.aux_supply_resources:
            resource = manager.open_resource(resource_name, timeout=timeout_ms)
            try:
                resource.read_termination = "\n"
                output = resource.query("OUTP?").strip()
                if output != "1":
                    raise RuntimeError(
                        f"bias supply {resource_name} output is not enabled "
                        "(interlock failure); refusing to transmit"
                    )
                setpoint = float(resource.query("VOLT?"))
                measured = float(resource.query("MEAS:VOLT?"))
                if abs(measured - setpoint) > _AUX_VOLTAGE_TOLERANCE_V:
                    raise RuntimeError(
                        f"bias supply {resource_name} voltage deviates from "
                        f"setpoint ({measured:.3f} V vs {setpoint:.3f} V); "
                        "refusing to transmit"
                    )
            finally:
                resource.close()

    def shutdown_drain(self, timeout_seconds: float) -> None:
        """Turn the 44 V drain output off (never on) from safe shutdown."""

        _validate_timeout(timeout_seconds)
        if not self._settings.enable_supply_shutdown:
            return
        if not self._opened or self._resource is None:
            return
        self._resource.write("OUTP OFF")


class Vst5842RFBench(RFBench):
    """Aggregated real bench for the local VST-based closed-loop station."""

    def __init__(self) -> None:
        settings = _settings_from_options(
            VST5842_DEVICE_SCHEMA.validate_options({})
        )
        self._settings = settings
        self._instrument = _Vst5842Instrument(settings)
        self._power_sensor = _N1912APowerSensor(settings)
        self._supply_guard = _N5767ASupplyGuard(settings)
        self._connected = False
        self._config: DeviceConfig | None = None

    @property
    def transmitter(self) -> Transmitter:
        return self._instrument

    @property
    def receiver(self) -> Receiver:
        return self._instrument

    @property
    def power_sensor(self) -> PowerSensor:
        return self._power_sensor

    @property
    def parameter_schema(self) -> DeviceParameterSchema:
        return VST5842_DEVICE_SCHEMA

    def connect(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if self._connected:
            raise RuntimeError("vst5842 RF bench is already connected")
        self._instrument.connect(timeout_seconds)
        try:
            self._power_sensor.connect(timeout_seconds)
            self._supply_guard.connect(timeout_seconds)
        except Exception:
            self._instrument.disconnect(timeout_seconds)
            raise
        self._connected = True

    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if not self._connected:
            raise RuntimeError("vst5842 RF bench is not connected")
        if not isinstance(config, DeviceConfig):
            raise TypeError("config must be a DeviceConfig")
        options = VST5842_DEVICE_SCHEMA.validate_options(config.device_options)
        effective_values = config.to_dict()
        effective_values["device_options"] = options
        effective_config = DeviceConfig(**effective_values)
        settings = _settings_from_options(options)

        self._instrument.stop_transmission(timeout_seconds)
        self._power_sensor.configure(effective_config, timeout_seconds)
        self._supply_guard.configure(effective_config, timeout_seconds)
        self._power_sensor.update_settings(settings)
        self._supply_guard.update_settings(settings)
        self._instrument.update_settings(settings)
        self._instrument.set_pretransmit_check(
            self._supply_guard.check_aux_supplies
            if settings.enable_supply_interlock
            else None
        )
        self._instrument.configure(effective_config, timeout_seconds)
        self._settings = settings
        self._config = effective_config

    def safe_shutdown(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if not self._connected:
            return
        self._instrument.stop_transmission(timeout_seconds)
        self._supply_guard.shutdown_drain(timeout_seconds)

    def disconnect(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if not self._connected:
            return
        self.safe_shutdown(timeout_seconds)
        self._supply_guard.disconnect(timeout_seconds)
        self._power_sensor.disconnect(timeout_seconds)
        self._instrument.disconnect(timeout_seconds)
        self._connected = False
        self._config = None


def _create_vst5842_bench() -> RFBench:
    return Vst5842RFBench()


register_rf_bench("vst5842", _create_vst5842_bench)
