"""Real RF bench adapter for the local NI RFIC test station.

The bench drives one NI PXIe-5842 VST through the NI RFIC SCPI server (the
loopback VXI-11 resource also used by InstrumentStudio and the station's
MATLAB client), plus one Agilent N1912A power meter and one Agilent N5767A
drain supply guard, all exposed through the standard capability contracts in
``remote_dpd.device`` and registered under ``vst5842``.

Why SCPI instead of the nirfsg/nirfsa driver API: on the 5842+5655 system
the receiver LO and downconverter resources are owned by the NI RFIC
software stack, and bare driver sessions cannot reliably reserve them
(persistent LO states and cross-switch conflicts observed on this station).
The RFIC SCPI server is the supported sharing path, so both transmit and
receive go through it:

- Transmitter: RFSG ARB playback of a TDMS waveform written into the RFIC
  SCPI Waveforms directory; TX attenuation maps to
  ``reference_power_dbm - attenuation`` at the RFSG power level.
- Receiver: RFmx SpecAn IQ acquisitions; the trace block returned by
  ``FETCh:SPECan:RESult:IQ:TRACe:DATA?`` is big-endian float32
  interleaved I/Q and is decoded accordingly.

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
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

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
)
from .preprocessing import CaptureBatch

_DEFAULT_WAVEFORMS_DIRECTORY = (
    r"C:\Users\Public\Documents\National Instruments\RFIC SCPI\Waveforms"
)
_WAVEFORM_FILE_NAME = "rdpd_wave.tdms"
_TDMS_GROUP_NAME = "waveforms"
_TDMS_CHANNEL_NAME = "Channel 0"
#: The SCPI server rounds the acquisition time to whole samples; a few
#: guard samples make sure the decoded trace never comes up short.
_CAPTURE_GUARD_SAMPLES = 32
#: Fixed overhead observed between INITiate:SPECan and a fetchable result.
_CAPTURE_SETTLE_SECONDS = 2.0
_DEFAULT_AUX_SUPPLY_RESOURCES = ("GPIB1::7::INSTR", "GPIB1::8::INSTR")
_AUX_VOLTAGE_TOLERANCE_V = 0.5
#: The SCPI server splices the waveform name into its playback script, so
#: only plain alphanumerics are accepted; matching is case-insensitive.
_WAVEFORM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9]+$")

VST5842_DEVICE_SCHEMA = DeviceParameterSchema(
    device_type="vst5842",
    schema_version=2,
    fields=(
        DeviceParameterField(
            name="scpi_resource",
            value_type=DeviceParameterType.STRING,
            default="TCPIP0::127.0.0.1::inst0::INSTR",
            description=(
                "VISA resource of the NI RFIC SCPI server loopback session "
                "that owns the PXIe-5842 transmit and receive chains."
            ),
        ),
        DeviceParameterField(
            name="instrument_config_file",
            value_type=DeviceParameterType.STRING,
            default="Instrument_2_PXI2Slot2.rfmxconfig",
            description=(
                "InstrumentStudio export (receiver ACP+IQ setup, reference "
                "level and external attenuation baseline) loaded on configure."
            ),
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
            maximum=120.0,
            default=50.0,
            unit="dBm",
            description=(
                "SpecAn reference level at the PA-output scale, matching the "
                "station GUI (InstrumentStudio displays 50.0 dBm)."
            ),
        ),
        DeviceParameterField(
            name="external_attenuation_db",
            value_type=DeviceParameterType.NUMBER,
            minimum=0.0,
            maximum=100.0,
            default=53.5,
            unit="dB",
            description=(
                "Attenuation between the PA output and the VST RF input; "
                "applied as the SpecAn external attenuation so decoded "
                "traces stay on the PA-output scale."
            ),
        ),
        DeviceParameterField(
            name="waveform_name",
            value_type=DeviceParameterType.STRING,
            default="RDPD1",
            description=(
                "Server-side waveform name for ARB playback. Alphanumeric "
                "only (the server embeds the name in its playback script, "
                "so underscores and symbols are rejected) and matched "
                "case-insensitively; the adapter upper-cases it."
            ),
        ),
        DeviceParameterField(
            name="waveforms_directory",
            value_type=DeviceParameterType.STRING,
            default=_DEFAULT_WAVEFORMS_DIRECTORY,
            description=(
                "RFIC SCPI Waveforms directory; uploaded TDMS files are "
                "written here before MMEMory:LOAD:WAVeform."
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
            default=False,
            description=(
                "Turn the 44 V drain output off during safe shutdown. Defaults "
                "to false: the drain supply is opened and closed manually by "
                "the operator, so runs only stop RF unless this is enabled."
            ),
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


def _write_tdms_waveform(
    path: Path,
    waveform: np.ndarray,
    sample_rate_hz: float,
) -> None:
    """Write one NI RFmx-style InterleavedIQCluster TDMS waveform file."""

    # Deferred: nptdms belongs to the optional real-hardware dependency group.
    from nptdms import ChannelObject, GroupObject, TdmsWriter

    interleaved = np.empty(waveform.size * 2, dtype=np.float32)
    interleaved[0::2] = waveform.real
    interleaved[1::2] = waveform.imag
    magnitude = np.abs(waveform)
    peak = float(np.max(magnitude))
    rms = float(np.sqrt(np.mean(magnitude**2)))
    papr_db = 20.0 * math.log10(peak / rms) if rms > 0.0 else 0.0
    group = GroupObject(
        _TDMS_GROUP_NAME,
        {
            "description": "",
            "Application": "NI-RFmx Waveform Creator",
            "NI_RF_WaveformFileVersion": "2.0.0",
        },
    )
    channel = ChannelObject(
        _TDMS_GROUP_NAME,
        _TDMS_CHANNEL_NAME,
        interleaved,
        {
            "description": "",
            "unit_string": "",
            "NI_RF_IQRate": float(sample_rate_hz),
            "NI_RF_SignalBandwidth": 0.8 * float(sample_rate_hz),
            "NI_RF_WaveformType": "InterleavedIQCluster",
            "NI_RF_PAPR": papr_db,
            "NI_RF_RuntimeScaling": 0.0,
            # Native RFmx waveform files carry the complex-sample period here
            # (NI_RF_IQRate^-1) even though the channel stores interleaved
            # float32; MEMory:WAVeform:DX? echoes this value.
            "dt": 1.0 / float(sample_rate_hz),
            "t0": 0.0,
        },
    )
    with TdmsWriter(str(path)) as writer:
        writer.write_segment([group, channel])


@dataclass(frozen=True, slots=True)
class _BenchSettings:
    scpi_resource: str
    instrument_config_file: str
    reference_power_dbm: float
    reference_level_dbm: float
    external_attenuation_db: float
    waveform_name: str
    waveforms_directory: str
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
    waveform_name = str(options["waveform_name"])
    if not _WAVEFORM_NAME_PATTERN.fullmatch(waveform_name):
        raise ValueError(
            "waveform_name must contain only ASCII letters and digits "
            "(the SCPI server embeds it in its playback script)"
        )
    return _BenchSettings(
        scpi_resource=str(options["scpi_resource"]),
        instrument_config_file=str(options["instrument_config_file"]),
        reference_power_dbm=_finite_real(
            "reference_power_dbm", options["reference_power_dbm"]
        ),
        reference_level_dbm=_finite_real(
            "reference_level_dbm", options["reference_level_dbm"]
        ),
        external_attenuation_db=_finite_real(
            "external_attenuation_db", options["external_attenuation_db"]
        ),
        waveform_name=waveform_name.upper(),
        waveforms_directory=str(options["waveforms_directory"]),
        power_meter_resource=str(options["power_meter_resource"]),
        power_meter_average=int(options["power_meter_average"]),
        supply_resource=str(options["supply_resource"]),
        aux_supply_resources=tuple(aux),
        enable_supply_shutdown=bool(options["enable_supply_shutdown"]),
        enable_supply_interlock=bool(options["enable_supply_interlock"]),
        max_capture_samples=int(options["max_capture_samples"]),
    )


class _Vst5842Instrument(Transmitter, Receiver):
    """Integrated transmitter and receiver on one RFIC SCPI server session."""

    def __init__(self, settings: _BenchSettings) -> None:
        self._settings = settings
        self._resource: object | None = None
        self._connected = False
        self._config: DeviceConfig | None = None
        self._waveform: np.ndarray | None = None
        self._transmitting = False
        self._generation_initiated = False
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
        timeout = _validate_timeout(timeout_seconds)
        if self._connected:
            raise RuntimeError("VST instrument is already connected")
        import pyvisa  # deferred: driver packages are optional dependencies

        try:
            manager = pyvisa.ResourceManager()
            resource = manager.open_resource(
                self._settings.scpi_resource, timeout=int(timeout * 1000)
            )
            try:
                resource.read_termination = "\n"
                identity = resource.query("*IDN?")
            except Exception:
                resource.close()
                raise
        except Exception as exc:
            raise RuntimeError(
                "RFIC SCPI server is unreachable at "
                f"{self._settings.scpi_resource!r}; start "
                "'ni_grpc_device_server.exe server_config.json' first and "
                "'NIRficScpiServer.exe' second (see docs/real_bench_design.md)"
            ) from exc
        if not identity.strip():
            resource.close()
            raise RuntimeError("RFIC SCPI server returned an empty *IDN?")
        self._resource = resource
        self._connected = True

    def configure(self, config: DeviceConfig, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        self._require_connected()
        if self._transmitting:
            raise RuntimeError("cannot configure while transmission is running")
        if not isinstance(config, DeviceConfig):
            raise TypeError("config must be a DeviceConfig")

        settings = self._settings
        self._command("SOURce:RFSG:OUTPut:ENABled 0")
        # A generation task left running by a previous session (or an aborted
        # run) refuses GMODe/SELect; abort unconditionally to get a clean
        # state. Aborting an idle task is a no-op.
        self._command("ABORt:RFSG")
        self._command(
            f'MMEMory:INSTr:LOAD:STATe "{settings.instrument_config_file}",1'
        )
        self._command(f"SOURce:RFSG:FREQuency {config.center_frequency_hz:.9E}")
        self._command("SOURce:RFSG:GMODe ARBWAVEFORM")
        self._command(
            "SOURce:RFSG:POWer:LEVel "
            f"{settings.reference_power_dbm - config.initial_attenuation_db:.6f}"
        )
        self._command(
            f"CONFigure:SPECan:FREQuency {config.center_frequency_hz:.9E}"
        )
        self._command(
            f"CONFigure:SPECan:RLEVel {settings.reference_level_dbm:.6f}"
        )
        self._command(
            f"CONFigure:SPECan:EATTenuation {settings.external_attenuation_db:.6f}"
        )

        self._config = config
        self._waveform = None
        self._transmitting = False
        # Reloading the instrument configuration resets the generator.
        self._generation_initiated = False

    def disconnect(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        if not self._connected:
            return
        # Leave the generation task aborted so a later session (or another
        # station client) can reconfigure the instrument.
        self._command("ABORt:RFSG")
        self._generation_initiated = False
        resource, self._resource = self._resource, None
        self._connected = False
        self._config = None
        self._waveform = None
        self._transmitting = False
        resource.close()

    # -- transmitter ------------------------------------------------------

    def upload_waveform(self, waveform: np.ndarray, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        config = self._require_configured()
        if self._transmitting:
            raise RuntimeError("cannot upload a waveform while transmission is running")
        data = _copy_waveform(waveform)

        settings = self._settings
        if self._generation_initiated:
            # The server refuses ARB reselection while the generation task is
            # running, so abort it; the next start re-initiates.
            self._command("ABORt:RFSG")
            self._generation_initiated = False
        target = Path(settings.waveforms_directory) / _WAVEFORM_FILE_NAME
        _write_tdms_waveform(target, data, config.sample_rate_hz)
        name = settings.waveform_name
        self._command(f'MMEMory:LOAD:WAVeform "{target}", "{name}", 0')
        # Binding the cached waveform into the RFSG ARB memory is what makes
        # INITiate:RFSG accept the name; loading into the cache alone is not
        # enough (verified against the station server, 2026-09-02).
        self._command(f'SOURce:RFSG:LOAD:WAVeform:MEMory "{name}"')
        self._command(f'SOURce:RFSG:WAVeform:REPeat:MODE "{name}", CONTINUOUS')
        self._command(f'SOURce:RFSG:ARB:WAVeform:SELect "{name}"')
        sample_period = float(
            self._query(f'MEMory:WAVeform:DX? "{settings.waveform_name}"')
        )
        expected_period = 1.0 / config.sample_rate_hz
        if not math.isclose(
            sample_period, expected_period, rel_tol=1e-6, abs_tol=1e-15
        ):
            raise RuntimeError(
                "loaded waveform sample period does not match the configured "
                f"sample rate ({sample_period:.9E} s vs {expected_period:.9E} s)"
            )
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
        if not self._generation_initiated:
            self._command("INITiate:RFSG")
            self._generation_initiated = True
        self._command("SOURce:RFSG:OUTPut:ENABled 1")

    def stop_transmission(self, timeout_seconds: float) -> None:
        _validate_timeout(timeout_seconds)
        self._require_connected()
        if not self._transmitting:
            return
        self._transmitting = False
        # Keep the generation running unloaded; only the RF output is gated
        # so a stopped bench can never radiate and a restart is one command.
        self._command("SOURce:RFSG:OUTPut:ENABled 0")

    def get_attenuation_db(self, timeout_seconds: float) -> float:
        _validate_timeout(timeout_seconds)
        self._require_configured()
        level = float(self._query("SOURce:RFSG:POWer:LEVel?"))
        return self._settings.reference_power_dbm - level

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
        self._command(
            f"SOURce:RFSG:POWer:LEVel "
            f"{self._settings.reference_power_dbm - attenuation:.6f}"
        )

    # -- receiver ---------------------------------------------------------

    @property
    def max_capture_samples(self) -> int:
        return self._settings.max_capture_samples

    def capture(self, request: CaptureRequest, timeout_seconds: float) -> CaptureBatch:
        timeout = _validate_timeout(timeout_seconds)
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

        sample_count = request.sample_count
        acquisition_seconds = (sample_count + _CAPTURE_GUARD_SAMPLES) / (
            config.sample_rate_hz
        )
        self._command("CONFigure:SPECan:MEASurement:SELect 1,IQ")
        self._command(f"CONFigure:SPECan:IQ:ACQuisition:TIME {acquisition_seconds:.9E}")
        self._command(f"CONFigure:SPECan:IQ:SRATe {config.sample_rate_hz:.9E}")
        self._command("INITiate:SPECan")
        time.sleep(acquisition_seconds + _CAPTURE_SETTLE_SECONDS)

        iq = self._fetch_iq_trace(timeout)
        if iq.size < sample_count:
            raise RuntimeError(
                "SpecAn IQ trace is shorter than requested "
                f"({iq.size} < {sample_count} samples); check that the "
                "transmission is running"
            )
        iq = np.ascontiguousarray(iq[:sample_count])
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

    def _command(self, command: str) -> None:
        """Send one command and fail on the first queued SCPI error."""

        self._resource.write(command)
        response = self._resource.query("SYSTem:ERRor?")
        try:
            code = int(response.split(",", 1)[0])
        except ValueError as exc:
            raise RuntimeError(
                f"malformed SYSTem:ERRor? response {response!r} after {command!r}"
            ) from exc
        if code != 0:
            raise RuntimeError(f"SCPI command {command!r} failed: {response.strip()}")

    def _query(self, command: str) -> str:
        return self._resource.query(command)

    def _fetch_iq_trace(self, timeout_seconds: float) -> np.ndarray:
        """Fetch one IQ trace and decode the big-endian float32 block."""

        resource = self._resource
        resource.timeout = int(timeout_seconds * 1000)
        resource.write("FETCh:SPECan:RESult:IQ:TRACe:DATA?")
        # Binary blocks contain 0x0A bytes; read raw with termination off.
        termination = resource.read_termination
        resource.read_termination = None
        try:
            raw = resource.read_raw()
        finally:
            resource.read_termination = termination
        marker = raw.find(b"#")
        if marker < 0:
            raise RuntimeError(
                f"SpecAn IQ response is not an IEEE block ({raw[:32]!r})"
            )
        digits = int(raw[marker + 1 : marker + 2])
        length = int(raw[marker + 2 : marker + 2 + digits])
        payload = raw[marker + 2 + digits : marker + 2 + digits + length]
        values = np.frombuffer(payload, dtype=">f4")
        usable = 2 * (values.size // 2)
        if usable == 0:
            raise RuntimeError(
                "SpecAn IQ trace is empty; check that the transmission is running"
            )
        return (
            values[:usable:2].astype(np.float64)
            + 1j * values[1:usable:2].astype(np.float64)
        )

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
        # The N1912A free-runs its averaging in continuous-init mode; READ?
        # conflicts with it and never completes, so settings are applied, one
        # averaging cycle is allowed to settle, and the latest finished
        # average is fetched with FETC1?.
        self._resource.timeout = int(timeout * 1000)
        self._resource.write(f"SENS1:FREQ {self._center_frequency_hz:.6E}")
        self._resource.write(f"SENS1:AVER:COUN {average}")
        self._resource.write("INIT1:CONT ON")
        time.sleep(min(average * 0.06 + 1.0, 10.0))
        response = self._resource.query("FETC1?")
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

    def quick_start_configuration(self) -> dict[str, object]:
        """One-click Web profile for the station's verified operating point.

        Values follow the 2026-09-02 closed-loop smoke plus the operator's
        confirmed decisions: target power at the +38 dBm working point, three
        ILC iterations of eight averaged segments at mu 0.1, and the 44 V
        drain supply left under manual operator control.
        """

        return {
            "device_type": "vst5842",
            "device_config": {
                "center_frequency_hz": 1.84e9,
                "sample_rate_hz": 491.52e6,
                "average_segment_count": 8,
                "target_power_dbm": 38.0,
                "safety_power_limit_dbm": 39.0,
                "initial_attenuation_db": 22.0,
                "min_attenuation_db": 0.0,
                "max_attenuation_db": 40.0,
                "settle_seconds": 0.5,
                "call_timeout_seconds": 90.0,
                "device_options": {
                    "enable_supply_shutdown": False,
                    "power_meter_average": 8,
                },
            },
            "normalize_reference_rms": True,
            "reference_target_rms_dbfs": -15.0,
            "runtime_name": "basic_ilc",
            "runtime_config": {"mu": 0.1},
            "max_iterations": 3,
        }
