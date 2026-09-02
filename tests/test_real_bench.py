import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from remote_dpd.device import (
    CaptureRequest,
    DeviceConfig,
    PowerSensor,
    Receiver,
    RFBench,
    Transmitter,
    create_rf_bench,
    list_rf_benches,
)
from remote_dpd.real_bench import (
    VST5842_DEVICE_SCHEMA,
    VST5842_RECOMMENDED_CONFIG,
    Vst5842RFBench,
)

TIMEOUT = 1.0
WAVEFORM_LENGTH = 64
SCPI_RESOURCE = "TCPIP0::127.0.0.1::inst0::INSTR"
WAVEFORM_NAME = "RDPD1"

# Commands that are legitimate read-only queries against the E3648A bias
# supplies; the power-safety red line forbids every write to those resources.
AUX_QUERY_WHITELIST = ("*IDN?", "OUTP?", "VOLT?", "MEAS:VOLT?", "MEAS:CURR?")


class FakeScpiResource:
    """Loopback SCPI session recording traffic with programmable responses."""

    def __init__(self, resource_name):
        self.resource_name = resource_name
        self.commands = []
        self.timeout = 2000
        self.read_termination = "\n"
        self.closed = False
        self.idn = "National Instruments,RFIC SCPI Server"
        self.idn_error = None
        self.error_queue = []
        self.state = {"power_level": -37.0}
        self.responses = {}
        self.fetch_payload = b""
        self.read_raw_terminations = []

    def write(self, command):
        self.commands.append(("write", command))
        if command.startswith("SOURce:RFSG:POWer:LEVel "):
            self.state["power_level"] = float(command.rsplit(" ", 1)[1])

    def query(self, command):
        self.commands.append(("query", command))
        if command == "*IDN?":
            if self.idn_error is not None:
                raise self.idn_error
            return self.idn
        if command == "SYSTem:ERRor?":
            if self.error_queue:
                return self.error_queue.pop(0)
            return '+0,"No error"'
        if command == "SOURce:RFSG:POWer:LEVel?":
            return f"{self.state['power_level']:.6E}"
        key = command.strip()
        if key in self.responses:
            return self.responses[key]
        raise RuntimeError(f"unexpected query {command!r}")

    def read_raw(self):
        self.read_raw_terminations.append(self.read_termination)
        payload = self.fetch_payload
        length = len(payload)
        header = b"#" + str(len(str(length))).encode() + str(length).encode()
        return b"0.0,+2.034505208E-09," + header + payload

    def close(self):
        self.closed = True

    def write_texts(self):
        return [command for kind, command in self.commands if kind == "write"]


class FakeVisaResource:
    def __init__(self, resource_name, behavior):
        self.resource_name = resource_name
        self.behavior = behavior
        self.commands = []
        self.timeout = 2000
        self.read_termination = None
        self.closed = False

    def write(self, command):
        self.commands.append(("write", command))
        if self.behavior is not None:
            self.behavior.on_write(command)

    def query(self, command):
        self.commands.append(("query", command))
        if self.behavior is None:
            raise RuntimeError(f"no behavior for {self.resource_name}")
        return self.behavior.on_query(command)

    def close(self):
        self.closed = True


class FakeResourceManager:
    def __init__(self):
        self.behaviors = {}
        self.objects = {}
        self.sessions = []

    def define(self, resource_name, responses=None):
        self.behaviors[resource_name] = InstrumentBehavior(responses or {})

    def define_object(self, resource_name, resource):
        self.objects[resource_name] = resource

    def open_resource(self, resource_name, timeout=None):
        if resource_name in self.objects:
            resource = self.objects[resource_name]
        elif resource_name in self.behaviors:
            resource = FakeVisaResource(resource_name, self.behaviors[resource_name])
        else:
            raise RuntimeError(f"resource {resource_name!r} not available")
        self.sessions.append(resource)
        return resource


class InstrumentBehavior:
    def __init__(self, responses):
        self.responses = responses
        self.writes = []

    def on_write(self, command):
        self.writes.append(command)

    def on_query(self, command):
        key = command.strip()
        if key in self.responses:
            return self.responses[key]
        raise RuntimeError(f"unexpected query {command!r}")


def make_fake_module_pyvisa(manager):
    module = types.ModuleType("pyvisa")
    module.ResourceManager = lambda: manager
    return module


def make_fake_module_nptdms(recorder):
    module = types.ModuleType("nptdms")

    class TdmsWriter:
        def __init__(self, path):
            self.path = path
            self.segments = []
            recorder["writers"].append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write_segment(self, objects=None):
            self.segments.append(list(objects or []))

    class GroupObject:
        def __init__(self, name, properties=None):
            self.name = name
            self.properties = dict(properties or {})

    class ChannelObject:
        def __init__(self, group_name, channel_name, data, properties=None):
            self.group_name = group_name
            self.channel_name = channel_name
            self.data = np.asarray(data)
            self.properties = dict(properties or {})

    module.TdmsWriter = TdmsWriter
    module.GroupObject = GroupObject
    module.ChannelObject = ChannelObject
    return module


def default_aux_responses():
    return {
        "*IDN?": "Agilent Technologies,E3648A,0,2.5-6.1-2.1",
        "OUTP?": "1",
        "VOLT?": "+8.00000000E+00",
        "MEAS:VOLT?": "+7.99860800E+00",
    }


def make_fetch_payload(samples):
    """Build the big-endian float32 interleaved I/Q block the server sends."""

    interleaved = np.empty(2 * samples.size, dtype=">f4")
    interleaved[0::2] = samples.real
    interleaved[1::2] = samples.imag
    return interleaved.tobytes()


def _nptdms_available():
    # The fake module is only installed during RealBenchTestCase runs; probe
    # the real environment here so round-trip tests skip cleanly without it.
    try:
        return importlib.util.find_spec("nptdms") is not None
    except (ImportError, ValueError):
        return False


class RealBenchTestCase(unittest.TestCase):
    """Base class injecting fake VISA / nptdms modules before each test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.waveforms_dir = Path(self.tmp.name)

        self.scpi = FakeScpiResource(SCPI_RESOURCE)
        self.scpi.responses = {
            f'MEMory:WAVeform:DX? "{WAVEFORM_NAME}"': repr(1.0 / 491.52e6),
        }
        self.visa = FakeResourceManager()
        self.visa.define_object(SCPI_RESOURCE, self.scpi)
        self.visa.define(
            "TCPIP0::192.168.255.40::inst0::INSTR", {"FETC1?": "+3.7960E+001"}
        )
        self.visa.define("GPIB1::5::INSTR", {"OUTP?": "1"})
        self.visa.define("GPIB1::7::INSTR", default_aux_responses())
        self.visa.define("GPIB1::8::INSTR", default_aux_responses())

        self.nptdms_recorder = {"writers": []}
        self._saved_modules = {}
        for name, module in (
            ("pyvisa", make_fake_module_pyvisa(self.visa)),
            ("nptdms", make_fake_module_nptdms(self.nptdms_recorder)),
        ):
            self._saved_modules[name] = sys.modules.get(name)
            sys.modules[name] = module

        sleep_patcher = mock.patch(
            "remote_dpd.real_bench.time.sleep", lambda seconds: None
        )
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def tearDown(self):
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def make_config(self, **overrides):
        values = {
            "center_frequency_hz": 1.84e9,
            "sample_rate_hz": 491.52e6,
            "average_segment_count": 4,
            "target_power_dbm": 38.0,
            "safety_power_limit_dbm": 39.0,
            "initial_attenuation_db": 20.0,
            "min_attenuation_db": 0.0,
            "max_attenuation_db": 40.0,
            "device_options": {
                "waveforms_directory": str(self.waveforms_dir),
            },
        }
        device_options = values["device_options"]
        device_options.update(overrides.pop("device_options", {}))
        values.update(overrides)
        return DeviceConfig(**values)

    def make_waveform(self):
        phases = np.linspace(0.0, 2.0 * np.pi, WAVEFORM_LENGTH, endpoint=False)
        return 0.25 * (np.cos(phases) + 1j * np.sin(phases))

    def connect_and_configure(self, bench, config=None):
        bench.connect(TIMEOUT)
        bench.configure(self.make_config() if config is None else config, TIMEOUT)
        return bench

    def start_transmitting(self, bench, waveform=None):
        waveform = self.make_waveform() if waveform is None else waveform
        bench.transmitter.upload_waveform(waveform, TIMEOUT)
        bench.transmitter.start_transmission(TIMEOUT)
        return waveform

    def sessions_for(self, prefix):
        return [s for s in self.visa.sessions if s.resource_name.startswith(prefix)]

    def scpi_writes(self):
        return self.scpi.write_texts()


class Vst5842RegistryAndSchemaTests(RealBenchTestCase):
    def test_registry_contains_vst5842(self):
        self.assertIn("vst5842", list_rf_benches())
        bench = create_rf_bench("vst5842")
        self.assertIsInstance(bench, Vst5842RFBench)
        self.assertIsInstance(bench, RFBench)
        self.assertIsInstance(bench.transmitter, Transmitter)
        self.assertIsInstance(bench.receiver, Receiver)
        self.assertIsInstance(bench.power_sensor, PowerSensor)
        self.assertIs(bench.transmitter, bench.receiver)

    def test_quick_start_profile_matches_smoke_verified_operating_point(self):
        profile = Vst5842RFBench().quick_start_configuration()
        self.assertEqual(profile["device_type"], "vst5842")
        common = profile["device_config"]
        self.assertEqual(common["center_frequency_hz"], 1.84e9)
        self.assertEqual(common["sample_rate_hz"], 491.52e6)
        self.assertEqual(common["target_power_dbm"], 38.0)
        self.assertEqual(common["safety_power_limit_dbm"], 39.0)
        self.assertEqual(common["initial_attenuation_db"], 22.0)
        self.assertEqual(common["call_timeout_seconds"], 90.0)
        self.assertEqual(common["average_segment_count"], 8)
        self.assertEqual(profile["max_iterations"], 3)
        self.assertEqual(profile["runtime_config"], {"mu": 0.1})
        self.assertFalse(common["device_options"]["enable_supply_shutdown"])
        self.assertEqual(common["device_options"]["power_meter_average"], 8)

    def test_schema_defaults_fill_and_unknown_options_rejected(self):
        options = VST5842_DEVICE_SCHEMA.validate_options({})
        self.assertEqual(options["scpi_resource"], SCPI_RESOURCE)
        self.assertEqual(
            options["instrument_config_file"], "Instrument_2_PXI2Slot2.rfmxconfig"
        )
        self.assertEqual(options["reference_power_dbm"], -17.0)
        self.assertEqual(options["reference_level_dbm"], 50.0)
        self.assertEqual(options["external_attenuation_db"], 53.5)
        self.assertEqual(options["waveform_name"], WAVEFORM_NAME)
        self.assertEqual(options["power_meter_average"], 64)
        self.assertFalse(options["enable_supply_shutdown"])
        self.assertTrue(options["enable_supply_interlock"])
        self.assertEqual(
            options["aux_supply_resources"],
            ["GPIB1::7::INSTR", "GPIB1::8::INSTR"],
        )
        with self.assertRaises(ValueError):
            VST5842_DEVICE_SCHEMA.validate_options({"unknown_option": 1})
        # The bare-driver option was removed with the SCPI rewrite.
        with self.assertRaises(ValueError):
            VST5842_DEVICE_SCHEMA.validate_options({"vst_resource": "PXI2Slot2"})

    def test_waveform_name_charset_enforced_and_upper_cased(self):
        from remote_dpd.real_bench import _settings_from_options

        for invalid in ("rpd_low", "wave 1", "x;y", 'q"'):
            with self.assertRaises(ValueError):
                _settings_from_options(
                    VST5842_DEVICE_SCHEMA.validate_options(
                        {"waveform_name": invalid}
                    )
                )
        settings = _settings_from_options(
            VST5842_DEVICE_SCHEMA.validate_options({"waveform_name": "rdpd1"})
        )
        self.assertEqual(settings.waveform_name, "RDPD1")

    def test_recommended_config_matches_station_operating_point(self):
        self.assertEqual(VST5842_RECOMMENDED_CONFIG.center_frequency_hz, 1.84e9)
        self.assertEqual(VST5842_RECOMMENDED_CONFIG.sample_rate_hz, 491.52e6)
        self.assertEqual(VST5842_RECOMMENDED_CONFIG.safety_power_limit_dbm, 39.0)


class Vst5842ConnectTests(RealBenchTestCase):
    def test_connect_probes_scpi_server_identity(self):
        bench = Vst5842RFBench()
        bench.connect(TIMEOUT)
        self.assertIn(("query", "*IDN?"), self.scpi.commands)
        self.assertFalse(self.scpi.closed)
        bench.disconnect(TIMEOUT)
        self.assertTrue(self.scpi.closed)

    def test_connect_failure_mentions_service_startup_order(self):
        self.scpi.idn_error = RuntimeError("connection refused")
        bench = Vst5842RFBench()
        with self.assertRaises(RuntimeError) as ctx:
            bench.connect(TIMEOUT)
        message = str(ctx.exception)
        self.assertIn("ni_grpc_device_server", message)
        self.assertIn("NIRficScpiServer", message)

    def test_connect_failure_when_resource_missing(self):
        self.visa.objects.pop(SCPI_RESOURCE)
        bench = Vst5842RFBench()
        with self.assertRaises(RuntimeError) as ctx:
            bench.connect(TIMEOUT)
        self.assertIn("ni_grpc_device_server", str(ctx.exception))


class Vst5842ConfigureTests(RealBenchTestCase):
    def test_configure_sends_full_scpi_sequence(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        writes = self.scpi_writes()
        expected = [
            "SOURce:RFSG:OUTPut:ENABled 0",
            "ABORt:RFSG",
            'MMEMory:INSTr:LOAD:STATe "Instrument_2_PXI2Slot2.rfmxconfig",1',
            "SOURce:RFSG:FREQuency 1.840000000E+09",
            "SOURce:RFSG:GMODe ARBWAVEFORM",
            "SOURce:RFSG:POWer:LEVel -37.000000",
            "CONFigure:SPECan:FREQuency 1.840000000E+09",
            "CONFigure:SPECan:RLEVel 50.000000",
            "CONFigure:SPECan:EATTenuation 53.500000",
        ]
        for command in expected:
            self.assertIn(command, writes)
        first_index = writes.index(expected[0])
        last_index = writes.index(expected[-1])
        ordered = writes[first_index : last_index + 1]
        self.assertEqual(ordered, expected)
        bench.disconnect(TIMEOUT)

    def test_configure_polls_scpi_error_after_each_command(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        commands = self.scpi.commands
        write_indexes = [
            index for index, (kind, _) in enumerate(commands) if kind == "write"
        ]
        for index in write_indexes:
            self.assertEqual(commands[index + 1], ("query", "SYSTem:ERRor?"))
        bench.disconnect(TIMEOUT)

    def test_configure_propagates_scpi_errors(self):
        self.scpi.error_queue.append('-113,"Undefined header"')
        bench = Vst5842RFBench()
        bench.connect(TIMEOUT)
        with self.assertRaises(RuntimeError) as ctx:
            bench.configure(self.make_config(), TIMEOUT)
        self.assertIn("-113", str(ctx.exception))
        self.assertIn("SOURce:RFSG:OUTPut:ENABled 0", str(ctx.exception))

    def test_configure_requires_connection_and_rejects_bad_options(self):
        bench = Vst5842RFBench()
        with self.assertRaises(RuntimeError):
            bench.configure(self.make_config(), TIMEOUT)
        bench.connect(TIMEOUT)
        with self.assertRaises((ValueError, TypeError)):
            bench.configure(
                self.make_config(device_options={"scpi_resource": 123}), TIMEOUT
            )

    def test_configure_while_transmitting_stops_rf_first(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        bench.configure(self.make_config(), TIMEOUT)
        writes = self.scpi_writes()
        output_off_index = writes.index("SOURce:RFSG:OUTPut:ENABled 0")
        load_index = writes.index(
            'MMEMory:INSTr:LOAD:STATe "Instrument_2_PXI2Slot2.rfmxconfig",1'
        )
        # The bench-level configure stops RF before reconfiguring, and the
        # reload resets generation so the next start re-initiates.
        self.assertLess(output_off_index, load_index)
        self.start_transmitting(bench)
        self.assertEqual(self.scpi_writes().count("INITiate:RFSG"), 2)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)


class Vst5842WaveformUploadTests(RealBenchTestCase):
    def uploaded_channel(self):
        writers = self.nptdms_recorder["writers"]
        self.assertEqual(len(writers), 1)
        group, channel = writers[0].segments[0]
        return group, channel

    def test_upload_writes_tdms_and_loads_waveform(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        waveform = self.make_waveform()
        bench.transmitter.upload_waveform(waveform, TIMEOUT)

        group, channel = self.uploaded_channel()
        self.assertEqual(group.name, "waveforms")
        self.assertEqual(group.properties["Application"], "NI-RFmx Waveform Creator")
        self.assertEqual(channel.group_name, "waveforms")
        self.assertEqual(channel.channel_name, "Channel 0")
        self.assertEqual(channel.data.dtype, np.float32)
        self.assertEqual(channel.data.size, 2 * waveform.size)
        np.testing.assert_allclose(channel.data[0::2], waveform.real, rtol=1e-6)
        np.testing.assert_allclose(channel.data[1::2], waveform.imag, rtol=1e-6)
        self.assertEqual(channel.properties["NI_RF_IQRate"], 491.52e6)
        self.assertEqual(
            channel.properties["NI_RF_WaveformType"], "InterleavedIQCluster"
        )
        self.assertEqual(channel.properties["NI_RF_RuntimeScaling"], 0.0)

        target = self.waveforms_dir / "rdpd_wave.tdms"
        writes = self.scpi_writes()
        self.assertIn(f'MMEMory:LOAD:WAVeform "{target}", "{WAVEFORM_NAME}", 0', writes)
        self.assertIn(
            f'SOURce:RFSG:LOAD:WAVeform:MEMory "{WAVEFORM_NAME}"', writes
        )
        self.assertIn(
            f'SOURce:RFSG:WAVeform:REPeat:MODE "{WAVEFORM_NAME}", CONTINUOUS', writes
        )
        self.assertIn(
            f'SOURce:RFSG:ARB:WAVeform:SELect "{WAVEFORM_NAME}"', writes
        )
        self.assertIn(
            ("query", f'MEMory:WAVeform:DX? "{WAVEFORM_NAME}"'), self.scpi.commands
        )
        # The generation task was never initiated, so the upload itself must
        # not abort anything (the ABORt seen earlier comes from configure).
        configure_abort_count = writes.index("ABORt:RFSG")
        self.assertEqual(
            writes[configure_abort_count + 1 :].count("ABORt:RFSG"), 0
        )
        bench.disconnect(TIMEOUT)

    def test_upload_aborts_running_generation_before_reselect(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        bench.transmitter.stop_transmission(TIMEOUT)
        before_count = len(self.scpi_writes())
        bench.transmitter.upload_waveform(self.make_waveform(), TIMEOUT)

        writes = self.scpi_writes()[before_count:]
        abort_index = writes.index("ABORt:RFSG")
        load_index = writes.index(
            f'MMEMory:LOAD:WAVeform "{self.waveforms_dir / "rdpd_wave.tdms"}", '
            f'"{WAVEFORM_NAME}", 0'
        )
        bind_index = writes.index(
            f'SOURce:RFSG:LOAD:WAVeform:MEMory "{WAVEFORM_NAME}"'
        )
        select_index = writes.index(
            f'SOURce:RFSG:ARB:WAVeform:SELect "{WAVEFORM_NAME}"'
        )
        self.assertLess(abort_index, load_index)
        self.assertLess(load_index, bind_index)
        self.assertLess(bind_index, select_index)

        # The abort reset the task, so the next start re-initiates it.
        bench.transmitter.start_transmission(TIMEOUT)
        self.assertEqual(self.scpi_writes().count("INITiate:RFSG"), 2)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_upload_rejects_invalid_waveforms(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        with self.assertRaises(ValueError):
            bench.transmitter.upload_waveform(
                np.array([1.0, np.nan, 2.0], dtype=np.complex128), TIMEOUT
            )
        with self.assertRaises(ValueError):
            bench.transmitter.upload_waveform(np.zeros(0, dtype=np.complex128), TIMEOUT)
        self.assertFalse(self.nptdms_recorder["writers"])
        bench.disconnect(TIMEOUT)

    def test_upload_rejected_while_transmitting_or_unconfigured(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        with self.assertRaises(RuntimeError):
            bench.transmitter.upload_waveform(self.make_waveform(), TIMEOUT)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

        unconfigured = Vst5842RFBench()
        unconfigured.connect(TIMEOUT)
        with self.assertRaises(RuntimeError):
            unconfigured.transmitter.upload_waveform(self.make_waveform(), TIMEOUT)
        unconfigured.disconnect(TIMEOUT)

    def test_upload_fails_when_sample_period_mismatches(self):
        bench = self.connect_and_configure(
            Vst5842RFBench(), self.make_config(sample_rate_hz=983.04e6)
        )
        # The fake DX response still reports the 491.52 MS/s period.
        with self.assertRaises(RuntimeError) as ctx:
            bench.transmitter.upload_waveform(self.make_waveform(), TIMEOUT)
        self.assertIn("sample period", str(ctx.exception))
        bench.disconnect(TIMEOUT)


@unittest.skipUnless(_nptdms_available(), "nptdms is not installed")
class TdmsWaveformWriterTests(unittest.TestCase):
    def test_written_file_round_trips_with_native_structure(self):
        from nptdms import TdmsFile

        from remote_dpd.real_bench import _write_tdms_waveform

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rdpd_wave.tdms"
            phases = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
            waveform = 0.5 * (np.cos(phases) + 1j * np.sin(phases))
            _write_tdms_waveform(path, waveform, 491.52e6)

            file = TdmsFile.read(str(path))
            group = file["waveforms"]
            self.assertEqual(
                group.properties["Application"], "NI-RFmx Waveform Creator"
            )
            self.assertEqual(
                group.properties["NI_RF_WaveformFileVersion"], "2.0.0"
            )
            channel = group.channels()[0]
            self.assertEqual(channel.name, "Channel 0")
            self.assertEqual(channel.dtype, np.float32)
            self.assertEqual(len(channel), 256)
            self.assertEqual(channel.properties["NI_RF_IQRate"], 491.52e6)
            self.assertEqual(
                channel.properties["NI_RF_WaveformType"], "InterleavedIQCluster"
            )
            self.assertAlmostEqual(
                channel.properties["dt"], 1.0 / 491.52e6, places=18
            )
            data = np.asarray(channel)
            np.testing.assert_allclose(data[0::2], waveform.real, rtol=1e-6)
            np.testing.assert_allclose(data[1::2], waveform.imag, rtol=1e-6)


class Vst5842TransmitTests(RealBenchTestCase):
    def test_start_runs_interlock_then_initiate_then_output(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        aux_queries = [
            command
            for session in self.sessions_for("GPIB1::")
            if session.resource_name != "GPIB1::5::INSTR"
            for _, command in session.commands
        ]
        self.assertIn("OUTP?", aux_queries)
        self.assertIn("MEAS:VOLT?", aux_queries)
        writes = self.scpi_writes()
        initiate_index = writes.index("INITiate:RFSG")
        output_index = writes.index("SOURce:RFSG:OUTPut:ENABled 1")
        self.assertLess(initiate_index, output_index)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_stop_gates_output_without_halting_generation(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        started_at = len(self.scpi_writes())
        bench.transmitter.stop_transmission(TIMEOUT)
        writes = self.scpi_writes()[started_at:]
        self.assertEqual(writes, ["SOURce:RFSG:OUTPut:ENABled 0"])
        # Restart only gates the output again; the generator stays initiated.
        bench.transmitter.start_transmission(TIMEOUT)
        writes = self.scpi_writes()
        self.assertEqual(writes.count("INITiate:RFSG"), 1)
        self.assertEqual(writes.count("SOURce:RFSG:OUTPut:ENABled 1"), 2)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_reconfigure_reinitiates_generation_on_next_start(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        bench.transmitter.stop_transmission(TIMEOUT)
        bench.configure(self.make_config(), TIMEOUT)
        self.start_transmitting(bench)
        writes = self.scpi_writes()
        self.assertEqual(writes.count("INITiate:RFSG"), 2)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_start_requires_uploaded_waveform(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        with self.assertRaises(RuntimeError):
            bench.transmitter.start_transmission(TIMEOUT)
        self.assertNotIn("INITiate:RFSG", self.scpi_writes())
        bench.disconnect(TIMEOUT)

    def test_attenuation_maps_onto_power_level(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        bench.transmitter.set_attenuation_db(5.0, TIMEOUT)
        self.assertIn("SOURce:RFSG:POWer:LEVel -22.000000", self.scpi_writes())
        self.scpi.state["power_level"] = -30.0
        self.assertAlmostEqual(bench.transmitter.get_attenuation_db(TIMEOUT), 13.0)
        with self.assertRaises(ValueError):
            bench.transmitter.set_attenuation_db(41.0, TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_interlock_failure_blocks_transmission(self):
        self.visa.define("GPIB1::7::INSTR", {**default_aux_responses(), "OUTP?": "0"})
        bench = self.connect_and_configure(Vst5842RFBench())
        bench.transmitter.upload_waveform(self.make_waveform(), TIMEOUT)
        with self.assertRaises(RuntimeError):
            bench.transmitter.start_transmission(TIMEOUT)
        self.assertNotIn("INITiate:RFSG", self.scpi_writes())
        self.assertNotIn("SOURce:RFSG:OUTPut:ENABled 1", self.scpi_writes())
        bench.disconnect(TIMEOUT)

    def test_bias_voltage_deviation_blocks_transmission(self):
        responses = default_aux_responses()
        responses["MEAS:VOLT?"] = "+5.000E+00"
        self.visa.define("GPIB1::8::INSTR", responses)
        bench = self.connect_and_configure(Vst5842RFBench())
        bench.transmitter.upload_waveform(self.make_waveform(), TIMEOUT)
        with self.assertRaises(RuntimeError):
            bench.transmitter.start_transmission(TIMEOUT)
        self.assertNotIn("SOURce:RFSG:OUTPut:ENABled 1", self.scpi_writes())
        bench.disconnect(TIMEOUT)


class Vst5842CaptureTests(RealBenchTestCase):
    def test_capture_decodes_big_endian_trace_and_echoes_request(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        waveform = self.start_transmitting(bench)
        period = waveform * 0.5
        self.scpi.fetch_payload = make_fetch_payload(np.tile(period, 8))

        batch = bench.receiver.capture(
            CaptureRequest(segment_length=WAVEFORM_LENGTH, segment_count=8), TIMEOUT
        )
        self.assertEqual(batch.segment_length, WAVEFORM_LENGTH)
        self.assertEqual(batch.segment_count, 8)
        self.assertEqual(batch.iq.size, WAVEFORM_LENGTH * 8)
        self.assertTrue(batch.coherent_within_batch)
        self.assertEqual(batch.sample_rate_hz, 491.52e6)
        # The trace block is big-endian float32; allow its quantization.
        np.testing.assert_allclose(
            batch.iq[:WAVEFORM_LENGTH], period, rtol=1e-6, atol=1e-9
        )
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_capture_sends_specan_sequence_and_trims_guard_samples(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        waveform = self.start_transmitting(bench)
        expected_samples = waveform.size * 4 + 32
        expected_time = f"{expected_samples / 491.52e6:.9E}"
        self.scpi.fetch_payload = make_fetch_payload(np.tile(waveform, 4))

        bench.receiver.capture(
            CaptureRequest(segment_length=WAVEFORM_LENGTH, segment_count=4), TIMEOUT
        )
        writes = self.scpi_writes()
        self.assertIn("CONFigure:SPECan:MEASurement:SELect 1,IQ", writes)
        self.assertIn(f"CONFigure:SPECan:IQ:ACQuisition:TIME {expected_time}", writes)
        self.assertIn("CONFigure:SPECan:IQ:SRATe 4.915200000E+08", writes)
        initiate_index = writes.index("INITiate:SPECan")
        fetch_index = writes.index("FETCh:SPECan:RESult:IQ:TRACe:DATA?")
        self.assertLess(initiate_index, fetch_index)
        # Binary-block reads must run with termination disabled and restore it.
        self.assertEqual(self.scpi.read_raw_terminations, [None])
        self.assertEqual(self.scpi.read_termination, "\n")
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_capture_truncates_extra_samples(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        waveform = self.start_transmitting(bench)
        extra = np.concatenate([np.tile(waveform, 2), np.full(7, 9.0 + 9.0j)])
        self.scpi.fetch_payload = make_fetch_payload(extra)

        batch = bench.receiver.capture(
            CaptureRequest(segment_length=WAVEFORM_LENGTH, segment_count=2), TIMEOUT
        )
        self.assertEqual(batch.iq.size, 2 * WAVEFORM_LENGTH)
        np.testing.assert_allclose(batch.iq, np.tile(waveform, 2), rtol=1e-6)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_capture_rejects_short_trace(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        waveform = self.start_transmitting(bench)
        self.scpi.fetch_payload = make_fetch_payload(waveform[:10])

        with self.assertRaises(RuntimeError) as ctx:
            bench.receiver.capture(
                CaptureRequest(segment_length=WAVEFORM_LENGTH, segment_count=4), TIMEOUT
            )
        self.assertIn("shorter than requested", str(ctx.exception))
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_capture_rejects_empty_trace(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        self.scpi.fetch_payload = b""

        with self.assertRaises(RuntimeError) as ctx:
            bench.receiver.capture(
                CaptureRequest(segment_length=WAVEFORM_LENGTH, segment_count=1), TIMEOUT
            )
        self.assertIn("empty", str(ctx.exception))
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_capture_rejects_wrong_segment_length_and_oversize(self):
        bench = self.connect_and_configure(
            Vst5842RFBench(),
            self.make_config(device_options={"max_capture_samples": 128}),
        )
        self.start_transmitting(bench)
        with self.assertRaises(ValueError):
            bench.receiver.capture(
                CaptureRequest(segment_length=WAVEFORM_LENGTH - 1, segment_count=4), TIMEOUT
            )
        with self.assertRaises(ValueError):
            bench.receiver.capture(
                CaptureRequest(segment_length=WAVEFORM_LENGTH, segment_count=8), TIMEOUT
            )
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_capture_requires_active_transmission(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        bench.transmitter.upload_waveform(self.make_waveform(), TIMEOUT)
        with self.assertRaises(RuntimeError):
            bench.receiver.capture(
                CaptureRequest(segment_length=WAVEFORM_LENGTH, segment_count=1), TIMEOUT
            )
        self.assertNotIn("INITiate:SPECan", self.scpi_writes())
        bench.disconnect(TIMEOUT)


class N1912APowerSensorTests(RealBenchTestCase):
    def test_measure_sets_frequency_and_averaging_then_reads(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        value = bench.power_sensor.measure_power_dbm(TIMEOUT)
        self.assertAlmostEqual(value, 37.96, places=4)
        meter = self.sessions_for("TCPIP0::192.")[0]
        commands = meter.commands
        writes = [command for kind, command in commands if kind == "write"]
        self.assertTrue(any(command.startswith("SENS1:FREQ ") for command in writes))
        self.assertIn("SENS1:AVER:COUN 64", writes)
        self.assertIn("INIT1:CONT ON", writes)
        self.assertIn(("query", "FETC1?"), commands)
        bench.disconnect(TIMEOUT)

    def test_measure_requires_configuration(self):
        bench = Vst5842RFBench()
        bench.connect(TIMEOUT)
        with self.assertRaises(RuntimeError):
            bench.power_sensor.measure_power_dbm(TIMEOUT)
        bench.disconnect(TIMEOUT)


class PowerSafetyRedLineTests(RealBenchTestCase):
    def test_safe_shutdown_turns_drain_off_and_never_enables_outputs(self):
        # The schema default keeps the drain supply untouched; this test pins
        # the shutdown path itself, so it opts in explicitly.
        bench = self.connect_and_configure(
            Vst5842RFBench(),
            self.make_config(device_options={"enable_supply_shutdown": True}),
        )
        self.start_transmitting(bench)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

        drain_sessions = self.sessions_for("GPIB1::5::INSTR")
        self.assertTrue(drain_sessions)
        for session in drain_sessions:
            self.assertIn(("write", "OUTP OFF"), session.commands)

        for session in self.visa.sessions:
            for kind, command in session.commands:
                normalized = command.strip().upper()
                self.assertFalse(
                    normalized in ("OUTP 1", "OUTP ON"),
                    f"forbidden output-enable command {command!r} on "
                    f"{session.resource_name}",
                )

    def test_aux_supplies_only_receive_read_only_queries(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)
        for session in self.sessions_for("GPIB1::"):
            if session.resource_name == "GPIB1::5::INSTR":
                continue
            for kind, command in session.commands:
                self.assertEqual(kind, "query")
                self.assertIn(command.strip(), AUX_QUERY_WHITELIST)

    def test_disabled_shutdown_does_not_touch_drain_supply(self):
        bench = self.connect_and_configure(
            Vst5842RFBench(),
            self.make_config(device_options={"enable_supply_shutdown": False}),
        )
        self.start_transmitting(bench)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)
        drain_sessions = self.sessions_for("GPIB1::5::INSTR")
        self.assertTrue(drain_sessions)
        for session in drain_sessions:
            self.assertNotIn(("write", "OUTP OFF"), session.commands)

    def test_module_source_contains_no_output_enable_literal(self):
        import remote_dpd

        source_path = Path(remote_dpd.__file__).parent / "real_bench.py"
        self.assertTrue(source_path.is_file())
        text = source_path.read_text(encoding="utf-8")
        self.assertNotIn("OUTP ON", text.upper())


class Vst5842LifecycleTests(RealBenchTestCase):
    def test_disconnect_releases_visa_resources(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        bench.disconnect(TIMEOUT)
        self.assertTrue(self.scpi.closed)
        meter = self.sessions_for("TCPIP0::192.")[0]
        self.assertTrue(meter.closed)
        with self.assertRaises(RuntimeError):
            bench.transmitter.upload_waveform(self.make_waveform(), TIMEOUT)

    def test_double_connect_and_disconnect_are_rejected_or_idempotent(self):
        bench = Vst5842RFBench()
        bench.connect(TIMEOUT)
        with self.assertRaises(RuntimeError):
            bench.connect(TIMEOUT)
        bench.disconnect(TIMEOUT)
        bench.disconnect(TIMEOUT)  # idempotent


if __name__ == "__main__":
    unittest.main()
