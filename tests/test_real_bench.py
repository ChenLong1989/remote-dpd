import sys
import types
import unittest
from pathlib import Path

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

# Commands that are legitimate read-only queries against the E3648A bias
# supplies; the power-safety red line forbids every write to those resources.
AUX_QUERY_WHITELIST = ("*IDN?", "OUTP?", "VOLT?", "MEAS:VOLT?", "MEAS:CURR?")


class FakeRfsgSession:
    def __init__(self, resource_name):
        self._resource_name = resource_name
        self._log = []
        self._props = {"power_level": 0.0}
        self._closed = False
        self.waveform = None
        self.script = None

    def __setattr__(self, name, value):
        if name.startswith("_") or name in ("waveform", "script"):
            object.__setattr__(self, name, value)
            return
        self._props[name] = value
        self._log.append(("set", name, value))

    def __getattr__(self, name):
        try:
            return self.__dict__["_props"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def write_arb_waveform(self, waveform_name, data, more_data_pending=False):
        self.waveform = (waveform_name, np.array(data, dtype=np.complex128))
        self._log.append(("call", "write_arb_waveform", waveform_name))

    def write_script(self, script):
        self.script = script
        self._log.append(("call", "write_script"))

    def initiate(self):
        self._log.append(("call", "initiate"))

    def abort(self):
        self._log.append(("call", "abort"))

    def close(self):
        self._closed = True
        self._log.append(("call", "close"))

    def operation_names(self):
        return [entry[1] for entry in self._log]


class FakeRfsaSession:
    def __init__(self, resource_name):
        self._resource_name = resource_name
        self._log = []
        self._props = {}
        self._closed = False
        self.iq_buffer = None

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._props[name] = value
        self._log.append(("set", name, value))

    def __getattr__(self, name):
        try:
            return self.__dict__["_props"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def read_iq_single_record_into(self, buffer, timeout=None):
        number_of_samples = self._props.get("number_of_samples")
        if number_of_samples is not None and number_of_samples != buffer.size:
            raise AssertionError(
                f"number_of_samples {number_of_samples} != buffer {buffer.size}"
            )
        if self.iq_buffer is not None:
            buffer[:] = self.iq_buffer[: buffer.size]
        self._log.append(("call", "read_iq_single_record_into", int(buffer.size)))

    def close(self):
        self._closed = True
        self._log.append(("call", "close"))


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
        self.sessions = []

    def define(self, resource_name, responses=None):
        self.behaviors[resource_name] = InstrumentBehavior(responses or {})

    def open_resource(self, resource_name, timeout=None):
        resource = FakeVisaResource(resource_name, self.behaviors.get(resource_name))
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


def make_fake_module_nirfsg(session):
    module = types.ModuleType("nirfsg")

    class GenerationMode:
        SCRIPT = "SCRIPT"
        ARB_WAVEFORM = "ARB_WAVEFORM"
        CW = "CW"

    module.GenerationMode = GenerationMode
    module.Session = lambda resource_name, **kwargs: session
    return module


def make_fake_module_nirfsa(session):
    module = types.ModuleType("nirfsa")

    class AcquisitionType:
        IQ = "IQ"
        SPECTRUM = "SPECTRUM"

    class StartTriggerType:
        NONE = "NONE"
        DIGITAL_EDGE = "DIGITAL_EDGE"
        SOFTWARE_EDGE = "SOFTWARE_EDGE"

    module.AcquisitionType = AcquisitionType
    module.StartTriggerType = StartTriggerType
    module.Session = lambda resource_name, **kwargs: session
    return module


def make_fake_module_hightime():
    module = types.ModuleType("hightime")

    class timedelta:
        def __init__(self, seconds=0.0):
            self.seconds = seconds

    module.timedelta = timedelta
    return module


def make_fake_module_pyvisa(manager):
    module = types.ModuleType("pyvisa")
    module.ResourceManager = lambda: manager
    return module


def default_aux_responses():
    return {
        "*IDN?": "Agilent Technologies,E3648A,0,2.5-6.1-2.1",
        "OUTP?": "1",
        "VOLT?": "+8.00000000E+00",
        "MEAS:VOLT?": "+7.99860800E+00",
    }


class RealBenchTestCase(unittest.TestCase):
    """Base class injecting fake driver modules before each test."""

    def setUp(self):
        self.rfsg = FakeRfsgSession("PXI2Slot2")
        self.rfsa = FakeRfsaSession("PXI2Slot2")
        self.visa = FakeResourceManager()
        self.visa.define("TCPIP0::192.168.255.40::inst0::INSTR", {"READ1?": "+3.7960E+001"})
        self.visa.define("GPIB1::5::INSTR", {"OUTP?": "1"})
        self.visa.define("GPIB1::7::INSTR", default_aux_responses())
        self.visa.define("GPIB1::8::INSTR", default_aux_responses())
        self._saved_modules = {}
        for name, module in (
            ("nirfsg", make_fake_module_nirfsg(self.rfsg)),
            ("nirfsa", make_fake_module_nirfsa(self.rfsa)),
            ("hightime", make_fake_module_hightime()),
            ("pyvisa", make_fake_module_pyvisa(self.visa)),
        ):
            self._saved_modules[name] = sys.modules.get(name)
            sys.modules[name] = module

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
        }
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

    def test_schema_defaults_fill_and_unknown_options_rejected(self):
        options = VST5842_DEVICE_SCHEMA.validate_options({})
        self.assertEqual(options["vst_resource"], "PXI2Slot2")
        self.assertEqual(options["reference_power_dbm"], -17.0)
        self.assertEqual(options["reference_level_dbm"], 55.0)
        self.assertEqual(options["power_meter_average"], 64)
        self.assertTrue(options["enable_supply_shutdown"])
        self.assertTrue(options["enable_supply_interlock"])
        self.assertEqual(
            options["aux_supply_resources"],
            ["GPIB1::7::INSTR", "GPIB1::8::INSTR"],
        )
        with self.assertRaises(ValueError):
            VST5842_DEVICE_SCHEMA.validate_options({"unknown_option": 1})

    def test_recommended_config_matches_station_operating_point(self):
        self.assertEqual(VST5842_RECOMMENDED_CONFIG.center_frequency_hz, 1.84e9)
        self.assertEqual(VST5842_RECOMMENDED_CONFIG.sample_rate_hz, 491.52e6)
        self.assertEqual(VST5842_RECOMMENDED_CONFIG.safety_power_limit_dbm, 39.0)


class Vst5842ConfigureTests(RealBenchTestCase):
    def test_configure_maps_common_settings_to_driver_state(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.assertEqual(self.rfsg._props["generation_mode"], "SCRIPT")
        self.assertEqual(self.rfsg._props["frequency"], 1.84e9)
        self.assertEqual(self.rfsg._props["iq_rate"], 491.52e6)
        self.assertEqual(self.rfsg._props["power_level"], -17.0 - 20.0)
        self.assertFalse(self.rfsg._props["output_enabled"])
        self.assertEqual(self.rfsa._props["acquisition_type"], "IQ")
        self.assertEqual(self.rfsa._props["center_frequency"], 1.84e9)
        self.assertEqual(self.rfsa._props["iq_rate"], 491.52e6)
        self.assertEqual(self.rfsa._props["reference_level"], 55.0)
        self.assertEqual(self.rfsa._props["start_trigger_type"], "NONE")
        bench.disconnect(TIMEOUT)

    def test_configure_requires_connection_and_rejects_bad_options(self):
        bench = Vst5842RFBench()
        with self.assertRaises(RuntimeError):
            bench.configure(self.make_config(), TIMEOUT)
        bench.connect(TIMEOUT)
        with self.assertRaises((ValueError, TypeError)):
            bench.configure(
                self.make_config(device_options={"vst_resource": 123}), TIMEOUT
            )

    def test_upload_waveform_writes_exact_cyclic_script(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        waveform = self.make_waveform()
        bench.transmitter.upload_waveform(waveform, TIMEOUT)
        name, data = self.rfsg.waveform
        self.assertEqual(name, "rdpdWave")
        self.assertEqual(data.dtype, np.complex128)
        np.testing.assert_array_equal(data, waveform)
        waveform[0] = 99.0 + 99.0j
        self.assertEqual(self.rfsg.waveform[1][0], 0.25 + 0.0j)
        self.assertIn("repeat forever", self.rfsg.script)
        self.assertIn("generate rdpdWave", self.rfsg.script)
        with self.assertRaises(ValueError):
            bench.transmitter.upload_waveform(
                np.array([1.0, np.nan, 2.0], dtype=np.complex128), TIMEOUT
            )
        bench.disconnect(TIMEOUT)


class Vst5842TransmitTests(RealBenchTestCase):
    def test_start_runs_interlock_then_rf_then_initiate(self):
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
        log = self.rfsg._log
        output_on_index = next(
            i for i, entry in enumerate(log) if entry[:2] == ("set", "output_enabled") and entry[2]
        )
        initiate_index = next(i for i, entry in enumerate(log) if entry[1] == "initiate")
        self.assertLess(output_on_index, initiate_index)
        bench.safe_shutdown(TIMEOUT)
        bench.disconnect(TIMEOUT)

    def test_stop_breaks_rf_before_halting_generation(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        bench.transmitter.stop_transmission(TIMEOUT)
        log = self.rfsg._log
        output_off_index = next(
            i for i, entry in enumerate(log) if entry[:2] == ("set", "output_enabled") and not entry[2]
        )
        abort_index = next(i for i, entry in enumerate(log) if entry[1] == "abort")
        self.assertLess(output_off_index, abort_index)
        bench.disconnect(TIMEOUT)

    def test_attenuation_maps_onto_power_level(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        bench.transmitter.set_attenuation_db(5.0, TIMEOUT)
        self.assertEqual(self.rfsg._props["power_level"], -17.0 - 5.0)
        self.rfsg._props["power_level"] = -30.0
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
        self.assertNotIn("initiate", self.rfsg.operation_names())
        bench.disconnect(TIMEOUT)

    def test_bias_voltage_deviation_blocks_transmission(self):
        responses = default_aux_responses()
        responses["MEAS:VOLT?"] = "+5.000E+00"
        self.visa.define("GPIB1::8::INSTR", responses)
        bench = self.connect_and_configure(Vst5842RFBench())
        bench.transmitter.upload_waveform(self.make_waveform(), TIMEOUT)
        with self.assertRaises(RuntimeError):
            bench.transmitter.start_transmission(TIMEOUT)
        self.assertNotIn("initiate", self.rfsg.operation_names())
        bench.disconnect(TIMEOUT)


class Vst5842CaptureTests(RealBenchTestCase):
    def test_capture_returns_segmented_coherent_batch(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        waveform = self.start_transmitting(bench)
        period = waveform * 0.5
        self.rfsa.iq_buffer = np.tile(period, 8)
        batch = bench.receiver.capture(
            CaptureRequest(segment_length=WAVEFORM_LENGTH, segment_count=8), TIMEOUT
        )
        self.assertEqual(batch.segment_length, WAVEFORM_LENGTH)
        self.assertEqual(batch.segment_count, 8)
        self.assertEqual(batch.iq.size, WAVEFORM_LENGTH * 8)
        self.assertTrue(batch.coherent_within_batch)
        self.assertEqual(batch.sample_rate_hz, 491.52e6)
        # The VST acquisition buffer is complex64; allow its quantization.
        np.testing.assert_allclose(
            batch.iq[:WAVEFORM_LENGTH], period, rtol=1e-6, atol=1e-9
        )
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
        bench.disconnect(TIMEOUT)


class N1912APowerSensorTests(RealBenchTestCase):
    def test_measure_sets_frequency_and_averaging_then_reads(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        value = bench.power_sensor.measure_power_dbm(TIMEOUT)
        self.assertAlmostEqual(value, 37.96, places=4)
        meter = self.sessions_for("TCPIP0::")[0]
        commands = meter.commands
        writes = [command for kind, command in commands if kind == "write"]
        self.assertTrue(any(command.startswith("SENS1:FREQ ") for command in writes))
        self.assertIn("SENS1:AVER:COUN 64", writes)
        self.assertIn(("query", "READ1?"), commands)
        bench.disconnect(TIMEOUT)

    def test_measure_requires_configuration(self):
        bench = Vst5842RFBench()
        bench.connect(TIMEOUT)
        with self.assertRaises(RuntimeError):
            bench.power_sensor.measure_power_dbm(TIMEOUT)
        bench.disconnect(TIMEOUT)


class PowerSafetyRedLineTests(RealBenchTestCase):
    def test_safe_shutdown_turns_drain_off_and_never_enables_outputs(self):
        bench = self.connect_and_configure(Vst5842RFBench())
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
    def test_disconnect_releases_driver_sessions_and_visa_resources(self):
        bench = self.connect_and_configure(Vst5842RFBench())
        self.start_transmitting(bench)
        bench.disconnect(TIMEOUT)
        self.assertTrue(self.rfsg._closed)
        self.assertTrue(self.rfsa._closed)
        meter = self.sessions_for("TCPIP0::")[0]
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
