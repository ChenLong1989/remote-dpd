import unittest

import numpy as np

from remote_dpd.device import (
    CaptureRequest,
    DeviceConfig,
    DeviceRegistrationError,
    PowerSensor,
    Receiver,
    RFBench,
    Transmitter,
    create_rf_bench,
    list_rf_benches,
    register_rf_bench,
)
from remote_dpd.dsp import fractional_shift
from remote_dpd.preprocessing import FeedbackPreprocessor
from remote_dpd.runtime import BasicILCRuntime, RuntimeStepInput
from remote_dpd.simulation import SIMULATED_DEVICE_SCHEMA, SimulatedRFBench

TIMEOUT = 1.0
SAMPLE_RATE_HZ = 245.76e6


def linear_options(**overrides):
    options = {
        "pa_coefficients": [
            {"p": 1, "m": 0, "real": 1.0, "imag": 0.0},
        ],
        "system_gain_db": 0.0,
        "system_phase_rad": 0.0,
        "delay_samples": 0.0,
        "noise_dbfs": -300.0,
        "random_seed": 7,
        "power_reference_dbm": 10.0,
        "max_capture_samples": 1024,
    }
    options.update(overrides)
    return options


def make_config(options=None, **overrides):
    values = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "target_power_dbm": -10.0,
        "safety_power_limit_dbm": 10.0,
        "initial_attenuation_db": 0.0,
        "min_attenuation_db": 0.0,
        "max_attenuation_db": 60.0,
        "device_options": linear_options() if options is None else options,
    }
    values.update(overrides)
    return DeviceConfig(**values)


def prepare_running_bench(waveform, config=None):
    bench = SimulatedRFBench()
    bench.connect(TIMEOUT)
    bench.configure(make_config() if config is None else config, TIMEOUT)
    bench.upload_waveform(waveform, TIMEOUT)
    bench.start_transmission(TIMEOUT)
    return bench


class SimulatedSchemaTests(unittest.TestCase):
    def test_registry_creates_isolated_simulated_benches_and_extensions(self):
        self.assertIn("simulated", list_rf_benches())
        first = create_rf_bench(" SIMULATED ")
        second = create_rf_bench("simulated")
        self.assertIsInstance(first, SimulatedRFBench)
        self.assertIsInstance(second, SimulatedRFBench)
        self.assertIsNot(first, second)

        register_rf_bench(
            "simulation_alias",
            SimulatedRFBench,
            replace=True,
        )
        self.assertIsInstance(create_rf_bench("simulation_alias"), SimulatedRFBench)
        with self.assertRaises(DeviceRegistrationError):
            register_rf_bench("simulation_alias", SimulatedRFBench)
        register_rf_bench("invalid_factory", lambda: object(), replace=True)
        with self.assertRaises(DeviceRegistrationError):
            create_rf_bench("invalid_factory")

    def test_publishes_versioned_schema_with_complete_editable_pa_defaults(self):
        bench = SimulatedRFBench()

        self.assertIs(bench.parameter_schema, SIMULATED_DEVICE_SCHEMA)
        self.assertEqual(SIMULATED_DEVICE_SCHEMA.device_type, "simulated")
        self.assertEqual(SIMULATED_DEVICE_SCHEMA.schema_version, 2)
        self.assertEqual(
            {field.name for field in SIMULATED_DEVICE_SCHEMA.fields},
            {
                "pa_coefficients",
                "system_gain_db",
                "system_phase_rad",
                "delay_samples",
                "noise_dbfs",
                "random_seed",
                "power_reference_dbm",
                "max_capture_samples",
            },
        )
        defaults = SIMULATED_DEVICE_SCHEMA.validate_options({})
        self.assertEqual(
            defaults["pa_coefficients"],
            [
                {"p": 1, "m": 0, "real": 1.0, "imag": 0.0},
                {"p": 1, "m": 1, "real": 0.04, "imag": 0.015},
                {"p": 3, "m": 0, "real": -0.36, "imag": 0.075},
                {"p": 3, "m": 1, "real": -0.06, "imag": 0.03},
            ],
        )
        self.assertEqual(defaults["power_reference_dbm"], 1.0)

    def test_schema_rejects_unknown_and_invalid_options_during_configuration(self):
        invalid_options = (
            ({"unknown": 1}, ValueError, "unknown device options"),
            (
                {"pa_coefficients": [{"p": 2, "m": 0, "real": 1.0, "imag": 0.0}]},
                ValueError,
                "step",
            ),
            (
                {"pa_coefficients": [{"p": 1, "m": -1, "real": 1.0, "imag": 0.0}]},
                ValueError,
                "at least",
            ),
            ({"random_seed": True}, TypeError, "integer"),
            ({"noise_dbfs": 1.0}, ValueError, "at most"),
            ({"pa_coefficients": []}, ValueError, "at least one"),
        )
        for options, error_type, message in invalid_options:
            with self.subTest(options=options):
                bench = SimulatedRFBench()
                bench.connect(TIMEOUT)
                with self.assertRaisesRegex(error_type, message):
                    bench.configure(make_config(options), TIMEOUT)


class SimulatedSignalPathTests(unittest.TestCase):
    def test_memory_polynomial_uses_all_terms_and_periodic_boundaries(self):
        waveform = np.array([0.2 + 0.1j, -0.4 + 0.3j, 0.6 - 0.2j, -0.1 - 0.5j])
        coefficients = [
            {"p": 1, "m": 1, "real": 0.5, "imag": -0.25},
            {"p": 3, "m": 0, "real": -0.2, "imag": 0.1},
        ]
        config = make_config(linear_options(pa_coefficients=coefficients))
        bench = prepare_running_bench(waveform, config)

        batch = bench.capture(CaptureRequest(waveform.size, 2), TIMEOUT)

        expected = (0.5 - 0.25j) * np.roll(waveform, 1)
        expected += (-0.2 + 0.1j) * waveform * np.abs(waveform) ** 2
        np.testing.assert_allclose(batch.segments[0], expected, atol=3e-15)
        np.testing.assert_allclose(batch.segments[1], expected, atol=3e-15)
        self.assertTrue(batch.coherent_within_batch)
        self.assertEqual(batch.sample_rate_hz, SAMPLE_RATE_HZ)

    def test_feedback_applies_gain_phase_and_fractional_periodic_delay(self):
        samples = np.arange(32)
        waveform = np.exp(2j * np.pi * 3 * samples / samples.size)
        options = linear_options(
            system_gain_db=-12.0,
            system_phase_rad=0.7,
            delay_samples=1.375,
        )
        bench = prepare_running_bench(waveform, make_config(options))

        actual = bench.capture(CaptureRequest(waveform.size, 1), TIMEOUT).iq

        expected = fractional_shift(
            waveform * 10.0 ** (-12.0 / 20.0) * np.exp(0.7j),
            1.375,
        )
        np.testing.assert_allclose(actual, expected, atol=3e-15)

    def test_attenuation_precedes_pa_and_lower_attenuation_raises_power(self):
        waveform = np.ones(16, dtype=np.complex128)
        options = linear_options(system_gain_db=-40.0, power_reference_dbm=10.0)
        config = make_config(
            options,
            initial_attenuation_db=20.0,
            min_attenuation_db=0.0,
        )
        bench = prepare_running_bench(waveform, config)

        power_at_20_db = bench.measure_power_dbm(TIMEOUT)
        feedback_at_20_db = bench.capture(CaptureRequest(16, 1), TIMEOUT).iq
        bench.set_attenuation_db(10.0, TIMEOUT)
        power_at_10_db = bench.measure_power_dbm(TIMEOUT)
        feedback_at_10_db = bench.capture(CaptureRequest(16, 1), TIMEOUT).iq

        self.assertAlmostEqual(power_at_20_db, -10.0, places=12)
        self.assertAlmostEqual(power_at_10_db, 0.0, places=12)
        self.assertAlmostEqual(power_at_10_db - power_at_20_db, 10.0, places=12)
        np.testing.assert_allclose(np.abs(feedback_at_20_db), 0.001, atol=3e-15)
        np.testing.assert_allclose(
            np.abs(feedback_at_10_db),
            10.0 ** (-50.0 / 20.0),
            atol=3e-15,
        )

    def test_waveform_is_copied_without_agc_or_clipping(self):
        source = np.array([2.0 + 0.5j, -1.5j, 1.25 - 0.25j])
        expected = source.copy()
        bench = prepare_running_bench(source)
        source[:] = 0.0

        capture = bench.capture(CaptureRequest(expected.size, 1), TIMEOUT)

        np.testing.assert_allclose(capture.iq, expected, atol=3e-15)
        self.assertGreater(float(np.max(np.abs(capture.iq))), 1.0)

    def test_seed_is_reproducible_and_reconfiguration_resets_noise_sequence(self):
        waveform = np.linspace(0.1, 0.8, 16).astype(np.complex128)
        options = linear_options(noise_dbfs=-20.0, random_seed=123)
        config = make_config(options)
        bench = prepare_running_bench(waveform, config)

        first = bench.capture(CaptureRequest(16, 2), TIMEOUT).iq
        second = bench.capture(CaptureRequest(16, 2), TIMEOUT).iq
        bench.stop_transmission(TIMEOUT)
        bench.configure(config, TIMEOUT)
        bench.upload_waveform(waveform, TIMEOUT)
        bench.start_transmission(TIMEOUT)
        replay = bench.capture(CaptureRequest(16, 2), TIMEOUT).iq

        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(first, replay)

    def test_configuration_is_detached_from_the_supplied_config(self):
        waveform = np.ones(8)
        config = make_config(linear_options(system_gain_db=0.0))
        bench = SimulatedRFBench()
        bench.connect(TIMEOUT)
        bench.configure(config, TIMEOUT)

        dict.__setitem__(config.device_options, "system_gain_db", -40.0)
        bench.upload_waveform(waveform, TIMEOUT)
        bench.start_transmission(TIMEOUT)
        capture = bench.capture(CaptureRequest(8, 1), TIMEOUT)

        np.testing.assert_allclose(capture.iq, waveform, atol=3e-15)

    def test_capture_enforces_complete_segments_and_maximum_sample_count(self):
        waveform = np.ones(4)
        config = make_config(linear_options(max_capture_samples=8))
        bench = prepare_running_bench(waveform, config)

        accepted = bench.capture(CaptureRequest(4, 2), TIMEOUT)

        self.assertEqual(bench.max_capture_samples, 8)
        self.assertEqual(accepted.iq.size, 8)
        with self.assertRaisesRegex(ValueError, "segment_length"):
            bench.capture(CaptureRequest(2, 2), TIMEOUT)
        with self.assertRaisesRegex(ValueError, "max_capture_samples"):
            bench.capture(CaptureRequest(4, 3), TIMEOUT)
        with self.assertRaisesRegex(TypeError, "CaptureRequest"):
            bench.capture(object(), TIMEOUT)

    def test_default_memory_pa_improves_with_basic_ilc(self):
        samples = np.arange(64)
        reference = (
            0.3 * np.exp(2j * np.pi * 3 * samples / samples.size)
            + 0.18 * np.exp(2j * np.pi * 9 * samples / samples.size)
            + 0.08 * np.exp(-2j * np.pi * 13 * samples / samples.size)
        )
        config = make_config(
            {
                "system_gain_db": 0.0,
                "system_phase_rad": 0.0,
                "delay_samples": 0.0,
                "noise_dbfs": -300.0,
                "random_seed": 7,
                "power_reference_dbm": 10.0,
                "max_capture_samples": 1024,
            }
        )
        bench = prepare_running_bench(reference, config)
        preprocessor = FeedbackPreprocessor(reference, SAMPLE_RATE_HZ)
        runtime = BasicILCRuntime()
        runtime.initialize({"mu": 0.5})
        current = reference.copy()
        first_nmse = None
        gain = None

        for iteration in range(6):
            batch = bench.capture(CaptureRequest(reference.size, 1), TIMEOUT)
            result = preprocessor.process([batch], gain_correction=gain)
            if gain is None:
                gain = result.gain_correction
                first_nmse = result.nmse_db
            if iteration == 5:
                final_nmse = result.nmse_db
                break
            candidate = runtime.step(
                RuntimeStepInput(
                    x=reference,
                    y_current=current,
                    z_current=result.z,
                    iteration=iteration,
                    config={"mu": 0.5},
                )
            ).y_candidate
            bench.stop_transmission(TIMEOUT)
            bench.upload_waveform(candidate, TIMEOUT)
            bench.start_transmission(TIMEOUT)
            current = candidate

        self.assertIsNotNone(first_nmse)
        self.assertLess(final_nmse, first_nmse - 3.0)


class SimulatedLifecycleTests(unittest.TestCase):
    def test_capabilities_return_same_integrated_instance(self):
        bench = SimulatedRFBench()

        self.assertIsInstance(bench, RFBench)
        self.assertIsInstance(bench, Transmitter)
        self.assertIsInstance(bench, Receiver)
        self.assertIsInstance(bench, PowerSensor)
        self.assertIs(bench.transmitter, bench)
        self.assertIs(bench.receiver, bench)
        self.assertIs(bench.power_sensor, bench)

    def test_enforces_connect_configure_upload_and_run_order(self):
        waveform = np.ones(8)
        request = CaptureRequest(8, 1)
        config = make_config()
        bench = SimulatedRFBench()

        with self.assertRaisesRegex(RuntimeError, "not connected"):
            bench.configure(config, TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            bench.upload_waveform(waveform, TIMEOUT)
        bench.connect(TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "already connected"):
            bench.connect(TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            bench.upload_waveform(waveform, TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            bench.start_transmission(TIMEOUT)

        bench.configure(config, TIMEOUT)
        self.assertEqual(bench.get_attenuation_db(TIMEOUT), 0.0)
        with self.assertRaisesRegex(RuntimeError, "uploaded"):
            bench.start_transmission(TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "not transmitting"):
            bench.capture(request, TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "not transmitting"):
            bench.measure_power_dbm(TIMEOUT)

        bench.upload_waveform(waveform, TIMEOUT)
        bench.start_transmission(TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "while transmission"):
            bench.configure(config, TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "while transmission"):
            bench.upload_waveform(waveform, TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "already running"):
            bench.start_transmission(TIMEOUT)

        bench.stop_transmission(TIMEOUT)
        bench.stop_transmission(TIMEOUT)
        bench.start_transmission(TIMEOUT)
        bench.safe_shutdown(TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "not transmitting"):
            bench.capture(request, TIMEOUT)
        bench.start_transmission(TIMEOUT)
        bench.disconnect(TIMEOUT)
        bench.disconnect(TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            bench.get_attenuation_db(TIMEOUT)
        self.assertEqual(bench.max_capture_samples, 1_000_000)

    def test_reconfiguration_resets_attenuation_and_requires_a_new_upload(self):
        waveform = np.ones(8)
        bench = SimulatedRFBench()
        bench.connect(TIMEOUT)
        bench.configure(
            make_config(initial_attenuation_db=20.0),
            TIMEOUT,
        )
        bench.set_attenuation_db(7.0, TIMEOUT)
        bench.upload_waveform(waveform, TIMEOUT)

        bench.configure(
            make_config(initial_attenuation_db=15.0),
            TIMEOUT,
        )

        self.assertEqual(bench.get_attenuation_db(TIMEOUT), 15.0)
        with self.assertRaisesRegex(RuntimeError, "uploaded"):
            bench.start_transmission(TIMEOUT)

    def test_rejects_invalid_waveforms_and_attenuation(self):
        bench = SimulatedRFBench()
        bench.connect(TIMEOUT)
        bench.configure(make_config(), TIMEOUT)
        invalid_waveforms = (
            np.empty(0),
            np.ones((2, 2)),
            np.array([1.0, np.nan]),
            np.array(["1", "2"]),
            np.array([True, False]),
        )
        for waveform in invalid_waveforms:
            with (
                self.subTest(waveform=waveform),
                self.assertRaises((TypeError, ValueError)),
            ):
                bench.upload_waveform(waveform, TIMEOUT)

        for attenuation in (-0.1, 60.1, float("nan"), float("inf")):
            with self.subTest(attenuation=attenuation), self.assertRaises(ValueError):
                bench.set_attenuation_db(attenuation, TIMEOUT)
        with self.assertRaises(TypeError):
            bench.set_attenuation_db(True, TIMEOUT)

    def test_all_blocking_methods_reject_non_positive_or_non_finite_timeouts(self):
        waveform = np.ones(8)
        config = make_config()
        bench = prepare_running_bench(waveform, config)
        request = CaptureRequest(8, 1)
        calls = (
            lambda timeout: bench.connect(timeout),
            lambda timeout: bench.configure(config, timeout),
            lambda timeout: bench.upload_waveform(waveform, timeout),
            lambda timeout: bench.start_transmission(timeout),
            lambda timeout: bench.stop_transmission(timeout),
            lambda timeout: bench.get_attenuation_db(timeout),
            lambda timeout: bench.set_attenuation_db(0.0, timeout),
            lambda timeout: bench.capture(request, timeout),
            lambda timeout: bench.measure_power_dbm(timeout),
            lambda timeout: bench.safe_shutdown(timeout),
            lambda timeout: bench.disconnect(timeout),
        )
        for timeout in (0.0, -1.0, float("nan"), float("inf")):
            for call in calls:
                with (
                    self.subTest(timeout=timeout, call=call),
                    self.assertRaises(ValueError),
                ):
                    call(timeout)
        for timeout in (True, "1"):
            for call in calls:
                with (
                    self.subTest(timeout=timeout, call=call),
                    self.assertRaises(TypeError),
                ):
                    call(timeout)


if __name__ == "__main__":
    unittest.main()
