"""Tests for the noisy ILC seed waveform (reference plus white seed noise)."""

import unittest

import numpy as np

from remote_dpd.controller import (
    ClosedLoopConfig,
    ClosedLoopController,
    ControllerState,
    SEED_NOISE_DEFAULT_BANDWIDTH_HZ,
    SEED_NOISE_DEFAULT_ENABLED,
    SEED_NOISE_DEFAULT_PSD_DB,
    SEED_NOISE_DEFAULT_SEED,
    _generate_seed_waveform,
)
from remote_dpd.device import DeviceConfig, create_rf_bench
from remote_dpd.file_interface import parse_configuration_json
from remote_dpd.power_control import PowerController
from remote_dpd.safety import DigitalSafetyError


def _bandlimited_carrier(sample_count: int, radius: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    raw = generator.standard_normal(sample_count) + 1j * generator.standard_normal(
        sample_count
    )
    spectrum = np.fft.fft(raw)
    keep = np.ones(sample_count, dtype=bool)
    keep[radius : sample_count - radius + 1] = False
    signal = np.fft.ifft(spectrum * keep)
    rms = float(np.sqrt(np.mean(np.abs(signal) ** 2)))
    return signal / rms * 0.178


def _noise_to_carrier_db(seed: np.ndarray, x: np.ndarray, sample_rate_hz: float) -> float:
    """Measure the 1 MHz noise power against the carrier inside the seed."""

    scale = np.vdot(x, seed) / np.vdot(x, x)
    noise = seed - scale * x
    noise_power = float(np.mean(np.abs(noise) ** 2))
    carrier_power = float(np.mean(np.abs(scale * x) ** 2))
    return 10.0 * np.log10(noise_power * 1e6 / sample_rate_hz / carrier_power)


class SeedNoiseGenerationTests(unittest.TestCase):
    def setUp(self):
        self.x = _bandlimited_carrier(8192, 300, 11)

    def _config(self, **overrides):
        values = {
            "device_config": DeviceConfig(sample_rate_hz=491.52e6),
            "runtime_config": {"mu": 0.5},
        }
        values.update(overrides)
        return ClosedLoopConfig(**values)

    def test_noise_psd_matches_the_configured_offset_per_integration_band(self):
        for sample_rate_hz, psd_db in (
            (491.52e6, -25.0),
            (983.04e6, -25.0),
            (245.76e6, -40.0),
            (100.0e6, -10.0),
        ):
            with self.subTest(sample_rate_hz=sample_rate_hz, psd_db=psd_db):
                config = self._config(
                    device_config=DeviceConfig(sample_rate_hz=sample_rate_hz),
                    seed_noise_psd_db=psd_db,
                )
                seed = _generate_seed_waveform(self.x, config)
                measured = _noise_to_carrier_db(seed, self.x, sample_rate_hz)
                # The realized noise power of one deterministic draw scatters
                # by a fraction of a dB, so the tolerance covers estimator
                # variance rather than generator bias.
                self.assertAlmostEqual(measured, psd_db, delta=0.3)

    def test_seed_preserves_the_reference_rms_and_stays_within_the_peak_limit(self):
        seed = _generate_seed_waveform(self.x, self._config())
        reference_rms = float(np.sqrt(np.mean(np.abs(self.x) ** 2)))
        seed_rms = float(np.sqrt(np.mean(np.abs(seed) ** 2)))
        self.assertAlmostEqual(seed_rms, reference_rms, places=12)
        self.assertLessEqual(float(np.max(np.abs(seed))), 1.0)
        self.assertFalse(np.array_equal(seed, self.x))

    def test_generation_is_deterministic_per_seed(self):
        config = self._config()
        first = _generate_seed_waveform(self.x, config)
        second = _generate_seed_waveform(self.x, config)
        np.testing.assert_array_equal(first, second)
        other = _generate_seed_waveform(
            self.x, self._config(seed_noise_seed=config.seed_noise_seed + 1)
        )
        self.assertFalse(np.array_equal(first, other))

    def test_disabled_seed_returns_a_detached_copy_of_the_reference(self):
        config = self._config(seed_noise_enabled=False)
        seed = _generate_seed_waveform(self.x, config)
        np.testing.assert_array_equal(seed, self.x)
        self.assertIsNot(seed, self.x)

    def test_defaults_match_the_user_specification(self):
        config = self._config()
        self.assertTrue(SEED_NOISE_DEFAULT_ENABLED)
        self.assertEqual(config.seed_noise_enabled, True)
        self.assertEqual(config.seed_noise_psd_db, SEED_NOISE_DEFAULT_PSD_DB)
        self.assertEqual(SEED_NOISE_DEFAULT_PSD_DB, -25.0)
        self.assertEqual(
            config.seed_noise_bandwidth_hz, SEED_NOISE_DEFAULT_BANDWIDTH_HZ
        )
        self.assertEqual(
            SEED_NOISE_DEFAULT_BANDWIDTH_HZ, 1e6
        )
        self.assertEqual(config.seed_noise_seed, SEED_NOISE_DEFAULT_SEED)

    def test_invalid_seed_noise_configurations_are_rejected(self):
        cases = {
            "psd too low": {"seed_noise_psd_db": -120.0},
            "psd too high": {"seed_noise_psd_db": 30.0},
            "bandwidth zero": {"seed_noise_bandwidth_hz": 0.0},
            "bandwidth too large": {"seed_noise_bandwidth_hz": 1e12},
            "negative rng seed": {"seed_noise_seed": -1},
            "psd wrong type": {"seed_noise_psd_db": "loud"},
            "enabled wrong type": {"seed_noise_enabled": 1},
        }
        for name, overrides in cases.items():
            with self.subTest(case=name), self.assertRaises((TypeError, ValueError)):
                self._config(**overrides)

    def test_noise_to_carrier_ceiling_is_enforced(self):
        # -5 dB per 1 MHz at 983.04 MS/s integrates to ~+24.9 dB total,
        # which stays legal; -5 dB is valid but extreme. Push over the
        # ceiling instead with the maximum PSD and full-rate bandwidth.
        with self.assertRaises(ValueError):
            self._config(
                device_config=DeviceConfig(sample_rate_hz=983.04e6),
                seed_noise_psd_db=20.0,
            )


class SeedNoiseClosedLoopTests(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            "device_config": DeviceConfig(
                sample_rate_hz=491.52e6,
                average_segment_count=4,
                target_power_dbm=-15.0,
                safety_power_limit_dbm=0.0,
                initial_attenuation_db=30.0,
                min_attenuation_db=0.0,
                max_attenuation_db=60.0,
                settle_seconds=0.0,
                max_adjustments=100,
                call_timeout_seconds=5.0,
                device_options={
                    "max_capture_samples": 8192,
                    "noise_dbfs": -100.0,
                    "random_seed": 7,
                },
            ),
            "runtime_name": "basic_ilc",
            "runtime_config": {"mu": 0.35},
            "max_iterations": 6,
        }
        values.update(overrides)
        return ClosedLoopConfig(**values)

    def test_iteration_zero_transmits_and_records_the_noisy_seed(self):
        config = self._config()
        controller = ClosedLoopController(
            create_rf_bench("simulated"),
            power_controller=PowerController(sleep_fn=lambda _: None),
        )
        controller.connect()
        controller.apply_config(config)
        x = _bandlimited_carrier(4096, 100, 3)
        controller.load_reference(x)
        controller.start_reference_transmission()
        controller.tune_power()
        record = controller.calibrate()

        seed = _generate_seed_waveform(controller.snapshot().x, config)
        np.testing.assert_array_equal(record.y, seed)
        self.assertFalse(np.array_equal(record.y, controller.snapshot().x))
        self.assertTrue(record.digital_safety.passed)
        self.assertAlmostEqual(
            record.digital_safety.candidate_peak,
            float(np.max(np.abs(seed))),
            places=12,
        )
        controller.disconnect()

    def test_reference_restart_reuses_the_same_seed_within_one_run(self):
        config = self._config()
        controller = ClosedLoopController(
            create_rf_bench("simulated"),
            power_controller=PowerController(sleep_fn=lambda _: None),
        )
        controller.connect()
        controller.apply_config(config)
        controller.load_reference(_bandlimited_carrier(4096, 100, 3))
        controller.start_reference_transmission()
        controller.tune_power()
        first = controller.snapshot()

        controller.stop_transmission()
        controller.start_reference_transmission()
        record = controller.calibrate()
        seed = _generate_seed_waveform(first.x, config)
        np.testing.assert_array_equal(record.y, seed)
        controller.disconnect()

    def test_ilc_denoises_the_seed_noise_on_a_mildly_compressive_pa(self):
        # Mild third-order compression keeps the identity-direction ILC
        # stable, so the loop removes the injected seed noise and converges
        # toward the clean reference (the intended "denoising" behavior).
        pa = [
            {"p": 1, "m": 0, "real": 1.0, "imag": 0.0},
            {"p": 3, "m": 0, "real": -0.35, "imag": 0.05},
        ]
        device_config = dict(self._config().device_config.to_dict())
        device_config["device_options"] = dict(
            device_config["device_options"], pa_coefficients=pa
        )
        config = self._config(
            device_config=DeviceConfig(**device_config),
            max_iterations=8,
        )
        controller = ClosedLoopController(
            create_rf_bench("simulated"),
            power_controller=PowerController(sleep_fn=lambda _: None),
        )
        controller.connect()
        controller.apply_config(config)
        controller.load_reference(_bandlimited_carrier(4096, 100, 3))
        snapshot = controller.run_auto()

        self.assertIs(snapshot.state, ControllerState.COMPLETED)
        history = [record.preprocessing.nmse_db for record in snapshot.records]
        self.assertLess(history[0], -0.5, "seed noise must dominate iteration zero")
        self.assertLess(history[-1], history[0] - 10.0)
        self.assertTrue(all(record.digital_safety.passed for record in snapshot.records))
        controller.disconnect()

    def test_extreme_seed_noise_fails_closed_before_transmission(self):
        # A noise level whose normalized seed cannot fit the digital peak
        # envelope must be rejected at the TX boundary, never uploaded.
        device_config = dict(self._config().device_config.to_dict())
        device_config["device_options"] = dict(device_config["device_options"])
        config = self._config(
            device_config=DeviceConfig(**device_config),
            reference_target_rms_dbfs=-8.0,
            seed_noise_psd_db=-5.0,
        )
        controller = ClosedLoopController(
            create_rf_bench("simulated"),
            power_controller=PowerController(sleep_fn=lambda _: None),
        )
        controller.connect()
        controller.apply_config(config)
        # A low-PAPR carrier keeps the clean reference inside the digital
        # envelope while the noise-dominated seed exceeds the peak limit.
        controller.load_reference(_bandlimited_carrier(4096, 8, 3))
        with self.assertRaises(DigitalSafetyError):
            controller.start_reference_transmission()
        snapshot = controller.snapshot()
        self.assertIs(snapshot.state, ControllerState.FAILED)
        self.assertFalse(snapshot.transmitting)
        controller.disconnect()


class SeedNoiseConfigContractTests(unittest.TestCase):
    def test_config_json_accepts_seed_noise_fields(self):
        parsed = parse_configuration_json(
            """
            {
                "device_type": "simulated",
                "device_config": {"sample_rate_hz": 491520000},
                "seed_noise_enabled": true,
                "seed_noise_psd_db": -30.0,
                "seed_noise_bandwidth_hz": 1000000.0,
                "seed_noise_seed": 12
            }
            """
        )
        self.assertTrue(parsed.closed_loop.seed_noise_enabled)
        self.assertEqual(parsed.closed_loop.seed_noise_psd_db, -30.0)
        self.assertEqual(parsed.closed_loop.seed_noise_bandwidth_hz, 1e6)
        self.assertEqual(parsed.closed_loop.seed_noise_seed, 12)

    def test_config_json_applies_defaults_when_seed_noise_fields_are_absent(self):
        parsed = parse_configuration_json(
            """
            {
                "device_type": "simulated",
                "device_config": {"sample_rate_hz": 491520000}
            }
            """
        )
        self.assertTrue(parsed.closed_loop.seed_noise_enabled)
        self.assertEqual(parsed.closed_loop.seed_noise_psd_db, -25.0)
        self.assertEqual(parsed.closed_loop.seed_noise_bandwidth_hz, 1e6)
        self.assertEqual(parsed.closed_loop.seed_noise_seed, 0)

    def test_config_json_rejects_invalid_seed_noise_values(self):
        with self.assertRaises(Exception):
            parse_configuration_json(
                """
                {
                    "device_type": "simulated",
                    "device_config": {"sample_rate_hz": 491520000},
                    "seed_noise_enabled": "yes"
                }
                """
            )


if __name__ == "__main__":
    unittest.main()
