import unittest

import numpy as np

from experiments.metrics import (
    auec,
    bilateral_aclr_db,
    binned_amam_ampm_error,
    fixed_domain_nmse,
    fixed_domain_nmse_db,
    known_grid_evm,
    papr_db,
)
from experiments.scenarios import (
    GainRolloffPA,
    HardSaturationPA,
    RappPA,
    make_ampm_scenario,
    make_hammerstein_pa,
    make_wiener_pa,
    rapp_amam,
    rapp_inverse_amplitude,
    rapp_reachable,
)
from experiments.waveforms import (
    DEFAULT_NFFT,
    DEFAULT_OCCUPIED_PER_SIDE,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SYMBOL_COUNT,
    generate_ofdm_waveform,
    named_seed_sequence,
)


class WaveformPrimitiveTests(unittest.TestCase):
    def test_named_seed_and_waveform_are_deterministic(self):
        first_seed = named_seed_sequence("confirmatory", "waveform", 7)
        unrelated = named_seed_sequence("pilot", "waveform", 2)
        second_seed = named_seed_sequence("confirmatory", "waveform", 7)

        self.assertEqual(first_seed.entropy, 20260822)
        self.assertEqual(first_seed.spawn_key, second_seed.spawn_key)
        self.assertNotEqual(first_seed.spawn_key, unrelated.spawn_key)

        first = generate_ofdm_waveform(first_seed)
        second = generate_ofdm_waveform(second_seed)
        different = generate_ofdm_waveform(unrelated)
        np.testing.assert_array_equal(first.samples, second.samples)
        np.testing.assert_array_equal(first.grid, second.grid)
        self.assertFalse(np.array_equal(first.samples, different.samples))

    def test_frozen_frequency_mapping_and_unit_power(self):
        waveform = generate_ofdm_waveform(("mapping", 0))
        self.assertEqual(waveform.samples.shape, (DEFAULT_SAMPLE_COUNT,))
        self.assertEqual(waveform.grid.shape, (DEFAULT_SYMBOL_COUNT, DEFAULT_NFFT))
        self.assertAlmostEqual(float(np.mean(np.abs(waveform.samples) ** 2)), 1.0, places=14)

        occupied = np.zeros(DEFAULT_NFFT, dtype=bool)
        occupied[1 : DEFAULT_OCCUPIED_PER_SIDE + 1] = True
        occupied[-DEFAULT_OCCUPIED_PER_SIDE :] = True
        self.assertFalse(occupied[0])
        self.assertTrue(np.all(waveform.grid[:, occupied] != 0.0))
        np.testing.assert_array_equal(waveform.grid[:, ~occupied], 0.0)

        recovered = np.fft.fft(
            waveform.samples.reshape(DEFAULT_SYMBOL_COUNT, DEFAULT_NFFT),
            axis=1,
            norm="ortho",
        )
        np.testing.assert_allclose(recovered, waveform.grid, rtol=0.0, atol=3e-14)


class ScenarioPrimitiveTests(unittest.TestCase):
    def test_rapp_inverse_and_reachability_are_analytic(self):
        output_radius = np.array([0.0, 0.2, 0.55, 0.75, 0.90, 0.97])
        input_radius = rapp_inverse_amplitude(output_radius, a_sat=1.0, p=4.0)
        reconstructed = rapp_amam(input_radius, a_sat=1.0, p=4.0)
        np.testing.assert_allclose(reconstructed, output_radius, rtol=2e-14, atol=2e-14)
        np.testing.assert_array_equal(
            rapp_reachable(np.array([0.0, 0.999, 1.0, 1.1])),
            np.array([True, True, False, False]),
        )
        with self.assertRaises(ValueError):
            rapp_inverse_amplitude(np.array([1.0]))

    def test_rapp_ampm_jvp_matches_centered_finite_difference(self):
        rng = np.random.default_rng(91)
        input_signal = 0.05 + 0.75 * (
            rng.normal(size=64) + 1j * rng.normal(size=64)
        ) / np.sqrt(2.0)
        tangent = (rng.normal(size=64) + 1j * rng.normal(size=64)) / np.sqrt(2.0)
        pa = RappPA(a_sat=1.0, p=4.0, phase_max_deg=135.0, r0=0.21)

        epsilon = 2e-6
        finite_difference = (
            pa.forward(input_signal + epsilon * tangent)
            - pa.forward(input_signal - epsilon * tangent)
        ) / (2.0 * epsilon)
        analytic = pa.jvp(input_signal, tangent)
        relative_error = np.linalg.norm(analytic - finite_difference) / np.linalg.norm(finite_difference)
        self.assertLess(relative_error, 2e-8)

    def test_rapp_ampm_vjp_is_real_adjoint_and_scalar_derivative(self):
        rng = np.random.default_rng(92)
        input_signal = 0.1 + 0.5 * (
            rng.normal(size=48) + 1j * rng.normal(size=48)
        )
        tangent = rng.normal(size=48) + 1j * rng.normal(size=48)
        cotangent = rng.normal(size=48) + 1j * rng.normal(size=48)
        pa = RappPA(a_sat=1.0, p=4.0, phase_max_deg=90.0, r0=0.21)

        jvp_inner = float(np.vdot(cotangent, pa.jvp(input_signal, tangent)).real)
        vjp = pa.vjp(input_signal, cotangent)
        vjp_inner = float(np.vdot(vjp, tangent).real)
        self.assertAlmostEqual(jvp_inner, vjp_inner, places=11)

        epsilon = 2e-6
        objective_plus = float(
            np.vdot(cotangent, pa.forward(input_signal + epsilon * tangent)).real
        )
        objective_minus = float(
            np.vdot(cotangent, pa.forward(input_signal - epsilon * tangent)).real
        )
        finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon)
        self.assertAlmostEqual(finite_difference, vjp_inner, delta=2e-8 * max(1.0, abs(vjp_inner)))

    def test_origin_derivative_and_stress_pa_behavior(self):
        direction = np.array([1.0 + 2.0j, -0.4 + 0.3j])
        pa = RappPA(phase_max_deg=45.0)
        expected = np.exp(1j * np.deg2rad(45.0)) * direction
        np.testing.assert_allclose(pa.jvp(np.zeros(2), direction), expected, atol=1e-14)

        hard = HardSaturationPA(a_sat=1.0)
        hard_output = hard.forward(np.array([0.5 + 0.0j, 2.0j]))
        np.testing.assert_allclose(np.abs(hard_output), [0.5, 1.0])

        rolloff = GainRolloffPA(turnover=0.7)
        slopes = rolloff.amplitude_derivative(np.array([0.5, 0.7, 1.0]))
        self.assertGreater(slopes[0], 0.0)
        self.assertAlmostEqual(slopes[1], 0.0, places=14)
        self.assertLess(slopes[2], 0.0)

    def test_low_power_ampm_scenario_and_dynamic_pa_are_reproducible(self):
        waveform = generate_ofdm_waveform(("scenario", 3))
        scenario = make_ampm_scenario(waveform.samples, 135.0)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.abs(scenario.desired) ** 2))),
            0.35,
            places=14,
        )

        wiener_first = make_wiener_pa(5)
        wiener_second = make_wiener_pa(5)
        hammerstein = make_hammerstein_pa(5)
        np.testing.assert_array_equal(wiener_first.taps, wiener_second.taps)
        self.assertAlmostEqual(abs(np.sum(wiener_first.taps) - 1.0), 0.0, places=14)
        self.assertAlmostEqual(abs(np.sum(hammerstein.taps) - 1.0), 0.0, places=14)
        self.assertEqual(wiener_first.forward(waveform.samples[:128]).shape, (128,))
        self.assertEqual(hammerstein.forward(waveform.samples[:128]).shape, (128,))


class MetricPrimitiveTests(unittest.TestCase):
    def test_fixed_nmse_auec_and_papr_have_analytic_values(self):
        reference = np.ones(16, dtype=np.complex128)
        measured = 1.1 * reference
        self.assertAlmostEqual(fixed_domain_nmse(reference, measured), 0.01, places=14)
        self.assertAlmostEqual(fixed_domain_nmse_db(reference, measured), -20.0, places=12)
        self.assertAlmostEqual(auec(np.array([-10.0, -20.0])), 0.055, places=14)
        self.assertAlmostEqual(papr_db(np.array([2.0, 0.0, 0.0, 0.0])), 10.0 * np.log10(4.0))

    def test_binned_amam_ampm_error_has_known_values(self):
        pa_input = np.array([0.25 + 0.0j, 0.75 + 0.0j])
        measured = np.array([0.35, 0.95]) * np.exp(1j * np.deg2rad(30.0))
        result = binned_amam_ampm_error(
            pa_input,
            measured,
            bin_edges=np.array([0.0, 0.5, 1.0]),
        )
        np.testing.assert_array_equal(result.counts, [1, 1])
        np.testing.assert_allclose(result.amam_rmse, [0.10, 0.20], atol=1e-14)
        np.testing.assert_allclose(result.ampm_bias_deg, [30.0, 30.0], atol=1e-14)
        np.testing.assert_allclose(result.ampm_rmse_deg, [30.0, 30.0], atol=1e-14)

    def test_bilateral_aclr_uses_exact_lower_and_upper_bins(self):
        nfft = 32
        occupied_per_side = 4
        spectrum = np.zeros(nfft, dtype=np.complex128)
        spectrum[1] = 1.0
        spectrum[occupied_per_side + 1] = 0.1
        spectrum[nfft - occupied_per_side - 1] = 0.2
        samples = np.fft.ifft(spectrum, norm="ortho")

        result = bilateral_aclr_db(
            samples,
            nfft=nfft,
            occupied_per_side=occupied_per_side,
            adjacent_bins_per_side=4,
        )
        self.assertAlmostEqual(result.main_power, 1.0, places=14)
        self.assertAlmostEqual(result.upper_power, 0.01, places=14)
        self.assertAlmostEqual(result.lower_power, 0.04, places=14)
        self.assertAlmostEqual(result.upper_db, 20.0, places=12)
        self.assertAlmostEqual(result.lower_db, 10.0 * np.log10(25.0), places=12)

    def test_known_grid_raw_and_one_tap_evm(self):
        symbol_count = 2
        nfft = 8
        grid = np.zeros((symbol_count, nfft), dtype=np.complex128)
        grid[:, 1] = np.array([1.0 + 0.0j, -1.0 + 1.0j])
        grid[:, -1] = np.array([0.5j, 0.75 - 0.25j])
        reference_samples = np.fft.ifft(grid, axis=1, norm="ortho").reshape(-1)
        gain = 0.5j
        measured = gain * reference_samples

        result = known_grid_evm(measured, grid, occupied_bins=np.array([1, nfft - 1]))
        expected_raw = 100.0 * abs(gain - 1.0)
        self.assertAlmostEqual(result.raw_percent, expected_raw, places=13)
        self.assertLess(result.one_tap_percent, 1e-13)
        self.assertAlmostEqual(abs(result.fitted_gain - gain), 0.0, places=14)
        self.assertEqual(result.resource_element_count, 4)


if __name__ == "__main__":
    unittest.main()
