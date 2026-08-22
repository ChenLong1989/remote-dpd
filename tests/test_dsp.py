import unittest

import numpy as np

from remote_dpd.dsp import (
    align_signal,
    align_and_average,
    circular_fir,
    legacy_gain_phase_calibration,
    rms,
)


class DSPTests(unittest.TestCase):
    def test_alignment_corrects_integer_delay_and_gain(self):
        rng = np.random.default_rng(4)
        reference = rng.normal(size=128) + 1j * rng.normal(size=128)
        measured = np.roll(reference, 7) * (0.4 + 0.2j)
        aligned, delay, gain = align_signal(reference, measured)
        self.assertLess(np.linalg.norm(aligned - reference) / np.linalg.norm(reference), 1e-10)
        self.assertAlmostEqual(delay, -7.0, places=6)
        self.assertGreater(abs(gain), 1.0)

    def test_legacy_calibration_is_cross_phase_times_rms_ratio(self):
        rng = np.random.default_rng(20260822)
        reference = rng.normal(size=257) + 1j * rng.normal(size=257)
        envelope = np.abs(reference)
        nonlinear = (
            (0.12 + 0.8 * envelope**2)
            * reference
            * np.exp(1j * (0.7 + 0.9 * envelope**2))
        )
        measured = np.roll((0.21 - 0.37j) * nonlinear, 5)
        time_aligned, delay, _ = align_signal(reference, measured, gain_phase=False)

        cross = np.vdot(time_aligned, reference)
        expected = (cross / abs(cross)) * rms(reference) / rms(time_aligned)
        coefficient = legacy_gain_phase_calibration(reference, time_aligned)
        calibrated, calibrated_delay, aligned_coefficient = align_signal(reference, measured)

        self.assertAlmostEqual(delay, calibrated_delay, places=12)
        self.assertAlmostEqual(coefficient.real, expected.real, places=14)
        self.assertAlmostEqual(coefficient.imag, expected.imag, places=14)
        self.assertAlmostEqual(aligned_coefficient.real, expected.real, places=14)
        self.assertAlmostEqual(aligned_coefficient.imag, expected.imag, places=14)
        np.testing.assert_allclose(calibrated, coefficient * time_aligned, rtol=2e-15, atol=2e-15)

        least_squares = np.vdot(time_aligned, reference) / np.vdot(
            time_aligned,
            time_aligned,
        )
        self.assertGreater(abs(coefficient - least_squares), 1e-3)

    def test_packed_feedback_is_averaged(self):
        reference = np.arange(32, dtype=np.float64) + 1j
        captures = np.stack([reference, reference * (1 + 0.01j)], axis=1)
        feedback = np.tile(captures, 5).reshape(-1, order="F")
        averaged, delays, gains = align_and_average(reference, feedback)
        self.assertEqual(len(delays), 10)
        self.assertEqual(averaged.size, reference.size)

    def test_circular_fir_preserves_length(self):
        signal = np.arange(8, dtype=np.complex128)
        output = circular_fir(signal, np.array([0.25, 0.5, 0.25]))
        self.assertEqual(output.shape, signal.shape)


if __name__ == "__main__":
    unittest.main()
