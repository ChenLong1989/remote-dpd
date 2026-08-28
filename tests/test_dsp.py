import unittest

import numpy as np

from remote_dpd.dsp import align_signal, fractional_shift, nmse_db


class DSPTests(unittest.TestCase):
    def test_alignment_corrects_integer_delay_and_gain(self):
        rng = np.random.default_rng(4)
        reference = rng.normal(size=128) + 1j * rng.normal(size=128)
        measured = np.roll(reference, 7) * (0.4 + 0.2j)
        aligned, delay, gain = align_signal(reference, measured)
        self.assertLess(
            np.linalg.norm(aligned - reference) / np.linalg.norm(reference), 1e-10
        )
        self.assertAlmostEqual(delay, -7.0, places=6)
        self.assertGreater(abs(gain), 1.0)

    def test_fractional_shift_is_periodic_and_invertible(self):
        signal = np.arange(32) + 1j * np.arange(32)[::-1]
        shifted = fractional_shift(signal, 2.375)
        restored = fractional_shift(shifted, -2.375)
        np.testing.assert_allclose(restored, signal, atol=1e-12)

    def test_nmse_reports_zero_error_floor(self):
        signal = np.arange(16, dtype=np.float64) + 1j
        self.assertLess(nmse_db(signal, signal), -300.0)


if __name__ == "__main__":
    unittest.main()
