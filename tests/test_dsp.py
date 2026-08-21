import unittest

import numpy as np

from remote_dpd.dsp import align_signal, align_and_average, circular_fir


class DSPTests(unittest.TestCase):
    def test_alignment_corrects_integer_delay_and_gain(self):
        rng = np.random.default_rng(4)
        reference = rng.normal(size=128) + 1j * rng.normal(size=128)
        measured = np.roll(reference, 7) * (0.4 + 0.2j)
        aligned, delay, gain = align_signal(reference, measured)
        self.assertLess(np.linalg.norm(aligned - reference) / np.linalg.norm(reference), 1e-10)
        self.assertAlmostEqual(delay, -7.0, places=6)
        self.assertGreater(abs(gain), 1.0)

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
