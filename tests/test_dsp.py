import unittest
from unittest.mock import patch

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

    def test_alignment_matches_legacy_global_fractional_search(self):
        rng = np.random.default_rng(20260830)
        for sample_count in (31, 32, 127, 128):
            reference = rng.normal(size=sample_count) + 1j * rng.normal(
                size=sample_count
            )
            for injected_delay in (-7.3125, -0.40625, 2.25, 8.46875):
                measured = fractional_shift(reference, injected_delay) * (0.4 + 0.2j)

                expected = legacy_align_signal(reference, measured)
                actual = align_signal(reference, measured)

                self.assertEqual(actual[1], expected[1])
                np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
                self.assertAlmostEqual(actual[2].real, expected[2].real, places=12)
                self.assertAlmostEqual(actual[2].imag, expected[2].imag, places=12)

    def test_alignment_keeps_all_fft_work_at_the_original_length(self):
        rng = np.random.default_rng(8)
        reference = rng.normal(size=129) + 1j * rng.normal(size=129)
        measured = fractional_shift(reference, 3.21875)
        original_fft = np.fft.fft
        original_ifft = np.fft.ifft

        with (
            patch("remote_dpd.dsp.np.fft.fft", wraps=original_fft) as fft,
            patch("remote_dpd.dsp.np.fft.ifft", wraps=original_ifft) as ifft,
        ):
            _, delay, _ = align_signal(reference, measured)

        self.assertEqual(delay, -3.21875)
        self.assertTrue(fft.call_args_list)
        self.assertTrue(ifft.call_args_list)
        self.assertTrue(
            all(
                np.asarray(call.args[0]).size == reference.size
                for call in fft.call_args_list
            )
        )
        self.assertTrue(
            all(
                np.asarray(call.args[0]).size == reference.size
                for call in ifft.call_args_list
            )
        )

    def test_alignment_matches_legacy_for_multitone_and_period_boundary(self):
        sample_count = 96
        indices = np.arange(sample_count)
        reference = (
            np.exp(2j * np.pi * 5 * indices / sample_count)
            + 0.7 * np.exp(2j * np.pi * 6 * indices / sample_count)
            + 0.2 * np.exp(-2j * np.pi * 17 * indices / sample_count)
        )
        for injected_delay in (-47.71875, 12.34375, 47.65625):
            measured = fractional_shift(reference, injected_delay) * np.exp(0.8j)

            expected = legacy_align_signal(reference, measured)
            actual = align_signal(reference, measured)

            self.assertEqual(actual[1], expected[1])
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
            self.assertAlmostEqual(actual[2].real, expected[2].real, places=12)
            self.assertAlmostEqual(actual[2].imag, expected[2].imag, places=12)

    def test_nmse_reports_zero_error_floor(self):
        signal = np.arange(16, dtype=np.float64) + 1j
        self.assertLess(nmse_db(signal, signal), -300.0)


def legacy_align_signal(
    reference: np.ndarray,
    measured: np.ndarray,
) -> tuple[np.ndarray, float, complex]:
    factor = 32
    size = reference.size * factor
    reference_upsampled = legacy_fft_resample(reference, size)
    measured_upsampled = legacy_fft_resample(measured, size)
    correlation = np.fft.ifft(
        np.fft.fft(reference_upsampled) * np.conj(np.fft.fft(measured_upsampled))
    )
    peak = int(np.argmax(np.abs(correlation)))
    signed_peak = peak if peak < size / 2 else peak - size
    delay = float(signed_peak) / factor
    aligned = fractional_shift(measured, delay)
    cross = np.vdot(aligned, reference)
    phase = cross / abs(cross)
    input_rms = np.sqrt(np.mean(np.abs(aligned) ** 2))
    reference_rms = np.sqrt(np.mean(np.abs(reference) ** 2))
    coefficient = phase * reference_rms / input_rms
    return aligned * coefficient, delay, complex(coefficient)


def legacy_fft_resample(signal: np.ndarray, size: int) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    output = np.zeros(size, dtype=np.complex128)
    start = (size - spectrum.size) // 2
    output[start : start + spectrum.size] = spectrum
    return np.fft.ifft(np.fft.ifftshift(output)) * (size / signal.size)


if __name__ == "__main__":
    unittest.main()
