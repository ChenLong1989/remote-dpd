import unittest

import numpy as np

from remote_dpd.dsp import fractional_shift
from remote_dpd.preprocessing import CaptureBatch, FeedbackPreprocessor

SAMPLE_RATE_HZ = 122.88e6


def make_reference(length: int = 128) -> np.ndarray:
    rng = np.random.default_rng(20260827)
    reference = rng.normal(size=length) + 1j * rng.normal(size=length)
    return reference / np.sqrt(np.mean(np.abs(reference) ** 2))


def make_capture(
    reference: np.ndarray,
    *,
    delay_samples: float,
    phase_radians: float,
    amplitude: float,
) -> np.ndarray:
    return (
        fractional_shift(reference, delay_samples)
        * amplitude
        * np.exp(1j * phase_radians)
    )


class CaptureBatchTests(unittest.TestCase):
    def test_copies_and_exposes_packed_segments(self):
        source = np.arange(24, dtype=np.complex128)
        batch = CaptureBatch(
            iq=source,
            segment_length=8,
            segment_count=3,
            sample_rate_hz=SAMPLE_RATE_HZ,
        )

        source[0] = 999.0

        self.assertEqual(batch.segments.shape, (3, 8))
        self.assertEqual(batch.iq[0], 0.0)
        self.assertFalse(batch.iq.flags.writeable)
        self.assertFalse(batch.segments.flags.writeable)

    def test_rejects_invalid_shape_length_and_values(self):
        valid = {
            "segment_length": 4,
            "segment_count": 2,
            "sample_rate_hz": SAMPLE_RATE_HZ,
        }
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            CaptureBatch(iq=np.ones((2, 4)), **valid)
        with self.assertRaisesRegex(ValueError, r"segment_length \* segment_count"):
            CaptureBatch(iq=np.ones(7), **valid)
        with self.assertRaisesRegex(ValueError, "finite"):
            CaptureBatch(iq=np.array([1, 2, 3, 4, 5, 6, 7, np.nan]), **valid)
        with self.assertRaisesRegex(TypeError, "numeric"):
            CaptureBatch(iq=np.array(["1"] * 8), **valid)

    def test_rejects_invalid_metadata(self):
        iq = np.ones(8)
        with self.assertRaisesRegex(ValueError, "segment_length"):
            CaptureBatch(iq, 0, 2, SAMPLE_RATE_HZ)
        with self.assertRaisesRegex(TypeError, "segment_count"):
            CaptureBatch(iq, 4, 2.0, SAMPLE_RATE_HZ)
        with self.assertRaisesRegex(ValueError, "sample_rate_hz"):
            CaptureBatch(iq, 4, 2, float("inf"))
        with self.assertRaisesRegex(TypeError, "coherent_within_batch"):
            CaptureBatch(iq, 4, 2, SAMPLE_RATE_HZ, coherent_within_batch=1)


class FeedbackPreprocessorTests(unittest.TestCase):
    def test_reference_and_results_are_defensive_copies(self):
        reference = make_reference()
        expected = reference.copy()
        capture = 0.5 * expected
        preprocessor = FeedbackPreprocessor(reference, SAMPLE_RATE_HZ)
        batch = CaptureBatch(capture, reference.size, 1, SAMPLE_RATE_HZ)

        reference[:] = 0.0
        capture[:] = 0.0
        result = preprocessor.process([batch])

        np.testing.assert_allclose(result.z, expected, atol=1e-12)
        self.assertFalse(np.shares_memory(result.z, preprocessor.reference))
        self.assertFalse(np.shares_memory(result.aligned_average, batch.iq))
        self.assertFalse(result.z.flags.writeable)
        self.assertFalse(result.aligned_average.flags.writeable)
        with self.assertRaises(ValueError):
            preprocessor.reference.setflags(write=True)
        with self.assertRaises(ValueError):
            batch.iq.setflags(write=True)
        with self.assertRaises(ValueError):
            result.z.setflags(write=True)

    def test_coherent_batch_estimates_first_segment_and_reuses_alignment(self):
        reference = make_reference()
        capture = make_capture(
            reference,
            delay_samples=2.25,
            phase_radians=0.7,
            amplitude=0.4,
        )
        batch = CaptureBatch(
            np.tile(capture, 3),
            reference.size,
            3,
            SAMPLE_RATE_HZ,
            coherent_within_batch=True,
        )

        result = FeedbackPreprocessor(reference, SAMPLE_RATE_HZ).process([batch])
        diagnostic = result.batch_diagnostics[0]

        np.testing.assert_allclose(result.aligned_average, 0.4 * reference, atol=1e-11)
        np.testing.assert_allclose(result.z, reference, atol=1e-11)
        self.assertAlmostEqual(result.gain_correction, 2.5, places=11)
        self.assertEqual(diagnostic.alignment_estimate_count, 1)
        self.assertEqual(
            tuple(segment.alignment_estimated for segment in diagnostic.segments),
            (True, False, False),
        )
        np.testing.assert_allclose(diagnostic.delays_samples, [-2.25] * 3, atol=1e-12)
        np.testing.assert_allclose(diagnostic.phase_radians, [-0.7] * 3, atol=1e-12)

    def test_noncoherent_batch_estimates_every_segment(self):
        reference = make_reference()
        delays = (2.0, -3.5, 0.3125)
        phases = (0.3, -1.0, 1.2)
        captures = [
            make_capture(
                reference,
                delay_samples=delay,
                phase_radians=phase,
                amplitude=0.7,
            )
            for delay, phase in zip(delays, phases)
        ]
        batch = CaptureBatch(
            np.concatenate(captures),
            reference.size,
            len(captures),
            SAMPLE_RATE_HZ,
            coherent_within_batch=False,
        )

        result = FeedbackPreprocessor(reference, SAMPLE_RATE_HZ).process([batch])
        diagnostic = result.batch_diagnostics[0]

        np.testing.assert_allclose(result.z, reference, atol=1e-11)
        self.assertEqual(diagnostic.alignment_estimate_count, 3)
        np.testing.assert_allclose(diagnostic.delays_samples, [-2.0, 3.5, -0.3125])
        np.testing.assert_allclose(
            diagnostic.phase_radians, -np.asarray(phases), atol=1e-12
        )

    def test_different_batches_estimate_alignment_independently(self):
        reference = make_reference()
        first_capture = make_capture(
            reference,
            delay_samples=1.5,
            phase_radians=0.4,
            amplitude=0.5,
        )
        second_capture = make_capture(
            reference,
            delay_samples=-4.25,
            phase_radians=-0.8,
            amplitude=0.5,
        )
        batches = [
            CaptureBatch(
                np.tile(first_capture, 2), reference.size, 2, SAMPLE_RATE_HZ, True
            ),
            CaptureBatch(
                np.tile(second_capture, 2), reference.size, 2, SAMPLE_RATE_HZ, True
            ),
        ]

        result = FeedbackPreprocessor(reference, SAMPLE_RATE_HZ).process(batches)

        np.testing.assert_allclose(result.z, reference, atol=1e-11)
        self.assertEqual(result.segment_count, 4)
        self.assertEqual(
            tuple(batch.alignment_estimate_count for batch in result.batch_diagnostics),
            (1, 1),
        )
        np.testing.assert_allclose(result.delays_samples, [-1.5, -1.5, 4.25, 4.25])
        np.testing.assert_allclose(result.phase_radians, [-0.4, -0.4, 0.8, 0.8])

    def test_explicit_gain_is_reused_without_reestimation(self):
        reference = make_reference()
        preprocessor = FeedbackPreprocessor(reference, SAMPLE_RATE_HZ)
        calibration_capture = make_capture(
            reference,
            delay_samples=1.0,
            phase_radians=0.2,
            amplitude=0.25,
        )
        calibration = preprocessor.process(
            [CaptureBatch(calibration_capture, reference.size, 1, SAMPLE_RATE_HZ)]
        )
        changed_capture = make_capture(
            reference,
            delay_samples=-2.0,
            phase_radians=-0.6,
            amplitude=0.5,
        )

        result = preprocessor.process(
            [CaptureBatch(changed_capture, reference.size, 1, SAMPLE_RATE_HZ)],
            gain_correction=calibration.gain_correction,
        )

        self.assertAlmostEqual(calibration.gain_correction, 4.0, places=11)
        self.assertAlmostEqual(result.gain_correction, 4.0, places=11)
        np.testing.assert_allclose(result.aligned_average, 0.5 * reference, atol=1e-11)
        np.testing.assert_allclose(result.z, 2.0 * reference, atol=1e-11)
        self.assertAlmostEqual(result.nmse_db, 0.0, places=10)

    def test_coherent_average_reduces_noise(self):
        reference = make_reference(256)
        rng = np.random.default_rng(42)
        captures = np.stack(
            [
                reference
                + 0.3
                * (
                    rng.normal(size=reference.size)
                    + 1j * rng.normal(size=reference.size)
                )
                for _ in range(16)
            ]
        )
        preprocessor = FeedbackPreprocessor(reference, SAMPLE_RATE_HZ)
        single = preprocessor.process(
            [CaptureBatch(captures[0], reference.size, 1, SAMPLE_RATE_HZ)]
        )
        averaged = preprocessor.process(
            [
                CaptureBatch(
                    captures.reshape(-1),
                    reference.size,
                    captures.shape[0],
                    SAMPLE_RATE_HZ,
                    coherent_within_batch=True,
                )
            ]
        )

        self.assertLess(averaged.nmse_db, single.nmse_db - 8.0)

    def test_rejects_batch_contract_mismatches(self):
        reference = make_reference()
        preprocessor = FeedbackPreprocessor(reference, SAMPLE_RATE_HZ)
        wrong_length = CaptureBatch(
            np.ones(reference.size // 2),
            reference.size // 2,
            1,
            SAMPLE_RATE_HZ,
        )
        wrong_rate = CaptureBatch(
            np.ones(reference.size), reference.size, 1, SAMPLE_RATE_HZ / 2
        )

        with self.assertRaisesRegex(ValueError, "at least one"):
            preprocessor.process([])
        with self.assertRaisesRegex(TypeError, "CaptureBatch"):
            preprocessor.process([object()])
        with self.assertRaisesRegex(ValueError, "segment_length"):
            preprocessor.process([wrong_length])
        with self.assertRaisesRegex(ValueError, "sample_rate_hz"):
            preprocessor.process([wrong_rate])

    def test_rejects_invalid_reference_and_gain(self):
        reference = make_reference()
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            FeedbackPreprocessor(reference.reshape(2, -1), SAMPLE_RATE_HZ)
        with self.assertRaisesRegex(ValueError, "finite"):
            FeedbackPreprocessor(np.array([1.0, np.inf]), SAMPLE_RATE_HZ)
        with self.assertRaisesRegex(ValueError, "non-zero finite RMS"):
            FeedbackPreprocessor(np.zeros(8), SAMPLE_RATE_HZ)
        with self.assertRaisesRegex(ValueError, "sample_rate_hz"):
            FeedbackPreprocessor(reference, 0.0)

        preprocessor = FeedbackPreprocessor(reference, SAMPLE_RATE_HZ)
        batch = CaptureBatch(reference, reference.size, 1, SAMPLE_RATE_HZ)
        for gain in (0.0, -1.0, float("nan"), float("inf")):
            with (
                self.subTest(gain=gain),
                self.assertRaisesRegex(ValueError, "gain_correction"),
            ):
                preprocessor.process([batch], gain_correction=gain)
        with self.assertRaisesRegex(TypeError, "gain_correction"):
            preprocessor.process([batch], gain_correction=1.0 + 0.0j)

    def test_rejects_zero_feedback_when_calibrating_gain(self):
        reference = make_reference()
        batch = CaptureBatch(
            np.zeros(reference.size), reference.size, 1, SAMPLE_RATE_HZ
        )

        with self.assertRaisesRegex(ValueError, "zero-RMS feedback"):
            FeedbackPreprocessor(reference, SAMPLE_RATE_HZ).process([batch])


if __name__ == "__main__":
    unittest.main()
