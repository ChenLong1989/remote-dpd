from __future__ import annotations

import unittest

import numpy as np

from remote_dpd.algorithms import (
    ILCConfig,
    _apply_fir_torch,
    _to_numpy,
    _to_torch,
    create_engine,
    legacy_ilc_update,
)
from remote_dpd.dsp import align_and_average, circular_fir, rms
from remote_dpd.state import SessionState


class AlgorithmStrategyTests(unittest.TestCase):
    def setUp(self):
        samples = np.arange(1280)
        self.reference = (
            0.7 * np.exp(2j * np.pi * samples / 31)
            + 0.2 * np.exp(2j * np.pi * samples / 47)
        ).astype(np.complex128)

    def test_legacy_default_formula_is_unchanged(self):
        current = 0.8 * self.reference
        config = ILCConfig(mu=0.3, alpha=0.2, dtype="complex128")
        result = create_engine("legacy_ilc", config).process(
            self.reference,
            current,
            self.reference,
            SessionState(),
        )
        expected = 0.2 * self.reference + 0.8 * current
        np.testing.assert_allclose(result.output, expected, rtol=2e-15, atol=2e-15)

    def test_legacy_engine_and_numpy_primitive_match_frozen_complex_fir_formula(self):
        rng = np.random.default_rng(20260823)
        reference = (
            self.reference
            + 0.08 * (rng.normal(size=self.reference.size) + 1j * rng.normal(size=self.reference.size))
        )
        current = reference * (0.45 + 0.25 * np.abs(reference) ** 2)
        current *= np.exp(1j * (0.2 + 0.7 * np.abs(reference) ** 2))
        feedback = current * (0.18 + 1.1 * np.abs(current) ** 2)
        feedback *= np.exp(1j * (-0.5 + 1.3 * np.abs(current) ** 2))
        feedback = np.roll((0.31 - 0.22j) * feedback, 4)
        aligned_feedback, _, _ = align_and_average(reference, feedback)

        error_fir = np.array([0.17 + 0.31j, 0.82 - 0.08j, -0.29 + 0.16j])
        tx_fir = np.array([0.09 - 0.13j, 0.76 + 0.22j, 0.18 - 0.27j, -0.05 + 0.11j])
        common = dict(
            mu=0.37,
            alpha=0.23,
            gain_db=2.7,
            phase_threshold=0.19,
            error_fir=error_fir,
            tx_fir=tx_fir,
        )

        outputs = {}
        for phase_compensate in (False, True):
            with self.subTest(phase_compensate=phase_compensate):
                primitive = legacy_ilc_update(
                    reference,
                    current,
                    aligned_feedback,
                    phase_compensate=phase_compensate,
                    numeric_dtype="complex128",
                    **common,
                )
                expected = self._frozen_legacy_formula(
                    reference,
                    current,
                    aligned_feedback,
                    phase_compensate=phase_compensate,
                    **common,
                )
                np.testing.assert_allclose(primitive, expected, rtol=3e-15, atol=3e-15)

                config = ILCConfig(
                    phase_compensate=phase_compensate,
                    dtype="complex128",
                    **common,
                )
                production = create_engine("legacy_ilc", config).process(
                    reference,
                    current,
                    feedback,
                    SessionState(),
                )
                np.testing.assert_allclose(production.aligned_feedback, aligned_feedback, rtol=0.0, atol=0.0)
                np.testing.assert_allclose(production.output, primitive, rtol=3e-15, atol=3e-15)
                outputs[phase_compensate] = primitive

        self.assertGreater(rms(outputs[True] - outputs[False]), 1e-3)

    def test_legacy_default_complex64_phase_path_and_single_tap_noop(self):
        current = self.reference * np.exp(0.6j * np.abs(self.reference) ** 2)
        feedback = current * (0.35 + 0.9 * np.abs(current) ** 2)
        aligned_feedback, _, _ = align_and_average(self.reference, feedback)
        config = ILCConfig(mu=0.29, phase_compensate=True)
        production = create_engine("legacy_ilc", config).process(
            self.reference,
            current,
            feedback,
            SessionState(),
        )
        primitive = legacy_ilc_update(
            self.reference,
            current,
            aligned_feedback,
            mu=config.mu,
            phase_compensate=True,
            phase_threshold=config.phase_threshold,
            numeric_dtype="complex64",
        )
        ignored_single_taps = legacy_ilc_update(
            self.reference,
            current,
            aligned_feedback,
            mu=config.mu,
            phase_compensate=True,
            phase_threshold=config.phase_threshold,
            error_fir=np.array([2.0 - 3.0j]),
            tx_fir=np.array([-4.0 + 5.0j]),
            numeric_dtype="complex64",
        )

        self.assertEqual(primitive.dtype, np.dtype(np.complex64))
        self.assertTrue(np.all(np.isfinite(production.output)))
        np.testing.assert_array_equal(production.output, primitive.astype(np.complex128))
        np.testing.assert_array_equal(ignored_single_taps, primitive)

    @staticmethod
    def _frozen_legacy_formula(
        reference,
        current,
        aligned_feedback,
        *,
        mu,
        alpha,
        gain_db,
        phase_compensate,
        phase_threshold,
        error_fir,
        tx_fir,
    ):
        gain = 10.0 ** (gain_db / 20.0)
        desired = gain * reference
        error = aligned_feedback - desired
        if phase_compensate:
            threshold = max(phase_threshold * rms(desired), np.finfo(np.float64).eps)
            magnitude_current = np.abs(current)
            magnitude_feedback = np.abs(aligned_feedback)
            weight = magnitude_current**2 / (magnitude_current**2 + threshold**2)
            weight *= magnitude_feedback**2 / (magnitude_feedback**2 + threshold**2)
            phase = np.ones_like(error)
            safe = magnitude_current > max(0.1 * threshold, np.finfo(np.float64).eps)
            ratio = aligned_feedback[safe] / current[safe]
            phase[safe] = np.conj(
                np.divide(
                    ratio,
                    np.abs(ratio),
                    out=np.zeros_like(ratio),
                    where=np.abs(ratio) > 0.0,
                )
            )
            error *= weight * phase + (1.0 - weight)
        filtered_error = circular_fir(error, error_fir)
        proposed = (
            gain * alpha * desired
            + (1.0 - alpha) * current
            - gain * mu * filtered_error
        )
        return circular_fir(proposed, tx_fir)

    def test_linear_engine_is_the_public_scalar_update(self):
        current = self.reference.copy()
        measured = 0.8 * self.reference
        config = ILCConfig(
            mu=0.5,
            calibration_mode="explicit",
            calibration_coefficient=1.0,
            dtype="complex128",
        )
        result = create_engine("linear_ilc", config).process(
            self.reference,
            current,
            measured,
            SessionState(),
        )
        np.testing.assert_allclose(result.output, 1.1 * self.reference, rtol=2e-15, atol=2e-15)
        self.assertEqual(result.metrics["algorithm"], "linear_ilc")

    def test_identity_model_vjp_degenerates_to_linear_update(self):
        desired_gain_db = 20.0 * np.log10(2.0)
        common = dict(
            mu=0.5,
            gain_db=desired_gain_db,
            calibration_mode="explicit",
            calibration_coefficient=1.0,
            pa_model_order=1,
            pa_model_memory_depth=1,
            pa_model_ridge=0.0,
            pa_model_min_validation_nmse_db=-80.0,
            dtype="complex128",
        )
        linear = create_engine("linear_ilc", ILCConfig(**common)).process(
            self.reference,
            self.reference,
            self.reference,
            SessionState(),
        )
        model = create_engine("model_vjp_ilc", ILCConfig(**common)).process(
            self.reference,
            self.reference,
            self.reference,
            SessionState(),
        )
        np.testing.assert_allclose(model.output, linear.output, rtol=2e-12, atol=2e-12)
        self.assertIsNone(model.metrics["pa_model_fallback_reason"])

    def test_frozen_first_calibration_is_not_reestimated(self):
        state = SessionState()
        coefficient = 0.5 * np.exp(0.4j)
        config = ILCConfig(mu=0.5, calibration_mode="frozen_first", dtype="complex128")
        engine = create_engine("linear_ilc", config)
        first = engine.process(
            self.reference,
            self.reference,
            coefficient * self.reference,
            state,
        )
        np.testing.assert_allclose(first.aligned_feedback, self.reference, rtol=2e-12, atol=2e-12)
        self.assertIsNotNone(state.feedback_calibration)

        second = engine.process(
            self.reference,
            first.output,
            0.5 * coefficient * self.reference,
            state,
        )
        np.testing.assert_allclose(second.aligned_feedback, 0.5 * self.reference, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(second.output, 1.25 * self.reference, rtol=2e-12, atol=2e-12)

    def test_failed_short_model_fit_holds_when_configured(self):
        reference = self.reference[:512]
        config = ILCConfig(
            calibration_mode="explicit",
            calibration_coefficient=1.0,
            pa_model_fallback="hold",
            dtype="complex128",
        )
        result = create_engine("model_lm_ilc", config).process(
            reference,
            reference,
            0.8 * reference,
            SessionState(),
        )
        np.testing.assert_array_equal(result.output, reference)
        self.assertEqual(result.metrics["pa_model_fallback_reason"], "insufficient_validation_samples")
        self.assertEqual(result.metrics["stop_reason"], "model_fallback_model_failure")

    def test_model_engine_rejects_legacy_preconditioners(self):
        config = ILCConfig(phase_compensate=True, backward_mode="legacy")
        with self.assertRaisesRegex(ValueError, "phase_compensate"):
            create_engine("model_lm_ilc", config)

    def test_ten_capture_transport_uses_first_block_and_tiles_numerically(self):
        sample_count = 32768
        samples = np.arange(sample_count)
        reference = np.exp(2j * np.pi * samples / 127)
        captures = np.column_stack(
            [(0.7 + 0.01 * index) * reference for index in range(10)]
        )
        packed_feedback = captures.reshape(-1, order="F")
        config = ILCConfig(mu=0.2, calibration_mode="legacy_dynamic", dtype="complex128")
        result = create_engine("linear_ilc", config).process(
            reference,
            reference,
            packed_feedback,
            SessionState(),
        )
        self.assertEqual(result.output.size, 327680)
        blocks = result.output.reshape(10, sample_count)
        for block in blocks:
            np.testing.assert_allclose(block, reference, rtol=2e-12, atol=2e-12)
        self.assertEqual(result.metrics["capture_count"], 10)


class FIRParityTests(unittest.TestCase):
    def test_torch_and_numpy_circular_fir_match_for_complex_even_and_odd_taps(self):
        rng = np.random.default_rng(20260822)
        signal = rng.normal(size=29) + 1j * rng.normal(size=29)
        config = ILCConfig(dtype="complex128")
        for taps in (
            np.array([1.0 + 0.2j, -0.3 + 0.7j]),
            np.array([0.2 - 0.4j, 1.0 + 0.0j, -0.5 + 0.1j]),
            np.array([0.1 + 0.3j, 0.8 - 0.2j, -0.4 + 0.5j, 0.2 - 0.1j]),
        ):
            with self.subTest(tap_count=taps.size):
                torch_result = _to_numpy(_apply_fir_torch(_to_torch(signal, config), taps))
                np.testing.assert_allclose(
                    torch_result,
                    circular_fir(signal, taps),
                    rtol=2e-15,
                    atol=2e-15,
                )


if __name__ == "__main__":
    unittest.main()
