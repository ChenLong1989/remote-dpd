import unittest

import numpy as np

from remote_dpd.learning import (
    InputSafetyLimits,
    StopReason,
    anchored_prediction,
    apply_rms_trust_region,
    damped_lm_cg,
    damped_normal_matvec,
    instantaneous_gain_ilc_step,
    linear_ilc_step,
    model_lm_ilc_step,
    project_input_safety,
    signal_peak,
    real_inner,
    signal_papr_db,
    signal_rms,
)
from remote_dpd.pa_model import MemoryPolynomialModel


class AffineRealLinearModel:
    """Small non-holomorphic model y = A u + B conj(u) + bias."""

    def __init__(self, a, b, bias=None):
        self.a = np.asarray(a, dtype=np.complex128)
        self.b = np.asarray(b, dtype=np.complex128)
        if bias is None:
            bias = np.zeros(self.a.shape[0], dtype=np.complex128)
        self.bias = np.asarray(bias, dtype=np.complex128)

    def predict(self, input_signal):
        value = np.asarray(input_signal, dtype=np.complex128)
        return self.a @ value + self.b @ np.conjugate(value) + self.bias

    def jvp(self, input_signal, tangent):
        del input_signal
        tangent = np.asarray(tangent, dtype=np.complex128)
        return self.a @ tangent + self.b @ np.conjugate(tangent)

    def vjp(self, input_signal, cotangent):
        del input_signal
        cotangent = np.asarray(cotangent, dtype=np.complex128)
        return self.a.conj().T @ cotangent + self.b.T @ np.conjugate(cotangent)


class ZeroSlopeModel:
    def __init__(self, output_size):
        self.output_size = output_size

    def predict(self, input_signal):
        del input_signal
        return np.zeros(self.output_size, dtype=np.complex128)

    def jvp(self, input_signal, tangent):
        del input_signal
        del tangent
        return np.zeros(self.output_size, dtype=np.complex128)

    def vjp(self, input_signal, cotangent):
        del cotangent
        return np.zeros_like(np.asarray(input_signal, dtype=np.complex128))


def explicit_real_jacobian(a, b):
    """Stack complex vectors as [real entries, imaginary entries]."""

    top_left = np.real(a + b)
    top_right = np.imag(b - a)
    bottom_left = np.imag(a + b)
    bottom_right = np.real(a - b)
    return np.block([[top_left, top_right], [bottom_left, bottom_right]])


def stack_real(value):
    value = np.asarray(value, dtype=np.complex128)
    return np.concatenate((value.real, value.imag))


def unstack_real(value):
    half = len(value) // 2
    return value[:half] + 1j * value[half:]


class BaselineLearningTests(unittest.TestCase):
    def test_linear_ilc_is_public_scalar_update(self):
        current = np.array([1 + 2j, -0.5j])
        desired = np.array([0.5 - 1j, 2 + 0.25j])
        measured = np.array([-0.25 + 0.5j, 1 - 0.75j])
        result = linear_ilc_step(current, desired, measured, 0.3)
        expected = current + 0.3 * (desired - measured)
        np.testing.assert_allclose(result.next_input, expected, rtol=0.0, atol=0.0)
        self.assertEqual(result.next_input.dtype, np.complex128)
        self.assertEqual(result.stop_reason, StopReason.ACCEPTED)

    def test_instantaneous_gain_uses_damped_complex_inverse(self):
        current = np.array([1 + 0.5j, 2 - 1j], dtype=np.complex128)
        gain = np.array([0.4 + 0.8j, -0.2 + 0.5j])
        measured = gain * current
        desired = np.array([1.2 - 0.4j, -0.5 + 0.7j])
        damping = 0.05
        learning_rate = 0.4
        result = instantaneous_gain_ilc_step(
            current,
            desired,
            measured,
            learning_rate,
            damping=damping,
            input_threshold=0.0,
        )
        expected_update = (
            learning_rate
            * np.conjugate(gain)
            * (desired - measured)
            / (np.abs(gain) ** 2 + damping)
        )
        np.testing.assert_allclose(result.update, expected_update, rtol=2e-15, atol=2e-15)

    def test_instantaneous_gain_zero_slope_is_saturation_limited(self):
        current = np.ones(8, dtype=np.complex128)
        measured = np.zeros(8, dtype=np.complex128)
        desired = np.ones(8, dtype=np.complex128)
        result = instantaneous_gain_ilc_step(current, desired, measured, 0.5)
        self.assertFalse(result.accepted)
        self.assertTrue(result.saturation_limited)
        self.assertEqual(result.stop_reason, StopReason.SATURATION_LIMITED)
        np.testing.assert_array_equal(result.next_input, current)

    def test_thresholded_zero_input_never_divides_by_zero(self):
        current = np.zeros(4, dtype=np.complex128)
        result = instantaneous_gain_ilc_step(
            current,
            np.ones(4),
            np.zeros(4),
            0.5,
            input_threshold=1e-4,
        )
        self.assertTrue(np.all(np.isfinite(result.next_input)))
        self.assertEqual(result.stop_reason, StopReason.SATURATION_LIMITED)


class RealSpaceCGTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260822)
        self.n = 5
        self.a = (
            rng.normal(size=(self.n, self.n))
            + 1j * rng.normal(size=(self.n, self.n))
        ) / np.sqrt(2 * self.n)
        # B is deliberately material: this test must not collapse to a
        # holomorphic complex-linear solve.
        self.b = 0.35 * (
            rng.normal(size=(self.n, self.n))
            + 1j * rng.normal(size=(self.n, self.n))
        ) / np.sqrt(2 * self.n)
        self.model = AffineRealLinearModel(self.a, self.b)
        self.point = rng.normal(size=self.n) + 1j * rng.normal(size=self.n)

    def test_nonholomorphic_jvp_vjp_adjoint_and_normal_spd(self):
        rng = np.random.default_rng(9)
        tangent = rng.normal(size=self.n) + 1j * rng.normal(size=self.n)
        cotangent = rng.normal(size=self.n) + 1j * rng.normal(size=self.n)
        left = real_inner(cotangent, self.model.jvp(self.point, tangent))
        right = real_inner(self.model.vjp(self.point, cotangent), tangent)
        self.assertAlmostEqual(left, right, places=12)

        p = rng.normal(size=self.n) + 1j * rng.normal(size=self.n)
        q = rng.normal(size=self.n) + 1j * rng.normal(size=self.n)
        damping = 0.07
        normal_p = damped_normal_matvec(self.model, self.point, p, damping)
        normal_q = damped_normal_matvec(self.model, self.point, q, damping)
        self.assertAlmostEqual(real_inner(p, normal_q), real_inner(normal_p, q), places=11)
        self.assertGreater(real_inner(p, normal_p), damping * real_inner(p, p) * (1 - 1e-12))

        # A nonzero B makes the differential fail complex linearity.
        mismatch = self.model.jvp(self.point, 1j * p) - 1j * self.model.jvp(self.point, p)
        self.assertGreater(np.linalg.norm(mismatch), 1e-3)

    def test_matrix_free_cg_matches_explicit_real_jacobian_solve(self):
        rng = np.random.default_rng(17)
        error = rng.normal(size=self.n) + 1j * rng.normal(size=self.n)
        damping = 0.13
        result = damped_lm_cg(
            self.model,
            self.point,
            error,
            damping=damping,
            max_iterations=2 * self.n + 4,
            relative_tolerance=1e-13,
        )
        jacobian = explicit_real_jacobian(self.a, self.b)
        expected_real = np.linalg.solve(
            jacobian.T @ jacobian + damping * np.eye(2 * self.n),
            -jacobian.T @ stack_real(error),
        )
        expected = unstack_real(expected_real)
        relative_error = np.linalg.norm(result.solution - expected) / np.linalg.norm(expected)
        self.assertLess(relative_error, 1e-8)
        self.assertTrue(result.converged)
        self.assertLess(result.relative_residual, 1e-12)

    def test_cg_freezes_model_linearization_once(self):
        parent = self

        class Frozen:
            def jvp(self, tangent):
                return parent.model.jvp(parent.point, tangent)

            def vjp(self, cotangent):
                return parent.model.vjp(parent.point, cotangent)

        class CountingModel:
            def __init__(self):
                self.linearize_calls = 0

            def linearize(self, input_signal):
                np.testing.assert_array_equal(input_signal, parent.point)
                self.linearize_calls += 1
                return Frozen()

        model = CountingModel()
        error = np.arange(self.n) + 1j * np.arange(self.n)[::-1]
        result = damped_lm_cg(
            model,
            self.point,
            error,
            damping=0.1,
            max_iterations=2 * self.n,
            relative_tolerance=1e-12,
        )
        self.assertEqual(model.linearize_calls, 1)
        self.assertTrue(np.all(np.isfinite(result.solution)))

    def test_non_positive_curvature_returns_last_finite_iterate(self):
        class IndefiniteAdjoint:
            def predict(self, input_signal):
                return np.asarray(input_signal, dtype=np.complex128)

            def jvp(self, input_signal, tangent):
                del input_signal
                return np.asarray(tangent, dtype=np.complex128)

            def vjp(self, input_signal, cotangent):
                del input_signal
                return -np.asarray(cotangent, dtype=np.complex128)

        result = damped_lm_cg(
            IndefiniteAdjoint(),
            np.ones(3),
            np.ones(3),
            damping=0.1,
            max_iterations=3,
            relative_tolerance=1e-6,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, StopReason.CG_NON_POSITIVE_CURVATURE)
        self.assertTrue(np.all(np.isfinite(result.solution)))
        np.testing.assert_array_equal(result.solution, np.zeros(3))

    def test_nonfinite_normal_operator_fails_without_nonfinite_solution(self):
        class NonfiniteOperator:
            def predict(self, input_signal):
                return np.asarray(input_signal, dtype=np.complex128)

            def jvp(self, input_signal, tangent):
                del input_signal
                return np.full_like(np.asarray(tangent, dtype=np.complex128), np.nan)

            def vjp(self, input_signal, cotangent):
                del input_signal
                return np.asarray(cotangent, dtype=np.complex128)

        result = damped_lm_cg(
            NonfiniteOperator(),
            np.ones(3),
            np.ones(3),
            damping=0.1,
            max_iterations=3,
            relative_tolerance=1e-6,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, StopReason.CG_NONFINITE_OPERATOR)
        self.assertTrue(np.all(np.isfinite(result.solution)))


class SafeguardedStepTests(unittest.TestCase):
    def test_anchored_prediction_cancels_model_bias(self):
        a = np.diag(np.array([0.7 + 0.2j, 1.1 - 0.3j]))
        b = np.diag(np.array([0.1 - 0.05j, -0.2 + 0.08j]))
        wrong_bias = np.array([8 - 3j, -4 + 6j])
        model = AffineRealLinearModel(a, b, wrong_bias)
        current = np.array([0.2 + 0.5j, -0.7 + 0.1j])
        update = np.array([0.05 - 0.12j, 0.2 + 0.09j])
        measured = np.array([1.5 + 0.4j, -0.2 + 0.7j])
        predicted = anchored_prediction(model, current, measured, update)
        expected = measured + a @ update + b @ np.conjugate(update)
        np.testing.assert_allclose(predicted, expected, rtol=1e-14, atol=1e-14)

    def test_zero_pa_slope_reports_saturation_without_runaway(self):
        current = np.full(16, 0.4 + 0.1j)
        desired = np.ones(16, dtype=np.complex128)
        measured = np.zeros(16, dtype=np.complex128)
        result = model_lm_ilc_step(
            current,
            desired,
            measured,
            ZeroSlopeModel(16),
            damping=1e-2,
        )
        self.assertFalse(result.accepted)
        self.assertTrue(result.saturation_limited)
        self.assertEqual(result.stop_reason, StopReason.SATURATION_LIMITED)
        np.testing.assert_array_equal(result.next_input, current)

    def test_lm_step_accepts_anchored_decrease_for_nonholomorphic_model(self):
        a = np.diag(np.array([0.8 + 0.2j, 0.5 - 0.1j]))
        b = np.diag(np.array([0.2 - 0.05j, -0.12 + 0.04j]))
        model = AffineRealLinearModel(a, b, bias=np.array([20 + 3j, -7 + 2j]))
        current = np.array([0.5 + 0.2j, -0.4 + 0.1j])
        measured = np.array([0.4 + 0.3j, -0.2 + 0.2j])
        desired = np.array([0.8 - 0.1j, 0.3 + 0.4j])
        result = model_lm_ilc_step(
            current,
            desired,
            measured,
            model,
            damping=1e-4,
            cg_max_iterations=8,
            cg_relative_tolerance=1e-12,
            trust_region_ratio=None,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.stop_reason, StopReason.ACCEPTED)
        self.assertLess(
            np.linalg.norm(result.predicted_output - desired),
            np.linalg.norm(measured - desired),
        )

    def test_prediction_backtracking_rejects_a_non_decreasing_model_step(self):
        class InconsistentModel:
            def predict(self, input_signal):
                return -np.asarray(input_signal, dtype=np.complex128)

            def jvp(self, input_signal, tangent):
                del input_signal
                return np.asarray(tangent, dtype=np.complex128)

            def vjp(self, input_signal, cotangent):
                del input_signal
                return np.asarray(cotangent, dtype=np.complex128)

        current = np.ones(4, dtype=np.complex128)
        measured = np.zeros(4, dtype=np.complex128)
        desired = np.ones(4, dtype=np.complex128)
        result = model_lm_ilc_step(
            current,
            desired,
            measured,
            InconsistentModel(),
            damping=1e-2,
            trust_region_ratio=None,
            max_backtracks=3,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.stop_reason, StopReason.PREDICTION_REJECTED)
        np.testing.assert_array_equal(result.next_input, current)

    def test_backtracking_scales_the_trust_limited_step_instead_of_repeating_it(self):
        class SineRealModel:
            def __init__(self, frequency):
                self.frequency = frequency

            def predict(self, input_signal):
                value = np.asarray(input_signal, dtype=np.complex128)
                return np.sin(self.frequency * value.real) + 1j * value.imag

            def jvp(self, input_signal, tangent):
                point = np.asarray(input_signal, dtype=np.complex128)
                direction = np.asarray(tangent, dtype=np.complex128)
                slope = self.frequency * np.cos(self.frequency * point.real)
                return slope * direction.real + 1j * direction.imag

            def vjp(self, input_signal, cotangent):
                point = np.asarray(input_signal, dtype=np.complex128)
                vector = np.asarray(cotangent, dtype=np.complex128)
                slope = self.frequency * np.cos(self.frequency * point.real)
                return slope * vector.real + 1j * vector.imag

        frequency = 27.91391737660467
        model = SineRealModel(frequency)
        current = np.array([0.05571834294977192], dtype=np.complex128)
        measured = model.predict(current)
        desired = np.array([1.587370908989144], dtype=np.complex128)
        result = model_lm_ilc_step(
            current,
            desired,
            measured,
            model,
            damping=1e-8,
            trust_region_ratio=0.021620546704993495,
            backtrack_factor=0.5,
            max_backtracks=8,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.backtracks, 1)
        self.assertTrue(result.trust_region_active)
        self.assertLess(
            np.linalg.norm(result.predicted_output - desired),
            np.linalg.norm(measured - desired),
        )

    def test_rms_trust_region_scales_with_real_coefficient(self):
        current = np.ones(32, dtype=np.complex128) * (0.3 + 0.4j)
        update = np.linspace(1, 3, 32) * (1 + 2j)
        result = apply_rms_trust_region(update, current, 0.2)
        self.assertTrue(result.active)
        self.assertAlmostEqual(signal_rms(result.update), 0.2 * signal_rms(current), places=14)
        ratio = result.update[5] / update[5]
        self.assertAlmostEqual(ratio.imag, 0.0, places=15)
        self.assertGreater(ratio.real, 0.0)

    def test_combined_safety_projection_enforces_all_limits(self):
        candidate = np.array(
            [12 + 3j, 0.3 - 0.2j, -0.5 + 0.1j, 0.7j, -0.1 - 0.4j, 0.2 + 0.2j],
            dtype=np.complex128,
        )
        limits = InputSafetyLimits(max_rms=0.45, max_peak=0.7, max_papr_db=2.5)
        result = project_input_safety(candidate, limits)
        self.assertTrue(result.feasible)
        self.assertTrue(result.active)
        self.assertTrue(result.papr_active)
        self.assertLessEqual(signal_rms(result.projected_input), limits.max_rms * (1 + 1e-12))
        self.assertLessEqual(np.max(np.abs(result.projected_input)), limits.max_peak * (1 + 1e-12))
        self.assertLessEqual(signal_papr_db(result.projected_input), limits.max_papr_db + 1e-9)
        self.assertTrue(np.all(np.isfinite(result.projected_input)))

    def test_complex64_safety_and_trust_boundaries_use_dtype_aware_tolerance(self):
        rng = np.random.default_rng(9157)
        for _ in range(128):
            candidate = (
                rng.normal(size=17) + 1j * rng.normal(size=17)
            ).astype(np.complex64)
            candidate[0] *= np.float32(8.0)
            papr_limit = float(rng.uniform(1.0, 5.0))
            projection = project_input_safety(
                candidate,
                InputSafetyLimits(
                    max_rms=0.8,
                    max_peak=1.5,
                    max_papr_db=papr_limit,
                ),
            )
            self.assertTrue(projection.feasible)
            self.assertEqual(projection.projected_input.dtype, np.dtype(np.complex64))
            self.assertLessEqual(signal_rms(projection.projected_input), 0.8 * (1.0 + 3e-6))
            self.assertLessEqual(np.max(np.abs(projection.projected_input)), 1.5 * (1.0 + 3e-6))
            self.assertLessEqual(signal_papr_db(projection.projected_input), papr_limit + 3e-6)

            current = (
                rng.normal(size=17) + 1j * rng.normal(size=17)
            ).astype(np.complex64)
            update = (10.0 * (
                rng.normal(size=17) + 1j * rng.normal(size=17)
            )).astype(np.complex64)
            ratio = float(rng.uniform(0.01, 0.5))
            trust = apply_rms_trust_region(update, current, ratio)
            self.assertEqual(trust.update.dtype, np.dtype(np.complex64))
            self.assertLessEqual(
                signal_rms(trust.update),
                ratio * signal_rms(current) * (1.0 + 3e-6),
            )

    def test_papr_checks_do_not_overflow_for_large_finite_inputs(self):
        for complex_dtype, real_dtype in (
            (np.complex64, np.float32),
            (np.complex128, np.float64),
        ):
            with self.subTest(dtype=np.dtype(complex_dtype).name):
                large = np.finfo(real_dtype).max / real_dtype(4.0)
                candidate = np.array([large, 1.0], dtype=complex_dtype)
                self.assertTrue(np.isfinite(signal_rms(candidate)))
                self.assertAlmostEqual(signal_papr_db(candidate), 10.0 * np.log10(2.0), places=5)
                projection = project_input_safety(
                    candidate,
                    InputSafetyLimits(max_papr_db=1.0),
                )
                self.assertTrue(projection.feasible)
                self.assertTrue(projection.active)
                self.assertLessEqual(signal_papr_db(projection.projected_input), 1.0 + 3e-6)

        finite_components = np.array(
            [3e38 + 3e38j, 3e38 - 3e38j],
            dtype=np.complex64,
        )
        expected_peak = np.hypot(3e38, 3e38)
        self.assertAlmostEqual(signal_peak(finite_components) / expected_peak, 1.0, places=6)
        limits = InputSafetyLimits(max_rms=5e38, max_peak=5e38, max_papr_db=1.0)
        projection = project_input_safety(finite_components, limits)
        self.assertTrue(projection.feasible)
        self.assertFalse(projection.active)
        np.testing.assert_array_equal(projection.projected_input, finite_components)

    def test_papr_checks_preserve_subnormal_signal_shape(self):
        for complex_dtype, real_dtype in (
            (np.complex64, np.float32),
            (np.complex128, np.float64),
        ):
            with self.subTest(dtype=np.dtype(complex_dtype).name):
                subnormal = np.nextafter(real_dtype(0.0), real_dtype(1.0))
                candidate = np.array([subnormal, 0.0], dtype=complex_dtype)
                self.assertGreater(signal_rms(candidate), 0.0)
                self.assertAlmostEqual(signal_papr_db(candidate), 10.0 * np.log10(2.0), places=5)
                projection = project_input_safety(
                    candidate,
                    InputSafetyLimits(max_papr_db=1.0),
                )
                self.assertTrue(projection.feasible)
                self.assertTrue(projection.active)
                self.assertLessEqual(signal_papr_db(projection.projected_input), 1.0 + 3e-6)

    def test_lm_final_candidate_obeys_safety_projection(self):
        identity = AffineRealLinearModel(np.eye(8), np.zeros((8, 8)))
        current = np.full(8, 0.1 + 0j)
        measured = current.copy()
        desired = np.array([5, 3, 2, 1, 1, 1, 1, 1], dtype=np.complex128)
        limits = InputSafetyLimits(max_rms=0.25, max_peak=0.35, max_papr_db=3.0)
        result = model_lm_ilc_step(
            current,
            desired,
            measured,
            identity,
            damping=1e-3,
            trust_region_ratio=None,
            safety_limits=limits,
        )
        self.assertTrue(result.accepted)
        self.assertTrue(result.input_projection_active)
        self.assertLessEqual(signal_rms(result.next_input), 0.25 * (1 + 1e-10))
        self.assertLessEqual(np.max(np.abs(result.next_input)), 0.35 * (1 + 1e-10))
        self.assertLessEqual(signal_papr_db(result.next_input), 3.0 + 1e-9)

    def test_safety_projection_cannot_push_effective_step_outside_trust_region(self):
        papr_limit_db = 2.5
        peak_to_rms = 10.0 ** (papr_limit_db / 20.0)
        current = np.array(
            [1.0, np.sqrt(2.0 / peak_to_rms**2 - 1.0) + 1e-3],
            dtype=np.complex128,
        )
        requested_update = np.full(2, -0.002, dtype=np.complex128)
        damping = 1e-2
        measured = current.copy()
        desired = current + (1.0 + damping) * requested_update
        trust_ratio = signal_rms(requested_update) / signal_rms(current)
        identity = AffineRealLinearModel(np.eye(2), np.zeros((2, 2)))

        result = model_lm_ilc_step(
            current,
            desired,
            measured,
            identity,
            damping=damping,
            trust_region_ratio=trust_ratio,
            safety_limits=InputSafetyLimits(max_papr_db=papr_limit_db),
        )

        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.backtracks, 1)
        self.assertGreaterEqual(result.diagnostics["post_projection_trust_rejections"], 1)
        self.assertLessEqual(
            signal_rms(result.update),
            trust_ratio * signal_rms(current) * (1.0 + 2e-12),
        )
        self.assertLessEqual(signal_papr_db(result.next_input), papr_limit_db + 1e-9)


class ValidationTests(unittest.TestCase):
    def test_complex64_learning_and_cg_preserve_operator_dtype(self):
        model = MemoryPolynomialModel(
            (1,),
            np.array([[1.0 + 0.0j]], dtype=np.complex64),
            1.0,
        )
        current = np.array([0.2 + 0.1j, -0.3 + 0.4j], dtype=np.complex64)
        measured = current.copy()
        desired = (np.complex64(1.1) * current).astype(np.complex64)

        linear = linear_ilc_step(current, desired, measured, 0.5)
        lm = model_lm_ilc_step(
            current,
            desired,
            measured,
            model,
            damping=1e-3,
            trust_region_ratio=None,
            cg_relative_tolerance=1e-5,
        )

        self.assertEqual(linear.next_input.dtype, np.dtype(np.complex64))
        self.assertEqual(linear.update.dtype, np.dtype(np.complex64))
        self.assertTrue(lm.accepted)
        self.assertEqual(lm.next_input.dtype, np.dtype(np.complex64))
        self.assertEqual(lm.update.dtype, np.dtype(np.complex64))
        assert lm.cg_result is not None
        self.assertEqual(lm.cg_result.solution.dtype, np.dtype(np.complex64))

    def test_illegal_safety_limits_are_rejected(self):
        for kwargs in (
            {"max_rms": 0.0},
            {"max_peak": -1.0},
            {"max_papr_db": -0.01},
            {"max_rms": np.inf},
            {"max_papr_db": np.nan},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    InputSafetyLimits(**kwargs)

    def test_illegal_solver_and_step_configuration_is_rejected(self):
        identity = AffineRealLinearModel(np.eye(2), np.zeros((2, 2)))
        vector = np.ones(2, dtype=np.complex128)
        with self.assertRaises(ValueError):
            damped_lm_cg(identity, vector, vector, damping=0.0)
        damped_lm_cg(identity, vector, vector, damping=1e-8)
        with self.assertRaises(ValueError):
            damped_lm_cg(
                identity,
                vector,
                vector,
                damping=np.nextafter(1e-8, 0.0),
            )
        with self.assertRaises(ValueError):
            damped_lm_cg(identity, vector, vector, damping=1e-2, max_iterations=0)
        with self.assertRaises(ValueError):
            damped_lm_cg(identity, vector, vector, damping=1e-2, relative_tolerance=1.0)
        with self.assertRaises(ValueError):
            instantaneous_gain_ilc_step(vector, vector, vector, 0.2, damping=-1.0)
        with self.assertRaises(ValueError):
            model_lm_ilc_step(vector, vector, vector, identity, damping=1e-2, trust_region_ratio=0.0)
        with self.assertRaises(ValueError):
            model_lm_ilc_step(vector, vector, vector, identity, damping=1e-2, backtrack_factor=1.0)
        with self.assertRaises(ValueError):
            linear_ilc_step(vector, vector, vector, 0.0)

    def test_nonfinite_runtime_data_holds_current_input(self):
        current = np.ones(3, dtype=np.complex128)
        measured = np.array([1.0, np.nan, 1.0], dtype=np.complex128)
        result = linear_ilc_step(current, np.zeros(3), measured, 0.2)
        self.assertFalse(result.accepted)
        self.assertEqual(result.stop_reason, StopReason.NONFINITE_INPUT)
        np.testing.assert_array_equal(result.next_input, current)

        unsafe_current = np.array([1.0, np.inf, 2.0j], dtype=np.complex128)
        result = linear_ilc_step(unsafe_current, np.zeros(3), np.zeros(3), 0.2)
        self.assertEqual(result.stop_reason, StopReason.NONFINITE_INPUT)
        self.assertTrue(np.all(np.isfinite(result.next_input)))


if __name__ == "__main__":
    unittest.main()
