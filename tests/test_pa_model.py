from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import numpy as np
import torch

from remote_dpd.pa_model import (
    MemoryPolynomialLinearization,
    MemoryPolynomialModel,
    PAForwardModelConfig,
    deterministic_block_split,
    explicit_real_jacobian,
    fit_pa_model,
)


def _complex_to_real(value: np.ndarray) -> np.ndarray:
    return np.concatenate((value.real, value.imag))


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(actual - expected) / denominator)


class PAForwardModelConfigTests(unittest.TestCase):
    def test_default_split_holds_out_every_fifth_256_sample_block(self):
        train_mask, validation_mask = deterministic_block_split(6 * 256 + 17)
        expected_validation = np.zeros(train_mask.size, dtype=bool)
        expected_validation[4 * 256 : 5 * 256] = True
        np.testing.assert_array_equal(validation_mask, expected_validation)
        np.testing.assert_array_equal(train_mask, ~expected_validation)

    def test_configuration_rejects_invalid_values(self):
        invalid_arguments = (
            {"orders": ()},
            {"orders": (1, 2)},
            {"orders": (3, 1)},
            {"memory_depth": 0},
            {"ridge": -1.0},
            {"ridge": float("nan")},
            {"envelope_quantile": 0.0},
            {"block_size": 0},
            {"validation_every": 1},
            {"minimum_scale": 0.0},
            {"column_rms_epsilon": float("inf")},
            {"max_condition_number": 0.5},
            {"max_validation_nmse_db": float("nan")},
            {"numeric_dtype": "float32"},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    PAForwardModelConfig(**arguments)

    def test_configuration_rejects_non_real_scalar_float_fields(self):
        field_names = (
            "ridge",
            "envelope_quantile",
            "minimum_scale",
            "column_rms_epsilon",
            "max_condition_number",
            "max_validation_nmse_db",
        )
        invalid_values = (True, np.bool_(False), 1.0 + 0.0j, np.array(1.0), [1.0])
        for field_name in field_names:
            for invalid_value in invalid_values:
                with self.subTest(field_name=field_name, invalid_value=invalid_value):
                    with self.assertRaises((TypeError, ValueError)):
                        PAForwardModelConfig(**{field_name: invalid_value})


class MemoryPolynomialDerivativeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260822)

    def test_identity_forward_jvp_and_vjp_are_exact(self):
        input_signal = self.rng.normal(size=19) + 1j * self.rng.normal(size=19)
        tangent = self.rng.normal(size=19) + 1j * self.rng.normal(size=19)
        cotangent = self.rng.normal(size=19) + 1j * self.rng.normal(size=19)
        model = MemoryPolynomialModel((1,), np.array([[1.0 + 0.0j]]), 0.7)

        np.testing.assert_array_equal(model.forward(input_signal), input_signal)
        np.testing.assert_array_equal(model.jvp(input_signal, tangent), tangent)
        np.testing.assert_array_equal(model.vjp(input_signal, cotangent), cotangent)

    def test_complex_gain_uses_the_real_adjoint(self):
        gain = 0.6 - 1.3j
        input_signal = self.rng.normal(size=11) + 1j * self.rng.normal(size=11)
        tangent = self.rng.normal(size=11) + 1j * self.rng.normal(size=11)
        cotangent = self.rng.normal(size=11) + 1j * self.rng.normal(size=11)
        model = MemoryPolynomialModel((1,), np.array([[gain]]), 1.0)

        np.testing.assert_allclose(model.predict(input_signal), gain * input_signal, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(model.jvp(input_signal, tangent), gain * tangent, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            model.vjp(input_signal, cotangent),
            np.conjugate(gain) * cotangent,
            rtol=0.0,
            atol=0.0,
        )

    def test_conjugate_map_oracle_has_the_expected_jvp_and_vjp(self):
        size = 13
        tangent = self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        cotangent = self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        linearization = MemoryPolynomialLinearization(
            np.zeros((1, size), dtype=np.complex128),
            np.ones((1, size), dtype=np.complex128),
        )

        np.testing.assert_array_equal(linearization.jvp(tangent), np.conjugate(tangent))
        np.testing.assert_array_equal(linearization.vjp(cotangent), np.conjugate(cotangent))

    def test_cubic_oracle_matches_closed_form_derivatives(self):
        scale = 0.8
        coefficient = 0.7 - 0.2j
        input_signal = self.rng.normal(size=9) + 1j * self.rng.normal(size=9)
        tangent = self.rng.normal(size=9) + 1j * self.rng.normal(size=9)
        cotangent = self.rng.normal(size=9) + 1j * self.rng.normal(size=9)
        model = MemoryPolynomialModel((3,), np.array([[coefficient]]), scale)
        a_coefficient = 2.0 * coefficient * np.abs(input_signal) ** 2 / scale**2
        b_coefficient = coefficient * input_signal**2 / scale**2

        expected_jvp = a_coefficient * tangent + b_coefficient * np.conjugate(tangent)
        expected_vjp = np.conjugate(a_coefficient) * cotangent + b_coefficient * np.conjugate(cotangent)
        np.testing.assert_allclose(model.jvp(input_signal, tangent), expected_jvp, rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(model.vjp(input_signal, cotangent), expected_vjp, rtol=2e-15, atol=2e-15)

    def test_jvp_matches_centered_finite_difference_and_explicit_real_jacobian(self):
        size = 5
        input_signal = 0.3 * (self.rng.normal(size=size) + 1j * self.rng.normal(size=size))
        tangent = self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        coefficients = np.array(
            [
                [0.9 + 0.2j, 0.15 - 0.1j],
                [0.12 - 0.04j, -0.03 + 0.02j],
            ],
            dtype=np.complex128,
        )
        model = MemoryPolynomialModel((1, 3), coefficients, 0.55)
        step = 1e-6
        finite_difference = (
            model.predict(input_signal + step * tangent)
            - model.predict(input_signal - step * tangent)
        ) / (2.0 * step)
        analytic = model.jvp(input_signal, tangent)
        self.assertLess(_relative_error(analytic, finite_difference), 1e-9)

        analytic_jacobian = explicit_real_jacobian(model, input_signal)
        finite_difference_jacobian = np.empty_like(analytic_jacobian)
        for column in range(2 * size):
            direction = np.zeros(size, dtype=np.complex128)
            if column < size:
                direction[column] = 1.0
            else:
                direction[column - size] = 1.0j
            output_difference = (
                model.predict(input_signal + step * direction)
                - model.predict(input_signal - step * direction)
            ) / (2.0 * step)
            finite_difference_jacobian[:, column] = _complex_to_real(output_difference)
        self.assertLess(_relative_error(analytic_jacobian, finite_difference_jacobian), 1e-9)

    def test_vjp_matches_explicit_jacobian_transpose_and_adjoint_identity(self):
        size = 7
        input_signal = 0.25 * (self.rng.normal(size=size) + 1j * self.rng.normal(size=size))
        tangent = self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        cotangent = self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        coefficients = np.array(
            [
                [0.8 + 0.1j, 0.12 - 0.07j, -0.04 + 0.02j],
                [0.15 - 0.03j, -0.03 + 0.01j, 0.01 + 0.005j],
                [-0.02 + 0.01j, 0.006 - 0.002j, -0.001 + 0.001j],
            ],
            dtype=np.complex128,
        )
        model = MemoryPolynomialModel((1, 3, 5), coefficients, 0.6)
        jacobian = explicit_real_jacobian(model, input_signal)
        jvp = model.jvp(input_signal, tangent)
        vjp = model.vjp(input_signal, cotangent)

        np.testing.assert_allclose(
            _complex_to_real(jvp),
            jacobian @ _complex_to_real(tangent),
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            _complex_to_real(vjp),
            jacobian.T @ _complex_to_real(cotangent),
            rtol=2e-14,
            atol=2e-14,
        )
        left = float(np.real(np.vdot(cotangent, jvp)))
        right = float(np.real(np.vdot(vjp, tangent)))
        self.assertLess(abs(left - right), 1e-12)

    def test_vjp_applies_the_inverse_circular_memory_roll(self):
        size = 10
        gain = -0.4 + 0.3j
        input_signal = np.ones(size, dtype=np.complex128)
        tangent = self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        cotangent = self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        model = MemoryPolynomialModel((1,), np.array([[0.0, 0.0, gain]]), 1.0)

        np.testing.assert_allclose(model.jvp(input_signal, tangent), gain * np.roll(tangent, 2))
        np.testing.assert_allclose(
            model.vjp(input_signal, cotangent),
            np.roll(np.conjugate(gain) * cotangent, -2),
        )

    def test_complex64_path_preserves_dtype_and_adjoint_accuracy(self):
        size = 257
        input_signal = (0.2 * (
            self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        )).astype(np.complex64)
        tangent = (
            self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        ).astype(np.complex64)
        cotangent = (
            self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        ).astype(np.complex64)
        coefficients = np.array(
            [[0.9 + 0.1j, 0.1 - 0.04j], [0.08 - 0.03j, -0.02 + 0.01j]],
            dtype=np.complex64,
        )
        model = MemoryPolynomialModel((1, 3), coefficients, 0.6)
        jvp = model.jvp(input_signal, tangent)
        vjp = model.vjp(input_signal, cotangent)

        self.assertEqual(model.numeric_dtype, "complex64")
        self.assertEqual(model.predict(input_signal).dtype, np.dtype(np.complex64))
        self.assertEqual(jvp.dtype, np.dtype(np.complex64))
        self.assertEqual(vjp.dtype, np.dtype(np.complex64))
        left = float(np.real(np.vdot(cotangent, jvp)))
        right = float(np.real(np.vdot(vjp, tangent)))
        relative_error = abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)
        self.assertLess(relative_error, 1e-4)

    def test_pytorch_autograd_oracle_matches_nonholomorphic_jvp_and_vjp(self):
        size = 11
        orders = (1, 3, 5)
        envelope_scale = 0.73
        input_base = 0.22 * (
            self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        )
        tangent_base = self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        cotangent_base = self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
        coefficient_base = np.array(
            [
                [0.91 + 0.13j, 0.12 - 0.08j, -0.035 + 0.021j],
                [0.17 - 0.055j, -0.041 + 0.018j, 0.009 - 0.006j],
                [-0.027 + 0.014j, 0.006 - 0.003j, -0.0015 + 0.0008j],
            ]
        )

        dtype_cases = (
            (np.complex128, torch.complex128),
            (np.complex64, torch.complex64),
        )
        for numpy_dtype, torch_dtype in dtype_cases:
            with self.subTest(dtype=np.dtype(numpy_dtype).name):
                input_signal = np.asarray(input_base, dtype=numpy_dtype)
                tangent = np.asarray(tangent_base, dtype=numpy_dtype)
                cotangent = np.asarray(cotangent_base, dtype=numpy_dtype)
                coefficients = np.asarray(coefficient_base, dtype=numpy_dtype)
                model = MemoryPolynomialModel(orders, coefficients, envelope_scale)

                torch_input = torch.tensor(
                    input_signal,
                    dtype=torch_dtype,
                    requires_grad=True,
                )
                torch_tangent = torch.tensor(tangent, dtype=torch_dtype)
                torch_cotangent = torch.tensor(cotangent, dtype=torch_dtype)
                torch_coefficients = torch.tensor(coefficients, dtype=torch_dtype)

                def torch_forward(value):
                    result = torch.zeros_like(value)
                    for order_index, order in enumerate(orders):
                        for delay in range(coefficients.shape[1]):
                            delayed = torch.roll(value, shifts=delay, dims=0)
                            radial = (torch.abs(delayed) / envelope_scale) ** (order - 1)
                            result = (
                                result
                                + torch_coefficients[order_index, delay]
                                * delayed
                                * radial
                            )
                    return result

                _, oracle_jvp = torch.autograd.functional.jvp(
                    torch_forward,
                    torch_input,
                    torch_tangent,
                    create_graph=False,
                    strict=True,
                )
                # PyTorch's gradient of this real scalar is the real-adjoint
                # complex VJP directly; no factor of two is removed.
                objective = torch.real(
                    torch.vdot(torch_cotangent.detach(), torch_forward(torch_input))
                )
                (oracle_vjp,) = torch.autograd.grad(objective, torch_input)

                analytic_jvp = model.jvp(input_signal, tangent)
                analytic_vjp = model.vjp(input_signal, cotangent)
                expected_jvp = oracle_jvp.detach().cpu().numpy()
                expected_vjp = oracle_vjp.detach().cpu().numpy()
                if numpy_dtype is np.complex128:
                    np.testing.assert_allclose(
                        analytic_jvp,
                        expected_jvp,
                        rtol=2e-14,
                        atol=2e-14,
                    )
                    np.testing.assert_allclose(
                        analytic_vjp,
                        expected_vjp,
                        rtol=2e-14,
                        atol=2e-14,
                    )
                else:
                    self.assertLess(_relative_error(analytic_jvp, expected_jvp), 1e-4)
                    self.assertLess(_relative_error(analytic_vjp, expected_vjp), 1e-4)

    def test_fitted_model_and_derivative_arrays_are_immutable(self):
        model = MemoryPolynomialModel((1,), np.array([[1.0 + 0.0j]]), 1.0)
        linearization = model.linearize(np.ones(4, dtype=np.complex128))

        with self.assertRaises(FrozenInstanceError):
            model.envelope_scale = 2.0
        with self.assertRaises(ValueError):
            model.coefficients[0, 0] = 2.0
        with self.assertRaises(ValueError):
            linearization.a_coefficients[0, 0] = 2.0


class PAForwardModelFitTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(918273)
        self.sample_count = 10 * 256
        self.input_signal = 0.28 * (
            self.rng.normal(size=self.sample_count) + 1j * self.rng.normal(size=self.sample_count)
        )

    def _exact_problem(self):
        train_mask, _ = deterministic_block_split(self.sample_count)
        scale = float(np.quantile(np.abs(self.input_signal[train_mask]), 0.999))
        coefficients = np.array(
            [
                [0.95 + 0.08j, 0.13 - 0.05j, -0.04 + 0.02j],
                [0.11 - 0.03j, -0.025 + 0.012j, 0.009 + 0.004j],
                [-0.018 + 0.008j, 0.004 - 0.002j, -0.001 + 0.0005j],
            ],
            dtype=np.complex128,
        )
        model = MemoryPolynomialModel((1, 3, 5), coefficients, scale)
        return model, model.predict(self.input_signal)

    def test_augmented_least_squares_recovers_an_exact_memory_polynomial(self):
        truth, output_signal = self._exact_problem()
        config = PAForwardModelConfig(
            orders=(1, 3, 5),
            memory_depth=3,
            ridge=0.0,
            max_validation_nmse_db=-80.0,
        )
        with patch("numpy.linalg.inv", side_effect=AssertionError("normal-equation inverse is forbidden")):
            result = fit_pa_model(self.input_signal, output_signal, config)

        self.assertTrue(result.succeeded, result.diagnostics)
        self.assertIsNotNone(result.model)
        fitted = result.model
        assert fitted is not None
        self.assertLess(result.diagnostics.train_nmse_db, -200.0)
        self.assertLess(result.diagnostics.validation_nmse_db, -200.0)
        self.assertEqual(result.diagnostics.rank, config.coefficient_count)
        self.assertEqual(result.diagnostics.sample_count, self.sample_count)
        self.assertEqual(result.diagnostics.train_sample_count, 8 * 256)
        self.assertEqual(result.diagnostics.validation_sample_count, 2 * 256)
        self.assertEqual(result.diagnostics.coefficient_count, 9)
        self.assertIsNone(result.diagnostics.fallback_reason)
        self.assertLess(_relative_error(fitted.coefficients, truth.coefficients), 1e-11)
        self.assertLess(_relative_error(fitted.predict(self.input_signal), output_signal), 1e-11)

        metrics = result.diagnostics.as_metrics()
        self.assertEqual(metrics["pa_model_rank"], 9)
        self.assertIsNone(metrics["pa_model_fallback_reason"])

    def test_default_ridge_fit_exceeds_the_required_noiseless_nmse(self):
        _, output_signal = self._exact_problem()
        result = fit_pa_model(
            self.input_signal,
            output_signal,
            PAForwardModelConfig(orders=(1, 3, 5), memory_depth=3),
        )

        self.assertTrue(result.succeeded, result.diagnostics)
        self.assertLess(result.diagnostics.validation_nmse_db, -80.0)

    def test_complex64_fit_is_a_real_numeric_path_not_an_input_only_cast(self):
        input_signal = self.input_signal.astype(np.complex64)
        train_mask, _ = deterministic_block_split(self.sample_count)
        scale = float(np.quantile(np.abs(input_signal[train_mask]), 0.999))
        coefficients = np.array(
            [
                [0.95 + 0.08j, 0.13 - 0.05j, -0.04 + 0.02j],
                [0.11 - 0.03j, -0.025 + 0.012j, 0.009 + 0.004j],
            ],
            dtype=np.complex64,
        )
        truth = MemoryPolynomialModel((1, 3), coefficients, scale)
        output_signal = truth.predict(input_signal)
        result = fit_pa_model(
            input_signal,
            output_signal,
            PAForwardModelConfig(
                orders=(1, 3),
                memory_depth=3,
                ridge=0.0,
                max_validation_nmse_db=-60.0,
                numeric_dtype="complex64",
            ),
        )

        self.assertTrue(result.succeeded, result.diagnostics)
        assert result.model is not None
        self.assertEqual(result.model.numeric_dtype, "complex64")
        self.assertEqual(result.model.coefficients.dtype, np.dtype(np.complex64))
        self.assertEqual(result.model.predict(input_signal).dtype, np.dtype(np.complex64))
        self.assertLess(result.diagnostics.validation_nmse_db, -80.0)

    def test_constant_envelope_input_returns_rank_deficient_fallback(self):
        phases = self.rng.uniform(-np.pi, np.pi, size=self.sample_count)
        input_signal = np.exp(1j * phases)
        output_signal = (0.8 + 0.1j) * input_signal
        result = fit_pa_model(
            input_signal,
            output_signal,
            PAForwardModelConfig(orders=(1, 3, 5), memory_depth=1),
        )

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.model)
        self.assertEqual(result.diagnostics.fallback_reason, "rank_deficient")
        self.assertLess(result.diagnostics.rank, result.diagnostics.coefficient_count)

    def test_excessive_condition_number_returns_explicit_fallback(self):
        output_signal = 0.7 * self.input_signal
        result = fit_pa_model(
            self.input_signal,
            output_signal,
            PAForwardModelConfig(
                orders=(1, 3),
                memory_depth=1,
                max_condition_number=1.01,
            ),
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.diagnostics.rank, 2)
        self.assertGreater(result.diagnostics.condition_number, 1.01)
        self.assertEqual(result.diagnostics.fallback_reason, "condition_number_exceeded")

    def test_validation_blocks_do_not_enter_fit_and_can_reject_the_model(self):
        _, output_signal = self._exact_problem()
        _, validation_mask = deterministic_block_split(self.sample_count)
        corrupted_output = output_signal.copy()
        corrupted_output[validation_mask] += 4.0 * (
            self.rng.normal(size=np.count_nonzero(validation_mask))
            + 1j * self.rng.normal(size=np.count_nonzero(validation_mask))
        )
        result = fit_pa_model(
            self.input_signal,
            corrupted_output,
            PAForwardModelConfig(
                orders=(1, 3, 5),
                memory_depth=3,
                ridge=0.0,
                max_validation_nmse_db=-20.0,
            ),
        )

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.model)
        self.assertLess(result.diagnostics.train_nmse_db, -200.0)
        self.assertGreater(result.diagnostics.validation_nmse_db, -20.0)
        self.assertEqual(result.diagnostics.fallback_reason, "validation_nmse_exceeded")

    def test_short_input_without_a_validation_block_returns_safe_fallback(self):
        input_signal = np.ones(4 * 256, dtype=np.complex128)
        result = fit_pa_model(
            input_signal,
            input_signal,
            PAForwardModelConfig(orders=(1,), memory_depth=1),
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.diagnostics.validation_sample_count, 0)
        self.assertEqual(result.diagnostics.fallback_reason, "insufficient_validation_samples")

    def test_empty_mismatched_and_non_finite_data_return_structured_fallbacks(self):
        cases = (
            (np.array([], dtype=np.complex128), np.array([], dtype=np.complex128), "insufficient_samples"),
            (np.ones(1280), np.ones(1279), "length_mismatch"),
            (
                np.concatenate((np.ones(1279), np.array([np.nan]))),
                np.ones(1280),
                "non_finite_input",
            ),
            (
                np.ones(1280),
                np.concatenate((np.ones(1279), np.array([np.inf]))),
                "non_finite_output",
            ),
        )
        config = PAForwardModelConfig(orders=(1,), memory_depth=1)
        for input_signal, output_signal, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                result = fit_pa_model(input_signal, output_signal, config)
                self.assertFalse(result.succeeded)
                self.assertIsNone(result.model)
                self.assertEqual(result.diagnostics.fallback_reason, expected_reason)


if __name__ == "__main__":
    unittest.main()
