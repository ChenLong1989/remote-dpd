import json
import unittest
from collections.abc import Mapping
from unittest import mock

import numpy as np

from remote_dpd import runtime as runtime_module
from remote_dpd.runtime import (
    ForwardModelILCRuntime,
    RuntimeConfigurationError,
    RuntimeInputError,
    RuntimeStepInput,
    create_runtime,
    list_runtimes,
)

SEVERE_PA = (
    (1, 0, 1.0 + 0.0j),
    (1, 1, 0.04 + 0.015j),
    (3, 0, -1.44 + 0.30j),
    (3, 1, -0.24 + 0.12j),
)
ATTENUATION = 10.0 ** (-0.7 / 20.0)


def memory_polynomial(signal, coefficients):
    output = np.zeros_like(signal)
    for order, memory, value in coefficients:
        delayed = np.roll(signal, memory)
        output += value * delayed * np.abs(delayed) ** (order - 1)
    return output


def band_limited_reference(rng, size, rms, papr_db=8.0):
    """Band-limited Gaussian reference with magnitude-clipped OFDM-like PAPR.

    Raw Gaussian peaks would exceed the PA slope turning point and make the
    inverse nonexistent; clipping keeps the test in the invertible region
    like the real multi-carrier waveform.
    """
    spectrum = np.zeros(size, dtype=np.complex128)
    carriers = np.zeros(size, dtype=bool)
    carrier_count = size // 4
    start = size // 2 - carrier_count // 2
    carriers[start : start + carrier_count] = True
    spectrum[carriers] = (
        rng.normal(size=carrier_count) + 1j * rng.normal(size=carrier_count)
    ) / np.sqrt(2.0)
    signal = np.fft.ifft(spectrum)
    signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2)) * rms
    ceiling = rms * 10.0 ** (papr_db / 20.0)
    return signal * np.minimum(1.0, ceiling / np.abs(signal))


def run_iterations(runtime_name, config, reference, pa, iterations):
    runtime = create_runtime(runtime_name)
    runtime.initialize(config)
    y = reference.copy()
    nmse_history = []
    for iteration in range(1, iterations + 1):
        z = pa(y)
        error = z - reference
        nmse_history.append(
            10.0
            * np.log10(np.mean(np.abs(error) ** 2) / np.mean(np.abs(reference) ** 2))
        )
        result = runtime.step(
            RuntimeStepInput(
                x=reference,
                y_current=y,
                z_current=z,
                iteration=iteration,
                config=config,
            )
        )
        y = result.y_candidate
    runtime.close()
    return np.asarray(nmse_history), y


def severe_simulated_pa(signal):
    return memory_polynomial(signal * ATTENUATION, SEVERE_PA)


class ForwardModelGradientTests(unittest.TestCase):
    def test_adjacent_gradient_matches_central_finite_differences(self):
        rng = np.random.default_rng(11)
        size = 96
        y = (rng.normal(size=size) + 1j * rng.normal(size=size)) * 0.3
        x = (rng.normal(size=size) + 1j * rng.normal(size=size)) * 0.25
        coefficients = (
            (1, 0, 1.0 + 0.1j),
            (1, 1, 0.3 - 0.2j),
            (3, 0, -0.8 + 0.3j),
            (3, 2, 0.4 - 0.2j),
            (5, 1, 0.2 + 0.4j),
        )
        error = memory_polynomial(y, coefficients) - x

        def energy(perturbed):
            residual = memory_polynomial(perturbed, coefficients) - x
            return float(np.sum(np.abs(residual) ** 2))

        epsilon = 1e-5
        finite = np.zeros_like(y)
        for index in range(size):
            for part in (0, 1):
                delta = epsilon if part == 0 else 1j * epsilon
                forward = y.copy()
                forward[index] += delta
                backward = y.copy()
                backward[index] -= delta
                derivative = (energy(forward) - energy(backward)) / (2.0 * epsilon)
                finite[index] += derivative * (1.0 if part == 0 else 1j)

        from remote_dpd.runtime import _adjoint_gradient, _BasisTerm, _ForwardModelFit

        fit = _ForwardModelFit(
            terms=tuple(_BasisTerm(order, memory) for order, memory, _ in coefficients),
            coefficients=np.asarray([value for _, _, value in coefficients]),
            residual_rms=0.0,
        )
        gradient = _adjoint_gradient(y, error, fit)

        np.testing.assert_allclose(finite, 2.0 * gradient, rtol=1e-5, atol=1e-7)

    def test_fitted_coefficients_recover_known_memory_polynomial(self):
        rng = np.random.default_rng(23)
        size = 8192
        y = band_limited_reference(rng, size, 0.2)
        expected = {
            (1, 0): 0.9 + 0.1j,
            (1, 1): -0.05 + 0.02j,
            (3, 0): -1.2 + 0.4j,
            (3, 1): 0.1 - 0.3j,
        }
        coefficients = tuple(
            (order, memory, value) for (order, memory), value in expected.items()
        )
        z = memory_polynomial(y, coefficients)

        fit = runtime_module._fit_memory_polynomial(y, z, (1, 3), (0, 1), ridge=1e-10)

        recovered = {
            (term.order, term.memory): value
            for term, value in zip(fit.terms, fit.coefficients)
        }
        for key, value in expected.items():
            np.testing.assert_allclose(recovered[key], value, rtol=1e-6, atol=1e-9)
        self.assertLess(fit.residual_rms, 1e-10)

    def test_fit_absorbs_linear_gain_phase_and_residual_rms_matches(self):
        rng = np.random.default_rng(31)
        size = 4096
        y = band_limited_reference(rng, size, 0.2)
        scale = 1.4 * np.exp(1j * 0.6)
        z = scale * memory_polynomial(y, ((1, 0, 1.0 + 0j), (3, 0, -0.7 + 0.2j)))
        noise = (rng.normal(size=size) + 1j * rng.normal(size=size)) * 1e-3
        noisy = z + noise

        fit = runtime_module._fit_memory_polynomial(
            y, noisy, (1, 3, 5), (0, 1, 2), ridge=1e-8
        )

        recovered = {
            (term.order, term.memory): value
            for term, value in zip(fit.terms, fit.coefficients)
        }
        np.testing.assert_allclose(recovered[(1, 0)], scale, rtol=2e-3)
        np.testing.assert_allclose(
            recovered[(3, 0)], scale * (-0.7 + 0.2j), rtol=5e-3, atol=5e-3
        )
        np.testing.assert_allclose(recovered[(5, 0)], 0.0 + 0.0j, atol=1e-2)

        columns = np.stack(
            [runtime_module._basis_column(y, term) for term in fit.terms],
            axis=1,
        )
        model = columns @ fit.coefficients
        direct_residual = float(np.sqrt(np.mean(np.abs(noisy - model) ** 2)))
        self.assertLessEqual(fit.residual_rms, direct_residual * 1.001 + 1e-12)

    def test_blockwise_accumulation_matches_single_block(self):
        rng = np.random.default_rng(47)
        size = 5000
        y = band_limited_reference(rng, size, 0.2)
        z = severe_simulated_pa(y)

        single = runtime_module._fit_memory_polynomial(
            y, z, (1, 3, 5), (0, 1, 2), ridge=1e-8
        )
        with mock.patch.object(runtime_module, "_FORWARD_MODEL_BLOCK_TERMS", 9 * 997):
            blocked = runtime_module._fit_memory_polynomial(
                y, z, (1, 3, 5), (0, 1, 2), ridge=1e-8
            )

        np.testing.assert_allclose(
            single.coefficients, blocked.coefficients, rtol=1e-9, atol=1e-12
        )
        # Near-exact fits leave only floating-point cancellation noise, so the
        # residual is compared against an absolute bound rather than between runs.
        self.assertLess(single.residual_rms, 1e-6)
        self.assertLess(blocked.residual_rms, 1e-6)

    def test_periodic_memory_uses_circular_roll(self):
        y = np.zeros(8, dtype=np.complex128)
        y[0] = 0.5 + 0.1j
        column = runtime_module._basis_column(y, runtime_module._BasisTerm(3, 2))
        expected = np.zeros(8, dtype=np.complex128)
        expected[2] = (0.5 + 0.1j) * np.abs(0.5 + 0.1j) ** 2
        np.testing.assert_allclose(column, expected)

    def test_zero_input_produces_zero_gradient_and_identity_candidate(self):
        zeros = np.zeros(16, dtype=np.complex128)
        reference = np.zeros(16, dtype=np.complex128)
        runtime = ForwardModelILCRuntime()
        runtime.initialize({})
        result = runtime.step(
            RuntimeStepInput(
                x=reference, y_current=zeros, z_current=zeros, iteration=1, config={}
            )
        )
        np.testing.assert_array_equal(result.y_candidate, zeros)

    def test_constant_input_stays_finite_with_collinear_basis(self):
        constant = np.full(32, 0.2 + 0.05j)
        reference = np.full(32, 0.1 - 0.1j)
        runtime = ForwardModelILCRuntime()
        runtime.initialize({})
        result = runtime.step(
            RuntimeStepInput(
                x=reference,
                y_current=constant.copy(),
                z_current=constant * 2.0,
                iteration=1,
                config={},
            )
        )
        self.assertTrue(np.all(np.isfinite(result.y_candidate)))


class ForwardModelConvergenceTests(unittest.TestCase):
    def test_converges_where_basic_ilc_diverges_on_rotated_loop_gain(self):
        rng = np.random.default_rng(5)
        size = 8192
        reference = band_limited_reference(rng, size, 0.2)
        loop_gain = 3.0j

        def pa(signal):
            return loop_gain * signal

        # Basic ILC needs |1 - mu*c| < 1; at mu=0.5 the rotated gain gives
        # |1 - 1.5j| ~ 1.8 and the iteration diverges.
        basic_history, _ = run_iterations("basic_ilc", {"mu": 0.5}, reference, pa, 25)
        # The model update converges for any loop phase as long as
        # mu * |c|^2 < 2 holds (0.1 * 9 = 0.9 here).
        model_history, _ = run_iterations(
            "forward_model_ilc", {"mu": 0.1}, reference, pa, 25
        )

        self.assertGreater(basic_history[-1], basic_history[0])
        self.assertLess(model_history[-1], -60.0)
        self.assertLess(model_history[-1], model_history[0] - 20.0)

    def test_severe_compression_converges_monotonically(self):
        rng = np.random.default_rng(7)
        size = 24576
        reference = band_limited_reference(rng, size, 10.0 ** (-15 / 20))

        history, final_y = run_iterations(
            "forward_model_ilc", {"mu": 1.0}, reference, severe_simulated_pa, 15
        )

        self.assertTrue(
            np.all(np.diff(history) <= 0.02),
            f"NMSE history must be non-increasing, got {np.round(history, 3)}",
        )
        self.assertLess(history[-1], history[0] - 10.0)
        self.assertLess(np.max(np.abs(final_y)), 0.75)


class ForwardModelContractTests(unittest.TestCase):
    def make_runtime(self, config):
        runtime = ForwardModelILCRuntime()
        runtime.initialize({} if config is None else config)
        return runtime

    def step_once(self, runtime, config, size=64):
        rng = np.random.default_rng(13)
        y = band_limited_reference(rng, size, 0.2)
        z = severe_simulated_pa(y)
        return runtime.step(
            RuntimeStepInput(
                x=y * 0.9,
                y_current=y,
                z_current=z,
                iteration=1,
                config={} if config is None else config,
            )
        )

    def test_default_config_matches_documented_values(self):
        prepared = ForwardModelILCRuntime()._prepare_config({})
        self.assertEqual(prepared["mu"], 1.0)
        self.assertEqual(prepared["orders"], (1, 3, 5))
        self.assertEqual(prepared["memory_depths"], (0, 1, 2))
        self.assertEqual(prepared["ridge"], 1e-8)

    def test_metrics_are_finite_and_report_model(self):
        runtime = self.make_runtime(None)
        result = self.step_once(runtime, None)
        metrics = result.metrics

        def unwrap(value):
            if isinstance(value, Mapping):
                return {key: unwrap(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [unwrap(item) for item in value]
            return value

        json.dumps(unwrap(dict(metrics)), allow_nan=False)
        self.assertEqual(len(metrics["model_terms"]), 9)
        for term in metrics["model_terms"]:
            self.assertIn("p", term)
            self.assertIn("m", term)
            self.assertIn("real", term)
            self.assertIn("imag", term)
        self.assertEqual(metrics["mu"], 1.0)
        self.assertEqual(metrics["runtime_step"], 1)
        self.assertGreaterEqual(metrics["gradient_rms"], 0.0)
        self.assertGreaterEqual(metrics["model_residual_rms"], 0.0)

    def test_rejects_unknown_fields_and_invalid_mu(self):
        cases = [
            {"mu": 0.0},
            {"mu": -0.5},
            {"mu": float("inf")},
            {"mu": True},
            {"mu": "1.0"},
            {"mu": 1.0, "unexpected": 1},
        ]
        for config in cases:
            with (
                self.subTest(config=config),
                self.assertRaises(RuntimeConfigurationError),
            ):
                ForwardModelILCRuntime().initialize(config)

    def test_rejects_invalid_orders_and_memory_depths(self):
        cases = [
            {"orders": []},
            {"orders": (2, 3)},
            {"orders": (3, 1)},
            {"orders": (1, 1)},
            {"orders": (1, 11)},
            {"orders": (1, 3, 5, 7, 9, 11)},
            {"orders": "135"},
            {"orders": (1.0, 3.0)},
            {"memory_depths": []},
            {"memory_depths": (-1,)},
            {"memory_depths": (2, 1)},
            {"memory_depths": (1, 17)},
            {"memory_depths": tuple(range(9))},
            {"memory_depths": 3},
            {"orders": (1, 3, 5, 7, 9), "memory_depths": (0, 1, 2, 3)},
        ]
        for config in cases:
            with (
                self.subTest(config=config),
                self.assertRaises(RuntimeConfigurationError),
            ):
                ForwardModelILCRuntime().initialize(config)

    def test_rejects_invalid_ridge(self):
        for ridge in (0.0, -1e-8, 1e-2 + 1e-12, float("inf"), True):
            with (
                self.subTest(ridge=ridge),
                self.assertRaises(RuntimeConfigurationError),
            ):
                ForwardModelILCRuntime().initialize({"ridge": ridge})

    def test_accepts_valid_custom_structure(self):
        runtime = self.make_runtime(
            {"mu": 0.5, "orders": [1, 3], "memory_depths": [0, 1, 2, 3], "ridge": 1e-6}
        )
        result = self.step_once(
            runtime,
            {"mu": 0.5, "orders": [1, 3], "memory_depths": [0, 1, 2, 3], "ridge": 1e-6},
        )
        self.assertEqual(len(result.metrics["model_terms"]), 8)

    def test_step_config_must_match_initialization(self):
        runtime = self.make_runtime({"mu": 0.7})
        with self.assertRaises(RuntimeConfigurationError):
            self.step_once(runtime, {"mu": 0.9})
        # Equivalent list/tuple forms must compare equal after normalization.
        self.step_once(runtime, {"mu": 0.7, "orders": (1, 3, 5)})

    def test_nonfinite_candidate_is_rejected(self):
        runtime = self.make_runtime({})
        magnitude = np.asarray(np.full(32, 1e150), dtype=np.complex128)
        with self.assertRaises(RuntimeInputError):
            runtime.step(
                RuntimeStepInput(
                    x=magnitude * 0.9,
                    y_current=magnitude.copy(),
                    z_current=magnitude.copy(),
                    iteration=1,
                    config={},
                )
            )

    def test_registry_lists_and_isolates_instances(self):
        self.assertIn("forward_model_ilc", list_runtimes())
        first = create_runtime("forward_model_ilc")
        second = create_runtime("forward_model_ilc")
        first.initialize({})
        second.initialize({})
        rng = np.random.default_rng(19)
        y = band_limited_reference(rng, 64, 0.2)
        z = severe_simulated_pa(y)
        request = RuntimeStepInput(
            x=y * 0.9, y_current=y, z_current=z, iteration=1, config={}
        )
        first.step(request)
        first.step(request)
        result = second.step(request)
        self.assertEqual(result.metrics["runtime_step"], 1)
        first.reset()
        first.initialize({})
        self.assertEqual(first.step(request).metrics["runtime_step"], 1)


class ForwardModelClosedLoopTests(unittest.TestCase):
    """Controller-level A/B: classic ILC diverges, forward-model ILC converges."""

    SEVERE_PA = (
        {"p": 1, "m": 0, "real": 1.0, "imag": 0.0},
        {"p": 1, "m": 1, "real": 0.04, "imag": 0.015},
        {"p": 3, "m": 0, "real": -2.5, "imag": 0.3},
        {"p": 3, "m": 1, "real": -0.24, "imag": 0.12},
    )

    @staticmethod
    def make_reference():
        rng = np.random.default_rng(11)
        size = 8192
        spectrum = np.zeros(size, dtype=np.complex128)
        carrier_count = size // 4
        start = size // 2 - carrier_count // 2
        spectrum[start : start + carrier_count] = (
            rng.normal(size=carrier_count) + 1j * rng.normal(size=carrier_count)
        ) / np.sqrt(2.0)
        signal = np.fft.ifft(spectrum)
        signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2)) * 10.0 ** (-15 / 20)
        ceiling = 10.0 ** (-15 / 20) * 10.0 ** (7 / 20)
        return signal * np.minimum(1.0, ceiling / np.abs(signal))

    def run_closed_loop(self, runtime_name, mu, pa_coefficients, power_reference):
        from remote_dpd import (
            ClosedLoopConfig,
            ClosedLoopController,
            DeviceConfig,
            create_rf_bench,
        )
        from remote_dpd.power_control import PowerController

        config = ClosedLoopConfig(
            device_config=DeviceConfig(
                sample_rate_hz=245.76e6,
                average_segment_count=2,
                settle_seconds=0.0,
                call_timeout_seconds=5.0,
                device_options={
                    "max_capture_samples": 8192 * 4,
                    "noise_dbfs": -100.0,
                    "random_seed": 3,
                    "pa_coefficients": list(pa_coefficients),
                    "power_reference_dbm": power_reference,
                },
            ),
            runtime_name=runtime_name,
            runtime_config={"mu": mu},
            max_iterations=15,
        )
        bench = create_rf_bench("simulated")
        controller = ClosedLoopController(
            bench,
            power_controller=PowerController(sleep_fn=lambda _: None),
        )
        controller.connect()
        controller.apply_config(config)
        controller.load_reference(self.make_reference())
        return controller

    def test_basic_ilc_diverges_under_severe_compression(self):
        from remote_dpd import ControllerState
        from remote_dpd.safety import DigitalSafetyError

        controller = self.run_closed_loop(
            "basic_ilc", 0.35, self.SEVERE_PA, power_reference=3.0
        )
        with self.assertRaises(DigitalSafetyError):
            controller.run_auto()

        snapshot = controller.snapshot()
        self.assertIs(snapshot.state, ControllerState.FAILED)
        self.assertEqual(snapshot.last_error.code, "digital_safety")
        history = [record.preprocessing.nmse_db for record in snapshot.records]
        self.assertGreater(min(history), -30.0)
        self.assertGreater(history[-1], history[0])

    def test_forward_model_ilc_converges_where_basic_diverges(self):
        from remote_dpd import ControllerState

        controller = self.run_closed_loop(
            "forward_model_ilc", 1.0, self.SEVERE_PA, power_reference=3.0
        )
        result = controller.run_auto()

        self.assertIs(result.state, ControllerState.COMPLETED)
        history = [record.preprocessing.nmse_db for record in result.records]
        self.assertEqual(len(history), 16)
        self.assertTrue(
            np.all(np.diff(history) <= 0.02),
            f"NMSE history must be non-increasing, got {np.round(history, 3)}",
        )
        self.assertLess(history[-1], history[0] - 5.0)
        self.assertTrue(all(record.digital_safety.passed for record in result.records))

    def test_forward_model_ilc_converges_fast_on_mild_pa(self):
        from remote_dpd import ControllerState

        mild_pa = (
            {"p": 1, "m": 0, "real": 1.0, "imag": 0.0},
            {"p": 1, "m": 1, "real": 0.04, "imag": 0.015},
            {"p": 3, "m": 0, "real": -0.6, "imag": 0.075},
            {"p": 3, "m": 1, "real": -0.12, "imag": 0.03},
        )
        controller = self.run_closed_loop(
            "forward_model_ilc", 1.0, mild_pa, power_reference=15.0
        )
        result = controller.run_auto()

        self.assertIs(result.state, ControllerState.COMPLETED)
        history = [record.preprocessing.nmse_db for record in result.records]
        self.assertLess(min(history[1:6]), -55.0)


if __name__ == "__main__":
    unittest.main()
