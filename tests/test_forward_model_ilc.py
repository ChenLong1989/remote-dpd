import json
import unittest
from collections.abc import Mapping
from unittest import mock

import numpy as np

from remote_dpd import runtime as runtime_module
from remote_dpd.forward_model import (
    TAP_COUNTS,
    FLFResidualModel,
    tap_pairs_for_count,
)
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


class FLFStructureTests(unittest.TestCase):
    def test_tap_ladder_matches_configured_set(self):
        # Three diagonal delays: tap_count 1 keeps the aligned tap, 3 the
        # full set; U = |unique(d_x)|.
        expected = {1: (1, 1), 3: (3, 3)}
        for tap_count, (taps, linear) in expected.items():
            with self.subTest(tap_count=tap_count):
                model = FLFResidualModel(tap_count, 32)
                self.assertEqual(model.tap_total, taps)
                self.assertEqual(model.linear_count, linear)
                self.assertEqual(
                    model.n_columns(), 2 * (linear + taps * (32 - 2))
                )

    def test_tap_selection_is_nested(self):
        selected = set()
        for tap_count in TAP_COUNTS:
            taps = set(tap_pairs_for_count(tap_count))
            self.assertTrue(selected <= taps)
            selected = taps
        self.assertEqual(
            tap_pairs_for_count(3), ((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0))
        )

    def test_sparse_lut_weights_match_dense_hats(self):
        rng = np.random.default_rng(3)
        amplitude = np.abs(rng.standard_normal(20000)) * 1.05
        for lut_size in (8, 16, 32, 64):
            with self.subTest(lut_size=lut_size):
                model = FLFResidualModel(1, lut_size)
                lo, hi, w_lo, w_hi = model.lut_weights(amplitude)
                dense = model._dense_hats(amplitude)
                sparse = np.zeros_like(dense)
                rows = np.arange(amplitude.size)
                sparse[rows, lo - 1] = w_lo
                sparse[rows, np.clip(hi - 1, 0, lut_size - 3)] += w_hi
                np.testing.assert_allclose(dense, sparse, atol=1e-12)
        # Amplitudes beyond (Q-1)/Q deactivate every hat.
        model = FLFResidualModel(1, 32)
        _, _, w_lo, w_hi = model.lut_weights(np.asarray([0.999]))
        self.assertEqual((w_lo[0], w_hi[0]), (0.0, 0.0))

    def test_rejects_invalid_constructor_arguments(self):
        for tap_count in (0, 2, 9, 47, True, "17"):
            with (
                self.subTest(tap_count=tap_count),
                self.assertRaises(ValueError),
            ):
                FLFResidualModel(tap_count, 32)
        for lut_size in (2, 0, True, 32.0, "32"):
            with (
                self.subTest(lut_size=lut_size),
                self.assertRaises(ValueError),
            ):
                FLFResidualModel(17, lut_size)


class FLFGradientTests(unittest.TestCase):
    def test_adjoint_matches_central_finite_differences(self):
        rng = np.random.default_rng(11)
        size = 1024
        model = FLFResidualModel(3, 16)
        y = band_limited_reference(rng, size, 0.3)
        x = band_limited_reference(rng, size, 0.25)
        weights = (rng.normal(size=model.n_columns()) + 1j * rng.normal(size=model.n_columns())) * 0.05

        def energy(perturbed):
            error = model.evaluate(perturbed, weights) - x
            return float(np.vdot(error, error).real)

        error = model.evaluate(y, weights) - x
        gradient = model.adjoint(y, error, weights)

        ratios = []
        for _ in range(4):
            direction = rng.normal(size=size) + 1j * rng.normal(size=size)
            direction /= np.sqrt(np.vdot(direction, direction).real)
            for epsilon in (1e-6, 1e-7):
                forward = energy(y + epsilon * direction)
                backward = energy(y - epsilon * direction)
                finite = (forward - backward) / (2.0 * epsilon)
                analytic = 2.0 * float(np.vdot(gradient, direction).real)
                ratios.append(finite / analytic)
        np.testing.assert_allclose(ratios, 1.0, rtol=2e-5)

    def test_zero_coefficients_reduce_gradient_to_identity_error(self):
        rng = np.random.default_rng(5)
        size = 512
        model = FLFResidualModel(3, 16)
        y = band_limited_reference(rng, size, 0.3)
        x = band_limited_reference(rng, size, 0.3)
        error = y - x
        gradient = model.adjoint(y, error, np.zeros(model.n_columns(), dtype=np.complex128))
        np.testing.assert_allclose(gradient, error)


class FLFFitTests(unittest.TestCase):
    def test_fit_recovers_smooth_known_model(self):
        rng = np.random.default_rng(23)
        size = 4096
        model = FLFResidualModel(3, 16)
        y = band_limited_reference(rng, size, 0.2)
        # Smooth LUT rows (low-frequency along the knot axis) so the
        # difference penalty does not bias the recovery.
        knots = np.arange(1, model.lut_size - 1)
        beta_rows = 0.3 * np.exp(2j * np.pi * knots / (model.lut_size - 1))
        weights = np.zeros(model.n_columns(), dtype=np.complex128)
        weights[: model.linear_count] = 0.8 - 0.1j
        weights[model.linear_count : 2 * model.linear_count] = 0.05 + 0.02j
        for r in (0, 1):
            base = 2 * model.linear_count + r * model.tap_total * (model.lut_size - 2)
            for t in range(model.tap_total):
                weights[base + t * (model.lut_size - 2) : base + (t + 1) * (model.lut_size - 2)] = (
                    beta_rows * (0.5 + 0.2 * t + 0.3 * r)
                )
        z = model.evaluate(y, weights)

        fit = model.fit(y, z, ridge=1e-10, lut_ridge=1e-8)

        # The basis is structurally redundant (linear and LUT columns can
        # express each other on band-limited data), so individual coefficients
        # are not identifiable; the fitted model must reproduce the data.
        self.assertLess(fit.residual_rms, 1e-5)
        prediction = model.evaluate(y, fit.coefficients)
        np.testing.assert_allclose(prediction, z, rtol=1e-5, atol=1e-7)

    def test_fit_residual_tracks_noise_floor(self):
        rng = np.random.default_rng(31)
        size = 4096
        model = FLFResidualModel(3, 16)
        y = band_limited_reference(rng, size, 0.2)
        z = severe_simulated_pa(y)
        noise = (rng.normal(size=size) + 1j * rng.normal(size=size)) * 2e-3 / np.sqrt(2.0)

        fit = model.fit(y, z + noise, ridge=1e-8, lut_ridge=1e-6)

        self.assertLess(fit.residual_rms, 3e-3)
        self.assertGreater(fit.residual_rms, 1e-3)

    def test_blockwise_accumulation_matches_single_block(self):
        rng = np.random.default_rng(47)
        size = 5000
        model = FLFResidualModel(3, 16)
        y = band_limited_reference(rng, size, 0.2)
        z = severe_simulated_pa(y)

        single = model.fit(y, z, ridge=1e-8, lut_ridge=1e-3)
        import remote_dpd.forward_model as forward_model_module

        with mock.patch.object(
            forward_model_module, "_BLOCK_TERMS", 97 * model.n_columns()
        ):
            blocked = model.fit(y, z, ridge=1e-8, lut_ridge=1e-3)
            reevaluated = model.evaluate(y, blocked.coefficients)

        # The FLF basis is structurally rank-deficient, so the solve is
        # sensitive to the block summation order: coefficients agree to
        # ~1e-6 and the model output and residual agree far tighter.
        np.testing.assert_allclose(
            single.coefficients, blocked.coefficients, rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            reevaluated, model.evaluate(y, single.coefficients), atol=1e-7
        )
        self.assertAlmostEqual(
            blocked.residual_rms / single.residual_rms, 1.0, places=6
        )

    def test_identity_passthrough_yields_zero_coefficients(self):
        rng = np.random.default_rng(53)
        size = 2048
        model = FLFResidualModel(3, 16)
        y = band_limited_reference(rng, size, 0.2)

        fit = model.fit(y, y, ridge=1e-8, lut_ridge=1e-3)

        self.assertEqual(fit.residual_rms, 0.0)
        self.assertLess(float(np.max(np.abs(fit.coefficients))), 1e-10)


class ForwardModelRuntimeTests(unittest.TestCase):
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

    def test_identity_passthrough_degrades_to_basic_ilc_update(self):
        rng = np.random.default_rng(59)
        size = 1024
        y = band_limited_reference(rng, size, 0.2)
        x = band_limited_reference(rng, size, 0.2)
        runtime = ForwardModelILCRuntime()
        runtime.initialize({})
        result = runtime.step(
            RuntimeStepInput(
                x=x, y_current=y.copy(), z_current=y.copy(), iteration=1, config={}
            )
        )
        # A perfectly linear PA leaves z - y = 0, so the fitted basis is zero
        # and the update reduces exactly to the basic ILC step.
        np.testing.assert_allclose(result.y_candidate, y - 1.0 * (y - x), atol=1e-12)


class ForwardModelConvergenceTests(unittest.TestCase):
    def setUp(self):
        # These tests pin the regular per-iteration refitting behaviour; the
        # temporary debug freeze (runtime._FORWARD_MODEL_FIT_FREEZE_AFTER_
        # ITERATION) must stay disabled for them.
        patcher = mock.patch.object(
            runtime_module, "_FORWARD_MODEL_FIT_FREEZE_AFTER_ITERATION", None
        )
        patcher.start()
        self.addCleanup(patcher.stop)

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
        size = 12288
        reference = band_limited_reference(rng, size, 10.0 ** (-15 / 20))

        history, final_y = run_iterations(
            "forward_model_ilc", {"mu": 1.0}, reference, severe_simulated_pa, 12
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
        self.assertEqual(prepared["tap_count"], 3)
        self.assertEqual(prepared["lut_size"], 32)
        self.assertEqual(prepared["ridge"], 1e-8)
        self.assertEqual(prepared["lut_ridge"], 1e-3)

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
        summary = metrics["model_coefficients"]
        self.assertEqual(summary["tap_count"], 3)
        self.assertEqual(summary["lut_size"], 32)
        self.assertEqual(summary["columns"], 2 * (3 + 3 * 30))
        self.assertEqual(len(summary["alpha"]), 6)
        for entry in summary["alpha"]:
            self.assertIn("phase", entry)
            self.assertIn("delay", entry)
            self.assertIn("real", entry)
            self.assertIn("imag", entry)
        self.assertEqual(summary["beta_count"], 2 * 3 * 30)
        self.assertGreaterEqual(summary["beta_rms"], 0.0)
        self.assertGreaterEqual(summary["beta_max"], 0.0)
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
            {"orders": [1, 3, 5]},
            {"memory_depths": [0, 1, 2]},
        ]
        for config in cases:
            with (
                self.subTest(config=config),
                self.assertRaises(RuntimeConfigurationError),
            ):
                ForwardModelILCRuntime().initialize(config)

    def test_rejects_invalid_tap_count_and_lut_size(self):
        cases = [
            {"tap_count": 0},
            {"tap_count": 2},
            {"tap_count": 8},
            {"tap_count": 17},
            {"tap_count": 46},
            {"tap_count": True},
            {"tap_count": "3"},
            {"tap_count": 3.0},
            {"lut_size": 2},
            {"lut_size": 0},
            {"lut_size": True},
            {"lut_size": 32.0},
            {"lut_size": "32"},
            # Column budget: 46 taps x 128 knots far exceeds the cap.
            {"tap_count": 46, "lut_size": 128},
        ]
        for config in cases:
            with (
                self.subTest(config=config),
                self.assertRaises(RuntimeConfigurationError),
            ):
                ForwardModelILCRuntime().initialize(config)

    def test_rejects_invalid_ridges(self):
        for field in ("ridge", "lut_ridge"):
            for value in (0.0, -1e-8, 1e-2 + 1e-12, float("inf"), True):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(RuntimeConfigurationError),
                ):
                    ForwardModelILCRuntime().initialize({field: value})

    def test_accepts_valid_custom_structure(self):
        config = {"mu": 0.5, "tap_count": 1, "lut_size": 16, "lut_ridge": 1e-4}
        runtime = self.make_runtime(config)
        result = self.step_once(runtime, config)
        self.assertEqual(
            result.metrics["model_coefficients"]["columns"], 2 * (1 + 1 * 14)
        )

    def test_step_config_must_match_initialization(self):
        runtime = self.make_runtime({"mu": 0.7})
        with self.assertRaises(RuntimeConfigurationError):
            self.step_once(runtime, {"mu": 0.9})
        # Equivalent forms must compare equal after normalization.
        self.step_once(runtime, {"mu": 0.7, "tap_count": 3})

    def test_nonfinite_candidate_is_rejected(self):
        runtime = self.make_runtime({})
        magnitude = np.asarray(np.full(32, 1e155), dtype=np.complex128)
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

    def setUp(self):
        # See ForwardModelConvergenceTests.setUp: keep the debug freeze off.
        patcher = mock.patch.object(
            runtime_module, "_FORWARD_MODEL_FIT_FREEZE_AFTER_ITERATION", None
        )
        patcher.start()
        self.addCleanup(patcher.stop)


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
        # The effective runtime configuration recorded after the run must
        # include every normalized default, not only the caller-supplied mu.
        recorded = dict(result.config.runtime_config)
        self.assertEqual(
            recorded,
            {
                "mu": 1.0,
                "tap_count": 3,
                "lut_size": 32,
                "ridge": 1e-8,
                "lut_ridge": 1e-3,
            },
        )
        json.dumps(result.config.to_dict(), allow_nan=False)

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
