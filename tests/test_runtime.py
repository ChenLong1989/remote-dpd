import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from remote_dpd.runtime import (
    RUNTIME_API_VERSION,
    BasicILCRuntime,
    DPDRuntime,
    RuntimeConfigurationError,
    RuntimeInputError,
    RuntimeLifecycleError,
    RuntimeStepInput,
    RuntimeStepResult,
    create_runtime,
    list_runtimes,
    register_runtime,
)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray([0.1 + 0.2j, -0.2 + 0.1j], dtype=np.complex128)
        self.y = np.asarray([0.2 + 0.2j, -0.1 + 0.1j], dtype=np.complex128)
        self.z = np.asarray([0.15 + 0.25j, -0.25 + 0.05j], dtype=np.complex128)

    def request(self, *, config=None, iteration=0):
        return RuntimeStepInput(
            x=self.x,
            y_current=self.y,
            z_current=self.z,
            iteration=iteration,
            config={} if config is None else config,
        )

    def test_basic_ilc_uses_default_mu_and_formula(self):
        runtime = BasicILCRuntime()
        runtime.initialize({})

        result = runtime.step(self.request())

        expected = self.y - 0.5 * (self.z - self.x)
        np.testing.assert_allclose(result.y_candidate, expected)
        self.assertEqual(result.metrics["mu"], 0.5)
        self.assertEqual(result.metrics["iteration"], 0)
        self.assertEqual(result.metrics["runtime_step"], 1)

    def test_basic_ilc_accepts_positive_finite_mu(self):
        runtime = BasicILCRuntime()
        runtime.initialize({"mu": np.float64(0.25)})

        result = runtime.step(self.request(config={"mu": 0.25}, iteration=3))

        np.testing.assert_allclose(
            result.y_candidate,
            self.y - 0.25 * (self.z - self.x),
        )
        self.assertEqual(result.metrics["iteration"], 3)

    def test_basic_ilc_rejects_invalid_mu_and_unknown_fields(self):
        invalid_values = [0.0, -0.1, float("nan"), float("inf"), True, "0.5", 0.5j]
        for value in invalid_values:
            with self.subTest(mu=value), self.assertRaises(RuntimeConfigurationError):
                BasicILCRuntime().initialize({"mu": value})
        with self.assertRaises(RuntimeConfigurationError):
            BasicILCRuntime().initialize({"mu": 0.5, "alpha": 0.0})

    def test_lifecycle_requires_initialize_and_reinitialize_after_reset(self):
        runtime = BasicILCRuntime()
        with self.assertRaises(RuntimeLifecycleError):
            runtime.step(self.request())

        runtime.initialize({})
        runtime.step(self.request())
        runtime.reset()
        self.assertFalse(runtime.initialized)
        with self.assertRaises(RuntimeLifecycleError):
            runtime.step(self.request())

        runtime.initialize({})
        result = runtime.step(self.request())
        self.assertEqual(result.metrics["runtime_step"], 1)

    def test_close_is_idempotent_and_terminal(self):
        runtime = BasicILCRuntime()
        runtime.initialize({})
        runtime.close()
        runtime.close()
        self.assertTrue(runtime.closed)
        with self.assertRaises(RuntimeLifecycleError):
            runtime.initialize({})
        with self.assertRaises(RuntimeLifecycleError):
            runtime.reset()

    def test_step_config_must_match_initialization(self):
        runtime = BasicILCRuntime()
        runtime.initialize({})
        with self.assertRaises(RuntimeConfigurationError):
            runtime.step(self.request(config={"mu": 0.25}))

    def test_step_input_validates_shape_length_finiteness_and_iteration(self):
        with self.assertRaises(RuntimeInputError):
            RuntimeStepInput(self.x.reshape(1, -1), self.y, self.z, 0, {})
        with self.assertRaises(RuntimeInputError):
            RuntimeStepInput(np.asarray([], dtype=complex), self.y, self.z, 0, {})
        with self.assertRaises(RuntimeInputError):
            RuntimeStepInput(self.x, self.y[:1], self.z, 0, {})
        with self.assertRaises(RuntimeInputError):
            RuntimeStepInput(self.x, self.y, np.asarray([np.nan, 0.0]), 0, {})
        for iteration in (-1, 1.5, True):
            with (
                self.subTest(iteration=iteration),
                self.assertRaises(RuntimeInputError),
            ):
                RuntimeStepInput(self.x, self.y, self.z, iteration, {})

    def test_inputs_and_outputs_are_defensively_immutable(self):
        source = self.x.copy()
        config = {
            "nested": {"items": [1, 2]},
            "array": np.asarray([1.0, 2.0]),
        }
        request = RuntimeStepInput(source, self.y, self.z, 0, config)
        source[0] = 99.0
        config["nested"]["items"][0] = 99
        config["array"][0] = 99.0
        self.assertEqual(request.x[0], self.x[0])
        self.assertEqual(request.config["nested"]["items"], (1, 2))
        np.testing.assert_array_equal(request.config["array"], [1.0, 2.0])
        with self.assertRaises(TypeError):
            request.config["new"] = 1
        with self.assertRaises(TypeError):
            request.config["nested"]["new"] = 1
        with self.assertRaises(TypeError):
            request.config["nested"]["items"][0] = 1
        with self.assertRaises(ValueError):
            request.config["array"].setflags(write=True)
        with self.assertRaises(ValueError):
            request.x[0] = 1.0
        with self.assertRaises(ValueError):
            request.x.setflags(write=True)

        runtime = BasicILCRuntime()
        runtime.initialize({})
        result = runtime.step(self.request())
        with self.assertRaises(TypeError):
            result.metrics["mu"] = 0.1
        with self.assertRaises(ValueError):
            result.y_candidate[0] = 1.0
        with self.assertRaises(ValueError):
            result.y_candidate.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            result.metrics = {}

    def test_runtime_config_rejects_mutable_or_non_finite_opaque_values(self):
        with self.assertRaises(RuntimeConfigurationError):
            RuntimeStepInput(self.x, self.y, self.z, 0, {"opaque": object()})
        with self.assertRaises(RuntimeConfigurationError):
            RuntimeStepInput(self.x, self.y, self.z, 0, {"set": {1, 2}})
        with self.assertRaises(RuntimeConfigurationError):
            RuntimeStepInput(self.x, self.y, self.z, 0, {"value": float("nan")})
        with self.assertRaises(RuntimeConfigurationError):
            RuntimeStepInput(
                self.x,
                self.y,
                self.z,
                0,
                {"array": np.asarray([1.0, np.inf])},
            )

    def test_registry_creates_isolated_runtime_instances(self):
        self.assertEqual(RUNTIME_API_VERSION, BasicILCRuntime.api_version)
        self.assertIn("basic_ilc", list_runtimes())
        first = create_runtime(" BASIC_ILC ")
        second = create_runtime("basic_ilc")
        self.assertIsInstance(first, BasicILCRuntime)
        self.assertIsNot(first, second)

        first.initialize({})
        first_result = first.step(self.request())
        second.initialize({})
        second_result = second.step(self.request())
        self.assertEqual(first_result.metrics["runtime_step"], 1)
        self.assertEqual(second_result.metrics["runtime_step"], 1)

    def test_registry_supports_a_version_compatible_extension(self):
        class PassthroughRuntime(DPDRuntime):
            name = "test_passthrough"

            def _step(self, step_input, config):
                return RuntimeStepResult(step_input.y_current, {"tag": "passthrough"})

        register_runtime(PassthroughRuntime.name, PassthroughRuntime, replace=True)
        runtime = create_runtime(PassthroughRuntime.name)
        runtime.initialize({})
        result = runtime.step(self.request())
        np.testing.assert_array_equal(result.y_candidate, self.y)
        self.assertEqual(result.metrics["tag"], "passthrough")


if __name__ == "__main__":
    unittest.main()
