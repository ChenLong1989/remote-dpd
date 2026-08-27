import unittest

import numpy as np

from remote_dpd import (
    BasicILCRuntime,
    CaptureBatch,
    FeedbackPreprocessor,
    RuntimeStepInput,
    validate_candidate,
)


class CorePipelineTests(unittest.TestCase):
    def test_fixed_gain_feedback_flows_into_basic_ilc_and_safety(self):
        samples = np.arange(128)
        x = (
            0.25 * np.exp(2j * np.pi * samples / 31)
            + 0.10 * np.exp(2j * np.pi * samples / 17)
        ).astype(np.complex128)
        sample_rate_hz = 245.76e6
        preprocessor = FeedbackPreprocessor(x, sample_rate_hz)

        calibration_capture = np.roll(x, 3) * 0.5 * np.exp(0.4j)
        calibration = preprocessor.process(
            [CaptureBatch(calibration_capture, x.size, 1, sample_rate_hz)]
        )
        np.testing.assert_allclose(calibration.z, x, atol=1e-10)

        changed_capture = np.roll(x, -2) * 0.6 * np.exp(-0.7j)
        current = preprocessor.process(
            [CaptureBatch(changed_capture, x.size, 1, sample_rate_hz)],
            gain_correction=calibration.gain_correction,
        )
        np.testing.assert_allclose(current.z, 1.2 * x, atol=1e-10)

        runtime = BasicILCRuntime()
        runtime.initialize({"mu": 0.5})
        result = runtime.step(
            RuntimeStepInput(
                x=x,
                y_current=x,
                z_current=current.z,
                iteration=1,
                config={"mu": 0.5},
            )
        )

        np.testing.assert_allclose(result.y_candidate, 0.9 * x, atol=1e-10)
        self.assertTrue(validate_candidate(x, result.y_candidate).passed)


if __name__ == "__main__":
    unittest.main()
