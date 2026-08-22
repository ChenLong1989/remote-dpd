from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from remote_dpd.config import algorithm_config_fingerprint, config_from_mat
from remote_dpd.exceptions import MatProtocolError
from remote_dpd.protocol import load_mat, save_mat
from remote_dpd.service import RemoteDPDService, ServiceOptions


class ConfigurationValidationTests(unittest.TestCase):
    def test_invalid_numeric_enum_and_model_conflict_values_are_rejected(self):
        invalid = (
            {"LearningRate": np.nan},
            {"InternalSamplingRate": 0.0},
            {"ILCBackwardMode": "unknown"},
            {"ILCCalibrationMode": "unknown"},
            {"ILCCalibrationMode": "explicit"},
            {"PAModelOrder": 4},
            {"PAModelMemoryDepth": 0},
            {"PAModelRidge": -1.0},
            {"ILCCGTolerance": 1.0},
            {"ILCLMDamping": 1e-9},
            {"ILCTrustRegionRatio": 1.1},
            {"PAModelFallback": "unsafe"},
            {"ILCBackwardMode": "model_lm", "phaseCompensate": 1},
            {"ILCBackwardMode": "model_vjp", "alpha": 0.1},
            {"ILCBackwardMode": "model_lm", "txFirHd": np.array([1.0, 0.1])},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    config_from_mat({"configDPD": values})

    def test_complex_fir_and_zero_ridge_are_preserved(self):
        taps = np.array([1.0 + 0.2j, -0.1 + 0.3j])
        config = config_from_mat(
            {"configDPD": {"txFirHd": taps, "PAModelRidge": 0.0}}
        )
        np.testing.assert_array_equal(config.tx_fir, taps)
        self.assertEqual(config.pa_model_ridge, 0.0)

    def test_algorithm_fingerprint_changes_only_with_algorithm_settings(self):
        first = config_from_mat({"configDPD": {"LearningRate": 0.2, "BW": 20.0}})
        metric_only = config_from_mat({"configDPD": {"LearningRate": 0.2, "BW": 40.0}})
        changed = config_from_mat({"configDPD": {"LearningRate": 0.3, "BW": 20.0}})
        self.assertEqual(
            algorithm_config_fingerprint(first),
            algorithm_config_fingerprint(metric_only),
        )
        self.assertNotEqual(
            algorithm_config_fingerprint(first),
            algorithm_config_fingerprint(changed),
        )


class FeedbackBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        samples = np.arange(1280)
        self.reference = (
            0.6 * np.exp(2j * np.pi * samples / 31)
            + 0.15 * np.exp(2j * np.pi * samples / 43)
        )
        save_mat(
            self.directory / "Config_file.mat",
            {"configDPD": {"LearningRate": 0.2, "BW": 20.0}},
        )
        self.service = RemoteDPDService(
            self.directory,
            supplier_name="test",
            engine_name="linear_ilc",
            options=ServiceOptions(stable_seconds=0.0),
        )
        self.service.process_file(self.directory / "Config_file.mat")

    def tearDown(self):
        self.temporary.cleanup()

    def _write_input(self, session: str = "session-a") -> str:
        save_mat(
            self.directory / "DPD_in.mat",
            {"DPD_In_cut": self.reference, "session_id": session},
        )
        self.service.process_file(self.directory / "DPD_in.mat")
        acknowledgement = load_mat(self.directory / "ACK_DPDin.mat")
        return str(np.asarray(acknowledgement["DPDInputID"]).reshape(-1)[0])

    def test_verified_binding_is_recorded_and_output_can_be_echoed(self):
        input_id = self._write_input()
        save_mat(
            self.directory / "FB_Signal.mat",
            {
                "FB_Signal_cut": 0.9 * self.reference,
                "session_id": "session-a",
                "iteration": 1,
                "DPDInputID": input_id,
            },
        )
        self.service.process_file(self.directory / "FB_Signal.mat")
        self.assertTrue(self.service.state.feedback_binding_verified)
        self.assertTrue(self.service.state.last_metrics["feedback_binding_verified"])
        output = load_mat(self.directory / "DPDout_Nokia.mat")
        self.assertEqual(str(output["session_id"]), "session-a")
        self.assertEqual(str(output["DPDOutputID"]), self.service.state.last_output_id)
        self.assertEqual(int(output["nextFeedbackIteration"]), 2)

    def test_stale_iteration_and_wrong_session_are_rejected(self):
        input_id = self._write_input()
        save_mat(
            self.directory / "FB_Signal.mat",
            {
                "FB_Signal_cut": 0.9 * self.reference,
                "session_id": "session-a",
                "iteration": 1,
                "DPDInputID": input_id,
            },
        )
        self.service.process_file(self.directory / "FB_Signal.mat")
        output_id = self.service.state.last_output_id

        save_mat(
            self.directory / "FB_Signal.mat",
            {
                "FB_Signal_cut": 0.85 * self.reference,
                "session_id": "session-a",
                "iteration": 1,
                "DPDOutputID": output_id,
            },
        )
        with self.assertRaisesRegex(MatProtocolError, "does not match expected 2"):
            self.service.process_file(self.directory / "FB_Signal.mat")

        save_mat(
            self.directory / "FB_Signal.mat",
            {
                "FB_Signal_cut": 0.85 * self.reference,
                "session_id": "session-b",
                "iteration": 2,
                "DPDOutputID": output_id,
            },
        )
        with self.assertRaisesRegex(MatProtocolError, "does not match"):
            self.service.process_file(self.directory / "FB_Signal.mat")

    def test_unbound_legacy_feedback_remains_compatible_and_observable(self):
        self._write_input()
        save_mat(
            self.directory / "FB_Signal.mat",
            {"FB_Signal_cut": self.reference},
        )
        self.service.process_file(self.directory / "FB_Signal.mat")
        self.assertFalse(self.service.state.feedback_binding_verified)
        self.assertFalse(self.service.state.last_metrics["feedback_binding_verified"])

    def test_algorithm_config_change_resets_nonlegacy_session(self):
        self._write_input()
        self.assertIsNotNone(self.service.state.reference)
        save_mat(
            self.directory / "Config_file.mat",
            {"configDPD": {"LearningRate": 0.3, "BW": 20.0}},
        )
        self.service.process_file(self.directory / "Config_file.mat")
        self.assertIsNone(self.service.state.reference)
        self.assertIsNotNone(self.service.state.last_config_id)


if __name__ == "__main__":
    unittest.main()
