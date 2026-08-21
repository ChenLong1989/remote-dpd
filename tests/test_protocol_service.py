import tempfile
import unittest
from pathlib import Path

import numpy as np

from remote_dpd.protocol import load_mat, save_mat
from remote_dpd.service import RemoteDPDService, ServiceOptions


class ProtocolServiceTests(unittest.TestCase):
    def test_mat_struct_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Config_file.mat"
            save_mat(path, {"configDPD": {"Reset": np.asarray(1), "LearningRate": np.asarray(0.3)}})
            payload = load_mat(path)
            self.assertIn("configDPD", payload)
            self.assertEqual(int(np.asarray(payload["configDPD"]["Reset"]).reshape(-1)[0]), 1)

    def test_ilc_file_exchange(self):
        # Importing the engine is intentionally skipped in environments where
        # the project dependencies have not yet been installed.
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            n = 256
            t = np.arange(n)
            reference = (0.5 * np.exp(2j * np.pi * t / 31) + 0.1 * np.exp(2j * np.pi * t / 17)).astype(np.complex128)
            save_mat(directory / "Config_file.mat", {"configDPD": {"Reset": np.asarray(0), "LearningRate": np.asarray(0.2), "BW": np.asarray(20.0)}})
            save_mat(directory / "DPD_in.mat", {"DPD_In_cut": reference.reshape(-1, 1)})
            save_mat(directory / "FB_Signal.mat", {"FB_Signal_cut": reference.reshape(-1, 1)})
            service = RemoteDPDService(directory, supplier_name="test", options=ServiceOptions(stable_seconds=0.0))
            service.process_file(directory / "Config_file.mat")
            service.process_file(directory / "DPD_in.mat")
            service.process_file(directory / "FB_Signal.mat")
            output = load_mat(directory / "DPDout_Nokia.mat")
            self.assertIn("DPDout_Nokia", output)
            self.assertEqual(np.asarray(output["DPDout_Nokia"]).size, n)
            self.assertIn("symbolEVM", load_mat(directory / "symbolEVM.mat"))
            # Replaying the same capture must not advance ILC state.
            iteration = service.state.iteration
            service.process_file(directory / "FB_Signal.mat")
            self.assertEqual(service.state.iteration, iteration)


if __name__ == "__main__":
    unittest.main()
