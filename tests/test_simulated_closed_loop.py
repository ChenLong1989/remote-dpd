import json
import unittest

import numpy as np

from remote_dpd import (
    ClosedLoopConfig,
    ClosedLoopController,
    ControllerState,
    DeviceConfig,
    create_rf_bench,
)
from remote_dpd.power_control import POWER_TOLERANCE_DB, PowerController


class SimulatedClosedLoopTests(unittest.TestCase):
    def test_automatic_loop_tunes_power_splits_captures_and_improves_nmse(self):
        sample_count = 128
        samples = np.arange(sample_count)
        x = (
            0.30 * np.exp(2j * np.pi * 3 * samples / sample_count)
            + 0.18 * np.exp(2j * np.pi * 9 * samples / sample_count)
            + 0.08 * np.exp(-2j * np.pi * 13 * samples / sample_count)
        )
        config = ClosedLoopConfig(
            device_config=DeviceConfig(
                sample_rate_hz=245.76e6,
                average_segment_count=5,
                target_power_dbm=-15.0,
                safety_power_limit_dbm=0.0,
                initial_attenuation_db=30.0,
                min_attenuation_db=0.0,
                max_attenuation_db=60.0,
                settle_seconds=0.0,
                max_adjustments=100,
                call_timeout_seconds=1.0,
                device_options={
                    "max_capture_samples": sample_count * 2,
                    "noise_dbfs": -100.0,
                    "random_seed": 7,
                },
            ),
            runtime_config={"mu": 0.35},
            max_iterations=5,
            seed_noise_enabled=False,
        )
        bench = create_rf_bench("simulated")
        controller = ClosedLoopController(
            bench,
            power_controller=PowerController(sleep_fn=lambda _: None),
        )

        controller.connect()
        controller.apply_config(config)
        controller.load_reference(x)
        result = controller.run_auto()

        self.assertEqual(result.state, ControllerState.COMPLETED)
        self.assertIn("pa_coefficients", result.config.device_config.device_options)
        self.assertIn("system_gain_db", result.config.device_config.device_options)
        json.dumps(result.config.to_dict(), allow_nan=False)
        self.assertEqual(len(result.records), config.max_iterations + 1)
        self.assertEqual(result.current_record.iteration, config.max_iterations)
        self.assertFalse(result.transmitting)
        self.assertGreater(len(result.power_trace), 1)
        self.assertGreaterEqual(result.power_trace[-1].gap_db, 0.0)
        self.assertLessEqual(result.power_trace[-1].gap_db, POWER_TOLERANCE_DB)
        self.assertTrue(
            all(
                record.attenuation_db == result.locked_attenuation_db
                for record in result.records
            )
        )
        self.assertTrue(
            all(
                len(record.preprocessing.batch_diagnostics) == 3
                for record in result.records
            )
        )
        self.assertLess(
            result.records[-1].preprocessing.nmse_db,
            result.records[0].preprocessing.nmse_db - 10.0,
        )
        self.assertTrue(result.records[-1].digital_safety.passed)


if __name__ == "__main__":
    unittest.main()
