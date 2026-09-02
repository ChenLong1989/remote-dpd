"""Quick-start default-configuration tests that do not build the web app.

These exercise the ``RFBench.quick_start_configuration()`` contract through
``web_bridge._web_default_configuration`` without constructing the waveform
repository, so they also run on Windows hosts where the anchored-directory
waveform root cannot be opened.
"""

import unittest

from remote_dpd.device import create_rf_bench
from remote_dpd.real_bench import Vst5842RFBench
from remote_dpd.simulation import SimulatedRFBench
from remote_dpd.web_bridge import _web_default_configuration


class WebQuickStartConfigurationTests(unittest.TestCase):
    def test_simulated_profile_keeps_historical_web_defaults(self):
        bench = SimulatedRFBench()
        configuration = _web_default_configuration(
            "simulated", bench, bench.parameter_schema
        )
        common = configuration["device_config"]
        self.assertEqual(configuration["device_type"], "simulated")
        self.assertEqual(common["sample_rate_hz"], 491.52e6)
        self.assertEqual(common["target_power_dbm"], -15.0)
        self.assertEqual(common["average_segment_count"], 10)
        self.assertEqual(
            common["device_options"]["max_capture_samples"], 10_000_000
        )
        self.assertTrue(configuration["normalize_reference_rms"])
        self.assertEqual(configuration["reference_target_rms_dbfs"], -15.0)
        self.assertEqual(configuration["runtime_config"], {"mu": 0.35})
        self.assertEqual(configuration["max_iterations"], 15)

    def test_vst5842_profile_matches_smoke_verified_operating_point(self):
        bench = Vst5842RFBench()
        configuration = _web_default_configuration(
            "vst5842", bench, bench.parameter_schema
        )
        common = configuration["device_config"]
        self.assertEqual(configuration["device_type"], "vst5842")
        self.assertEqual(common["center_frequency_hz"], 1.84e9)
        self.assertEqual(common["sample_rate_hz"], 491.52e6)
        self.assertEqual(common["average_segment_count"], 8)
        self.assertEqual(common["target_power_dbm"], 38.0)
        self.assertEqual(common["safety_power_limit_dbm"], 39.0)
        self.assertEqual(common["initial_attenuation_db"], 22.0)
        self.assertEqual(common["min_attenuation_db"], 0.0)
        self.assertEqual(common["max_attenuation_db"], 40.0)
        self.assertEqual(common["settle_seconds"], 0.5)
        self.assertEqual(common["call_timeout_seconds"], 90.0)
        self.assertEqual(
            common["device_options"]["scpi_resource"],
            "TCPIP0::127.0.0.1::inst0::INSTR",
        )
        self.assertFalse(common["device_options"]["enable_supply_shutdown"])
        self.assertEqual(common["device_options"]["power_meter_average"], 8)
        self.assertEqual(configuration["max_iterations"], 3)
        self.assertEqual(configuration["runtime_config"], {"mu": 0.1})
        self.assertTrue(configuration["normalize_reference_rms"])
        self.assertEqual(configuration["reference_target_rms_dbfs"], -15.0)

    def test_invalid_or_missing_profiles_fall_back_to_generic_defaults(self):
        base = create_rf_bench("vst5842")
        schema = base.parameter_schema

        class _NoProfileBench(Vst5842RFBench):
            def quick_start_configuration(self):
                return None

        class _BrokenProfileBench(Vst5842RFBench):
            def quick_start_configuration(self):
                return {"device_type": "elsewhere", "device_config": "not a map"}

        for bench_type in (_NoProfileBench, _BrokenProfileBench):
            with self.subTest(bench=bench_type.__name__):
                fallback = _web_default_configuration(
                    "vst5842", bench_type(), schema
                )
                self.assertEqual(
                    fallback["device_config"]["target_power_dbm"], -15.0
                )
                self.assertEqual(fallback["device_config"]["sample_rate_hz"], 983.04e6)
                self.assertEqual(fallback["device_config"]["call_timeout_seconds"], 10.0)
                self.assertEqual(fallback["max_iterations"], 10)
                self.assertEqual(fallback["runtime_config"], {"mu": 0.5})


if __name__ == "__main__":
    unittest.main()
