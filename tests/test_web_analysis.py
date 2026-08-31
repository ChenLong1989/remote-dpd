import json
import math
import threading
import unittest
from unittest import mock

import numpy as np

from remote_dpd.web_analysis import (
    AnalysisRecord,
    AnalysisRequest,
    RFAnalysisEngine,
    WebAnalysisError,
    analyze_waveforms,
)


def _tone(sample_count, bin_index, amplitude=1.0):
    samples = np.arange(sample_count)
    return amplitude * np.exp(2j * np.pi * bin_index * samples / sample_count)


def _request(**overrides):
    payload = {
        "schema_version": 1,
        "points": 256,
        "frequency_mode": "relative",
        "traces": ["baseline_z", "target_z", "reference_x", "target_y"],
        "bands": [],
    }
    payload.update(overrides)
    return AnalysisRequest.from_payload(payload)


class WebAnalysisTests(unittest.TestCase):
    def test_unit_complex_tone_is_zero_dbfs_per_bin(self):
        sample_count = 1024
        sample_rate = 1024.0
        x = _tone(sample_count, 100)
        result = analyze_waveforms(
            request=_request(traces=["reference_x"]),
            reference=x,
            baseline=None,
            target=None,
            sample_rate_hz=sample_rate,
            center_frequency_hz=3.5e9,
        )

        trace = result["traces"][0]
        self.assertEqual(result["fft_size"], sample_count)
        self.assertEqual(result["bin_width_hz"], 1.0)
        self.assertAlmostEqual(max(trace["values_dbfs"]), 0.0, places=10)
        self.assertAlmostEqual(trace["papr_db"], 0.0, places=10)
        self.assertEqual(trace["label"], "REFERENCE · X")

    def test_measurement_bands_report_dbfs_dbc_and_positive_aclr(self):
        sample_count = 1024
        sample_rate = 1024.0
        main = _tone(sample_count, 50)
        baseline_z = main + _tone(sample_count, 150, 0.1)
        target_z = main + _tone(sample_count, 150, 0.01)
        baseline = AnalysisRecord(0, main, baseline_z, power_dbm=-10.0)
        target = AnalysisRecord(3, main, target_z, power_dbm=-9.9)
        request = _request(
            traces=["baseline_z", "target_z"],
            bands=[
                {
                    "label": "Main",
                    "role": "main",
                    "center_offset_hz": 50.0,
                    "integration_bandwidth_hz": 1.0,
                    "enabled": True,
                },
                {
                    "label": "Adjacent R1",
                    "role": "adjacent",
                    "center_offset_hz": 150.0,
                    "integration_bandwidth_hz": 1.0,
                    "enabled": True,
                },
            ],
        )

        result = analyze_waveforms(
            request=request,
            reference=main,
            baseline=baseline,
            target=target,
            sample_rate_hz=sample_rate,
            center_frequency_hz=3.5e9,
        )

        adjacent = result["bands"][1]["traces"]
        self.assertAlmostEqual(adjacent["baseline_z"]["power_dbfs"], -20.0, places=8)
        self.assertAlmostEqual(
            adjacent["baseline_z"]["relative_power_dbc"], -20.0, places=8
        )
        self.assertAlmostEqual(adjacent["baseline_z"]["aclr_db"], 20.0, places=8)
        self.assertAlmostEqual(adjacent["target_z"]["aclr_db"], 40.0, places=8)
        self.assertTrue(result["aclr_available"])

    def test_target_error_trace_uses_complete_aligned_waveform_difference(self):
        sample_count = 1024
        reference = _tone(sample_count, 50)
        error = _tone(sample_count, 150, 0.1)
        target = AnalysisRecord(7, reference, reference + error)

        result = analyze_waveforms(
            request=_request(traces=["target_error"]),
            reference=reference,
            baseline=None,
            target=target,
            sample_rate_hz=1024.0,
            center_frequency_hz=3.5e9,
        )

        trace = result["traces"][0]
        self.assertEqual(trace["key"], "target_error")
        self.assertEqual(trace["label"], "ERROR · Z7 − X")
        self.assertEqual(trace["iteration"], 7)
        self.assertAlmostEqual(max(trace["values_dbfs"]), -20.0, places=8)

    def test_multicarrier_aclr_uses_each_adjacent_reference_tx(self):
        sample_count = 1024
        left_main = _tone(sample_count, -100)
        right_main = _tone(sample_count, 100, 0.5)
        left_adjacent = _tone(sample_count, -200, 0.1)
        right_adjacent = _tone(sample_count, 200, 0.05)
        z = left_main + right_main + left_adjacent + right_adjacent
        record = AnalysisRecord(0, left_main + right_main, z)
        request = _request(
            traces=["baseline_z"],
            bands=[
                {
                    "label": "TX1",
                    "role": "main",
                    "center_offset_hz": -100.0,
                    "integration_bandwidth_hz": 1.0,
                    "enabled": True,
                },
                {
                    "label": "TX2",
                    "role": "main",
                    "center_offset_hz": 100.0,
                    "integration_bandwidth_hz": 1.0,
                    "enabled": True,
                },
                {
                    "label": "Adjacent L1",
                    "role": "adjacent",
                    "center_offset_hz": -200.0,
                    "integration_bandwidth_hz": 1.0,
                    "enabled": True,
                    "reference_label": "TX1",
                },
                {
                    "label": "Adjacent R1",
                    "role": "adjacent",
                    "center_offset_hz": 200.0,
                    "integration_bandwidth_hz": 1.0,
                    "enabled": True,
                    "reference_label": "TX2",
                },
            ],
        )

        result = analyze_waveforms(
            request=request,
            reference=left_main + right_main,
            baseline=record,
            target=None,
            sample_rate_hz=1024.0,
            center_frequency_hz=3.5e9,
        )

        left = result["bands"][2]
        right = result["bands"][3]
        self.assertEqual(left["resolved_reference_label"], "TX1")
        self.assertEqual(right["resolved_reference_label"], "TX2")
        self.assertAlmostEqual(
            left["traces"]["baseline_z"]["relative_power_dbc"], -20.0, places=8
        )
        self.assertAlmostEqual(
            right["traces"]["baseline_z"]["relative_power_dbc"],
            -20.0,
            places=8,
        )

    def test_partial_bin_overlap_is_integrated_proportionally(self):
        sample_count = 1024
        x = _tone(sample_count, 0)
        record = AnalysisRecord(0, x, x)
        result = analyze_waveforms(
            request=_request(
                traces=["baseline_z"],
                bands=[
                    {
                        "label": "Half bin",
                        "role": "main",
                        "center_offset_hz": 0.25,
                        "integration_bandwidth_hz": 0.5,
                        "enabled": True,
                    }
                ],
            ),
            reference=x,
            baseline=record,
            target=None,
            sample_rate_hz=1024.0,
            center_frequency_hz=3.5e9,
        )

        power = result["bands"][0]["traces"]["baseline_z"]["power_dbfs"]
        self.assertAlmostEqual(power, 10.0 * math.log10(0.5), places=8)

    def test_full_nyquist_band_preserves_parseval_power(self):
        sample_count = 1024
        nyquist_tone = _tone(sample_count, sample_count // 2)
        record = AnalysisRecord(0, nyquist_tone, nyquist_tone)
        result = analyze_waveforms(
            request=_request(
                traces=["baseline_z"],
                bands=[
                    {
                        "label": "Full span",
                        "role": "main",
                        "center_offset_hz": 0.0,
                        "integration_bandwidth_hz": 1024.0,
                        "enabled": True,
                    }
                ],
            ),
            reference=nyquist_tone,
            baseline=record,
            target=None,
            sample_rate_hz=1024.0,
            center_frequency_hz=3.5e9,
        )

        power = result["bands"][0]["traces"]["baseline_z"]["power_dbfs"]
        self.assertAlmostEqual(power, 0.0, places=8)

    def test_large_stimulus_response_is_deterministically_bounded(self):
        sample_count = 300_000
        amplitude = np.linspace(0.01, 0.9, sample_count)
        y = amplitude.astype(np.complex128)
        record = AnalysisRecord(0, y, 0.9 * y)
        result = analyze_waveforms(
            request=_request(traces=["baseline_z"]),
            reference=y,
            baseline=record,
            target=None,
            sample_rate_hz=10e6,
            center_frequency_hz=3.5e9,
        )

        response = result["stimulus_response"]
        self.assertTrue(response["sampled"])
        self.assertEqual(response["input_sample_count"], sample_count)
        self.assertEqual(response["analyzed_sample_count"], 262_144)

    def test_absolute_frequency_adds_center_frequency(self):
        x = _tone(512, 0)
        relative = analyze_waveforms(
            request=_request(traces=["reference_x"], frequency_mode="relative"),
            reference=x,
            baseline=None,
            target=None,
            sample_rate_hz=512.0,
            center_frequency_hz=3.5e9,
        )
        absolute = analyze_waveforms(
            request=_request(traces=["reference_x"], frequency_mode="absolute"),
            reference=x,
            baseline=None,
            target=None,
            sample_rate_hz=512.0,
            center_frequency_hz=3.5e9,
        )

        delta = np.asarray(absolute["frequency_hz"]) - np.asarray(
            relative["frequency_hz"]
        )
        np.testing.assert_allclose(delta, 3.5e9)

    def test_stimulus_response_bins_amplitude_gain_and_phase(self):
        sample_count = 2048
        amplitude = np.linspace(0.01, 0.9, sample_count)
        phase = np.linspace(0.0, 6.0 * np.pi, sample_count)
        y = amplitude * np.exp(1j * phase)
        z = 0.8 * y * np.exp(1j * math.radians(15.0))
        record = AnalysisRecord(0, y, z)
        result = analyze_waveforms(
            request=_request(traces=["baseline_z"]),
            reference=y,
            baseline=record,
            target=None,
            sample_rate_hz=10e6,
            center_frequency_hz=3.5e9,
        )

        points = result["stimulus_response"]["baseline"]["points"]
        self.assertGreater(len(points), 40)
        middle = points[len(points) // 2]
        self.assertAlmostEqual(middle["gain_db"], 20.0 * math.log10(0.8), places=8)
        self.assertAlmostEqual(middle["phase_degrees"], 15.0, places=8)

    def test_profile_and_band_validation_is_strict(self):
        five_traces = [
            "baseline_z",
            "target_z",
            "target_error",
            "reference_x",
            "target_y",
        ]
        self.assertEqual(
            AnalysisRequest.from_payload({"traces": five_traces}).traces,
            tuple(five_traces),
        )
        with self.assertRaisesRegex(WebAnalysisError, "at most 5 traces"):
            AnalysisRequest.from_payload({"traces": [*five_traces, "baseline_z"]})
        invalid_payloads = [
            {"traces": ["unknown"]},
            {"traces": ["reference_x", "reference_x"]},
            {
                "bands": [
                    {
                        "label": "Main",
                        "role": "main",
                        "center_offset_hz": 0.0,
                        "integration_bandwidth_hz": 10.0,
                        "reference_label": "Main",
                    }
                ]
            },
            {
                "bands": [
                    {
                        "label": "Main",
                        "role": "main",
                        "center_offset_hz": 0.0,
                        "integration_bandwidth_hz": -1.0,
                    }
                ]
            },
            {"server_path": "/tmp/data"},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(WebAnalysisError):
                AnalysisRequest.from_payload(payload)

        request = _request(
            bands=[
                {
                    "label": "Main 1",
                    "role": "main",
                    "center_offset_hz": 0.0,
                    "integration_bandwidth_hz": 10.0,
                },
                {
                    "label": "Main 2",
                    "role": "main",
                    "center_offset_hz": 20.0,
                    "integration_bandwidth_hz": 10.0,
                },
                {
                    "label": "Adjacent",
                    "role": "adjacent",
                    "center_offset_hz": 40.0,
                    "integration_bandwidth_hz": 10.0,
                },
            ]
        )
        x = _tone(512, 0)
        with self.assertRaisesRegex(WebAnalysisError, "requires reference_label"):
            analyze_waveforms(
                request=request,
                reference=x,
                baseline=None,
                target=None,
                sample_rate_hz=512.0,
                center_frequency_hz=3.5e9,
            )

        unknown_reference = _request(
            traces=["reference_x"],
            bands=[
                {
                    "label": "Main",
                    "role": "main",
                    "center_offset_hz": 0.0,
                    "integration_bandwidth_hz": 10.0,
                },
                {
                    "label": "Adjacent",
                    "role": "adjacent",
                    "center_offset_hz": 20.0,
                    "integration_bandwidth_hz": 10.0,
                    "reference_label": "Missing",
                },
            ],
        )
        with self.assertRaisesRegex(WebAnalysisError, "unknown enabled main"):
            analyze_waveforms(
                request=unknown_reference,
                reference=x,
                baseline=None,
                target=None,
                sample_rate_hz=512.0,
                center_frequency_hz=3.5e9,
            )

    def test_result_and_cache_are_json_finite_and_detached(self):
        x = _tone(512, 10, 0.5)
        record = AnalysisRecord(0, x, x)
        engine = RFAnalysisEngine(max_cache_entries=2, max_cache_bytes=1024 * 1024)
        request = _request(traces=["baseline_z"])
        first = engine.analyze(
            source_key=("test", 1),
            request=request,
            reference=x,
            baseline=record,
            target=None,
            sample_rate_hz=512.0,
            center_frequency_hz=3.5e9,
        )
        first["traces"][0]["label"] = "mutated"
        second = engine.analyze(
            source_key=("test", 1),
            request=request,
            reference=x,
            baseline=record,
            target=None,
            sample_rate_hz=512.0,
            center_frequency_hz=3.5e9,
        )

        self.assertNotEqual(second["traces"][0]["label"], "mutated")
        json.dumps(second, allow_nan=False)

    def test_engine_rejects_concurrent_analysis_without_queueing(self):
        x = _tone(512, 10, 0.5)
        record = AnalysisRecord(0, x, x)
        request = _request(traces=["baseline_z"])
        engine = RFAnalysisEngine()
        entered = threading.Event()
        release = threading.Event()
        completed = threading.Event()

        def blocking_analysis(**_kwargs):
            entered.set()
            if not release.wait(2.0):
                raise TimeoutError("analysis test gate timed out")
            return {"schema_version": 1, "result": "complete"}

        def first_call():
            try:
                engine.analyze(
                    source_key=("concurrent", 1),
                    request=request,
                    reference=x,
                    baseline=record,
                    target=None,
                    sample_rate_hz=512.0,
                    center_frequency_hz=3.5e9,
                )
            finally:
                completed.set()

        with mock.patch(
            "remote_dpd.web_analysis.analyze_waveforms", side_effect=blocking_analysis
        ):
            worker = threading.Thread(target=first_call)
            worker.start()
            self.assertTrue(entered.wait(1.0))
            with self.assertRaises(WebAnalysisError) as caught:
                engine.analyze(
                    source_key=("concurrent", 2),
                    request=request,
                    reference=x,
                    baseline=record,
                    target=None,
                    sample_rate_hz=512.0,
                    center_frequency_hz=3.5e9,
                )
            self.assertEqual(caught.exception.code, "analysis_busy")
            self.assertEqual(caught.exception.status_code, 429)
            release.set()
            self.assertTrue(completed.wait(1.0))
            worker.join(1.0)


if __name__ == "__main__":
    unittest.main()
