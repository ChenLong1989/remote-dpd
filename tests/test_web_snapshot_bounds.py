import json
import unittest

import numpy as np

from remote_dpd.controller import (
    ClosedLoopConfig,
    ControllerSnapshot,
    ControllerState,
    IterationRecord,
)
from remote_dpd.device import DeviceConfig
from remote_dpd.preprocessing import (
    BatchDiagnostic,
    PreprocessingResult,
    SegmentDiagnostic,
)
from remote_dpd.safety import DigitalSafetyReport
from remote_dpd.web_bridge import (
    _bounded_iteration_metadata,
    _bounded_json_document,
    snapshot_payload,
)


class WebSnapshotBoundsTests(unittest.TestCase):
    def test_run_detail_documents_have_a_shared_recursive_budget(self):
        config, config_truncated = _bounded_json_document(
            {
                "runtime_config": {
                    "large": {
                        "$type": "ndarray",
                        "data": list(range(200_000)),
                    }
                }
            },
            max_depth=16,
            max_nodes=4_096,
            max_items=512,
            max_string=1_024,
        )
        events, events_truncated = _bounded_json_document(
            [
                {"kind": "metric", "details": {"values": list(range(10_000))}}
                for _ in range(100)
            ],
            max_depth=12,
            max_nodes=10_000,
            max_items=1_000,
            max_string=512,
        )

        self.assertTrue(config_truncated)
        self.assertTrue(events_truncated)
        self.assertLessEqual(
            len(config["runtime_config"]["large"]["data"]),
            512,
        )
        self.assertLess(len(json.dumps({"config": config, "events": events})), 500_000)

    def test_iteration_metadata_recursively_bounds_runtime_metrics(self):
        payload = _bounded_iteration_metadata(
            {
                "iteration": 1,
                "preprocessing": {"batch_diagnostics": []},
                "runtime_metrics": {"large": list(range(200_000))},
            }
        )

        self.assertTrue(payload["metadata_truncated"])
        self.assertLessEqual(len(payload["runtime_metrics"]["large"]), 16)
        self.assertLess(len(json.dumps(payload)), 20_000)

    def test_maximum_web_history_keeps_only_latest_bounded_diagnostics(self):
        snapshot = _maximum_history_snapshot()

        payload = snapshot_payload(snapshot)
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

        self.assertEqual(payload["record_count"], 1_001)
        self.assertEqual(len(payload["records"]), 256)
        for historical in payload["records"][:-1]:
            self.assertFalse(historical["diagnostics_included"])
            self.assertNotIn("batches", historical)
            self.assertLess(len(json.dumps(historical)), 2_000)
        bounded_history_metrics = payload["records"][-2]
        self.assertTrue(bounded_history_metrics["runtime_metrics_truncated"])
        self.assertIsNone(bounded_history_metrics["runtime_metrics"]["large_array"])
        self.assertLessEqual(
            len(bounded_history_metrics["runtime_metrics"]["long_text"]),
            64,
        )

        latest = payload["records"][-1]
        self.assertTrue(latest["diagnostics_included"])
        self.assertEqual(len(latest["batches"]), 8)
        self.assertEqual(latest["batches_truncated"], 8)
        for batch in latest["batches"]:
            self.assertEqual(len(batch["segments"]), 8)
            self.assertEqual(batch["segments_truncated"], 8)

        metrics = latest["runtime_metrics"]
        self.assertTrue(latest["runtime_metrics_truncated"])
        self.assertEqual(metrics["large_array"]["$type"], "ndarray")
        self.assertLessEqual(len(metrics["large_array"]["values"]), 16)
        self.assertGreater(metrics["large_array"]["truncated_items"], 0)
        self.assertLessEqual(len(metrics["long_text"]), 128)
        self.assertLessEqual(len(metrics["wide"]), 16)
        self.assertLessEqual(_maximum_json_depth(metrics), 7)
        self.assertLessEqual(_json_node_count(metrics), 64)
        self.assertLess(len(json.dumps(metrics)), 20_000)
        self.assertLess(len(encoded), 500_000)


def _maximum_history_snapshot() -> ControllerSnapshot:
    x = np.asarray(
        [0.20 + 0.01j, -0.10 + 0.03j, 0.08 - 0.04j, -0.03 - 0.02j],
        dtype=np.complex128,
    )
    segments = tuple(
        SegmentDiagnostic(
            segment_index=index,
            alignment_estimated=index == 0,
            delay_samples=2.25,
            phase_correction=np.exp(-0.35j),
            phase_radians=-0.35,
            input_rms=0.15,
            aligned_rms=0.15,
            aligned_nmse_db=-30.0,
        )
        for index in range(16)
    )
    batches = tuple(
        BatchDiagnostic(
            batch_index=index,
            coherent_within_batch=True,
            input_rms=0.15,
            aligned_average_rms=0.15,
            aligned_average_nmse_db=-30.0,
            segments=segments,
        )
        for index in range(16)
    )
    preprocessing = PreprocessingResult(
        z=x,
        aligned_average=x,
        gain_correction=1.0,
        gain_correction_db=0.0,
        reference_rms=0.15,
        aligned_average_rms=0.15,
        z_rms=0.15,
        aligned_average_nmse_db=-30.0,
        nmse_db=-35.0,
        segment_count=256,
        batch_diagnostics=batches,
    )
    safety = DigitalSafetyReport(
        signal_role="candidate",
        passed=True,
        reference_samples=x.size,
        reference_peak=0.21,
        reference_rms=0.15,
        candidate_samples=x.size,
        candidate_peak=0.21,
        candidate_rms=0.15,
        peak_limit=1.0,
        candidate_rms_limit=0.19,
        violations=(),
    )
    records = []
    for iteration in range(1_001):
        runtime_metrics = {
            "iteration": iteration,
            "mu": 0.5,
            "error_rms": 0.01,
            "candidate_rms": 0.15,
        }
        if iteration in {999, 1_000}:
            nested = {"leaf": "done"}
            for depth in range(20):
                nested = {f"depth_{depth}": nested}
            runtime_metrics = {
                "large_array": np.arange(100_000, dtype=np.float64),
                "long_text": "x" * 10_000,
                "deep": nested,
                "wide": {f"metric_{index}": index for index in range(100)},
            }
        records.append(
            IterationRecord(
                iteration=iteration,
                y=x,
                z=x,
                power_dbm=-10.0,
                attenuation_db=20.0,
                digital_safety=safety,
                preprocessing=preprocessing,
                runtime_metrics=runtime_metrics,
            )
        )
    config = ClosedLoopConfig(
        device_config=DeviceConfig(
            average_segment_count=256,
            settle_seconds=0.0,
        ),
        runtime_config={"mu": 0.5},
        max_iterations=1_000,
    )
    return ControllerSnapshot(
        state=ControllerState.COMPLETED,
        connected=True,
        configured=True,
        reference_loaded=True,
        transmitting=False,
        stop_requested=False,
        active_operation=None,
        iteration=1_000,
        max_iterations=1_000,
        gain_correction=1.0,
        locked_attenuation_db=20.0,
        latest_power_dbm=-10.0,
        config=config,
        device_type="simulated",
        completed_at="2026-08-28T00:00:00.000000Z",
        reference_safety=safety,
        x=x,
        records=tuple(records),
    )


def _maximum_json_depth(value):
    if isinstance(value, dict):
        return 1 + max(
            (_maximum_json_depth(item) for item in value.values()), default=0
        )
    if isinstance(value, list):
        return 1 + max((_maximum_json_depth(item) for item in value), default=0)
    return 1


def _json_node_count(value):
    if isinstance(value, dict):
        return 1 + sum(_json_node_count(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_json_node_count(item) for item in value)
    return 1


if __name__ == "__main__":
    unittest.main()
