"""Tests for deterministic experiment enumeration, recovery, and statistics."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import experiments.runner as runner_module

from experiments.config import (
    CORE_METHODS,
    PILOT_CANDIDATES,
    ExperimentProtocol,
    ResourceLimits,
    TrajectorySpec,
    canonical_json,
    enumerate_study,
    expected_study_counts,
    matrix_hash,
    resolved_method_parameters,
    stable_hash,
)
from experiments.runner import (
    CapacityGateError,
    CorruptArtifactError,
    ExperimentRunner,
    HashMismatchError,
    atomic_write_json,
    directory_size,
    load_completed_records,
    read_verified_shard,
    write_verified_shard,
)
from experiments.statistics import (
    encode_right_censored_convergence,
    evaluate_primary_criterion,
    holm_adjust,
    paired_bootstrap,
    paired_rate_difference_bootstrap,
    paired_relative_reduction_bootstrap,
    select_pilot_candidates,
)


class ExperimentConfigurationTests(unittest.TestCase):
    def test_frozen_matrix_counts_and_unique_ids(self) -> None:
        protocol = ExperimentProtocol()
        expected = {
            "smoke": 14,
            "pilot": 288,
            "confirmatory": 2240,
            "robustness": 960,
            "mismatch": 288,
            "ablation": 192,
            "dynamic": 160,
            "stress": 168,
        }
        self.assertEqual(expected_study_counts(protocol), expected)
        for study, count in expected.items():
            first = enumerate_study(study, protocol)
            second = enumerate_study(study, protocol)
            self.assertEqual(len(first), count)
            self.assertEqual(
                [item.trajectory_id for item in first],
                [item.trajectory_id for item in second],
            )
            self.assertEqual(len({item.trajectory_id for item in first}), count)
            self.assertEqual(matrix_hash(first), matrix_hash(second))

    def test_confirmatory_pairing_uses_same_seed_indices(self) -> None:
        specs = enumerate_study("confirmatory")
        cell = [
            spec
            for spec in specs
            if spec.scenario == "amam" and spec.severity == "0.97" and spec.waveform_seed_index == 7
        ]
        self.assertEqual({spec.algorithm for spec in cell}, set(CORE_METHODS))
        self.assertEqual({spec.pa_seed_index for spec in cell}, {7})
        self.assertEqual({spec.waveform_seed_index for spec in cell}, {7})

    def test_raw_vjp_ablation_records_the_resolved_vjp_learning_rate(self) -> None:
        specs = enumerate_study(
            "ablation",
            resolved={"model_vjp_ilc": {"learning_rate": 0.3}},
        )
        raw = [spec for spec in specs if spec.parameters.get("ablation") == "raw_vjp"]
        self.assertTrue(raw)
        self.assertTrue(all(spec.parameters["learning_rate"] == 0.3 for spec in raw))

    def test_cell_configuration_changes_trajectory_id(self) -> None:
        first = TrajectorySpec("smoke", "amam", "0.97", 0, 0, "linear_ilc", {"mu": 0.1})
        second = TrajectorySpec("smoke", "amam", "0.97", 0, 0, "linear_ilc", {"mu": 0.2})
        self.assertNotEqual(first.config_hash, second.config_hash)
        self.assertNotEqual(first.trajectory_id, second.trajectory_id)
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))


class ExperimentStatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_clustered(self) -> None:
        baseline = np.array([1.0, 1.2, 0.9, 1.1])
        treatment = np.array([0.5, 0.7, 0.45, 0.55])
        clusters = ("pa0", "pa0", "pa1", "pa1")
        first = paired_relative_reduction_bootstrap(
            baseline,
            treatment,
            cluster_ids=clusters,
            resamples=500,
            seed=123,
        )
        second = paired_relative_reduction_bootstrap(
            baseline,
            treatment,
            cluster_ids=clusters,
            resamples=500,
            seed=123,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.cluster_count, 2)
        self.assertAlmostEqual(first.estimate, 0.5)

        difference = paired_bootstrap(
            baseline,
            treatment,
            cluster_ids=clusters,
            resamples=200,
            seed=123,
        )
        self.assertGreater(difference.confidence_low, 0.0)

    def test_rate_difference_holm_and_primary_criterion(self) -> None:
        rate = paired_rate_difference_bootstrap(
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            resamples=200,
            seed=4,
        )
        self.assertEqual(rate.estimate, 50.0)
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"].adjusted_p_value, 0.03)
        self.assertAlmostEqual(adjusted["b"].adjusted_p_value, 0.06)
        self.assertTrue(adjusted["a"].rejected)
        self.assertFalse(adjusted["b"].rejected)

        criterion = evaluate_primary_criterion(
            linear_auec=[1.0, 1.0, 1.0, 1.0],
            model_auec=[0.5, 0.6, 0.7, 0.5],
            linear_final_nmse_db=[-20, -21, -22, -23],
            model_final_nmse_db=[-25, -25, -27, -28],
            linear_success=[0, 0, 1, 0],
            model_success=[1, 1, 1, 0],
            linear_diverged=[0, 1, 0, 0],
            model_diverged=[0, 0, 0, 0],
            linear_constraint_violation=[0, 0, 0, 0],
            model_constraint_violation=[0, 0, 0, 0],
        )
        self.assertTrue(criterion.passed)

    def test_statistics_reject_nonfinite_pairs_instead_of_dropping_failures(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap([1.0, np.inf], [0.5, 0.7], resamples=10)

    def test_nonconvergence_is_right_censored_at_final_iteration(self) -> None:
        encoded = encode_right_censored_convergence([3, None, 12, None], final_iteration=30)
        np.testing.assert_array_equal(encoded.iterations, [3, 30, 12, 30])
        np.testing.assert_array_equal(encoded.event_observed, [True, False, True, False])

    def test_pilot_selection_excludes_safety_failure_and_uses_table_order_tie(self) -> None:
        candidates = {"linear_ilc": PILOT_CANDIDATES["linear_ilc"][:3]}
        rows = []
        for index, value in enumerate((1.0, 1.01, 0.5)):
            for seed in range(2):
                rows.append(
                    {
                        "algorithm": "linear_ilc",
                        "candidate_index": index,
                        "auec": value,
                        "safety_failure": index == 2 and seed == 0,
                    }
                )
        selected = select_pilot_candidates(rows, candidate_parameters=candidates)
        self.assertEqual(selected["linear_ilc"].candidate_index, 0)
        self.assertEqual(selected["linear_ilc"].tie_set, (0, 1))

        cheaper = select_pilot_candidates(
            rows,
            candidate_parameters=candidates,
            candidate_costs={"linear_ilc": (2.0, 1.0, 0.5)},
        )
        self.assertEqual(cheaper["linear_ilc"].candidate_index, 1)


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = ExperimentProtocol(
            nfft=64,
            symbol_count=20,
            occupied_per_side=20,
            update_count=2,
            model_orders=(1, 3, 5),
            model_memory_depth=1,
            model_block_size=16,
            confirmatory_seed_count=1,
            pilot_seed_count=1,
            robustness_seed_count=1,
            mismatch_seed_count=1,
            ablation_seed_count=1,
            dynamic_seed_count=1,
            bootstrap_resamples=20,
        )
        self.resources = ResourceLimits(
            worker_count=1,
            minimum_free_disk_bytes=1,
            artifact_budget_bytes=100_000_000,
            infrastructure_retries=0,
        )
        pilot_hashes = {
            "code_hash": "c" * 64,
            "configuration_hash": "d" * 64,
            "protocol_hash": "e" * 64,
            "matrix_hash": "f" * 64,
            "environment_hash": "1" * 64,
        }
        pilot_provenance = {
            "run_directory": "pilot",
            "manifest_sha256": "2" * 64,
            "expected_ids_sha256": "3" * 64,
            "records_hash": "4" * 64,
        }
        resolved_payload = {
            "schema_version": 1,
            "protocol_hash": self.protocol.protocol_hash,
            "pilot_hashes": pilot_hashes,
            "pilot_provenance": pilot_provenance,
            "resolved_methods": resolved_method_parameters(),
            "selection": {},
        }
        self.pilot_lock = {
            "resolved_hash": stable_hash(resolved_payload),
            "resolved_config_sha256": "b" * 64,
            "resolved_config_path": "C:/verified/resolved_config.json",
            "pilot_hashes": pilot_hashes,
            "pilot_provenance": pilot_provenance,
            "resolved_payload": resolved_payload,
        }

    def test_runner_completes_and_resumes_all_methods(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            specs = [
                spec
                for spec in enumerate_study("smoke", self.protocol)
                if spec.scenario == "amam"
            ]
            first = runner.run(
                "smoke",
                specs,
                worker_count=1,
                enforce_estimated_time_gate=False,
            )
            self.assertTrue(first.complete)
            self.assertEqual(first.newly_completed_count, len(CORE_METHODS))
            records = load_completed_records(Path(temporary) / "smoke")
            self.assertEqual(len(records), len(CORE_METHODS))
            self.assertTrue(all(record["evaluation_count"] == 3 for record in records))
            self.assertTrue(all(record["status"] == "completed" for record in records))

            second = runner.run(
                "smoke",
                specs,
                worker_count=1,
                enforce_estimated_time_gate=False,
            )
            self.assertEqual(second.newly_completed_count, 0)
            self.assertEqual(second.resumed_count, len(CORE_METHODS))
            manifest = json.loads((Path(temporary) / "smoke" / "manifest.json").read_text())
            self.assertTrue(manifest["git"]["dirty"])
            self.assertEqual(manifest["expected_trajectory_count"], len(CORE_METHODS))
            self.assertTrue((Path(temporary) / "smoke" / "seeds.csv").exists())

    def test_algorithm_failure_is_a_shard_and_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            base = next(
                spec
                for spec in enumerate_study("smoke", self.protocol)
                if spec.scenario == "amam" and spec.algorithm == "linear_ilc"
            )
            invalid = TrajectorySpec(
                study=base.study,
                scenario=base.scenario,
                severity=base.severity,
                pa_seed_index=base.pa_seed_index,
                waveform_seed_index=base.waveform_seed_index,
                algorithm=base.algorithm,
                parameters={**base.parameters, "learning_rate": -1.0},
            )
            summary = runner.run(
                "smoke",
                [invalid],
                worker_count=1,
                enforce_estimated_time_gate=False,
            )
            self.assertEqual(summary.algorithm_failure_count, 1)
            self.assertEqual(summary.infrastructure_retry_count, 0)
            record = load_completed_records(Path(temporary) / "smoke")[0]
            self.assertEqual(record["status"], "algorithm_failure")
            self.assertEqual(record["evaluation_count"], self.protocol.evaluation_count)
            self.assertEqual(len(record["metrics"]), self.protocol.evaluation_count)
            self.assertIsNotNone(record["auec"])

    def test_nonfinite_pa_output_is_a_penalized_complete_failure_record(self) -> None:
        class NonfinitePA:
            def forward(self, input_signal):
                return np.full(np.asarray(input_signal).shape, np.nan + 0.0j)

            def jvp(self, input_signal, tangent):
                return np.full(np.asarray(tangent).shape, np.nan + 0.0j)

            def vjp(self, input_signal, cotangent):
                return np.full(np.asarray(cotangent).shape, np.nan + 0.0j)

        original_make_scenario = runner_module._make_scenario

        def make_nonfinite_scenario(spec, waveform, protocol):
            original = original_make_scenario(spec, waveform, protocol)
            return runner_module.PAScenario(
                name=original.name + "_nonfinite",
                desired=original.desired,
                initial_input=original.initial_input,
                pa=NonfinitePA(),
                metadata=original.metadata,
            )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            specs = enumerate_study("smoke", self.protocol)[:1]
            with patch("experiments.runner._make_scenario", side_effect=make_nonfinite_scenario):
                summary = runner.run(
                    "smoke",
                    specs,
                    worker_count=1,
                    enforce_estimated_time_gate=False,
                )
            self.assertEqual(summary.algorithm_failure_count, 1)
            record = load_completed_records(Path(temporary) / "smoke")[0]
            self.assertTrue(record["diverged"])
            self.assertFalse(record["success"])
            self.assertEqual(record["terminal_reason"], "nonfinite_evaluation")
            self.assertEqual(record["evaluation_count"], self.protocol.evaluation_count)
            self.assertTrue(
                all(
                    metric["nmse_db"] == self.protocol.numeric_failure_nmse_db
                    for metric in record["metrics"]
                )
            )
            np.testing.assert_allclose(
                record["auec"],
                10.0 ** (self.protocol.numeric_failure_nmse_db / 10.0),
                rtol=1e-15,
                atol=0.0,
            )

    def test_complex64_ablation_records_actual_learning_dtype(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            with patch("experiments.runner._validate_pilot_lock_context"):
                runner = ExperimentRunner(
                    temporary,
                    protocol=self.protocol,
                    resources=self.resources,
                    pilot_lock=self.pilot_lock,
                )
            specs = [
                spec
                for spec in enumerate_study("ablation", self.protocol)
                if spec.scenario == "amam" and spec.parameters.get("ablation") == "complex64"
            ]
            with patch("experiments.runner._validate_pilot_lock_context"):
                runner.run(
                    "ablation",
                    specs,
                    worker_count=1,
                    enforce_estimated_time_gate=False,
                )
            record = load_completed_records(Path(temporary) / "ablation")[0]
            update_rounds = record["metrics"][:-1]
            self.assertTrue(update_rounds)
            self.assertTrue(
                all(metric["learning_numeric_dtype"] == "complex64" for metric in update_rounds)
            )

    def test_manifest_hash_mismatch_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            specs = enumerate_study("smoke", self.protocol)[:1]
            run_directory, _, _ = runner.prepare("smoke", specs)
            manifest_path = run_directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["hashes"]["protocol_hash"] = "0" * 64
            atomic_write_json(manifest_path, manifest)
            with self.assertRaises(HashMismatchError):
                runner.prepare("smoke", specs)

    def test_environment_change_blocks_prepare_and_resume(self) -> None:
        environment_a = {"python": "A", "packages": {"numpy": "1"}}
        environment_b = {"python": "B", "packages": {"numpy": "2"}}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            specs = enumerate_study("smoke", self.protocol)[:1]
            with patch("experiments.runner.environment_manifest", return_value=environment_a):
                runner.prepare("smoke", specs)
            with patch("experiments.runner.environment_manifest", return_value=environment_b):
                with self.assertRaises(HashMismatchError):
                    runner.prepare("smoke", specs)

    def test_locked_study_requires_and_records_verified_pilot_chain(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            spec = enumerate_study("ablation", self.protocol)[:1]
            without_lock = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            with self.assertRaisesRegex(ValueError, "pilot-lock provenance"):
                without_lock.prepare("ablation", spec)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            with patch("experiments.runner._validate_pilot_lock_context"):
                with_lock = ExperimentRunner(
                    temporary,
                    protocol=self.protocol,
                    resources=self.resources,
                    pilot_lock=self.pilot_lock,
                )
            with patch("experiments.runner._validate_pilot_lock_context"):
                run_directory, hashes, _ = with_lock.prepare("ablation", spec)
            manifest = json.loads((run_directory / "manifest.json").read_text())
            self.assertEqual(manifest["pilot_lock"], self.pilot_lock)
            self.assertEqual(
                hashes.configuration_hash,
                stable_hash(
                    {
                        "protocol": self.protocol.as_dict(),
                        "resolved_methods": with_lock.resolved,
                        "resolved_hash": self.pilot_lock["resolved_hash"],
                    }
                ),
            )

    def test_output_root_rejects_a_different_valid_resolved_hash(self) -> None:
        second_lock = json.loads(canonical_json(self.pilot_lock))
        second_lock["resolved_payload"]["selection"] = {"alternate": True}
        second_lock["resolved_hash"] = stable_hash(second_lock["resolved_payload"])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            with patch("experiments.runner._validate_pilot_lock_context"):
                first = ExperimentRunner(
                    temporary,
                    protocol=self.protocol,
                    resources=self.resources,
                    pilot_lock=self.pilot_lock,
                )
                first.prepare(
                    "ablation",
                    enumerate_study("ablation", self.protocol)[:1],
                )
                second = ExperimentRunner(
                    temporary,
                    protocol=self.protocol,
                    resources=self.resources,
                    pilot_lock=second_lock,
                )
                with self.assertRaisesRegex(HashMismatchError, "different pilot"):
                    second.prepare(
                        "dynamic",
                        enumerate_study("dynamic", self.protocol)[:1],
                    )

    def test_manifest_materializes_reachability_and_safety_limits(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            spec = next(
                item
                for item in enumerate_study("smoke", self.protocol)
                if item.scenario == "amam" and item.algorithm == "linear_ilc"
            )
            run_directory, _, _ = runner.prepare("smoke", [spec])
            manifest = json.loads((run_directory / "manifest.json").read_text())
            self.assertEqual(manifest["fixed_calibration"]["coefficient_real"], 1.0)
            self.assertEqual(len(manifest["cell_instances"]), 1)
            cell = manifest["cell_instances"][0]
            self.assertTrue(cell["scenario_metadata"]["reachable"])
            self.assertLessEqual(
                cell["scenario_metadata"]["required_input_peak"],
                cell["safety_limits"]["max_peak"],
            )
            self.assertEqual(
                len(cell["envelope_bin_edges"]),
                self.protocol.envelope_bin_count + 1,
            )

    def test_live_code_hash_change_stops_before_scheduling(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            specs = enumerate_study("smoke", self.protocol)[:1]
            with patch(
                "experiments.runner.compute_code_hash",
                side_effect=("a" * 64, "b" * 64),
            ):
                with self.assertRaises(HashMismatchError):
                    runner.run(
                        "smoke",
                        specs,
                        worker_count=1,
                        enforce_estimated_time_gate=False,
                    )

    def test_convergence_event_does_not_stop_frozen_updates(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            protocol = ExperimentProtocol(
                nfft=64,
                symbol_count=20,
                occupied_per_side=20,
                update_count=4,
                convergence_nmse_db=100.0,
                divergence_margin_db=1000.0,
                model_orders=(1, 3, 5),
                model_memory_depth=1,
                model_block_size=16,
                confirmatory_seed_count=1,
                pilot_seed_count=1,
                robustness_seed_count=1,
                mismatch_seed_count=1,
                ablation_seed_count=1,
                dynamic_seed_count=1,
                stress_seed_count=1,
                bootstrap_resamples=20,
            )
            runner = ExperimentRunner(temporary, protocol=protocol, resources=self.resources)
            spec = next(
                item
                for item in enumerate_study("smoke", protocol)
                if item.scenario == "amam" and item.algorithm == "linear_ilc"
            )
            runner.run(
                "smoke",
                [spec],
                worker_count=1,
                enforce_estimated_time_gate=False,
            )
            record = load_completed_records(Path(temporary) / "smoke")[0]
            self.assertEqual(record["convergence_iteration"], 0)
            self.assertIn("step_stop_reason", record["metrics"][3])
            self.assertEqual(record["evaluation_count"], protocol.evaluation_count)

    def test_complete_resume_recreates_missing_run_summary(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            specs = enumerate_study("smoke", self.protocol)[:1]
            runner.run(
                "smoke",
                specs,
                worker_count=1,
                enforce_estimated_time_gate=False,
            )
            summary_path = Path(temporary) / "smoke" / "run_summary.json"
            summary_path.unlink()
            summary = runner.run(
                "smoke",
                specs,
                worker_count=1,
                enforce_estimated_time_gate=False,
            )
            self.assertTrue(summary.complete)
            self.assertTrue(summary_path.exists())

    def test_complete_checkpoint_recovers_when_final_rename_was_interrupted(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            specs = [
                spec
                for spec in enumerate_study("smoke", self.protocol)
                if spec.scenario == "ampm"
                and spec.severity == "135"
                and spec.algorithm == "linear_ilc"
            ]
            runner.run(
                "smoke",
                specs,
                worker_count=1,
                enforce_estimated_time_gate=False,
            )
            run_directory = Path(temporary) / "smoke"
            original = load_completed_records(run_directory)[0]
            shard = next((run_directory / "shards").glob("*.json"))
            shard.unlink()
            recovered_summary = runner.run(
                "smoke",
                specs,
                worker_count=1,
                enforce_estimated_time_gate=False,
            )
            recovered = load_completed_records(run_directory)[0]
            self.assertEqual(recovered_summary.newly_completed_count, 1)
            self.assertEqual(recovered["metrics"], original["metrics"])
            self.assertEqual(recovered["final_nmse_db"], original["final_nmse_db"])
            self.assertEqual(recovered["success"], original["success"])

    def test_verified_shard_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "shard.json"
            write_verified_shard(path, {"trajectory_id": "one", "value": 3})
            self.assertEqual(read_verified_shard(path)["value"], 3)
            wrapper = json.loads(path.read_text())
            wrapper["record"]["value"] = 4
            path.write_text(json.dumps(wrapper), encoding="utf-8")
            with self.assertRaises(CorruptArtifactError):
                read_verified_shard(path)

    def test_directory_size_tolerates_disappearing_directories_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            retained = root / "retained.bin"
            retained.write_bytes(b"12345")
            disappearing = root / "work" / "completed"
            disappearing.mkdir(parents=True)
            removed_file = disappearing / "checkpoint.bin"
            removed_file.write_bytes(b"removed")
            original_scandir = os.scandir

            def remove_before_scan(target: str | os.PathLike[str]):
                if Path(target) == disappearing and disappearing.exists():
                    os.unlink(removed_file)
                    os.rmdir(disappearing)
                return original_scandir(target)

            with patch("experiments.runner.os.scandir", side_effect=remove_before_scan):
                self.assertEqual(directory_size(root), retained.stat().st_size)

            with patch(
                "experiments.runner.os.scandir",
                side_effect=PermissionError("access denied"),
            ):
                with self.assertRaises(PermissionError):
                    directory_size(root)

    def test_capacity_gate_prevents_scheduling(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            limits = ResourceLimits(
                worker_count=1,
                minimum_free_disk_bytes=10**18,
                artifact_budget_bytes=100_000_000,
                infrastructure_retries=0,
            )
            runner = ExperimentRunner(temporary, protocol=self.protocol, resources=limits)
            report = runner.capacity_report()
            self.assertFalse(report.allowed)
            with self.assertRaises(CapacityGateError):
                runner.run(
                    "smoke",
                    enumerate_study("smoke", self.protocol)[:1],
                    worker_count=1,
                    enforce_estimated_time_gate=False,
                )

    def test_runner_uses_dtype_aware_safety_feasibility(self) -> None:
        from experiments.runner import _within_limits
        from remote_dpd.learning import InputSafetyLimits, project_input_safety

        candidate = np.asarray([1.0, 0.3], dtype=np.complex64)
        limits = InputSafetyLimits(max_papr_db=2.6360339487352635)
        projection = project_input_safety(candidate, limits)
        self.assertTrue(projection.feasible)
        self.assertTrue(_within_limits(projection.projected_input, limits))

        large = np.asarray([3e38 + 3e38j, 3e38 - 3e38j], dtype=np.complex64)
        large_limits = InputSafetyLimits(
            max_rms=5e38,
            max_peak=5e38,
            max_papr_db=1.0,
        )
        self.assertTrue(_within_limits(large, large_limits))

    def test_capacity_probe_records_throughput_and_peak_rss(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = ExperimentRunner(
                temporary,
                protocol=self.protocol,
                resources=self.resources,
            )
            report = runner.run_capacity_probe()
            self.assertGreater(report.trajectory_seconds, 0.0)
            self.assertGreater(report.evaluations_per_second, 0.0)
            self.assertGreaterEqual(report.peak_worker_rss_bytes, 0)
            self.assertEqual(report.numeric_backend_max_threads, 1)
            self.assertTrue(Path(report.probe_directory, "manifest.json").exists())
            cached = runner.run_capacity_probe()
            self.assertEqual(cached.trajectory_seconds, report.trajectory_seconds)
            self.assertEqual(cached.probe_directory, report.probe_directory)


if __name__ == "__main__":
    unittest.main()
