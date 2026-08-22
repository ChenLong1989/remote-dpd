"""Synthetic tests for verified aggregation and publication result export."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import experiments.analysis as analysis_module

from experiments.analysis import (
    AnalysisError,
    EXACT_ZERO_NMSE_DB_FLOOR,
    FAILURE_ENDPOINT_POLICY,
    VerificationError,
    analyze_records,
    load_verified_dataset,
    normalize_trajectory,
    write_analysis_artifacts,
)
from experiments.config import (
    ExperimentProtocol,
    TrajectorySpec,
    canonical_json,
    enumerate_study,
    matrix_hash,
    resolved_method_parameters,
    stable_hash,
)
from experiments.plot_results import plot_publication_figures


class ExperimentAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = ExperimentProtocol(
            update_count=4,
            convergence_hold=2,
            divergence_hold=2,
            confirmatory_seed_count=4,
            bootstrap_resamples=200,
        )
        self.hashes = {
            "code_hash": "1" * 64,
            "configuration_hash": "2" * 64,
            "protocol_hash": self.protocol.protocol_hash,
            "matrix_hash": "3" * 64,
        }

    def test_verified_shards_pair_seeds_and_retain_algorithm_failure(self) -> None:
        records = self._primary_records()
        random.Random(7).shuffle(records)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            run_directory = self._write_run(Path(temporary), records)
            dataset = load_verified_dataset(run_directory)
            self.assertTrue(dataset.completeness_verified)
            self.assertEqual(len(dataset.records), 16)

            result = analyze_records(
                dataset.records,
                protocol=dataset.protocol,
                bootstrap_resamples=200,
                source_metadata=dataset.metadata(),
            )
            self.assertEqual(len(result.primary_comparisons), 2)
            self.assertEqual(result.metadata["algorithm_failure_count"], 2)
            self.assertEqual(result.metadata["imputed_trajectory_count"], 0)
            self.assertTrue(result.metadata["primary_available"])
            self.assertIsNone(result.metadata["primary_unavailable_reason"])
            for comparison in result.primary_comparisons:
                self.assertEqual(comparison["pair_count"], 4)
                self.assertEqual(comparison["linear_algorithm_failure_count"], 1)
                self.assertEqual(comparison["model_algorithm_failure_count"], 0)
                self.assertGreater(
                    comparison["auec_relative_reduction"]["estimate"],
                    0.25,
                )
                self.assertGreaterEqual(
                    comparison["final_nmse_improvement_db"]["estimate"],
                    3.0,
                )
                self.assertTrue(comparison["criterion"]["passed"])
                self.assertIn("adjusted_p_value", comparison["holm"])

            failure = next(item for item in result.trajectories if item.algorithm_failure)
            self.assertEqual(failure.imputed_evaluation_count, 0)
            self.assertEqual(failure.endpoint_policy, FAILURE_ENDPOINT_POLICY)
            self.assertEqual(failure.nmse_db, (-10.0, -12.0, -12.0, -12.0, -12.0))
            self.assertFalse(failure.convergence_event_observed)
            self.assertEqual(failure.convergence_iteration, self.protocol.update_count)

            removed_path = sorted((run_directory / "shards").glob("*.json"))[0]
            removed_wrapper = json.loads(removed_path.read_text(encoding="utf-8"))
            removed_id = removed_wrapper["record"]["trajectory_id"]
            removed_path.unlink()
            expected_path = run_directory / "expected_ids.json"
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            expected["trajectory_ids"].remove(removed_id)
            expected_path.write_text(canonical_json(expected) + "\n", encoding="utf-8")
            manifest_path = run_directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["expected_trajectory_count"] -= 1
            manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "matrix hash"):
                load_verified_dataset(run_directory)

    def test_incomplete_primary_pair_is_an_error_not_a_dropped_seed(self) -> None:
        records = self._primary_records()
        records = [
            record
            for record in records
            if not (
                record["scenario"] == "amam"
                and record["algorithm"] == "model_lm_ilc"
                and record["pa_seed_index"] == 3
            )
        ]
        with self.assertRaisesRegex(AnalysisError, "primary pairs are incomplete"):
            analyze_records(records, protocol=self.protocol, bootstrap_resamples=20)

    def test_runner_manifest_rejects_a_fully_resigned_frozen_matrix_subset(self) -> None:
        protocol = ExperimentProtocol(
            update_count=4,
            convergence_hold=2,
            divergence_hold=2,
            pilot_seed_count=1,
            amam_severities=(0.97,),
            ampm_severities_deg=(135.0,),
            bootstrap_resamples=20,
        )
        resolved = resolved_method_parameters()
        specifications = list(
            enumerate_study("pilot", protocol, resolved=resolved)
        )
        records = [self._record_from_spec(specification) for specification in specifications]
        hashes = {
            **self.hashes,
            "protocol_hash": protocol.protocol_hash,
            "matrix_hash": matrix_hash(specifications),
        }
        for record in records:
            record["hashes"] = hashes

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            run_directory = Path(temporary) / "pilot"
            shard_directory = run_directory / "shards"
            shard_directory.mkdir(parents=True)

            def write_shards(values: list[dict[str, object]]) -> None:
                for value in values:
                    wrapper = {"checksum": stable_hash(value), "record": value}
                    (shard_directory / f"{value['trajectory_id']}.json").write_text(
                        canonical_json(wrapper) + "\n",
                        encoding="utf-8",
                    )

            write_shards(records)
            expected_path = run_directory / "expected_ids.json"
            expected_path.write_text(
                canonical_json(
                    {
                        "hashes": hashes,
                        "trajectory_ids": sorted(
                            str(record["trajectory_id"]) for record in records
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = run_directory / "manifest.json"
            manifest: dict[str, object] = {
                "schema_version": 1,
                "study": "pilot",
                "hashes": hashes,
                "expected_trajectory_count": len(records),
                "scientific_protocol": protocol.as_dict(),
                "resolved_methods": resolved,
                "runtime_limits": {},
                "generation": {
                    "argv": ["python", "-m", "experiments.run_experiments"],
                    "cwd": str(Path.cwd()),
                    "display_command": "python -m experiments.run_experiments",
                },
            }
            manifest_path.write_text(
                canonical_json(manifest) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(load_verified_dataset(run_directory).records), len(records))

            removed = records.pop()
            (shard_directory / f"{removed['trajectory_id']}.json").unlink()
            remaining_specs = [
                TrajectorySpec.from_dict(record["spec"])  # type: ignore[arg-type]
                for record in records
            ]
            resigned_hashes = {
                **hashes,
                "matrix_hash": matrix_hash(remaining_specs),
            }
            for record in records:
                record["hashes"] = resigned_hashes
            write_shards(records)
            expected_path.write_text(
                canonical_json(
                    {
                        "hashes": resigned_hashes,
                        "trajectory_ids": sorted(
                            str(record["trajectory_id"]) for record in records
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest["hashes"] = resigned_hashes
            manifest["expected_trajectory_count"] = len(records)
            manifest_path.write_text(
                canonical_json(manifest) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(VerificationError, "frozen enumeration"):
                load_verified_dataset(run_directory)

    def test_locked_runner_manifest_without_pilot_provenance_is_rejected(self) -> None:
        protocol = ExperimentProtocol(confirmatory_seed_count=1, bootstrap_resamples=20)
        resolved = resolved_method_parameters()
        specifications = enumerate_study("confirmatory", protocol, resolved=resolved)
        manifest = {
            "study": "confirmatory",
            "resolved_methods": resolved,
            "runtime_limits": {},
            "generation": {"argv": ["python", "-m", "experiments.run_experiments"]},
        }
        with self.assertRaisesRegex(VerificationError, "pilot provenance"):
            analysis_module._validate_runner_matrix_enumeration(
                manifest,
                protocol,
                specifications,
                matrix_hash(specifications),
            )

    def test_completed_short_record_and_unusable_failure_are_rejected(self) -> None:
        completed = self._record("amam", "0.97", "linear_ilc", 0, [-20.0, -21.0])
        with self.assertRaisesRegex(AnalysisError, "does not contain fixed k=0..K"):
            normalize_trajectory(completed, self.protocol)

        failed = self._record(
            "amam",
            "0.97",
            "linear_ilc",
            0,
            ["nan"],
            status="algorithm_failure",
        )
        with self.assertRaisesRegex(AnalysisError, "unusable NMSE"):
            normalize_trajectory(failed, self.protocol)

    def test_exact_zero_nmse_uses_linear_zero_and_a_recorded_db_floor(self) -> None:
        record = self._record(
            "ampm",
            "0",
            "no_dpd",
            0,
            ["-inf"] * self.protocol.evaluation_count,
            convergence_iteration=0,
        )
        endpoint = normalize_trajectory(record, self.protocol)
        self.assertEqual(endpoint.auec, 0.0)
        self.assertEqual(endpoint.final_nmse_db, float("-inf"))
        self.assertEqual(endpoint.final_nmse_db_for_statistics, EXACT_ZERO_NMSE_DB_FLOOR)
        self.assertEqual(endpoint.exact_zero_evaluation_count, self.protocol.evaluation_count)

    def test_metric_derived_endpoints_reject_self_consistent_top_level_tampering(self) -> None:
        record = self._record(
            "amam",
            "0.97",
            "model_lm_ilc",
            0,
            [-12.0, -22.0, -30.0, -37.0, -40.0],
            convergence_iteration=3,
        )
        changes = (
            ("auec", 1.0, "AUEC"),
            ("final_nmse_db", -99.0, "final NMSE"),
            ("convergence_iteration", 0, "convergence_iteration"),
            ("diverged", True, "diverged"),
            ("constraint_violation", True, "constraint_violation"),
        )
        for field, value, message in changes:
            with self.subTest(field=field):
                tampered = deepcopy(record)
                tampered[field] = value
                with self.assertRaisesRegex(AnalysisError, message):
                    normalize_trajectory(tampered, self.protocol)

    def test_numeric_failure_is_derived_before_threshold_checks(self) -> None:
        record = self._record(
            "amam",
            "0.97",
            "linear_ilc",
            0,
            [-20.0, 300.0, 300.0, 300.0, 300.0],
            status="algorithm_failure",
            terminal_reason="nonfinite_evaluation",
        )
        for metric in record["metrics"][1:]:  # type: ignore[index]
            metric["evaluation_failure"] = "nonfinite_pa_or_capture"
            metric["diverged"] = True
            metric.pop("step_stop_reason", None)
        record["diverged"] = True
        endpoint = normalize_trajectory(record, self.protocol)
        self.assertTrue(endpoint.algorithm_failure)
        self.assertTrue(endpoint.diverged)
        self.assertFalse(endpoint.convergence_event_observed)

    def test_post_failure_hold_does_not_create_a_new_convergence_event(self) -> None:
        record = self._record(
            "amam",
            "0.97",
            "linear_ilc",
            0,
            [-10.0, -40.0, -40.0, -40.0, -40.0],
            status="algorithm_failure",
            terminal_reason="cg_non_positive_curvature",
        )
        endpoint = normalize_trajectory(record, self.protocol)
        self.assertTrue(endpoint.algorithm_failure)
        self.assertFalse(endpoint.convergence_event_observed)
        self.assertFalse(endpoint.diverged)

        failure_round_qualifies = self._record(
            "amam",
            "0.97",
            "linear_ilc",
            1,
            [-40.0, -40.0, -40.0, -40.0, -40.0],
            status="algorithm_failure",
            terminal_reason="cg_non_positive_curvature",
            convergence_iteration=0,
        )
        qualified = normalize_trajectory(failure_round_qualifies, self.protocol)
        self.assertTrue(qualified.convergence_event_observed)
        self.assertEqual(qualified.convergence_iteration, 0)
        self.assertFalse(qualified.success)

    def test_partial_smoke_exports_descriptive_summaries_without_primary_inference(self) -> None:
        record = self._record(
            "amam",
            "0.97",
            "linear_ilc",
            0,
            [-12.0, -15.0, -18.0, -20.0, -21.0],
        )
        specification = TrajectorySpec(
            study="smoke",
            scenario="amam",
            severity="0.97",
            pa_seed_index=0,
            waveform_seed_index=0,
            algorithm="linear_ilc",
            parameters={},
        )
        record["trajectory_id"] = specification.trajectory_id
        record["spec"] = specification.as_dict()
        result = analyze_records([record], protocol=self.protocol, bootstrap_resamples=20)
        self.assertFalse(result.metadata["primary_available"])
        self.assertIn(
            "lacks one or both primary methods",
            result.metadata["primary_unavailable_reason"],
        )
        self.assertEqual(result.primary_comparisons, ())
        self.assertEqual(len(result.cell_summaries), 1)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            paths = write_analysis_artifacts(result, temporary)
            self.assertTrue(paths["summary_json"].is_file())
            self.assertEqual(plot_publication_figures(result, temporary), {})

    def test_stress_cells_summarize_guarded_stops_and_gradient_direction(self) -> None:
        records: list[dict[str, object]] = []
        for scenario, severity in (("hard_saturation", "2"), ("gain_rolloff", "0.4")):
            for algorithm in ("linear_ilc", "model_lm_ilc"):
                record = self._record(
                    scenario,
                    severity,
                    algorithm,
                    0,
                    [-5.0, -6.0, -7.0, -8.0, -9.0],
                )
                record["spec"]["study"] = "stress"  # type: ignore[index]
                identifier = str(record["trajectory_id"]).replace("confirmatory--", "stress--")
                record["trajectory_id"] = identifier
                record["spec"]["trajectory_id"] = identifier  # type: ignore[index]
                if algorithm == "model_lm_ilc":
                    metrics = record["metrics"]
                    metrics[0]["saturation_limited"] = True  # type: ignore[index]
                    metrics[0]["identity_gradient_cosine"] = -0.25  # type: ignore[index]
                    metrics[1]["identity_gradient_cosine"] = 0.75  # type: ignore[index]
                    metrics[0]["learned_gradient_cosine"] = 0.80  # type: ignore[index]
                    metrics[1]["learned_gradient_cosine"] = 0.90  # type: ignore[index]
                records.append(record)

        result = analyze_records(records, protocol=self.protocol, bootstrap_resamples=20)
        self.assertEqual(result.metadata["study"], "stress")
        self.assertEqual(result.primary_comparisons, ())
        model_summaries = [
            value for value in result.cell_summaries if value["algorithm"] == "model_lm_ilc"
        ]
        self.assertEqual(len(model_summaries), 2)
        for summary in model_summaries:
            self.assertEqual(summary["guarded_safe_stop_rate_percent"], 100.0)
            self.assertEqual(summary["identity_gradient_cosine_observation_count"], 5)
            self.assertEqual(summary["negative_identity_gradient_cosine_fraction"], 0.2)
            self.assertEqual(summary["learned_gradient_cosine_observation_count"], 5)
            self.assertEqual(summary["negative_learned_gradient_cosine_fraction"], 0.0)
            self.assertAlmostEqual(
                summary["median_final_identity_negative_local_fraction"],
                0.2,
            )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            paths = plot_publication_figures(result, temporary)
            self.assertIn("stress_diagnostics_png", paths)
            self.assertIn("stress_diagnostics_pdf", paths)

    def test_enumerated_stress_severity_flows_through_analysis_and_plotting(self) -> None:
        protocol = ExperimentProtocol(
            update_count=4,
            convergence_hold=2,
            divergence_hold=2,
            stress_seed_count=1,
            bootstrap_resamples=20,
        )
        specifications = enumerate_study("stress", protocol)
        self.assertIn(
            ("hard_saturation", "2"),
            {(item.scenario, item.severity) for item in specifications},
        )
        records = [self._record_from_spec(specification) for specification in specifications]
        result = analyze_records(records, protocol=protocol, bootstrap_resamples=20)
        self.assertFalse(result.metadata["primary_available"])
        hard_saturation = [
            item
            for item in result.cell_summaries
            if item["scenario"] == "hard_saturation" and item["severity"] == "2"
        ]
        self.assertEqual(
            len(hard_saturation),
            len({item.algorithm for item in specifications}),
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            paths = plot_publication_figures(result, temporary, formats=("png",))
            self.assertIn("stress_diagnostics_png", paths)

    def test_variant_plot_keeps_ablation_configs_separate(self) -> None:
        records: list[dict[str, object]] = []
        for scenario, severity in (("amam", "0.97"), ("ampm", "135")):
            for index, ablation in enumerate(("raw_vjp", "no_ridge")):
                record = self._record(
                    scenario,
                    severity,
                    "model_lm_ilc",
                    index,
                    [-10.0, -15.0, -20.0, -25.0, -30.0],
                )
                record["spec"]["study"] = "ablation"  # type: ignore[index]
                record["spec"]["parameters"] = {"ablation": ablation}  # type: ignore[index]
                record["spec"]["config_hash"] = stable_hash(  # type: ignore[index]
                    {"scenario": scenario, "ablation": ablation}
                )
                identifier = str(record["trajectory_id"]).replace(
                    "confirmatory--", "ablation--"
                )
                record["trajectory_id"] = identifier
                record["spec"]["trajectory_id"] = identifier  # type: ignore[index]
                records.append(record)

        result = analyze_records(records, protocol=self.protocol, bootstrap_resamples=20)
        self.assertEqual(len(result.cell_summaries), 4)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            paths = plot_publication_figures(result, temporary)
            self.assertEqual(set(paths), {"variant_endpoints_png", "variant_endpoints_pdf"})

    def test_jsonl_requires_checksum_wrappers_and_detects_tampering(self) -> None:
        records = self._primary_records()[:2]
        self._apply_matrix_hash(records)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            verified_path = root / "records.jsonl"
            wrappers = [
                {"checksum": stable_hash(record), "record": record}
                for record in records
            ]
            verified_path.write_text(
                "\n".join(canonical_json(wrapper) for wrapper in wrappers) + "\n",
                encoding="utf-8",
            )
            dataset = load_verified_dataset(
                verified_path,
                protocol=self.protocol,
                require_complete=False,
            )
            self.assertFalse(dataset.completeness_verified)
            self.assertEqual(len(dataset.records), 2)

            wrappers[0]["record"]["final_nmse_db"] = -999.0
            verified_path.write_text(
                "\n".join(canonical_json(wrapper) for wrapper in wrappers) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(VerificationError, "checksum mismatch"):
                load_verified_dataset(
                    verified_path,
                    protocol=self.protocol,
                    require_complete=False,
                )

            wrappers[0]["checksum"] = stable_hash(wrappers[0]["record"])
            verified_path.write_text(
                "\n".join(canonical_json(wrapper) for wrapper in wrappers) + "\n",
                encoding="utf-8",
            )
            self_consistent = load_verified_dataset(
                verified_path,
                protocol=self.protocol,
                require_complete=False,
            )
            with self.assertRaisesRegex(AnalysisError, "final NMSE"):
                analyze_records(
                    self_consistent.records,
                    protocol=self.protocol,
                    bootstrap_resamples=20,
                )

            raw_path = root / "raw.jsonl"
            raw_path.write_text(canonical_json(records[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "wrapper"):
                load_verified_dataset(raw_path, protocol=self.protocol, require_complete=False)

    def test_csv_json_and_png_pdf_exports_are_reproducible_from_result(self) -> None:
        result = analyze_records(
            self._primary_records(),
            protocol=self.protocol,
            bootstrap_resamples=100,
            source_metadata={"dataset_hash": "4" * 64, "completeness_verified": True},
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            output = Path(temporary)
            table_paths = write_analysis_artifacts(result, output)
            figure_paths = plot_publication_figures(result, output)

            self.assertEqual(set(table_paths), {
                "summary_json",
                "trajectories_csv",
                "per_iteration_metrics_csv",
                "cells_csv",
                "primary_csv",
                "ampm_fixed_r0_phase_csv",
            })
            summary = json.loads(table_paths["summary_json"].read_text(encoding="utf-8"))
            self.assertEqual(summary["metadata"]["bootstrap_resamples"], 100)
            self.assertEqual(len(summary["primary_comparisons"]), 2)
            model_cell = next(
                value
                for value in summary["cell_summaries"]
                if value["scenario"] == "ampm"
                and value["algorithm"] == "model_lm_ilc"
            )
            self.assertEqual(model_cell["pa_model_train_nmse_db_trajectory_coverage_count"], 4)
            self.assertEqual(model_cell["pa_model_train_nmse_db_final_coverage_count"], 0)
            self.assertIn("median_final_identity_negative_local_fraction", model_cell)
            trajectory_lines = table_paths["trajectories_csv"].read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(trajectory_lines), len(result.trajectories) + 1)
            iteration_text = table_paths["per_iteration_metrics_csv"].read_text(
                encoding="utf-8"
            )
            iteration_lines = iteration_text.splitlines()
            self.assertEqual(
                len(iteration_lines),
                len(result.trajectories) * self.protocol.evaluation_count + 1,
            )
            header = iteration_lines[0]
            for field in (
                "aclr_lower_db",
                "evm_one_tap_percent",
                "low_power_phase_rmse_deg",
                "pa_model_validation_nmse_db",
                "cg_relative_residual",
                "identity_negative_local_fraction",
                "learned_gradient_cosine",
            ):
                self.assertIn(field, header)
            self.assertIn("null", iteration_text)

            phase_lines = table_paths["ampm_fixed_r0_phase_csv"].read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(phase_lines[0].count("fixed_r0"), 1)
            self.assertEqual(len(phase_lines), 4 * 2 * self.protocol.evaluation_count + 1)

            self.assertEqual(len(figure_paths), 8)
            self.assertIn("ampm_fixed_r0_phase_png", figure_paths)
            self.assertIn("ampm_fixed_r0_phase_pdf", figure_paths)
            for name, path in figure_paths.items():
                self.assertGreater(path.stat().st_size, 1000)
                signature = path.read_bytes()[:8]
                if name.endswith("_png"):
                    self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")
                else:
                    self.assertTrue(signature.startswith(b"%PDF-"))

    def _primary_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for scenario, severity in (("amam", "0.97"), ("ampm", "135")):
            for seed in range(4):
                if seed == 3:
                    records.append(
                        self._record(
                            scenario,
                            severity,
                            "linear_ilc",
                            seed,
                            [-10.0, -12.0, -12.0, -12.0, -12.0],
                            status="algorithm_failure",
                            terminal_reason="cg_breakdown",
                        )
                    )
                else:
                    records.append(
                        self._record(
                            scenario,
                            severity,
                            "linear_ilc",
                            seed,
                            [-12.0, -15.0, -18.0, -20.0, -21.0],
                        )
                    )
                records.append(
                    self._record(
                        scenario,
                        severity,
                        "model_lm_ilc",
                        seed,
                        [-12.0, -22.0, -30.0, -37.0, -40.0],
                        convergence_iteration=3,
                    )
                )
        return records

    def _record(
        self,
        scenario: str,
        severity: str,
        algorithm: str,
        seed: int,
        nmse_values: list[float | str],
        *,
        status: str = "completed",
        terminal_reason: str = "maximum_iterations",
        convergence_iteration: int | None = None,
    ) -> dict[str, object]:
        trajectory_spec = TrajectorySpec(
            study="confirmatory",
            scenario=scenario,
            severity=severity,
            pa_seed_index=seed,
            waveform_seed_index=seed,
            algorithm=algorithm,
            parameters={},
        )
        identifier = trajectory_spec.trajectory_id
        spec = trajectory_spec.as_dict()
        numeric_values = [
            (
                float(value)
                if not isinstance(value, str)
                else {"-inf": float("-inf"), "inf": float("inf")}.get(
                    value,
                    float("nan"),
                )
            )
            for value in nmse_values
        ]
        final: float | str = nmse_values[-1]
        auec = float(
            np.mean(np.power(10.0, np.asarray(numeric_values, dtype=np.float64) / 10.0))
        )
        metrics: list[dict[str, object]] = []
        for iteration, value in enumerate(nmse_values):
            numeric_value = (
                float(value)
                if not isinstance(value, str)
                else {"-inf": float("-inf"), "inf": float("inf")}.get(
                    value,
                    float("nan"),
                )
            )
            linear_nmse: float | str = (
                10.0 ** (numeric_value / 10.0)
                if np.isfinite(numeric_value) or numeric_value == float("-inf")
                else "nan"
            )
            metric: dict[str, object] = {
                "iteration": iteration,
                "nmse": linear_nmse,
                "nmse_db": value,
                "input_rms": 0.30 + 0.01 * iteration,
                "input_peak": 0.80 + 0.01 * iteration,
                "input_papr_db": 8.0,
                "aclr_lower_db": -40.0 - iteration,
                "aclr_upper_db": -39.0 - iteration,
                "aclr_worst_db": -39.0 - iteration,
                "evm_raw_percent": 5.0 / (iteration + 1),
                "evm_one_tap_percent": 4.0 / (iteration + 1),
                "low_power_amplitude_threshold": self.protocol.ampm_r0,
                "low_power_phase_rmse_deg": 30.0 / (iteration + 1),
                "binned_amam_rmse": 0.10 / (iteration + 1),
                "envelope_bin_edges": [0.0, 0.1, 0.2, 0.3],
                "envelope_bin_counts": [10, 20, 30],
                "envelope_phase_counts": [8, 18, 28],
                "binned_amam_rmse_values": [0.1, 0.08, 0.05],
                "binned_ampm_rmse_deg_values": [30.0, 20.0, 10.0],
                "identity_gradient_cosine": -0.20 + 0.10 * iteration,
                "identity_negative_local_fraction": 0.40 - 0.05 * iteration,
                "identity_negative_inner_magnitude_fraction": 0.50 - 0.05 * iteration,
                "diverged": False,
                "constraint_violation": False,
            }
            if algorithm in {"model_vjp_ilc", "model_lm_ilc", "oracle_lm"}:
                metric["learned_gradient_cosine"] = 0.90
            if algorithm == "model_lm_ilc" and iteration < self.protocol.update_count:
                metric.update(
                    {
                        "pa_model_train_nmse_db": -45.0,
                        "pa_model_validation_nmse_db": -42.0,
                        "pa_model_rank": 15,
                        "pa_model_condition": 100.0,
                        "cg_iterations": 4,
                        "cg_relative_residual": 1e-4,
                    }
                )
            metrics.append(metric)
        if status == "algorithm_failure" and metrics and np.all(
            np.isfinite(np.asarray(numeric_values, dtype=np.float64))
        ):
            failure_iteration = min(1, len(metrics) - 1)
            metrics[failure_iteration]["step_stop_reason"] = terminal_reason
        return {
            "schema_version": 1,
            "trajectory_id": identifier,
            "spec": spec,
            "hashes": self.hashes,
            "status": status,
            "terminal_reason": terminal_reason,
            "algorithm": algorithm,
            "scenario": scenario,
            "severity": severity,
            "pa_seed_index": seed,
            "waveform_seed_index": seed,
            "evaluation_count": len(nmse_values),
            "auec": auec,
            "final_nmse_db": final,
            "convergence_iteration": convergence_iteration,
            "success": convergence_iteration is not None and status == "completed",
            "diverged": False,
            "constraint_violation": False,
            "model_fallback_count": 0,
            "metrics": metrics,
            "runtime_seconds": 0.1,
            "peak_rss_bytes": 1024,
        }

    def _record_from_spec(self, specification: TrajectorySpec) -> dict[str, object]:
        record = self._record(
            specification.scenario,
            specification.severity,
            specification.algorithm,
            specification.pa_seed_index,
            [-10.0, -12.0, -14.0, -16.0, -18.0],
        )
        record.update(
            {
                "trajectory_id": specification.trajectory_id,
                "spec": specification.as_dict(),
                "algorithm": specification.algorithm,
                "scenario": specification.scenario,
                "severity": specification.severity,
                "pa_seed_index": specification.pa_seed_index,
                "waveform_seed_index": specification.waveform_seed_index,
            }
        )
        return record

    def _write_run(self, root: Path, records: list[dict[str, object]]) -> Path:
        run_directory = root / "confirmatory"
        shards = run_directory / "shards"
        shards.mkdir(parents=True)
        run_hashes = self._apply_matrix_hash(records)
        for record in records:
            wrapper = {"checksum": stable_hash(record), "record": record}
            (shards / f"{record['trajectory_id']}.json").write_text(
                canonical_json(wrapper) + "\n",
                encoding="utf-8",
            )
        expected = {
            "hashes": run_hashes,
            "trajectory_ids": sorted(str(record["trajectory_id"]) for record in records),
        }
        (run_directory / "expected_ids.json").write_text(
            canonical_json(expected) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "study": "confirmatory",
            "hashes": run_hashes,
            "expected_trajectory_count": len(records),
            "scientific_protocol": self.protocol.as_dict(),
        }
        (run_directory / "manifest.json").write_text(
            canonical_json(manifest) + "\n",
            encoding="utf-8",
        )
        return run_directory

    def _apply_matrix_hash(
        self,
        records: list[dict[str, object]],
    ) -> dict[str, str]:
        specifications = [
            TrajectorySpec.from_dict(record["spec"])  # type: ignore[arg-type]
            for record in records
        ]
        hashes = {**self.hashes, "matrix_hash": matrix_hash(specifications)}
        for record in records:
            record["hashes"] = hashes
        return hashes


if __name__ == "__main__":
    unittest.main()
