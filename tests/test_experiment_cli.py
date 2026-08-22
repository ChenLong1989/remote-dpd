"""Regression tests for CLI environment initialization and pilot loading."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import run_experiments
from experiments.config import (
    DEFAULT_RESOLVED_METHODS,
    PILOT_CANDIDATES,
    PILOT_COMPUTE_COSTS,
    ExperimentProtocol,
    canonical_json,
    enumerate_study,
    matrix_hash,
    stable_hash,
)
from experiments.run_experiments import _load_resolved
from experiments.runner import (
    atomic_write_json,
    compute_code_hash,
    environment_manifest,
    file_sha256,
    load_completed_records,
    write_verified_shard,
)
from experiments.runtime import (
    NUMERIC_THREAD_ENVIRONMENT_VARIABLES,
    apply_numeric_thread_limits,
)
from experiments.statistics import select_pilot_candidates


class ResolvedEnvironmentInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = ExperimentProtocol(
            pilot_seed_count=1,
            confirmatory_seed_count=1,
            bootstrap_resamples=20,
        )

    def test_loader_initializes_thread_environment_before_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            resolved_path, pilot_hashes = self._write_valid_pilot(Path(temporary))
            with patch.dict(os.environ, {}, clear=False):
                for name in NUMERIC_THREAD_ENVIRONMENT_VARIABLES:
                    os.environ.pop(name, None)
                self.assertTrue(
                    all(os.environ.get(name) is None for name in NUMERIC_THREAD_ENVIRONMENT_VARIABLES)
                )

                loaded = _load_resolved(
                    resolved_path,
                    self.protocol,
                    pilot_hashes["code_hash"],
                )

                self.assertEqual(loaded.methods, loaded.pilot_lock["resolved_payload"]["resolved_methods"])
                self.assertTrue(
                    all(os.environ.get(name) == "1" for name in NUMERIC_THREAD_ENVIRONMENT_VARIABLES)
                )
                self.assertEqual(
                    stable_hash(environment_manifest()),
                    pilot_hashes["environment_hash"],
                )

    def test_locked_dry_run_initializes_environment_before_loading_resolved(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            output_root = Path(temporary)
            resolved_path, pilot_hashes = self._write_valid_pilot(output_root)
            stdout = io.StringIO()
            with patch.dict(os.environ, {}, clear=False):
                for name in NUMERIC_THREAD_ENVIRONMENT_VARIABLES:
                    os.environ.pop(name, None)
                with patch.object(
                    run_experiments,
                    "ExperimentProtocol",
                    return_value=self.protocol,
                ), redirect_stdout(stdout):
                    status = run_experiments.main(
                        [
                            "--study",
                            "confirmatory",
                            "--output",
                            str(output_root),
                            "--workers",
                            "1",
                            "--resolved-config",
                            str(resolved_path),
                            "--dry-run",
                        ]
                    )

                self.assertEqual(status, 0)
                self.assertTrue(
                    all(os.environ.get(name) == "1" for name in NUMERIC_THREAD_ENVIRONMENT_VARIABLES)
                )
                self.assertEqual(
                    stable_hash(environment_manifest()),
                    pilot_hashes["environment_hash"],
                )
            dry_run = json.loads(stdout.getvalue())
            self.assertFalse(dry_run["writes_artifacts"])
            self.assertFalse((output_root / "confirmatory").exists())

    def test_loader_still_rejects_a_different_numeric_environment(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            resolved_path, pilot_hashes = self._write_valid_pilot(Path(temporary))
            different_environment = json.loads(canonical_json(environment_manifest()))
            different_environment["packages"]["numpy"] = "different-version"
            with patch(
                "experiments.runner.environment_manifest",
                return_value=different_environment,
            ):
                with self.assertRaisesRegex(ValueError, "environment hash"):
                    _load_resolved(
                        resolved_path,
                        self.protocol,
                        pilot_hashes["code_hash"],
                    )

    def _write_valid_pilot(self, output_root: Path) -> tuple[Path, dict[str, str]]:
        apply_numeric_thread_limits()
        project_root = Path(__file__).resolve().parents[1]
        pilot_directory = output_root / "pilot"
        shard_directory = pilot_directory / "shards"
        shard_directory.mkdir(parents=True)
        specifications = enumerate_study("pilot", self.protocol)
        pilot_hashes = {
            "code_hash": compute_code_hash(project_root),
            "configuration_hash": stable_hash(
                {
                    "protocol": self.protocol.as_dict(),
                    "resolved_methods": DEFAULT_RESOLVED_METHODS,
                    "resolved_hash": None,
                }
            ),
            "protocol_hash": self.protocol.protocol_hash,
            "matrix_hash": matrix_hash(specifications),
            "environment_hash": stable_hash(environment_manifest()),
        }
        selection_rows = []
        for specification in specifications:
            candidate_index = int(specification.parameters["candidate_index"])
            record = {
                "trajectory_id": specification.trajectory_id,
                "algorithm": specification.algorithm,
                "spec": specification.as_dict(),
                "auec": float(candidate_index + 1),
                "constraint_violation": False,
                "diverged": False,
                "status": "completed",
                "hashes": pilot_hashes,
            }
            write_verified_shard(
                shard_directory / f"{specification.trajectory_id}.json",
                record,
            )
            selection_rows.append(
                {
                    "algorithm": specification.algorithm,
                    "candidate_index": candidate_index,
                    "auec": record["auec"],
                    "safety_failure": False,
                }
            )

        manifest_path = pilot_directory / "manifest.json"
        expected_ids_path = pilot_directory / "expected_ids.json"
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "study": "pilot",
                "hashes": pilot_hashes,
                "expected_trajectory_count": len(specifications),
            },
        )
        atomic_write_json(
            expected_ids_path,
            {
                "hashes": pilot_hashes,
                "trajectory_ids": sorted(
                    specification.trajectory_id for specification in specifications
                ),
            },
        )
        selections = select_pilot_candidates(
            selection_rows,
            candidate_parameters=PILOT_CANDIDATES,
            candidate_costs=PILOT_COMPUTE_COSTS,
        )
        resolved_methods = {
            name: dict(selection.parameters) for name, selection in selections.items()
        }
        for name in ("no_dpd", "legacy_ilc", "oracle_lm"):
            resolved_methods[name] = dict(DEFAULT_RESOLVED_METHODS[name])
        records = load_completed_records(pilot_directory)
        resolved_path = pilot_directory / "resolved_config.json"
        resolved_payload = {
            "schema_version": 1,
            "protocol_hash": self.protocol.protocol_hash,
            "pilot_hashes": pilot_hashes,
            "pilot_provenance": {
                "run_directory": os.path.relpath(pilot_directory, resolved_path.parent),
                "manifest_sha256": file_sha256(manifest_path),
                "expected_ids_sha256": file_sha256(expected_ids_path),
                "records_hash": stable_hash(records),
            },
            "resolved_methods": resolved_methods,
            "selection": {
                name: selection.as_dict() for name, selection in selections.items()
            },
        }
        resolved_payload["resolved_hash"] = stable_hash(resolved_payload)
        atomic_write_json(resolved_path, resolved_payload)
        return resolved_path, pilot_hashes


if __name__ == "__main__":
    unittest.main()
