"""Command-line entry point for reproducible experiment preparation and runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .runtime import apply_numeric_thread_limits

# Normalize the environment before importing NumPy-backed protocol modules.
apply_numeric_thread_limits()

from .config import (
    PILOT_CANDIDATES,
    PILOT_COMPUTE_COSTS,
    PILOT_LOCKED_STUDIES,
    STUDIES,
    ExperimentProtocol,
    ResourceLimits,
    enumerate_study,
    expected_study_counts,
    matrix_hash,
    stable_hash,
)
from .runner import (
    ExperimentRunner,
    atomic_write_json,
    compute_code_hash,
    file_sha256,
    load_completed_records,
    read_json,
)
from .statistics import select_pilot_candidates


@dataclass(frozen=True, slots=True)
class LoadedResolvedConfig:
    """Verified methods and the pilot chain bound into locked-study manifests."""

    methods: Mapping[str, Mapping[str, Any]]
    pilot_lock: Mapping[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen PA model-backpropagation ILC simulation protocol."
    )
    parser.add_argument("--study", choices=STUDIES, help="Study matrix to prepare or execute.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "experiments",
        help="Artifact root; each study uses its own subdirectory.",
    )
    parser.add_argument("--workers", type=int, default=6, help="Independent worker processes (1..8).")
    parser.add_argument(
        "--resolved-config",
        type=Path,
        help="Hash-locked pilot result for confirmatory and secondary studies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix without creating or changing run artifacts.",
    )
    parser.add_argument(
        "--capacity-only",
        action="store_true",
        help="Check disk/RSS capacity without preparing or running a matrix.",
    )
    parser.add_argument(
        "--capacity-probe",
        action="store_true",
        help="Run one learned-LM smoke trajectory and project full-protocol runtime.",
    )
    parser.add_argument(
        "--list-studies",
        action="store_true",
        help="Print preregistered trajectory counts and exit.",
    )
    parser.add_argument(
        "--select-pilot",
        action="store_true",
        help="Resolve pilot candidates from verified pilot shards and exit.",
    )
    parser.add_argument(
        "--resolved-output",
        type=Path,
        help="Destination for --select-pilot (default: pilot/resolved_config.json).",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        help="Run only the first N trajectories; allowed only for the smoke study.",
    )
    parser.add_argument(
        "--no-time-gate",
        action="store_true",
        help="Disable the 72-hour projection gate (does not alter scientific cells).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    apply_numeric_thread_limits()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    generation_argv = (
        sys.executable,
        "-m",
        "experiments.run_experiments",
        *raw_argv,
    )
    protocol = ExperimentProtocol()
    if args.list_studies:
        print(json.dumps(expected_study_counts(protocol), indent=2, sort_keys=True))
        return 0

    if args.select_pilot:
        return _select_pilot(args, protocol, generation_argv)
    if args.study is None and not (args.capacity_only or args.capacity_probe):
        raise SystemExit(
            "--study is required unless a list/capacity operation is requested"
        )
    if args.debug_limit is not None:
        if args.study != "smoke":
            raise SystemExit("--debug-limit is permitted only for the smoke study")
        if args.debug_limit <= 0:
            raise SystemExit("--debug-limit must be positive")

    project_root = Path(__file__).resolve().parents[1]
    loaded_resolved = (
        _load_resolved(
            args.resolved_config,
            protocol,
            compute_code_hash(project_root),
        )
        if args.resolved_config
        else None
    )
    resolved = loaded_resolved.methods if loaded_resolved is not None else None
    pilot_lock = loaded_resolved.pilot_lock if loaded_resolved is not None else None
    if args.study in PILOT_LOCKED_STUDIES and resolved is None:
        raise SystemExit(
            "confirmatory/secondary studies require --resolved-config from --select-pilot"
        )
    resources = ResourceLimits(worker_count=args.workers)
    runner = ExperimentRunner(
        args.output,
        protocol=protocol,
        resources=resources,
        resolved=resolved,
        pilot_lock=pilot_lock,
        generation_argv=generation_argv,
    )
    if args.capacity_only:
        print(json.dumps(asdict(runner.capacity_report()), indent=2, sort_keys=True))
        return 0
    if args.capacity_probe:
        print(json.dumps(runner.run_capacity_probe().as_dict(), indent=2, sort_keys=True))
        return 0

    specs = enumerate_study(args.study, protocol, resolved=resolved)
    if args.debug_limit is not None:
        specs = specs[: args.debug_limit]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study": args.study,
                    "trajectory_count": len(specs),
                    "matrix_hash": matrix_hash(specs),
                    "code_hash": compute_code_hash(project_root),
                    "configuration_hash": stable_hash(
                        {
                            "protocol": protocol.as_dict(),
                            "resolved_methods": runner.resolved,
                            "resolved_hash": (
                                runner.pilot_lock.get("resolved_hash")
                                if runner.pilot_lock is not None
                                else None
                            ),
                        }
                    ),
                    "protocol_hash": protocol.protocol_hash,
                    "would_run_directory": str((args.output.resolve() / args.study)),
                    "writes_artifacts": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    summary = runner.run(
        args.study,
        specs,
        worker_count=args.workers,
        enforce_estimated_time_gate=not args.no_time_gate,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


def _select_pilot(
    args: argparse.Namespace,
    protocol: ExperimentProtocol,
    generation_argv: tuple[str, ...],
) -> int:
    runner = ExperimentRunner(
        args.output,
        protocol=protocol,
        resources=ResourceLimits(worker_count=args.workers),
        generation_argv=generation_argv,
    )
    expected = enumerate_study("pilot", protocol)
    pilot_directory, hashes, expected = runner.prepare("pilot", expected)
    verified = runner._verified_completed(pilot_directory, hashes, expected)
    expected_ids = {spec.trajectory_id for spec in expected}
    if set(verified) != expected_ids:
        missing = len(expected_ids.difference(verified))
        unexpected = len(set(verified).difference(expected_ids))
        raise RuntimeError(
            f"pilot matrix is incomplete or mixed: {missing} missing, {unexpected} unexpected"
        )
    records = tuple(verified[trajectory_id] for trajectory_id in sorted(verified))
    selections = select_pilot_candidates(
        _pilot_selection_rows(records),
        candidate_parameters=PILOT_CANDIDATES,
        candidate_costs=PILOT_COMPUTE_COSTS,
    )
    resolved_methods: dict[str, Mapping[str, Any]] = {
        name: selection.parameters for name, selection in selections.items()
    }
    # The methods not tuned in the pilot use their preregistered fixed values.
    from .config import DEFAULT_RESOLVED_METHODS

    for name in ("no_dpd", "legacy_ilc", "oracle_lm"):
        resolved_methods[name] = DEFAULT_RESOLVED_METHODS[name]
    destination = args.resolved_output or pilot_directory / "resolved_config.json"
    payload = {
        "schema_version": 1,
        "protocol_hash": protocol.protocol_hash,
        "pilot_hashes": hashes.as_dict(),
        "pilot_provenance": {
            "run_directory": os.path.relpath(pilot_directory, destination.parent),
            "manifest_sha256": file_sha256(pilot_directory / "manifest.json"),
            "expected_ids_sha256": file_sha256(pilot_directory / "expected_ids.json"),
            "records_hash": stable_hash(records),
        },
        "resolved_methods": resolved_methods,
        "selection": {name: selection.as_dict() for name, selection in selections.items()},
    }
    payload["resolved_hash"] = stable_hash(payload)
    atomic_write_json(destination, payload)
    print(json.dumps({"resolved_config": str(destination), "resolved_hash": payload["resolved_hash"]}, indent=2))
    return 0


def _load_resolved(
    path: Path,
    protocol: ExperimentProtocol,
    expected_code_hash: str,
) -> LoadedResolvedConfig:
    apply_numeric_thread_limits()
    expected_protocol_hash = protocol.protocol_hash
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or "resolved_methods" not in payload:
        raise ValueError("resolved config must be a signed pilot selection object")
    expected_hash = payload.get("resolved_hash")
    unhashed = dict(payload)
    unhashed.pop("resolved_hash", None)
    if expected_hash != stable_hash(unhashed):
        raise ValueError("resolved config checksum mismatch")
    if payload.get("protocol_hash") != expected_protocol_hash:
        raise ValueError("resolved config protocol hash mismatch")
    pilot_hashes = payload.get("pilot_hashes")
    if not isinstance(pilot_hashes, dict):
        raise ValueError("resolved config has no pilot hash provenance")
    if pilot_hashes.get("protocol_hash") != expected_protocol_hash:
        raise ValueError("pilot protocol hash does not match the current protocol")
    if pilot_hashes.get("code_hash") != expected_code_hash:
        raise ValueError("pilot code hash does not match the current scientific code")
    from .runner import environment_manifest

    if pilot_hashes.get("environment_hash") != stable_hash(environment_manifest()):
        raise ValueError("pilot environment hash does not match the current environment")
    if set(pilot_hashes) != {
        "code_hash",
        "configuration_hash",
        "protocol_hash",
        "matrix_hash",
        "environment_hash",
    } or any(
        not isinstance(value, str) or len(value) != 64
        for value in pilot_hashes.values()
    ):
        raise ValueError("pilot hash provenance is malformed")
    provenance = payload.get("pilot_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("resolved config has no pilot artifact provenance")
    pilot_directory = (path.parent / str(provenance.get("run_directory", ""))).resolve()
    manifest_path = pilot_directory / "manifest.json"
    expected_ids_path = pilot_directory / "expected_ids.json"
    if provenance.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("pilot manifest checksum mismatch")
    if provenance.get("expected_ids_sha256") != file_sha256(expected_ids_path):
        raise ValueError("pilot expected-ID checksum mismatch")
    manifest = read_json(manifest_path)
    expected_ids = read_json(expected_ids_path)
    if manifest.get("hashes") != pilot_hashes or expected_ids.get("hashes") != pilot_hashes:
        raise ValueError("pilot manifest hashes do not match resolved provenance")
    records = load_completed_records(pilot_directory)
    if provenance.get("records_hash") != stable_hash(records):
        raise ValueError("pilot result-set provenance checksum mismatch")
    expected_trajectory_ids = expected_ids.get("trajectory_ids")
    expected_specs = enumerate_study("pilot", protocol)
    frozen_ids = sorted(spec.trajectory_id for spec in expected_specs)
    frozen_matrix_hash = matrix_hash(expected_specs)
    if expected_trajectory_ids != frozen_ids:
        raise ValueError("pilot expected IDs do not match the frozen pilot matrix")
    if pilot_hashes.get("matrix_hash") != frozen_matrix_hash:
        raise ValueError("pilot matrix hash does not match the frozen pilot matrix")
    if manifest.get("expected_trajectory_count") != len(expected_specs):
        raise ValueError("pilot manifest count does not match the frozen pilot matrix")
    from .config import DEFAULT_RESOLVED_METHODS

    expected_configuration_hash = stable_hash(
        {
            "protocol": protocol.as_dict(),
            "resolved_methods": DEFAULT_RESOLVED_METHODS,
            "resolved_hash": None,
        }
    )
    if pilot_hashes.get("configuration_hash") != expected_configuration_hash:
        raise ValueError("pilot configuration hash does not match the frozen pilot configuration")
    if not isinstance(expected_trajectory_ids, list) or {
        record.get("trajectory_id") for record in records
    } != set(expected_trajectory_ids):
        raise ValueError("pilot result set is incomplete or contains unexpected IDs")
    if any(record.get("hashes") != pilot_hashes for record in records):
        raise ValueError("pilot shard hashes do not match resolved provenance")
    recomputed_selections = select_pilot_candidates(
        _pilot_selection_rows(records),
        candidate_parameters=PILOT_CANDIDATES,
        candidate_costs=PILOT_COMPUTE_COSTS,
    )
    recomputed_payload = {
        name: selection.as_dict() for name, selection in recomputed_selections.items()
    }
    if payload.get("selection") != recomputed_payload:
        raise ValueError("resolved selection does not match the verified pilot results")
    if payload.get("schema_version") != 1 or not isinstance(payload.get("selection"), dict):
        raise ValueError("resolved config schema or selection is invalid")
    resolved_methods = payload["resolved_methods"]
    if not isinstance(resolved_methods, dict):
        raise ValueError("resolved_methods must be a JSON object")
    from .config import CORE_METHODS

    if set(resolved_methods) != set(CORE_METHODS):
        raise ValueError("resolved config must contain every frozen method exactly once")
    for algorithm, candidates in PILOT_CANDIDATES.items():
        selection = payload["selection"].get(algorithm)
        if not isinstance(selection, dict):
            raise ValueError(f"missing pilot selection for {algorithm}")
        candidate_index = selection.get("candidate_index")
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise ValueError(f"invalid pilot candidate index for {algorithm}")
        if not 0 <= candidate_index < len(candidates):
            raise ValueError(f"pilot candidate index is out of range for {algorithm}")
        expected_parameters = dict(candidates[candidate_index])
        if selection.get("parameters") != expected_parameters:
            raise ValueError(f"pilot selection parameters do not match frozen table for {algorithm}")
        if resolved_methods.get(algorithm) != expected_parameters:
            raise ValueError(f"resolved method does not match pilot selection for {algorithm}")
    for algorithm in ("no_dpd", "legacy_ilc", "oracle_lm"):
        if resolved_methods.get(algorithm) != dict(DEFAULT_RESOLVED_METHODS[algorithm]):
            raise ValueError(f"fixed method parameters changed for {algorithm}")
    pilot_lock = {
        "resolved_hash": expected_hash,
        "resolved_config_sha256": file_sha256(path),
        "resolved_config_path": str(path.resolve()),
        "pilot_hashes": dict(pilot_hashes),
        "pilot_provenance": dict(provenance),
        "resolved_payload": unhashed,
    }
    return LoadedResolvedConfig(resolved_methods, pilot_lock)


def _pilot_selection_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        spec = record.get("spec", {})
        parameters = spec.get("parameters", {}) if isinstance(spec, Mapping) else {}
        rows.append(
            {
                "algorithm": record.get("algorithm"),
                "candidate_index": parameters.get("candidate_index"),
                "auec": record.get("auec"),
                "safety_failure": bool(record.get("constraint_violation"))
                or bool(record.get("diverged"))
                or record.get("status") != "completed",
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
