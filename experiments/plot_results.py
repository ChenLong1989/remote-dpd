"""Publication-style figures and one-command result export."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis import (
    AnalysisResult,
    PRIMARY_CELLS,
    analyze_records,
    load_verified_dataset,
    write_analysis_artifacts,
)


METHOD_ORDER = (
    "no_dpd",
    "linear_ilc",
    "legacy_ilc",
    "instantaneous_gain_ilc",
    "oracle_lm",
    "model_vjp_ilc",
    "model_lm_ilc",
)
METHOD_LABELS = {
    "no_dpd": "No DPD",
    "linear_ilc": "Linear ILC",
    "legacy_ilc": "Legacy ILC",
    "instantaneous_gain_ilc": "Instantaneous gain",
    "oracle_lm": "Oracle LM",
    "model_vjp_ilc": "Learned raw VJP",
    "model_lm_ilc": "Learned safeguarded LM",
}
METHOD_COLORS = {
    "no_dpd": "#7F7F7F",
    "linear_ilc": "#D55E00",
    "legacy_ilc": "#CC79A7",
    "instantaneous_gain_ilc": "#E69F00",
    "oracle_lm": "#009E73",
    "model_vjp_ilc": "#56B4E9",
    "model_lm_ilc": "#0072B2",
}
SCENARIO_LABELS = {
    ("amam", "0.97"): "AM/AM near saturation (0.97)",
    ("ampm", "135"): "Low-power AM/PM (135 degrees)",
    ("hard_saturation", "2"): "Unreachable hard saturation (2.00)",
    ("gain_rolloff", "0.4"): "Gain roll-off beyond turnover (0.40)",
}
FIGURE_DPI = 300
NMSE_AXIS_DB = (-60.0, 10.0)
FIXED_CREATION_TIME = datetime(2026, 8, 22, tzinfo=timezone.utc)


def plot_publication_figures(
    result: AnalysisResult,
    output_directory: str | os.PathLike[str],
    *,
    formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, Path]:
    """Generate fixed-style convergence, effect, and endpoint-rate figures."""

    normalized_formats = tuple(dict.fromkeys(value.lower() for value in formats))
    if not normalized_formats or any(value not in {"png", "pdf"} for value in normalized_formats):
        raise ValueError("formats must contain png and/or pdf")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    outputs: dict[str, Path] = {}
    with plt.rc_context(_publication_style()):
        cells = _display_cells(result)
        figures: dict[str, Any] = {}
        if len(cells) == 2:
            if _has_unique_algorithm_configs(result, cells):
                figures = {
                    (
                        "convergence_main"
                        if result.primary_comparisons
                        else "convergence_study"
                    ): _convergence_figure(plt, result, cells),
                    "endpoint_rates": _endpoint_rate_figure(plt, result, cells),
                }
            else:
                figures = {
                    "variant_endpoints": _variant_endpoint_figure(plt, result, cells)
                }
            if result.primary_comparisons:
                figures["primary_effects"] = _primary_effect_figure(plt, result)
            if result.metadata.get("study") == "stress":
                figures["stress_diagnostics"] = _stress_diagnostic_figure(
                    plt,
                    result,
                    cells,
                )
        if _has_ampm_phase_figure_data(result):
            figures["ampm_fixed_r0_phase"] = _ampm_fixed_r0_phase_figure(
                plt,
                result,
            )
        try:
            for stem, figure in figures.items():
                for extension in normalized_formats:
                    destination = output / f"{stem}.{extension}"
                    _save_figure(figure, destination, result, stem)
                    outputs[f"{stem}_{extension}"] = destination
        finally:
            for figure in figures.values():
                plt.close(figure)
    return outputs


def export_publication_results(
    source: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    require_complete: bool = True,
    formats: Sequence[str] = ("png", "pdf"),
) -> tuple[AnalysisResult, dict[str, Path]]:
    """Verify source data and write every publication table and figure."""

    dataset = load_verified_dataset(source, require_complete=require_complete)
    result = analyze_records(
        dataset.records,
        protocol=dataset.protocol,
        source_metadata=dataset.metadata(),
    )
    paths = write_analysis_artifacts(result, output_directory)
    paths.update(plot_publication_figures(result, output_directory, formats=formats))
    return result, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify PA-backpropagation results and reproduce publication tables and figures."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Run directory, shards directory, or verified JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Publication artifact directory.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="Figure formats; the default emits both PNG and vector PDF.",
    )
    parser.add_argument(
        "--allow-unlisted-jsonl",
        action="store_true",
        help=(
            "Allow JSONL without expected_ids.json; per-record checksums and pairing remain strict."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, paths = export_publication_results(
        args.source,
        args.output,
        require_complete=not args.allow_unlisted_jsonl,
        formats=args.formats,
    )
    print(
        json.dumps(
            {
                "study": result.metadata["study"],
                "trajectory_count": len(result.trajectories),
                "primary_available": result.metadata["primary_available"],
                "primary_unavailable_reason": result.metadata[
                    "primary_unavailable_reason"
                ],
                "primary_comparison_count": len(result.primary_comparisons),
                "outputs": {name: str(path) for name, path in sorted(paths.items())},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _publication_style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 3.5,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _display_cells(result: AnalysisResult) -> tuple[tuple[str, str], ...]:
    available = {
        (item.scenario, item.severity)
        for item in result.trajectories
    }
    if all(cell in available for cell in PRIMARY_CELLS):
        return PRIMARY_CELLS
    ordered = sorted(available)
    return tuple(ordered) if len(ordered) == 2 else ()


def _scenario_label(scenario: str, severity: str) -> str:
    fallback = f"{scenario.replace('_', ' ').title()} ({severity})"
    return SCENARIO_LABELS.get((scenario, severity), fallback)


def _has_unique_algorithm_configs(
    result: AnalysisResult,
    cells: Sequence[tuple[str, str]],
) -> bool:
    for scenario, severity in cells:
        seen: set[str] = set()
        for summary in result.cell_summaries:
            if summary["scenario"] != scenario or summary["severity"] != severity:
                continue
            algorithm = str(summary["algorithm"])
            if algorithm in seen:
                return False
            seen.add(algorithm)
    return True


def _convergence_figure(
    plt: Any,
    result: AnalysisResult,
    cells: Sequence[tuple[str, str]],
) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.85), sharex=True, sharey=True)
    for axis, (scenario, severity) in zip(axes, cells):
        available = {
            item.algorithm
            for item in result.trajectories
            if item.scenario == scenario and item.severity == severity
        }
        for algorithm in METHOD_ORDER:
            if algorithm not in available:
                continue
            curves = np.asarray(
                [
                    item.nmse_db
                    for item in result.trajectories
                    if item.scenario == scenario
                    and item.severity == severity
                    and item.algorithm == algorithm
                ],
                dtype=np.float64,
            )
            if curves.ndim != 2 or curves.shape[1] != result.protocol.evaluation_count:
                raise ValueError(f"invalid convergence matrix for {scenario}/{algorithm}")
            curves = np.maximum(curves, NMSE_AXIS_DB[0])
            iterations = np.arange(result.protocol.evaluation_count)
            median = np.median(curves, axis=0)
            lower, upper = np.quantile(curves, (0.25, 0.75), axis=0)
            color = METHOD_COLORS[algorithm]
            axis.fill_between(iterations, lower, upper, color=color, alpha=0.12, linewidth=0.0)
            axis.plot(iterations, median, color=color, label=METHOD_LABELS[algorithm])
        axis.axhline(
            result.protocol.convergence_nmse_db,
            color="#333333",
            linestyle="--",
            linewidth=0.9,
            label="Convergence threshold" if axis is axes[0] else None,
        )
        axis.set_title(_scenario_label(scenario, severity))
        axis.set_xlim(0, result.protocol.update_count)
        axis.set_ylim(*NMSE_AXIS_DB)
        axis.set_xticks(_iteration_ticks(result.protocol.update_count))
        axis.set_xlabel("Evaluation iteration k")
        axis.grid(True, which="major")
    axes[0].set_ylabel("Tracking NMSE (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )
    figure.text(
        0.5,
        0.005,
        (
            "Lines show paired-seed medians; bands show IQR. Algorithm-stop trajectories "
            "remain on the recorded fixed grid; display values below -60 dB are clipped."
        ),
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.88))
    return figure


def _primary_effect_figure(plt: Any, result: AnalysisResult) -> Any:
    if len(result.primary_comparisons) != len(PRIMARY_CELLS):
        raise ValueError("primary effect figure requires both preregistered main-cell comparisons")
    comparisons = {value["scenario"]: value for value in result.primary_comparisons}
    labels = ["AM/AM", "AM/PM"]
    ordered = [comparisons["amam"], comparisons["ampm"]]
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.7))
    panels = (
        (
            axes[0, 0],
            "auec_relative_reduction",
            100.0,
            "AUEC reduction (%)",
            (-100.0, 100.0),
            25.0,
        ),
        (
            axes[0, 1],
            "final_nmse_improvement_db",
            1.0,
            "Final NMSE improvement (dB)",
            (-20.0, 20.0),
            3.0,
        ),
        (
            axes[1, 0],
            "success_rate_difference_points",
            1.0,
            "Success-rate difference (points)",
            (-100.0, 100.0),
            20.0,
        ),
    )
    x = np.arange(2)
    for axis, key, scale, ylabel, limits, threshold in panels:
        estimates, lows, highs = _effect_values(ordered, key, scale)
        axis.errorbar(
            x,
            estimates,
            yerr=np.vstack((estimates - lows, highs - estimates)),
            fmt="o",
            color=METHOD_COLORS["model_lm_ilc"],
            capsize=3,
            linewidth=1.2,
        )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.axhline(threshold, color="#D55E00", linestyle="--", linewidth=0.9)
        axis.set_xticks(x, labels)
        axis.set_ylim(*limits)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y")

    safety_axis = axes[1, 1]
    for offset, key, label, color, threshold in (
        (-0.12, "divergence_rate_difference_points", "Divergence", "#D55E00", 10.0),
        (0.12, "constraint_rate_difference_points", "Constraint", "#009E73", 0.0),
    ):
        estimates, lows, highs = _effect_values(ordered, key, 1.0)
        safety_axis.errorbar(
            x + offset,
            estimates,
            yerr=np.vstack((estimates - lows, highs - estimates)),
            fmt="o",
            label=label,
            color=color,
            capsize=3,
            linewidth=1.2,
        )
        safety_axis.axhline(threshold, color=color, linestyle="--", linewidth=0.8, alpha=0.8)
    safety_axis.axhline(0.0, color="#555555", linewidth=0.8)
    safety_axis.set_xticks(x, labels)
    safety_axis.set_ylim(-100.0, 100.0)
    safety_axis.set_ylabel("Treatment - baseline (points)")
    safety_axis.grid(True, axis="y")
    safety_axis.legend(frameon=False, loc="best")
    figure.suptitle("Learned safeguarded LM versus linear ILC: paired 95% bootstrap CI", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def _endpoint_rate_figure(
    plt: Any,
    result: AnalysisResult,
    cells: Sequence[tuple[str, str]],
) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True)
    width = 0.36
    for axis, (scenario, severity) in zip(axes, cells):
        summaries = {
            value["algorithm"]: value
            for value in result.cell_summaries
            if value["scenario"] == scenario and value["severity"] == severity
        }
        algorithms = [algorithm for algorithm in METHOD_ORDER if algorithm in summaries]
        x = np.arange(len(algorithms))
        success = [summaries[algorithm]["success_rate_percent"] for algorithm in algorithms]
        safety = [summaries[algorithm]["safety_rate_percent"] for algorithm in algorithms]
        axis.bar(x - width / 2, success, width, label="Success", color="#0072B2")
        axis.bar(x + width / 2, safety, width, label="Safety", color="#009E73")
        axis.set_xticks(x, [METHOD_LABELS[value] for value in algorithms], rotation=30, ha="right")
        axis.set_title(_scenario_label(scenario, severity))
        axis.set_ylim(0.0, 100.0)
        axis.set_ylabel("Trajectory rate (%)")
        axis.grid(True, axis="y")
    axes[0].legend(frameon=False, loc="lower left")
    figure.tight_layout()
    return figure


def _variant_endpoint_figure(
    plt: Any,
    result: AnalysisResult,
    cells: Sequence[tuple[str, str]],
) -> Any:
    """Plot secondary-study endpoints without pooling distinct configurations."""

    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.2), sharex="col")
    for column, (scenario, severity) in enumerate(cells):
        summaries = [
            value
            for value in result.cell_summaries
            if value["scenario"] == scenario and value["severity"] == severity
        ]
        category_map: dict[tuple[Any, ...], str] = {}
        for summary in summaries:
            key, label = _variant_identity(summary)
            category_map[key] = label
        categories = sorted(category_map)
        category_index = {key: index for index, key in enumerate(categories)}
        algorithms = [
            algorithm
            for algorithm in METHOD_ORDER
            if any(value["algorithm"] == algorithm for value in summaries)
        ]
        offsets = (
            np.linspace(-0.28, 0.28, len(algorithms))
            if len(algorithms) > 1
            else np.zeros(1)
        )
        for algorithm, offset in zip(algorithms, offsets):
            selected = [value for value in summaries if value["algorithm"] == algorithm]
            x_values = [category_index[_variant_identity(value)[0]] + offset for value in selected]
            final_values = [
                max(float(value["median_final_nmse_db_for_statistics"]), NMSE_AXIS_DB[0])
                for value in selected
            ]
            success_values = [float(value["success_rate_percent"]) for value in selected]
            color = METHOD_COLORS[algorithm]
            axes[0, column].plot(
                x_values,
                final_values,
                marker="o",
                linestyle="none",
                color=color,
                label=METHOD_LABELS[algorithm],
            )
            axes[1, column].plot(
                x_values,
                success_values,
                marker="o",
                linestyle="none",
                color=color,
            )
        ticks = np.arange(len(categories))
        labels = [category_map[key] for key in categories]
        axes[0, column].set_title(_scenario_label(scenario, severity))
        axes[0, column].set_ylabel("Median final NMSE (dB)")
        axes[0, column].set_ylim(*NMSE_AXIS_DB)
        axes[0, column].grid(True, axis="y")
        axes[1, column].set_ylabel("Success rate (%)")
        axes[1, column].set_ylim(0.0, 100.0)
        axes[1, column].set_xticks(ticks, labels, rotation=35, ha="right")
        axes[1, column].grid(True, axis="y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=max(1, min(4, len(labels))),
        frameon=False,
    )
    figure.suptitle("Secondary-study endpoints by locked configuration", y=0.955)
    figure.text(
        0.5,
        0.002,
        "Display values below -60 dB are clipped; CSV/JSON tables retain the registered floor.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.91))
    return figure


def _variant_identity(summary: Mapping[str, Any]) -> tuple[tuple[Any, ...], str]:
    parameters = summary.get("parameters")
    if not isinstance(parameters, Mapping):
        return ("config", str(summary["config_hash"])), str(summary["config_hash"])[:8]
    if "candidate_index" in parameters:
        index = int(parameters["candidate_index"])
        return ("pilot", index), f"Candidate {index}"
    if "snr_db" in parameters and "capture_count" in parameters:
        snr = parameters["snr_db"]
        snr_text = "Inf" if str(snr).lower() == "inf" else f"{float(snr):g}"
        snr_order = {"inf": 0, "50": 1, "40": 2, "30": 3}.get(str(snr).lower(), 99)
        captures = int(parameters["capture_count"])
        return ("robustness", snr_order, captures), f"{snr_text} dB / {captures} cap."
    if "ablation" in parameters:
        order = (
            "raw_vjp",
            "no_ridge",
            "frozen_first_model",
            "three_iteration_replay",
            "unanchored_prediction",
            "no_trust_region",
            "complex64",
            "legacy_dynamic_calibration",
        )
        name = str(parameters["ablation"])
        index = order.index(name) if name in order else len(order)
        return ("ablation", index, name), name.replace("_", " ")
    if "model_orders" in parameters and "model_memory_depth" in parameters:
        orders = tuple(int(value) for value in parameters["model_orders"])
        depth = int(parameters["model_memory_depth"])
        maximum_order = max(orders)
        return ("mismatch", maximum_order, depth), f"Order {maximum_order} / M={depth}"
    return ("config", str(summary["config_hash"])), str(summary["config_hash"])[:8]


def _stress_diagnostic_figure(
    plt: Any,
    result: AnalysisResult,
    cells: Sequence[tuple[str, str]],
) -> Any:
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.0), sharex="col")
    for column, (scenario, severity) in enumerate(cells):
        summaries = {
            value["algorithm"]: value
            for value in result.cell_summaries
            if value["scenario"] == scenario and value["severity"] == severity
        }
        algorithms = [algorithm for algorithm in METHOD_ORDER if algorithm in summaries]
        x = np.arange(len(algorithms))
        rate_axis = axes[0, column]
        width = 0.36
        safety = [summaries[algorithm]["safety_rate_percent"] for algorithm in algorithms]
        guarded = [
            summaries[algorithm]["guarded_safe_stop_rate_percent"] for algorithm in algorithms
        ]
        rate_axis.bar(x - width / 2, safety, width, label="Safety", color="#009E73")
        rate_axis.bar(x + width / 2, guarded, width, label="Guarded stop", color="#E69F00")
        rate_axis.set_ylim(0.0, 100.0)
        rate_axis.set_ylabel("Trajectory rate (%)")
        rate_axis.set_title(_scenario_label(scenario, severity))
        rate_axis.grid(True, axis="y")

        gradient_axis = axes[1, column]
        identity_label_pending = True
        learned_label_pending = True
        for index, algorithm in enumerate(algorithms):
            identity_value = summaries[algorithm][
                "median_trajectory_identity_gradient_cosine"
            ]
            learned_value = summaries[algorithm][
                "median_trajectory_learned_gradient_cosine"
            ]
            if identity_value is None and learned_value is None:
                gradient_axis.text(
                    index,
                    -0.98,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=6.0,
                    color="#666666",
                )
            if identity_value is not None:
                gradient_axis.plot(
                    index - 0.08,
                    identity_value,
                    marker="x",
                    color="#333333",
                    label="Identity vs oracle" if identity_label_pending else None,
                    linestyle="none",
                )
                identity_label_pending = False
            if learned_value is not None:
                gradient_axis.plot(
                    index + 0.08,
                    learned_value,
                    marker="o",
                    color=METHOD_COLORS[algorithm],
                    label="Learned vs oracle" if learned_label_pending else None,
                    linestyle="none",
                )
                learned_label_pending = False
        gradient_axis.axhline(0.0, color="#555555", linewidth=0.8)
        gradient_axis.set_ylim(-1.05, 1.05)
        gradient_axis.set_ylabel("Median gradient cosine")
        gradient_axis.set_xticks(
            x,
            [METHOD_LABELS[value] for value in algorithms],
            rotation=30,
            ha="right",
        )
        gradient_axis.grid(True, axis="y")
    axes[0, 0].legend(frameon=False, loc="lower left")
    axes[1, 0].legend(frameon=False, loc="lower left")
    figure.suptitle("Stress-cell guard behavior and recorded gradient direction", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def _has_ampm_phase_figure_data(result: AnalysisResult) -> bool:
    if result.metadata.get("study") not in {"confirmatory", "smoke", "dynamic"}:
        return False
    return any(
        "ampm" in item.scenario
        and any(
            _finite_optional_metric(metric.get("low_power_phase_rmse_deg")) is not None
            for metric in item.iteration_metrics
        )
        for item in result.trajectories
    )


def _ampm_fixed_r0_phase_figure(plt: Any, result: AnalysisResult) -> Any:
    preferred = ("ampm", "135")
    available = sorted(
        {
            (item.scenario, item.severity)
            for item in result.trajectories
            if "ampm" in item.scenario
        }
    )
    cell = preferred if preferred in available else available[0]
    scenario, severity = cell
    figure, axis = plt.subplots(1, 1, figsize=(4.8, 3.15))
    for algorithm in METHOD_ORDER:
        members = [
            item
            for item in result.trajectories
            if item.scenario == scenario
            and item.severity == severity
            and item.algorithm == algorithm
        ]
        if not members:
            continue
        config_hashes = {item.config_hash for item in members}
        if len(config_hashes) != 1:
            continue
        curves = np.full(
            (len(members), result.protocol.evaluation_count),
            np.nan,
            dtype=np.float64,
        )
        for row_index, member in enumerate(members):
            usable_count = len(member.nmse_db) - member.imputed_evaluation_count
            for iteration in range(min(usable_count, len(member.iteration_metrics))):
                curves[row_index, iteration] = _finite_optional_metric(
                    member.iteration_metrics[iteration].get(
                        "low_power_phase_rmse_deg"
                    )
                )
        medians = np.full(result.protocol.evaluation_count, np.nan)
        lowers = np.full_like(medians, np.nan)
        uppers = np.full_like(medians, np.nan)
        coverage = np.count_nonzero(np.isfinite(curves), axis=0)
        for iteration in range(result.protocol.evaluation_count):
            values = curves[np.isfinite(curves[:, iteration]), iteration]
            if values.size:
                medians[iteration] = float(np.median(values))
                lowers[iteration], uppers[iteration] = np.quantile(values, (0.25, 0.75))
        if not np.any(np.isfinite(medians)):
            continue
        iterations = np.arange(result.protocol.evaluation_count)
        color = METHOD_COLORS[algorithm]
        axis.fill_between(iterations, lowers, uppers, color=color, alpha=0.12, linewidth=0.0)
        minimum_coverage = int(np.min(coverage))
        axis.plot(
            iterations,
            medians,
            color=color,
            label=f"{METHOD_LABELS[algorithm]} (min n={minimum_coverage}/{len(members)})",
        )
    axis.set_xlim(0, result.protocol.update_count)
    axis.set_ylim(0.0, 180.0)
    axis.set_xticks(_iteration_ticks(result.protocol.update_count))
    axis.set_xlabel("Evaluation iteration k")
    axis.set_ylabel("Low-power phase RMSE (degrees)")
    axis.set_title(
        f"{_scenario_label(scenario, severity)}; fixed r0={result.protocol.ampm_r0:g}"
    )
    axis.grid(True)
    axis.legend(frameon=False, fontsize=6.5)
    figure.text(
        0.5,
        0.003,
        (
            "Median and IQR use available paired trajectories at each k; "
            "missing values are not interpolated."
        ),
        ha="center",
        va="bottom",
        fontsize=6.3,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    return figure


def _finite_optional_metric(value: Any) -> float | None:
    if value is None or (
        isinstance(value, str) and value in {"nan", "inf", "-inf"}
    ):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _effect_values(
    comparisons: Sequence[Mapping[str, Any]],
    key: str,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimates = (
        np.asarray([value[key]["estimate"] for value in comparisons], dtype=np.float64)
        * scale
    )
    lows = (
        np.asarray([value[key]["confidence_low"] for value in comparisons], dtype=np.float64)
        * scale
    )
    highs = (
        np.asarray([value[key]["confidence_high"] for value in comparisons], dtype=np.float64)
        * scale
    )
    if not np.all(np.isfinite(np.concatenate((estimates, lows, highs)))):
        raise ValueError(f"non-finite bootstrap result for {key}")
    return estimates, lows, highs


def _iteration_ticks(update_count: int) -> list[int]:
    candidates = (0, 1, 2, 5, 10, 20, 30)
    ticks = [value for value in candidates if value <= update_count]
    if update_count not in ticks:
        ticks.append(update_count)
    return sorted(set(ticks))


def _save_figure(figure: Any, destination: Path, result: AnalysisResult, stem: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset_hash = str(result.metadata.get("dataset_hash", "unknown"))
    common = {
        "Title": stem.replace("_", " ").title(),
        "Creator": "remote-dpd experiments.plot_results",
        "Subject": (
            "Preregistered PA-model-backpropagation ILC simulation; "
            f"dataset SHA-256 {dataset_hash}"
        ),
        "Keywords": "DPD, ILC, PA model, paired bootstrap, simulation",
    }
    if destination.suffix.lower() == ".pdf":
        metadata = {
            **common,
            "CreationDate": FIXED_CREATION_TIME,
            "ModDate": FIXED_CREATION_TIME,
        }
    else:
        metadata = {
            "Title": common["Title"],
            "Author": common["Creator"],
            "Description": common["Subject"],
            "Software": common["Creator"],
            "Creation Time": FIXED_CREATION_TIME.isoformat(),
        }
    figure.savefig(
        destination,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=metadata,
    )


__all__ = [
    "FIGURE_DPI",
    "METHOD_COLORS",
    "METHOD_LABELS",
    "METHOD_ORDER",
    "NMSE_AXIS_DB",
    "export_publication_results",
    "plot_publication_figures",
]


if __name__ == "__main__":
    raise SystemExit(main())
