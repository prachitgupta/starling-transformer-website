#!/usr/bin/env python3
"""Import the finalized Case B / GPT simulation evidence for the project page."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = Path(
    "/home/prachit/Desktop/starling_final_testing/src/llm_vision_planner/"
    "simulation/results/benchmark"
)
DEFAULT_REPORT = Path(
    "/home/prachit/Desktop/starling_final_testing/src/papers/report/simulation.pdf"
)

SURFACE = "#fbfaf7"
INK = "#16201c"
MUTED = "#63706a"
GRID = "#dfe6e1"
BLUE = "#2764d8"
ORANGE = "#ea6a47"
GREEN = "#0a8f68"
PURPLE = "#7756d8"


def method_label(name: str) -> str:
    labels = {
        "corridor_aware_residual_beta_2": "conditional residual + regret (ours)",
        "single_residual": "residual only (ablation)",
        "direct_distance": "trajectory CP, max score",
        "direct_position": "trajectory CP on position",
        "two_score": "two scores (ablation)",
        "weighted_max": "weighted-max CP",
        "ccm_assumed_gaussian": "CCM, Gaussian assumption",
        "ccm_assumed_worst_case": "CCM, worst-case bound",
        "single_residual_selected_expert": "selected-expert residual (ablation)",
    }
    return labels.get(name, name.replace("_", " "))


def style(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=0, colors=MUTED)


def proposed_method(table: dict) -> str:
    return next(name for name in table if name.startswith("corridor_aware"))


def entries(results: dict, method: str) -> list[dict]:
    regimes = results["config"]["regimes"]
    values = [results["cases"]["B"][regime]["gpt"][method] for regime in regimes]
    values.append(results["shifted_start"]["B"]["gpt"][method])
    return values


def coverage_plot(results: dict, output: Path) -> None:
    regimes = results["config"]["regimes"]
    table = results["cases"]["B"][regimes[0]]["gpt"]
    method = proposed_method(table)
    values = entries(results, method)
    labels = ["Gaussian", "Heavy-tailed", "State-dependent", "Shifted start"]
    coverage = np.array([100 * value["coverage"] for value in values])
    lower = coverage - np.array([100 * value["coverage_p05"] for value in values])
    upper = np.array([100 * value["coverage_p95"] for value in values]) - coverage
    radii = [value["radius_mean_m"] for value in values]

    figure, axis = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(labels))
    axis.errorbar(
        x,
        coverage,
        yerr=[lower, upper],
        fmt="o",
        color=BLUE,
        markersize=8,
        capsize=4,
        linewidth=2,
        label="5th-95th percentile over 1,000 scene-level splits",
    )
    axis.axhline(90, color=INK, linestyle="--", linewidth=1.2)
    axis.text(3.42, 90.5, "90% target", color=MUTED, ha="right", fontsize=9)
    for xx, yy, radius in zip(x, coverage, radii):
        axis.text(
            xx,
            99.15,
            f"{yy:.1f}%\n{radius:.2f} m mean radius",
            ha="center",
            va="center",
            fontsize=9,
            color=INK,
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(84, 101)
    axis.set_ylabel("coverage of test episodes [%]")
    axis.set_title(
        "Case B · GPT selector · conditional residual score",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    axis.legend(frameon=False, loc="lower left", fontsize=9)
    style(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=180, facecolor=SURFACE)
    plt.close(figure)


def tightness_plot(results: dict, output: Path) -> None:
    regimes = results["config"]["regimes"]
    table = results["cases"]["B"]
    methods = [
        proposed_method(table[regimes[0]]["gpt"]),
        "direct_distance",
        "direct_position",
        "two_score",
        "single_residual_selected_expert",
    ]
    colours = [BLUE, ORANGE, GREEN, PURPLE, "#8a918d"]
    positions = np.arange(len(methods))[::-1]
    width = 0.22

    figure, axis = plt.subplots(figsize=(9.2, 5.1))
    for index, regime in enumerate(regimes):
        values = [table[regime]["gpt"][method] for method in methods]
        radii = [value["radius_mean_m"] for value in values]
        y = positions + (index - 1) * width
        axis.barh(y, radii, height=width, color=colours[index], label=regime.replace("_", " "))
        for yy, radius, value in zip(y, radii, values):
            axis.text(
                radius + 0.035,
                yy,
                f"{radius:.2f} m · {100 * value['coverage']:.0f}%",
                va="center",
                fontsize=8,
                color=INK,
            )
    axis.set_yticks(positions, [method_label(method) for method in methods])
    axis.set_xlabel("mean tube radius around the Case B reference [m]")
    axis.set_xlim(0, 3.45)
    axis.set_title(
        "Case B · GPT selector · tube tightness",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.grid(axis="y", visible=False)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(length=0, colors=MUTED)
    axis.legend(frameon=False, loc="upper right", fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def finite(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


def export_data(results: dict, certificate: dict, output: Path) -> None:
    regimes = results["config"]["regimes"]
    table = results["cases"]["B"]
    method = proposed_method(table[regimes[0]]["gpt"])
    conditions = {
        regime: table[regime]["gpt"][method]
        for regime in regimes
    }
    conditions["shifted_start"] = results["shifted_start"]["B"]["gpt"][method]
    payload = {
        "case": "B",
        "expert_selector": "gpt",
        "score": "max(sup_t [D_dot + lambda D]_+, beta lambda L / sqrt(w))",
        "beta": results["beta"],
        "target_coverage": 1 - results["config"]["alpha"],
        "splits": results["config"]["splits"],
        "preferences": results["preferences"]["B"],
        "held_out_rows": results["dataset"]["rows_case_B"],
        "conditions": conditions,
        "certificate": {
            key: certificate[key]
            for key in (
                "alpha",
                "calibration_size",
                "rank",
                "quantile",
                "contraction_rate",
                "position_block_w",
                "record_step_s",
                "steady_state_radius_m",
            )
        },
        "evidence_scope": (
            "Offline 4D-car simulation. Transformer, HRRT* routes, GPT labels, and scenes are real; "
            "vehicle dynamics, disturbances, and rollouts are simulated."
        ),
    }
    output.write_text(json.dumps(finite(payload), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    source = args.benchmark_dir.resolve()
    results = json.loads((source / "benchmark.json").read_text())
    certificate = json.loads((source / "certificate_caseB_gpt.json").read_text())
    if certificate.get("case") != "B" or certificate.get("expert_selector") != "gpt":
        raise ValueError("Expected the finalized Case B certificate with the GPT expert selector")

    image_dir = ROOT / "assets/images/simulation"
    paper_dir = ROOT / "assets/papers"
    image_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "score_histogram_caseB_gpt.png",
        "scene_tube_caseB_gpt.png",
        "literature_comparison_caseB_gpt.png",
        "hardware_coverage_dummy.png",
    ):
        shutil.copy2(source / name, image_dir / name)
    coverage_plot(results, image_dir / "coverage_caseB_gpt.png")
    tightness_plot(results, image_dir / "tightness_caseB_gpt.png")
    export_data(results, certificate, ROOT / "data/case_b_gpt.json")
    shutil.copy2(args.report.resolve(), paper_dir / "simulation.pdf")
    print("Imported finalized Case B / GPT simulation evidence.")


if __name__ == "__main__":
    main()
