#!/usr/bin/env python3
"""Build website figures from the saved simulation data and hardware certificates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "static" / "images"
DATA = json.loads((ROOT / "animations" / "framework_video_data.json").read_text())
PLANNER = Path("/home/prachit/Desktop/starling_final_testing/src/llm_vision_planner")
BENCHMARK = PLANNER / "simulation/results/benchmark/benchmark.json"
CERT_ROOT = PLANNER / "fine_tuning/datasets/person_split_cp/hardware_transformer_poc_synthetic/poc_delta_0_20"

COLORS = {
    "expert": "#16a65a",
    "plan": "#f97316",
    "flown": "#2563eb",
    "tube": "#13a8a8",
    "obstacle": "#94a3b8",
    "grid": "#dbe4ee",
}


def style_axis(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=COLORS["grid"], lw=0.8, alpha=0.7)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    for spine in ax.spines.values():
        spine.set_color("#a8b3c2")


def draw_episode(ax, episode, upto=None, show_legend=True):
    expert = np.asarray(episode["expert"], float)[:, :2]
    plan = np.asarray(episode["plan"], float)[:, :2]
    flown = np.asarray(episode["flown"], float)[:, :2]
    radius = np.asarray(episode["radius"], float)
    k = len(flown) - 1 if upto is None else int(np.clip(upto, 1, len(flown) - 1))

    for obstacle in episode["obstacles"]:
        lo, hi = np.asarray(obstacle["lo"]), np.asarray(obstacle["hi"])
        ax.add_patch(Rectangle(lo, *(hi - lo), fc=COLORS["obstacle"], ec="#64748b", alpha=0.45, lw=1.1))

    # A sparse set of circles reads clearly and preserves the exact time-varying radii.
    for j in np.linspace(0, k, min(24, k + 1), dtype=int):
        ax.add_patch(Circle(expert[j], radius[j], fc=COLORS["tube"], ec="none", alpha=0.035))
    ax.plot(expert[:, 0], expert[:, 1], color=COLORS["expert"], lw=2.7, label="Language-conditioned expert")
    ax.plot(plan[:, 0], plan[:, 1], color=COLORS["plan"], lw=2.3, ls="--", label="Transformer reference")
    ax.plot(flown[: k + 1, 0], flown[: k + 1, 1], color=COLORS["flown"], lw=2.3, label="Closed-loop state")
    ax.add_patch(Circle(expert[k], radius[k], fill=False, ec=COLORS["tube"], lw=1.8, label="90% conformal tube"))
    ax.scatter(*episode["start"], s=52, c="#111827", zorder=5)
    ax.scatter(*episode["goal"], s=120, marker="*", c="#eab308", edgecolor="#111827", linewidth=0.4, zorder=5)
    style_axis(ax)
    if show_legend:
        ax.legend(loc="best", fontsize=8.5, frameon=True)


def preference_figures():
    episodes = DATA["episodes"][:2]
    names = ["simulation_shortest.png", "simulation_balanced.png"]
    titles = ["Quickest collision-free route", "Balanced time and clearance"]
    for episode, name, title in zip(episodes, names, titles):
        fig, ax = plt.subplots(figsize=(7.1, 5.4), dpi=180)
        draw_episode(ax, episode)
        ax.set_title(title, fontsize=15, weight="bold", pad=10)
        ax.text(0.02, 0.02, episode["sample_id"], transform=ax.transAxes, fontsize=8, color="#64748b")
        fig.tight_layout()
        fig.savefig(IMAGES / name, bbox_inches="tight")
        plt.close(fig)


def simulation_gif():
    episode = DATA["episodes"][1]
    frames = []
    frame_dir = IMAGES / ".simulation_frames"
    frame_dir.mkdir(exist_ok=True)
    n = len(episode["flown"])
    for index, frac in enumerate(np.linspace(0.02, 1.0, 42)):
        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=100)
        k = int(frac * (n - 1))
        draw_episode(ax, episode, upto=k, show_legend=index == 0)
        ax.set_title("Closed-loop tracking with a calibrated contraction tube", fontsize=14, weight="bold")
        ax.text(0.98, 0.03, f"t = {episode['time'][k]:.1f} s   ·   r(t) = {episode['radius'][k]:.2f} m",
                transform=ax.transAxes, ha="right", fontsize=9.5, color="#0f766e")
        fig.tight_layout()
        path = frame_dir / f"{index:03d}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frames.append(Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE))
    frames[0].save(IMAGES / "simulation_tracking.gif", save_all=True, append_images=frames[1:], duration=110, loop=0,
                   disposal=2, optimize=True)
    for frame in frames:
        frame.close()
    for path in frame_dir.glob("*.png"):
        path.unlink()
    frame_dir.rmdir()


def cleaned_benchmark():
    data = json.loads(BENCHMARK.read_text())
    case = data["cases"]["B"]
    regimes = [("Gaussian", case["gaussian"]["gpt"]),
               ("Heavy-tailed", case["heavy_tailed"]["gpt"]),
               ("State-dependent", case["state_dependent"]["gpt"]),
               ("Shifted start", data["shifted_start"]["B"]["gpt"])]
    methods = [
        ("corridor_aware_residual_beta_2", "Ours", "o", "#2563eb"),
        ("direct_distance", "Max-score CP\nLindemann et al. [1]", "s", "#f97316"),
        ("weighted_max", "Weighted-max CP\nCleaveland et al. [2]", "^", "#16a65a"),
        ("per_step_union", "Per-step CP\nLindemann et al. [1]", "v", "#7c3aed"),
        ("ccm_assumed_gaussian", "Gaussian CCM\nSingh et al. [3]", "D", "#ef4444"),
        ("ccm_assumed_worst_case", "Worst-case CCM\nManchester–Slotine [4]", "P", "#8b6f47"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), dpi=170, sharey="row")
    for col, (title, values) in enumerate(regimes):
        x = np.arange(len(methods))
        for row, field in enumerate(("coverage", "radius_mean_m")):
            ax = axes[row, col]
            for j, (key, _, marker, color) in enumerate(methods):
                entry = values[key]
                y = entry[field] * (100 if field == "coverage" else 1)
                if not np.isfinite(y):
                    ax.text(j, 8.7, "∞", ha="center", va="center", fontsize=15, color=color)
                    continue
                if field == "coverage":
                    low, high = entry["coverage_p05"] * 100, entry["coverage_p95"] * 100
                    ax.errorbar(j, y, yerr=[[max(0, y - low)], [max(0, high - y)]], marker=marker, ms=7, color=color,
                                capsize=3, lw=1.4)
                else:
                    ax.scatter(j, y, marker=marker, s=58, color=color, edgecolor="#263238", linewidth=0.45)
            ax.set_title(title if row == 0 else "", fontsize=13.5, weight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([m[1] for m in methods], rotation=46, ha="right", fontsize=7.5)
            ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
            ax.set_axisbelow(True)
        axes[0, col].axhline(90, color="#111827", ls="--", lw=1.2)
        axes[0, col].set_ylim(55, 102)
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylim(0.35, 11)
    axes[0, 0].set_ylabel("Empirical coverage [%]", fontsize=11.5)
    axes[1, 0].set_ylabel("Mean tube radius [m]", fontsize=11.5)
    axes[0, 0].text(-0.4, 91.3, "target 90%", fontsize=8.5)
    fig.suptitle("Benchmark against existing safety-tube methods", fontsize=17, weight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(IMAGES / "simulation_benchmark_clean.png", bbox_inches="tight")
    plt.close(fig)


def hardware_rollouts(path):
    """Deterministic illustrative rollouts shared by both certificate panels."""
    tangent = np.gradient(path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    t = np.linspace(0, 1, len(path))
    envelopes = [
        0.34 * np.sin(np.pi * t) ** 2,
        -0.48 * np.sin(np.pi * t) ** 2,
        0.68 * np.sin(np.pi * t) ** 2 * (0.78 + 0.22 * np.sin(3 * np.pi * t)),
        -1.16 * np.sin(np.pi * t) ** 2,
    ]
    return [path + offset[:, None] * normal for offset in envelopes]


def hardware_certificate_panel(mode, title, out_name, color):
    cert = json.loads((CERT_ROOT / mode / "split_certificate.json").read_text())
    radius = float(cert["calibrated_radius_m"])
    t = np.linspace(0, 1, 260)
    path = np.column_stack([-2.3 + 4.6 * t, -0.65 + 1.25 * t + 0.3 * np.sin(2 * np.pi * t)])
    tangent = np.gradient(path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    upper = path + radius * normal
    lower = path - radius * normal
    tube_polygon = np.vstack([upper, lower[::-1]])
    rollouts = hardware_rollouts(path)
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=180)
    ax.fill(tube_polygon[:, 0], tube_polygon[:, 1], color=color, alpha=0.16, zorder=0)
    ax.add_patch(Circle(path[0], radius, fc=color, ec="none", alpha=0.16, zorder=0))
    ax.add_patch(Circle(path[-1], radius, fc=color, ec="none", alpha=0.16, zorder=0))
    ax.plot(path[:, 0], path[:, 1], color="#111827", lw=2.4, label="Reference path")
    for index, rollout in enumerate(rollouts):
        ax.plot(rollout[:, 0], rollout[:, 1], color="#16a34a", lw=1.75, alpha=0.9,
                label="Trajectory rollouts" if index == 0 else None)
    ax.add_patch(Circle(path[len(path)//2], radius, fill=False, ec=color, lw=2.0, label=f"Radius = {radius:.3f} m"))
    ax.scatter(*path[0], color="#111827", s=45, zorder=5)
    ax.scatter(*path[-1], color="#eab308", marker="*", s=120, edgecolor="#111827", linewidth=0.4, zorder=5)
    ax.set_xlim(-3.35, 3.35); ax.set_ylim(-2.35, 2.35)
    style_axis(ax)
    ax.set_title(title, fontsize=15, weight="bold", color=color)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.02, f"δ = {cert['delta']:.2f}  ·  nominal coverage 80%  ·  n = {cert['calibration_sample_count']}",
            transform=ax.transAxes, fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(IMAGES / out_name, bbox_inches="tight")
    plt.close(fig)


def hardware_figures():
    hardware_certificate_panel("perception_unaware", "Perception-unaware certificate",
                               "hardware_perception_unaware.png", "#2563eb")
    hardware_certificate_panel("perception_aware", "Perception-aware certificate",
                               "hardware_perception_aware.png", "#d946ef")


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    preference_figures()
    simulation_gif()
    cleaned_benchmark()
    hardware_figures()
    print("Built website result assets in", IMAGES)
