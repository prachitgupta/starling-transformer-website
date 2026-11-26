#!/usr/bin/env python3
"""Framework video 2: the hardware case with environmental (perception) uncertainty.

    python video2.py --videos <dir with the source recordings> --out framework_hardware.mp4
    python video2.py --stills <dir>

Every drawn scene here is a schematic illustration (not measured data); the recorded clips are real.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from engine import (BG, BLUE, DIM, FG, GREEN, MUTED, ORANGE, PANEL, PURPLE, RED, TEAL, YELLOW, Clip, Movie, arrow, box,
                    canvas, draw_path, draw_tube, drone_icon, heading, label, pulse, ramp, still)

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)


def panel(ax, x0, y0, x1, y1, alpha=1.0, title=None, edge=DIM):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=PANEL, edgecolor=matplotlib.colors.to_rgba(edge, alpha),
                           lw=1.2, alpha=None, zorder=0))
    if title:
        label(ax, x0, y1 + 0.25, title, MUTED, 11.5, alpha)


def obstacle(ax, centre, half=0.32, colour=PURPLE, alpha=1.0, dashed=False, text=None):
    centre = np.asarray(centre)
    ax.add_patch(Rectangle(centre - half, 2 * half, 2 * half, facecolor=matplotlib.colors.to_rgba(colour, 0.3 * alpha),
                           edgecolor=matplotlib.colors.to_rgba(colour, alpha), lw=1.6, linestyle="--" if dashed else "-", zorder=2))
    if text:
        label(ax, centre[0], centre[1] + half + 0.2, text, colour, 10.5, alpha, ha="center")


def route(start, goal, centre, side=1.0, clearance=0.75, n=160):
    """A smooth path from start to goal passing the obstacle on one side."""
    start, goal, centre = map(np.asarray, (start, goal, centre))
    direction = (goal - start) / np.linalg.norm(goal - start)
    normal = np.array([-direction[1], direction[0]])
    via = centre + side * clearance * normal
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 2 * start + 2 * u * (1 - u) * (2 * via - 0.5 * (start + goal)) + u ** 2 * goal


# =============================================================================== title
def scene_title(fig, s):
    ax = canvas(fig)
    a = ramp(s, 0.2, 1.2)
    label(ax, 8, 5.65, "Perception Uncertainty-Aware Tracking", FG, 30, a, ha="center", weight="bold")
    label(ax, 8, 4.7, "of Language-Conditioned Oracle Motion Plans", FG, 27, a, ha="center", weight="bold")
    label(ax, 8, 3.6, "Hardware framework  ·  Starling 2", TEAL, 19, ramp(s, 1.0, 2.0), ha="center")


# =============================================================================== exchangeability
CARDS = [(1.3 + 1.75 * (k % 8), 5.1 - 1.55 * (k // 8)) for k in range(16)]
MISSED = {2, 5, 9, 12, 14}


def scene_exchangeability(fig, s):
    ax = canvas(fig)
    heading(ax, "Split conformal prediction needs exchangeability",
            "calibration and deployment scenes are sampled from the same distribution", ramp(s, 0, 0.6))
    drop = ramp(s, 6.0, 7.5)
    for k, (x, y) in enumerate(CARDS):
        a = ramp(s, 0.5 + 0.08 * k, 1.1 + 0.08 * k)
        missed = k in MISSED
        flag = ramp(s, 3.0, 3.8) if missed else 0.0
        fade = a * (1.0 - 0.85 * drop) if missed else a
        colour = matplotlib.colors.to_hex((1 - flag) * np.array(matplotlib.colors.to_rgb(DIM)) + flag * np.array(matplotlib.colors.to_rgb(RED)))
        ax.add_patch(Rectangle((x - 0.65, y - 0.55), 1.3, 1.1, facecolor=PANEL,
                               edgecolor=matplotlib.colors.to_rgba(colour, fade), lw=1.5))
        yy = y
        ox, oy = x + 0.25 * np.sin(3.1 * k), yy + 0.18 * np.cos(2.3 * k)
        ax.add_patch(Rectangle((ox - 0.13, oy - 0.13), 0.26, 0.26, facecolor=matplotlib.colors.to_rgba(PURPLE, 0.75 * fade), edgecolor="none"))
        ax.plot([x - 0.5, x + 0.5], [yy - 0.38, yy + 0.38], color=GREEN, lw=1.2, alpha=0.7 * fade)
        if flag > 0.01:
            label(ax, x, yy - 0.75, "missed", RED, 9.5, flag * (1 - 0.85 * drop), ha="center")
    label(ax, 15.0, 5.1, "calibration", MUTED, 11, ramp(s, 1.0, 1.8), ha="right", rotation=90)
    label(ax, 0.7, 2.35, "1.  The onboard detector sometimes misses the obstacle.", RED, 14.5, ramp(s, 3.0, 3.8), weight="bold")
    label(ax, 0.7, 1.7, "2.  Dropping those scenes breaks exchangeability.",
          FG, 14.5, ramp(s, 6.0, 6.8), weight="bold")
    label(ax, 0.7, 1.05, "3.  So the coverage guarantee of split conformal prediction is no longer valid.",
          ORANGE, 14.5, ramp(s, 8.6, 9.4), weight="bold")
    label(ax, 0.7, 0.4, "Fix:  assume 100 % detectability in our restricted workspace.",
          GREEN, 14.5, ramp(s, 10.8, 11.6), weight="bold")


# =============================================================================== workspace grid
GX, GY = np.meshgrid(np.arange(12), np.arange(7))
GRID = np.column_stack([GX.ravel(), GY.ravel()])
RATE = np.clip(1.25 - 0.045 * ((GRID[:, 0] - 5.5) ** 2 * 0.55 + (GRID[:, 1] - 3.0) ** 2 * 1.6) + 0.08 * RNG.standard_normal(len(GRID)), 0, 1)
RELIABLE = (np.abs(GRID[:, 0] - 5.5) <= 2.6) & (np.abs(GRID[:, 1] - 3) <= 2.1)
ORDER = RNG.permutation(len(GRID))
CELL = 0.62
G0 = np.array([1.2, 1.35])


def cell_xy(index):
    return G0 + CELL * GRID[index]


def workspace_rect():
    points = np.array([cell_xy(i) for i in np.flatnonzero(RELIABLE)])
    return points.min(0) - 0.3, points.max(0) + 0.3


def scene_grid(fig, s):
    ax = canvas(fig)
    heading(ax, "Finding the workspace", "place the obstacle at every grid point; keep the points detected in every trial", ramp(s, 0, 0.6))
    panel(ax, 0.7, 0.85, 8.6, 5.6, ramp(s, 0.2, 0.8), "survey grid (top view)")
    scan = ramp(s, 1.2, 7.2)
    for rank, index in enumerate(ORDER):
        x, y = cell_xy(index)
        a = ramp(s, 0.3 + 0.004 * rank, 0.9 + 0.004 * rank)
        tested = np.clip(scan * len(GRID) - rank, 0, 1)
        target = GREEN if RELIABLE[index] else RED
        colour = matplotlib.colors.to_hex((1 - tested) * np.array(matplotlib.colors.to_rgb("#6b7480")) + tested * np.array(matplotlib.colors.to_rgb(target)))
        size = 0.12 + 0.07 * pulse(tested, 0.0, 1.0)
        fade = 1.0 - 0.7 * ramp(s, 8.2, 9.2) * (not RELIABLE[index])
        ax.add_patch(Circle((x, y), size, color=colour, alpha=a * fade, zorder=3))
    final = ramp(s, 8.2, 9.4)
    lo, hi = workspace_rect()
    if final > 0.01:
        ax.add_patch(Rectangle(lo, *(hi - lo), facecolor=matplotlib.colors.to_rgba(GREEN, 0.14 * final),
                               edgecolor=matplotlib.colors.to_rgba(GREEN, final), lw=2.4, zorder=2))
        label(ax, (lo[0] + hi[0]) / 2, hi[1] + 0.25, "finalized workspace", GREEN, 13, final, ha="center", weight="bold")
    drone_icon(ax, 4.6, 1.0, size=0.2, alpha=ramp(s, 0.4, 1.0))
    label(ax, 5.0, 1.0, "camera", MUTED, 10.5, ramp(s, 0.4, 1.0))

    x = 9.3
    for k, (colour, text, t0) in enumerate([("#6b7480", "not surveyed yet", 0.6), (GREEN, "detected in 100 % of trials", 1.6),
                                            (RED, "at least one missed detection", 1.6)]):
        a = ramp(s, t0, t0 + 0.8)
        ax.add_patch(Circle((x + 0.15, 5.9 - 0.6 * k), 0.13, color=colour, alpha=a))
        label(ax, x + 0.5, 5.9 - 0.6 * k, text, FG, 13, a)
    done = int(scan * len(GRID))
    label(ax, x, 3.7, f"surveyed  {done} / {len(GRID)}", MUTED, 13, ramp(s, 1.2, 1.8))
    label(ax, x, 3.1, f"reliable  {int(RELIABLE[ORDER[:done]].sum())}", GREEN, 13, ramp(s, 1.2, 1.8))
    label(ax, x, 2.2, "Inside the green region no scene is ever", FG, 13, final)
    label(ax, x, 1.75, "discarded, so calibration and deployment", FG, 13, final)
    label(ax, x, 1.3, "scenes stay exchangeable.", FG, 13, final)
    label(ax, 0.7, 0.35, "schematic illustration of the survey procedure", DIM, 10, 1.0)


# =============================================================================== training data
N_SCENES = 9
SCENE_OBSTACLES = [cell_xy(i) + 0.12 * RNG.standard_normal(2) for i in RNG.choice(np.flatnonzero(RELIABLE), N_SCENES)]
SCENE_STARTS = [np.array([1.3, 1.6 + 3.0 * RNG.random()]) for _ in range(N_SCENES)]
SCENE_GOALS = [np.array([8.0, 1.5 + 3.4 * RNG.random()]) for _ in range(N_SCENES)]
SCENE_SIDES = [1, -1, 1, 1, -1, -1, 1, -1, 1]


def scene_training(fig, s):
    ax = canvas(fig)
    heading(ax, "Training data: synthetic scenes inside the workspace",
            "obstacle jittered over the reliable region, goal ablated, HRRT* expert labels every scene", ramp(s, 0, 0.6))
    a = ramp(s, 0.2, 0.8)
    panel(ax, 0.7, 0.85, 8.6, 5.6, a, "generated scene (top view)")
    lo, hi = workspace_rect()
    ax.add_patch(Rectangle(lo, *(hi - lo), facecolor=matplotlib.colors.to_rgba(GREEN, 0.08 * a),
                           edgecolor=matplotlib.colors.to_rgba(GREEN, 0.6 * a), lw=1.4, linestyle="--", zorder=1))
    period = 1.25
    index = min(int(max(s - 1.0, 0.0) / period), N_SCENES - 1)
    local = (max(s - 1.0, 0.0) - index * period) / period if s < 1.0 + N_SCENES * period else 1.0
    for k in range(index):                                                      # ghosts of earlier samples
        obstacle(ax, SCENE_OBSTACLES[k], 0.28, PURPLE, 0.22)
        ax.plot(*SCENE_GOALS[k], marker="*", color=YELLOW, markersize=10, alpha=0.3)
        path = route(SCENE_STARTS[k], SCENE_GOALS[k], SCENE_OBSTACLES[k], SCENE_SIDES[k])
        ax.plot(path[:, 0], path[:, 1], color=GREEN, lw=1.0, alpha=0.22)
    if s > 1.0:
        obstacle(ax, SCENE_OBSTACLES[index], 0.28, PURPLE, ramp(local, 0.0, 0.2), text="obstacle")
        ax.add_patch(Circle(SCENE_STARTS[index], 0.1, color=FG, alpha=ramp(local, 0.0, 0.2), zorder=5))
        ax.plot(*SCENE_GOALS[index], marker="*", color=YELLOW, markersize=17, alpha=ramp(local, 0.1, 0.3), markeredgecolor=BG, zorder=5)
        draw_path(ax, route(SCENE_STARTS[index], SCENE_GOALS[index], SCENE_OBSTACLES[index], SCENE_SIDES[index]),
                  ramp(local, 0.25, 0.9), GREEN, 2.6, zorder=4)
    x = 9.3
    label(ax, x, 5.9, f"scene {index + 1 if s > 1.0 else 0:02d}  →  dataset", FG, 15, a, weight="bold")
    for k, (colour, text) in enumerate([(PURPLE, "obstacle position: sampled over the reliable region"),
                                        (YELLOW, "goal: ablated across the far side"),
                                        (GREEN, "label: language-conditioned HRRT* expert"),
                                        (MUTED, "language: goal sentence $L_1$ + preference $L_2$")]):
        ax.add_patch(Circle((x + 0.12, 5.1 - 0.55 * k), 0.1, color=colour, alpha=a))
        label(ax, x + 0.45, 5.1 - 0.55 * k, text, FG, 12, a)
    a2 = ramp(s, 6.0, 7.0)
    box(ax, 10.9, 2.2, 3.0, 1.2, "$D_{train}$,  $D_{val}$\nsynthetic", BLUE, a2, 13)
    box(ax, 14.25, 2.2, 3.0, 1.2, "$D_{cal}$,  $D_{test}$\nmeasured on the drone", ORANGE, ramp(s, 8.0, 9.0), 12)
    label(ax, x, 1.1, "same scene distribution for all four sets — only the source of the", MUTED, 11.5, ramp(s, 9.0, 10.0))
    label(ax, x, 0.7, "obstacle box differs (simulated vs. perceived by the real camera)", MUTED, 11.5, ramp(s, 9.0, 10.0))
    label(ax, 0.7, 0.35, "schematic illustration of the generation procedure", DIM, 10, 1.0)


# =============================================================================== hardware formulation
def scene_formulation(fig, s):
    ax = canvas(fig)
    heading(ax, "Hardware problem formulation", "contraction between the transformer plan and the expert; ideal tracking by PX4", ramp(s, 0, 0.6))
    a = ramp(s, 0.2, 0.9)
    panel(ax, 0.7, 3.5, 7.3, 7.2, a)
    centre = np.array([4.0, 5.25])
    obstacle(ax, centre, 0.42, PURPLE, a)
    expert = route([1.2, 4.3], [6.8, 6.3], centre, 1.0, 1.0)
    u = np.linspace(0, 1, len(expert))
    normal = np.gradient(expert, axis=0)
    normal = np.column_stack([-normal[:, 1], normal[:, 0]]) / np.linalg.norm(normal, axis=1, keepdims=True)
    plan = expert + (8.0 * u * np.exp(-5.5 * u))[:, None] * normal
    draw_path(ax, expert, ramp(s, 0.8, 2.4), GREEN, 3.0, zorder=4)
    f = ramp(s, 2.6, 5.4)
    tip = draw_path(ax, plan, f, YELLOW, 2.4, style=(0, (4, 2.5)), zorder=5)
    k = int(f * (len(plan) - 1))
    ax.plot(plan[: k + 1, 0], plan[: k + 1, 1], color=BLUE, lw=5.5, alpha=0.3 * ramp(s, 2.6, 3.0), zorder=3, solid_capstyle="round")
    if tip is not None and 0 < f < 1:
        drone_icon(ax, *tip)
    label(ax, 6.6, 6.75, r"$x_d$", GREEN, 15, ramp(s, 0.8, 1.6))
    label(ax, 1.5, 5.5, r"$\hat{x}_d$", YELLOW, 15, ramp(s, 2.6, 3.4))
    label(ax, 0.9, 3.75, r"flown path  $x\approx\hat{x}_d$  (blue halo)", BLUE, 11.5, ramp(s, 2.8, 3.6))

    x = 7.8
    label(ax, x, 6.95, "ideal planar double integrator", FG, 14, ramp(s, 1.0, 1.8), weight="bold")
    label(ax, x, 6.25, r"$\dot{x}=A\,x+B\,u,\qquad x=(p,\dot{p}),\ \ u=\ddot{p}$", FG, 17, ramp(s, 1.4, 2.2))
    label(ax, x, 5.4, "ideal tracking", FG, 14, ramp(s, 3.6, 4.4), weight="bold")
    label(ax, x + 2.3, 5.4, r"$x(t)=\hat{x}_d(t)$", BLUE, 17, ramp(s, 3.6, 4.4))
    label(ax, x, 4.85, "PX4 cascaded PID runs far faster than the planner horizon", MUTED, 11.5, ramp(s, 4.0, 4.8))
    label(ax, x, 4.05, "tracking error and residual (planner + perception)", FG, 14, ramp(s, 5.6, 6.4), weight="bold")
    label(ax, x, 3.35, r"$e=x-x_d,\quad \dot e=A_c e+R,\quad R=B[\hat u_d-u_d+K(\hat x_d-x_d)]$", FG, 15, ramp(s, 6.0, 6.8))
    label(ax, x, 2.45, "nonconformity score (one number per flight)", FG, 14, ramp(s, 7.6, 8.4), weight="bold")
    label(ax, x, 1.75, r"$S=\sup_t\ \|P^{1/2}R(t)\|_2$", TEAL, 20, ramp(s, 8.0, 8.8))
    af = ramp(s, 9.6, 10.6)
    box(ax, 4.0, 1.75, 6.9, 2.2, "", TEAL, af, glow=0.25 * pulse(s, 10.0, 11.6))
    label(ax, 4.0, 2.1, r"$\Pr\left[\ \|x(t)-x_d(t)\|_2\leq\frac{W(0)e^{-\lambda_P t}+\frac{q}{\lambda_P}(1-e^{-\lambda_P t})}{\sqrt{\lambda_{\min}(P)}}\ \ \forall t\ \right]\geq 1-\alpha$", FG, 14.5, af, ha="center")
    label(ax, 4.0, 1.05, r"$W(t)=\|P^{1/2}(x-x_d)\|_2$,   $q$ = conformal quantile of the measured $D_{cal}$ scores", MUTED, 10.5, af, ha="center")
    label(ax, x, 0.75, "$A_c=A-BK$,  $A_c^{\\top}P+PA_c\\preceq-2\\lambda_P P$;  perception enters through $\\hat{x}_d$", MUTED, 11.5, ramp(s, 8.8, 9.6))


# =============================================================================== formulation equations
def scene_equations(fig, s):
    ax = canvas(fig)
    heading(ax, "Hardware problem formulation", "the expert is planned on the known environment; the transformer plans on the perceived environment", ramp(s, 0, 0.6))
    rows = [
        (0.8, r"$E$: ground-truth environment (Vicon estimates of the obstacles)    $\hat E$: perceived environment (perception pipeline)", FG, 14.5),
        (1.6, r"$L_1$: goal-selection language      $L_2$: HRRT$^*$ route-selection language", FG, 14.5),
        (2.6, r"language-conditioned expert on the known environment:   $\dot x_d=A\,x_d+B\,u_d,\quad x_d=x_d(E,L_1,L_2,t)$", GREEN, 15),
        (3.6, r"transformer approximation on the perceived environment:   $\dot{\hat x}_d=A\,\hat x_d+B\,\hat u_d,\quad \hat x_d=\hat x_d(\hat E,L_1,L_2,t)$", YELLOW, 15),
        (4.7, r"errors:   $e:=x-x_d(E,L_1,L_2,t)$  (expert-relative, not available online)      $\hat e:=x-\hat x_d(\hat E,L_1,L_2,t)$  (available online)", FG, 14),
        (5.7, r"flown controller:   $u=\hat u_d-K\hat e$        tracking error:   $\dot e=A_c e+R,\quad A_c=A-BK$", BLUE, 15),
        (6.7, r"residual:   $R=B\left[\hat u_d-u_d+K(\hat x_d-x_d)\right]=B(\delta u_{plan}+K\delta x_{plan})+B(\delta u_{env}+K\delta x_{env})$   (planner + perception mismatch)", FG, 14),
        (7.9, r"score:   $S=\sup_{t\in[0,T]}\ \|P^{1/2}R(t)\|_2$        tube:   $\|x(t)-x_d(t)\|_2\leq\dfrac{W(0)e^{-\lambda_P t}+\frac{q}{\lambda_P}(1-e^{-\lambda_P t})}{\sqrt{\lambda_{\min}(P)}}$  with probability $\geq 1-\alpha$", TEAL, 14),
    ]
    for k, (t0, text, colour, size) in enumerate(rows):
        label(ax, 0.8, 7.0 - 0.82 * k, text, colour, size, ramp(s, t0, t0 + 0.7))
    label(ax, 0.8, 0.45, "ideal double integrator with ideal tracking by the PX4 cascaded PID: only the planner and perception mismatch remain", MUTED, 11.5, ramp(s, 9.0, 9.8))


# =============================================================================== one calibration sample
def scene_sample(fig, s):
    ax = canvas(fig)
    heading(ax, "Example calibration sample", "ground-truth obstacle vs. perceived obstacle from the perception pipeline (YOLOv8 object detection integrated with ToF point-cloud estimates) — and the score this sample contributes", ramp(s, 0, 0.6))
    a = ramp(s, 0.2, 0.9)
    true_centre, seen_shift = np.array([3.9, 3.9]), np.array([0.3, -0.22])
    start, goal = np.array([1.2, 2.3]), np.array([6.6, 5.4])
    expert = route(start, goal, true_centre, 1.0, 1.0)
    u = np.linspace(0, 1, len(expert))
    normal = np.gradient(expert, axis=0)
    normal = np.column_stack([-normal[:, 1], normal[:, 0]]) / np.linalg.norm(normal, axis=1, keepdims=True)
    gap = 6.0 * u * np.exp(-6.0 * u) + 0.2 * np.exp(-((u - 0.55) / 0.14) ** 2)
    plan = expert + gap[:, None] * normal

    for column, title in enumerate(["ground-truth environment  →  expert", "perceived environment  →  transformer"]):
        offset = np.array([7.6 * column, 0.0])
        panel(ax, 0.7 + offset[0], 1.6, 7.3 + offset[0], 6.3, a, title)
        obstacle(ax, true_centre + offset, 0.4, PURPLE, a * (1.0 if column == 0 else 0.35), text="true" if column == 0 else None)
        ax.add_patch(Circle(start + offset, 0.1, color=FG, alpha=a, zorder=5))
        ax.plot(*(goal + offset), marker="*", color=YELLOW, markersize=17, alpha=a, markeredgecolor=BG, zorder=5)
    right = np.array([7.6, 0.0])
    obstacle(ax, true_centre + seen_shift + right, 0.46, ORANGE, ramp(s, 1.2, 2.0), dashed=True, text="perceived")
    draw_path(ax, expert, ramp(s, 1.0, 3.0), GREEN, 3.0, zorder=4)
    draw_path(ax, expert + right, 1.0, GREEN, 1.6, ramp(s, 3.0, 3.8) * 0.6, zorder=3)
    f = ramp(s, 4.0, 9.0)
    k = int(f * (len(plan) - 1))
    tube = ramp(s, 10.0, 11.5)
    if tube > 0.01:
        draw_tube(ax, expert + right, np.full(len(expert), 0.62), tube, alpha=0.16)
    draw_path(ax, plan + right, f, YELLOW, 2.4, style=(0, (4, 2.5)), zorder=5)
    ax.plot(plan[: k + 1, 0] + right[0], plan[: k + 1, 1], color=BLUE, lw=5.5, alpha=0.3 * ramp(s, 4.0, 4.4), zorder=3, solid_capstyle="round")
    if 0 < f < 1:
        drone_icon(ax, *(plan[k] + right))
        ax.plot([plan[k, 0] + right[0], expert[k, 0] + right[0]], [plan[k, 1], expert[k, 1]], color=FG, lw=1.2, linestyle=":", zorder=6)

    # The score trace.
    px, py, pw, ph = 2.1, 0.35, 5.0, 0.95
    ax.plot([px, px + pw], [py, py], color=DIM, lw=1.0, alpha=a)
    ax.plot(px + u[: k + 1] * pw, py + gap[: k + 1] / gap.max() * ph * 0.9, color=TEAL, lw=2.0, alpha=ramp(s, 4.0, 4.4))
    label(ax, px - 0.15, py + ph * 0.6, r"$\|P^{1/2}R\|_2$", TEAL, 11.5, a, ha="right")
    peak = int(np.argmax(gap))
    if k >= peak:
        ax.add_patch(Circle((px + u[peak] * pw, py + ph * 0.9), 0.08, color=FG, zorder=6))
        label(ax, px + u[peak] * pw + 0.18, py + ph * 0.95, "$S_i$ = the peak", FG, 11.5, 1.0)
    for j, (colour, text, t0) in enumerate([(GREEN, "expert $x_d$ (true obstacle)", 1.0), (YELLOW, "transformer plan $\\hat{x}_d$ (perceived obstacle)", 4.0),
                                            (BLUE, "path flown, $x\\approx\\hat{x}_d$", 4.2), (TEAL, "tube from the calibrated quantile $q$", 10.0)]):
        aj = ramp(s, t0, t0 + 0.8)
        ax.plot([8.4 + 0.0, 8.9], [1.15 - 0.3 * j] * 2, color=colour, lw=3.2, alpha=aj, solid_capstyle="round")
        label(ax, 9.1, 1.15 - 0.3 * j, text, MUTED, 11, aj)
    label(ax, 15.3, 1.3, "schematic illustration", DIM, 10, 1.0, ha="right")


# =============================================================================== end card
def scene_end(fig, s):
    ax = canvas(fig)
    a = ramp(s, 0.1, 0.9)
    label(ax, 8, 5.6, "Perception error is calibrated, not modelled.", FG, 24, a, ha="center", weight="bold")
    for k, text in enumerate(["workspace restricted to 100 %-detection regions  →  no scene is discarded  →  exchangeability holds",
                              "training on synthetic scenes; calibration on scenes perceived by the real drone, same distribution",
                              "one score per flight, one conformal quantile, one tube around the language-conditioned expert"]):
        label(ax, 8, 4.4 - 0.7 * k, text, MUTED, 13, ramp(s, 0.6 + 0.5 * k, 1.4 + 0.5 * k), ha="center")
    label(ax, 8, 1.9, r"$\Pr\left[\ \|p(t)-p_d(t)\|\leq r\ \ \forall t\ \right]\geq 1-\alpha$", TEAL, 22, ramp(s, 2.2, 3.0), ha="center")


SCENES = [("0_title", scene_title, 3.5, [3.0]), ("1_exch", scene_exchangeability, 14.0, [4.5, 13.5]),
          ("3_grid", scene_grid, 11.5, [4.0, 11.0]), ("4_training", scene_training, 12.5, [5.0, 12.0]),
          ("5_formulation", scene_formulation, 13.5, [4.5, 13.0]), ("6_sample", scene_sample, 13.5, [6.5, 13.0])]


def build(videos: Path, out: Path) -> None:
    workspace = Clip(videos / "human_restricted trajectory.mp4", 28.0, 36.0)
    perception = Clip(videos / "perception_uncertainity.mp4", 0.0, 6.0)
    hardware_a = Clip(videos / "starling2_rrt_tube_stage10.29-1.00.mp4", 18.0, 40.0, speed=1.4)
    hardware_b = Clip(videos / "starling2_rrt_tube_stage10.29-1.00.mp4", 40.0, 75.0, speed=1.4)
    table = {name: (scene, seconds) for name, scene, seconds, _ in SCENES}
    movie = Movie(out)
    for scene, seconds in [
            table["0_title"],
            (perception.scene("Perception uncertainty", "we tighten the safe space and incorporate the effect of perception uncertainty in our probabilistically robust conformal tubes without any assumptions on the sensor model or the nature of disturbance"), perception.duration),
            table["1_exch"],
            (workspace.scene("Surveying the workspace on hardware", "recorded survey: where does the onboard detector see the obstacle every time?"), workspace.duration),
            table["3_grid"], table["4_training"], table["5_formulation"], table["6_sample"],
            (hardware_a.scene("Hardware demonstration", "the human prescribes the goal position through the interactive gateway; the transformer plan and its conformal tube are shown for approval  (1.4× speed)"), hardware_a.duration),
            (hardware_b.scene("Hardware demonstration", "the drone flies the approved plan; the final conformal tube around the flown trajectory is shown  (1.4× speed)"), hardware_b.duration)]:
        movie.play(scene, seconds)
    movie.close()


def stills(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, scene, _, times in SCENES:
        for time in times:
            still(scene, time, directory / f"{name}_{time:04.1f}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, default=HERE.parent / "static" / "videos" / "clips")
    parser.add_argument("--out", type=Path, default=HERE.parent / "static" / "videos" / "framework_hardware.mp4")
    parser.add_argument("--stills", type=Path, default=None)
    args = parser.parse_args()
    if args.stills:
        stills(args.stills)
    else:
        build(args.videos, args.out)
