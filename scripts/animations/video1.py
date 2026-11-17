#!/usr/bin/env python3
"""Framework video 1: the general nonlinear case (simulation framework).

    python video1.py --videos <dir with the source recordings> --out framework_nonlinear.mp4
    python video1.py --stills <dir>      # one PNG per scene and time, for checking layouts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Circle, Rectangle

from engine import (BG, BLUE, DIM, FG, GREEN, MUTED, ORANGE, PANEL, PURPLE, RED, TEAL, YELLOW, Clip, Movie, WorldMap,
                    arrow, box, canvas, car_icon, draw_path, draw_tube, drone_icon, heading, label, partial, pulse,
                    ramp, stick_figure, still)

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "framework_video_data.json").read_text())
EP_A, EP_B, EP_C = DATA["episodes"]            # widest berth (2 obstacles); balanced (3 obstacles); wrong corridor
CERT = DATA["certificate"]
FAMILY_COLOURS = [BLUE, PURPLE, ORANGE, TEAL, "#d98fbf", "#8fa3d9", "#c9c26b", "#7fc9a0"]


def bounds_of(episode, pad=0.0):
    points = np.vstack([np.vstack(route) for route in episode["family"]] + [episode["start"], episode["goal"]]
                       + [o["lo"] for o in episode["obstacles"]] + [o["hi"] for o in episode["obstacles"]])
    return (points[:, 0].min() - pad, points[:, 0].max() + pad), (points[:, 1].min() - pad, points[:, 1].max() + pad)


def xy(states):
    return np.asarray(states, float)[:, :2]


# =============================================================================== 0. title
def scene_title(fig, s):
    ax = canvas(fig)
    a = ramp(s, 0.2, 1.2)
    label(ax, 8, 5.5, "Probabilistically Robust and Safe Tracking", FG, 27, a, ha="center", weight="bold")
    label(ax, 8, 4.6, "of Language-Conditioned Natural-Language Tasks", FG, 27, a, ha="center", weight="bold")
    label(ax, 8, 3.45, "Framework  ·  general nonlinear case", TEAL, 19, ramp(s, 1.0, 2.0), ha="center")
    for k, (colour, name) in enumerate([(YELLOW, "transformer plan"), (GREEN, "language-conditioned expert"),
                                        (BLUE, "flown state"), (TEAL, "conformal contraction tube")]):
        a2 = ramp(s, 1.4 + 0.2 * k, 2.2 + 0.2 * k)
        x0 = [1.0, 4.0, 8.4, 11.2][k]
        ax.plot([x0, x0 + 0.6], [2.2, 2.2], color=colour, lw=4, alpha=a2, solid_capstyle="round")
        label(ax, x0 + 0.75, 2.2, name, MUTED, 11.5, a2)


# =============================================================================== 1. problem formulation
def scene_problem(fig, s):
    ax = canvas(fig)
    heading(ax, "Problem formulation", "three trajectories, one probabilistic safety tube", ramp(s, 0, 0.6))
    ep = EP_B
    world = WorldMap((0.6, 0.6, 9.0, 7.3), bounds_of(ep, 0.7))
    a = ramp(s, 0.3, 1.2)
    world.frame(ax, a)
    world.obstacles(ax, ep["obstacles"], a)
    world.endpoints(ax, ep["start"], ep["goal"], a)
    expert, plan, flown = world.pt(xy(ep["expert"])), world.pt(xy(ep["plan"])), world.pt(xy(ep["flown"]))
    radius = np.asarray(ep["radius"]) * world.scale

    f_tube = ramp(s, 10.0, 13.0)
    draw_tube(ax, expert, radius, f_tube)
    draw_path(ax, expert, ramp(s, 1.2, 3.4), GREEN, 3.2, zorder=4)
    draw_path(ax, plan, ramp(s, 4.0, 6.2), YELLOW, 2.6, style=(0, (4, 2.5)), zorder=5)
    f_flown = ramp(s, 6.9, 9.6)
    tip = draw_path(ax, flown, f_flown, BLUE, 2.4, zorder=6)
    if tip is not None and f_flown < 1.0:
        drone_icon(ax, *tip)

    x = 9.5
    rows = [(1.2, GREEN, r"$x_d(t)$", "language-conditioned expert", "HRRT* route chosen by $L_2$  ·  offline only"),
            (4.0, YELLOW, r"$\hat{x}_d(t)$", "transformer plan", "predicted online, tracked by the controller"),
            (6.9, BLUE, r"$x(t)$", "flown state", "true dynamics with unknown disturbance")]
    for k, (t0, colour, symbol, name, note) in enumerate(rows):
        a = ramp(s, t0, t0 + 0.8)
        y = 6.75 - 1.25 * k
        ax.plot([x, x + 0.55], [y, y], color=colour, lw=4, alpha=a, solid_capstyle="round",
                linestyle=(0, (2, 1.4)) if colour == YELLOW else "-")
        label(ax, x + 0.75, y, symbol, colour, 18, a)
        label(ax, x + 1.85, y + 0.02, name, FG, 13.5, a, weight="bold")
        label(ax, x, y - 0.42, note, MUTED, 11.5, a)
    a = ramp(s, 10.0, 11.0)
    ax.add_patch(Rectangle((x, 2.55), 0.55, 0.3, facecolor=matplotlib.colors.to_rgba(TEAL, 0.3), edgecolor=TEAL, alpha=a))
    label(ax, x + 0.75, 2.72, "conformal contraction tube", TEAL, 13.5, a, weight="bold")
    label(ax, x, 2.2, "around the expert, valid with probability 1 − α", MUTED, 11.5, a)
    a = ramp(s, 11.5, 12.6)
    label(ax, 12.6, 1.25, r"$\Pr\left[\ \|p(t)-p_d(t)\|\leq r(t)\ \ \forall t\ \right]\ \geq\ 1-\alpha$", FG, 17, a, ha="center")


# =============================================================================== 2. expert generation
def scene_expert(fig, s):
    ax = canvas(fig)
    heading(ax, "The language-conditioned expert", "HRRT* proposes safe routes; the preference selects one", ramp(s, 0, 0.6))
    ep = EP_A
    world = WorldMap((8.5, 0.9, 15.5, 6.75), bounds_of(ep, 0.25))
    a = ramp(s, 0.2, 1.0)
    world.frame(ax, a)
    label(ax, 8.5, 7.0, "scene (ground truth)", MUTED, 11, a)
    world.obstacles(ax, ep["obstacles"], a)
    goal_alpha = ramp(s, 3.9, 4.6)
    world.endpoints(ax, ep["start"], ep["goal"], a, goal_alpha)

    # The operator.
    throw = max(pulse(s, 2.0, 3.0), pulse(s, 8.6, 9.6))
    hand = stick_figure(ax, 1.5, 2.2, 1.25, FG, a, wave=throw)
    label(ax, 1.5, 1.8, "operator", MUTED, 11.5, a, ha="center")

    planner, selector = np.array([5.6, 6.2]), np.array([5.6, 2.3])
    box(ax, *planner, 3.0, 1.15, "HRRT* planner\nhomotopy-aware RRT*", BLUE, a, 12.5, glow=pulse(s, 3.3, 4.6))
    box(ax, *selector, 3.0, 1.15, "GPT route selector\nreads the route cards", GREEN, a, 12.5, glow=pulse(s, 9.9, 11.2))
    arrow(ax, planner + [1.55, 0], [8.45, 6.2], BLUE, ramp(s, 3.6, 4.2) * 0.9, frac=ramp(s, 3.6, 4.4))
    arrow(ax, selector + [1.55, 0], [8.45, 2.3], GREEN, ramp(s, 10.0, 10.6) * 0.9, frac=ramp(s, 10.0, 10.8))

    def thrown(text, colour, t_show, t_throw, t_land, target):
        """A prompt that appears as a speech bubble, then flies along an arc to ``target``."""
        appear, fly = ramp(s, t_show, t_show + 0.6), ramp(s, t_throw, t_land)
        if appear <= 0.01 or s > t_land + 0.35:
            return
        start = np.array([4.1, 4.3])
        position = start + fly * (target - start) + np.array([0.0, 1.3 * np.sin(np.pi * fly)])
        shrink = 1.0 - 0.45 * fly
        fade = appear * (1.0 - ramp(s, t_land, t_land + 0.35))
        box(ax, *position, 4.3 * shrink, 0.95 * shrink, text, colour, fade, 12.5 * shrink)
    thrown("$L_1$:  “fly to the far side of the chair”", BLUE, 1.0, 2.2, 3.4, planner)
    thrown("$L_2$:  “give every obstacle\nthe widest practical berth”", GREEN, 7.4, 8.8, 10.0, selector)

    # Family of routes, then the selection.
    selected = np.asarray(ep["selected"])
    chosen = ramp(s, 10.6, 11.8)
    for k, route in enumerate(ep["family"]):
        route = np.asarray(route)
        is_selected = len(route) == len(selected) and np.allclose(route, selected)
        f = ramp(s, 4.4 + 0.33 * k, 5.4 + 0.33 * k)
        if is_selected:
            colour = matplotlib.colors.to_hex((1 - chosen) * np.array(matplotlib.colors.to_rgb(FAMILY_COLOURS[k % 8]))
                                              + chosen * np.array(matplotlib.colors.to_rgb(GREEN)))
            draw_path(ax, world.pt(route), f, colour, 2.0 + 2.2 * chosen, zorder=5)
        else:
            draw_path(ax, world.pt(route), f, FAMILY_COLOURS[k % 8], 2.0, alpha=1.0 - 0.75 * chosen, zorder=3)
    label(ax, 12.0, 0.5, "a family of safe, semantically aware routes",
          MUTED, 12.5, ramp(s, 5.0, 6.0) * (1 - chosen), ha="center")
    label(ax, 12.0, 0.5, r"language-conditioned expert trajectory  $x_d(t)$", GREEN, 13.5, chosen, ha="center", weight="bold")


# =============================================================================== 4. transformer
def scene_transformer(fig, s):
    ax = canvas(fig)
    heading(ax, "The transformer approximator", "scene layout + language preference  →  eight waypoints", ramp(s, 0, 0.6))
    ep = EP_A
    a = ramp(s, 0.3, 1.2)
    # Inputs.
    scene_map = WorldMap((0.5, 4.3, 3.5, 7.2), bounds_of(ep))
    scene_map.frame(ax, a, "scene layout")
    scene_map.obstacles(ax, ep["obstacles"], a, labels=False)
    start, goal = scene_map.pt(ep["start"]), scene_map.pt(ep["goal"])
    drone_icon(ax, *start, size=0.12, alpha=a)
    ax.plot(*goal, marker="*", color=YELLOW, markersize=12, alpha=a, markeredgecolor=BG)
    box(ax, 2.0, 2.3, 3.0, 1.3, "$L_2$:  “give every obstacle\nthe widest practical berth”", GREEN, a, 11.5)
    label(ax, 2.0, 1.4, "natural-language preference", MUTED, 10.5, a, ha="center")

    nodes = {"scene": (5.6, 5.75), "text": (5.6, 2.3), "fusion": (8.55, 4.0), "decoder": (11.35, 4.0)}
    box(ax, *nodes["scene"], 2.5, 1.2, "Scene encoder\nscene tokens → MLP", BLUE, a, 11.5, glow=pulse(s, 2.3, 4.0))
    box(ax, *nodes["text"], 2.5, 1.2, "MiniLM\nlanguage encoder", GREEN, a, 11.5, glow=pulse(s, 2.3, 4.0))
    box(ax, *nodes["fusion"], 2.4, 1.7, "Fusion\nTransformer\n3 layers", PURPLE, a, 12, glow=pulse(s, 4.3, 6.0))
    box(ax, *nodes["decoder"], 2.4, 3.3, "", ORANGE, a, glow=pulse(s, 6.2, 8.0))
    label(ax, 11.35, 5.35, "8 query-position", FG, 11.5, a, ha="center")
    label(ax, 11.35, 5.0, "decoder heads", FG, 11.5, a, ha="center")
    for k in range(8):
        lit = ramp(s, 6.4 + 0.18 * k, 6.8 + 0.18 * k)
        qx, qy = 10.55 + 0.53 * (k % 4), 4.2 - 0.62 * (k // 4)
        ax.add_patch(Rectangle((qx - 0.2, qy - 0.2), 0.4, 0.4, facecolor=matplotlib.colors.to_rgba(ORANGE, 0.15 + 0.7 * lit),
                               edgecolor=ORANGE, lw=1.2, alpha=a))
        label(ax, qx, qy, f"$q_{k + 1}$", FG, 12, a, ha="center")

    def link(p0, p1, colour, t0, t1):
        arrow(ax, p0, p1, DIM, a, 1.4)
        travel = (s - t0) / (t1 - t0)
        if 0.0 < travel < 1.0:                                   # a pulse of data moving along the arrow
            position = np.asarray(p0) + travel * (np.asarray(p1) - np.asarray(p0))
            ax.add_patch(Circle(position, 0.11, color=colour, zorder=8))
    link((3.55, 5.75), (4.3, 5.75), BLUE, 1.5, 2.5)
    link((3.55, 2.3), (4.3, 2.3), GREEN, 1.5, 2.5)
    link((6.9, 5.6), (7.3, 4.5), BLUE, 3.4, 4.4)
    link((6.9, 2.5), (7.3, 3.5), GREEN, 3.4, 4.4)
    link((9.8, 4.0), (10.1, 4.0), PURPLE, 5.4, 6.3)
    link((12.6, 4.0), (12.95, 4.0), ORANGE, 8.0, 8.6)

    # Output: the eight waypoints on the scene.
    out = WorldMap((13.0, 2.2, 15.7, 5.8), bounds_of(ep))
    out.frame(ax, a, "waypoints")
    out.obstacles(ax, ep["obstacles"], a * 0.8, labels=False)
    waypoints = out.pt(ep["transformer_waypoints"])
    for k, point in enumerate(waypoints):
        pop = ramp(s, 8.5 + 0.35 * k, 8.9 + 0.35 * k)
        if pop > 0.01:
            if k > 0:
                previous = waypoints[k - 1]
                ax.plot([previous[0], previous[0] + pop * (point[0] - previous[0])],
                        [previous[1], previous[1] + pop * (point[1] - previous[1])], color=YELLOW, lw=1.5, alpha=0.8, zorder=4)
            ax.add_patch(Circle(point, 0.05 + 0.07 * pop + 0.05 * pulse(s, 8.5 + 0.35 * k, 9.2 + 0.35 * k), color=YELLOW, zorder=5))
    label(ax, 14.35, 1.75, r"$\hat{w}_1,\ \dots,\ \hat{w}_8$", YELLOW, 16, ramp(s, 11.2, 12.0), ha="center")
    label(ax, 8.0, 0.55, "trained by imitation of the language-conditioned expert on 21,114 examples",
          MUTED, 11.5, ramp(s, 11.6, 12.6), ha="center")


# =============================================================================== 5. waypoints -> trajectory
def scene_trajectory(fig, s):
    ax = canvas(fig)
    heading(ax, "From waypoints to a dynamically feasible reference",
            "the same construction is applied to the transformer waypoints and to the expert route", ramp(s, 0, 0.6))
    ep = EP_A
    world = WorldMap((0.6, 0.7, 7.6, 7.2), bounds_of(ep))
    a = ramp(s, 0.2, 1.0)
    world.frame(ax, a)
    world.obstacles(ax, ep["obstacles"], a)
    world.endpoints(ax, ep["start"], ep["goal"], a)
    waypoints, plan = world.pt(ep["transformer_waypoints"]), world.pt(xy(ep["plan"]))
    ax.plot(waypoints[:, 0], waypoints[:, 1], color=YELLOW, lw=1.2, alpha=0.45 * a, linestyle=":", zorder=3)
    for k, point in enumerate(waypoints):
        ax.add_patch(Circle(point, 0.1, color=YELLOW, alpha=a, zorder=5))
        label(ax, point[0] + 0.16, point[1] + 0.2, rf"$\hat{{w}}_{k + 1}$", YELLOW, 10.5, a)
    f = ramp(s, 3.0, 7.0)
    tip = draw_path(ax, plan, f, YELLOW, 3.0, zorder=6)
    if tip is not None and 0.0 < f < 1.0:
        drone_icon(ax, *tip, color=FG)

    x = 8.2
    label(ax, x, 6.75, "1.  minimum-control QP through the waypoints", FG, 14.5, ramp(s, 1.0, 1.8), weight="bold")
    label(ax, x + 0.4, 5.9, r"$\min_{u_0,\dots,u_{N-1}}\ \ \sum_{k=0}^{N-1}\ \|u_k\|^2$", FG, 19, ramp(s, 1.6, 2.4))
    label(ax, x + 0.4, 5.0, r"$\mathrm{s.t.}\ \ \ x_{k+1}=A\,x_k+B\,u_k,\ \ \ \ p_{k_i}=\hat{w}_i,\ \ i=1,\dots,8$", FG, 16.5, ramp(s, 2.2, 3.0))
    label(ax, x, 3.85, "2.  nominal reference that solves the vehicle model", FG, 14.5, ramp(s, 5.0, 5.8), weight="bold")
    label(ax, x + 0.4, 3.05, r"$\dot{\hat{x}}_d\ =\ f(\hat{x}_d)+B\,\hat{u}_d$", YELLOW, 20, ramp(s, 5.6, 6.4))
    label(ax, x + 5.0, 3.05, r"$\Rightarrow\ \ (\hat{x}_d,\ \hat{u}_d)$", YELLOW, 18, ramp(s, 6.2, 7.0))
    label(ax, x, 1.9, "3.  same construction for the expert route", FG, 14.5, ramp(s, 7.4, 8.2), weight="bold")
    label(ax, x + 6.1, 1.9, r"$\Rightarrow\ (x_d,\ u_d)$", GREEN, 17, ramp(s, 7.4, 8.2))
    label(ax, x, 1.15, "both references are fixed before the flight; only the first is known online", MUTED, 11.5, ramp(s, 8.2, 9.0))


# =============================================================================== 6. dynamics and disturbances
def scene_dynamics(fig, s):
    ax = canvas(fig)
    heading(ax, "Vehicle model and disturbances", "4D car with an unknown, unmodelled disturbance ζ(x, u)", ramp(s, 0, 0.6))
    a = ramp(s, 0.2, 1.0)
    # The car driving; wind pushes it off the nominal path.
    ax.add_patch(Rectangle((0.6, 4.25), 6.0, 3.0, facecolor=PANEL, edgecolor=DIM, lw=1.2, alpha=a))
    u = np.linspace(0, 1, 300)
    nominal = np.column_stack([1.0 + 5.2 * u, 5.75 + 0.75 * np.sin(2 * np.pi * u)])
    drift = np.column_stack([np.zeros_like(u), -0.35 * np.sin(np.pi * u) ** 2 - 0.12 * np.sin(6 * np.pi * u) * u])
    actual = nominal + drift
    travel = (s % 7.0) / 7.0 if s > 1.0 else 0.0
    k = int(travel * 299)
    ax.plot(nominal[:, 0], nominal[:, 1], color=YELLOW, lw=1.8, alpha=0.8 * a, linestyle=(0, (4, 2.5)))
    ax.plot(actual[: k + 1, 0], actual[: k + 1, 1], color=BLUE, lw=2.4, alpha=a)
    theta = np.arctan2(*(np.gradient(actual, axis=0)[k][::-1]))
    car_icon(ax, *actual[k], theta, 0.24, BLUE, a)
    arrow(ax, actual[k], actual[k] + 0.75 * np.array([np.cos(theta), np.sin(theta)]), FG, a, 1.5)
    label(ax, *(actual[k] + [0.3, 0.42]), r"$v,\ \theta$", FG, 12, a)
    for wx in (1.6, 3.2, 4.8):                                               # wind arrows
        arrow(ax, (wx, 7.0), (wx + 0.15, 6.55), RED, 0.75 * a, 1.3)
    label(ax, 6.35, 7.0, r"$\zeta$", RED, 15, a, ha="right")
    label(ax, 0.8, 4.5, r"state $x=(p_x,p_y,\theta,v)$   input $u=(\omega,a)$", MUTED, 11.5, a)

    x = 7.3
    for k2, (t0, text) in enumerate([(1.2, r"$\dot{p}_x=v\cos\theta\ +\ \zeta_{p_x}$"), (1.7, r"$\dot{p}_y=v\sin\theta\ +\ \zeta_{p_y}$"),
                                     (2.2, r"$\dot{\theta}=\omega\ +\ \zeta_{\theta}$"), (2.7, r"$\dot{v}=a\ +\ \zeta_{v}$")]):
        label(ax, x + (4.3 if k2 >= 2 else 0.0), 6.75 - 0.75 * (k2 % 2), text, FG, 18, ramp(s, t0, t0 + 0.7))
    label(ax, x, 4.95, r"$\dot{x}=f(x)+B\,u+\zeta(x,u)$", TEAL, 21, ramp(s, 3.4, 4.2))
    label(ax, x + 5.3, 4.95, "no model of ζ is assumed", MUTED, 12, ramp(s, 3.8, 4.6))

    # Three disturbance regimes used in the benchmark.
    cards = [("Gaussian", BLUE, "wind, bias $\\sim\\mathcal{N}(0,\\sigma^2)$", "light tails: large pushes are rare"),
             ("Heavy-tailed", ORANGE, "wind, bias $\\sim$ Student-$t_3$", "occasional very large gusts"),
             ("State-dependent", PURPLE, "drag $-c_d\\,v|v|$,  slip $\\propto v\\,\\tanh\\omega$", "grows with speed and turning")]
    grid = np.linspace(-4.5, 4.5, 200)
    for k3, (name, colour, law, note) in enumerate(cards):
        a3 = ramp(s, 5.0 + 1.3 * k3, 6.0 + 1.3 * k3)
        x0 = 0.6 + 5.0 * k3
        ax.add_patch(Rectangle((x0, 0.45), 4.75, 3.35, facecolor=PANEL, edgecolor=colour, lw=1.4, alpha=a3))
        label(ax, x0 + 0.2, 3.45, name, colour, 14.5, a3, weight="bold")
        label(ax, x0 + 0.2, 1.15, law, FG, 12, a3)
        label(ax, x0 + 0.2, 0.75, note, MUTED, 10.5, a3)
        if a3 <= 0.01:
            continue
        px, py, pw, ph = x0 + 0.35, 1.55, 4.05, 1.55                         # mini plot area
        if k3 < 2:
            density = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi) if k3 == 0 else \
                (2 / (np.pi * np.sqrt(3))) / (1 + grid ** 2 / 3) ** 2
            ax.fill_between(px + (grid + 4.5) / 9 * pw, py, py + density / 0.42 * ph, color=colour, alpha=0.35 * a3)
            ax.plot(px + (grid + 4.5) / 9 * pw, py + density / 0.42 * ph, color=colour, lw=1.8, alpha=a3)
            if k3 == 1:                                                      # mark the fat tails
                for side in (-1, 1):
                    arrow(ax, (px + pw / 2 + side * 1.55, py + 0.75), (px + pw / 2 + side * 1.75, py + 0.2), colour, a3, 1.2)
                label(ax, px + pw / 2 + 1.0, py + 0.95, "fat tails", colour, 10, a3)
        else:
            speed = np.linspace(0, 1, 100)
            ax.plot(px + speed * pw, py + speed ** 2 * ph * 0.95, color=colour, lw=2.0, alpha=a3)
            ax.plot(px + speed * pw, py + np.tanh(3 * speed) * speed * ph * 0.6, color=colour, lw=2.0, alpha=a3, linestyle="--")
            label(ax, px + pw - 0.05, py + ph * 0.98, "drag", colour, 10, a3, ha="right")
            label(ax, px + pw - 0.05, py + ph * 0.45, "slip", colour, 10, a3, ha="right")
        ax.plot([px, px + pw], [py, py], color=DIM, lw=1.0, alpha=a3)


# =============================================================================== 7. controller and contraction
def scene_controller(fig, s):
    ax = canvas(fig)
    heading(ax, "Contracting controller", "min-norm geodesic feedback from the SOS contraction metric  $M(x)$", ramp(s, 0, 0.6))
    a = ramp(s, 0.2, 1.0)
    ax.add_patch(Rectangle((0.6, 3.6), 7.0, 3.7, facecolor=PANEL, edgecolor=DIM, lw=1.2, alpha=a))
    u = np.linspace(0, 1, 240)
    reference = np.column_stack([1.7 + 5.5 * u, 5.4 + 0.9 * np.sin(2.2 * np.pi * u) * (1 - 0.3 * u)])
    normal = np.gradient(reference, axis=0)
    normal = np.column_stack([-normal[:, 1], normal[:, 0]]) / np.linalg.norm(normal, axis=1, keepdims=True)
    ax.plot(reference[:, 0], reference[:, 1], color=YELLOW, lw=2.2, alpha=a, linestyle=(0, (4, 2.5)), zorder=4)
    label(ax, 7.4, 6.95, r"$\hat{x}_d(t)$", YELLOW, 14, a, ha="right")
    travel = ramp(s, 1.5, 8.0)
    k = int(travel * 239)
    for offset, colour in zip((0.95, 0.55, -0.6, -1.0), (BLUE, "#7fd3e6", "#3fa9c9", "#9adcf0")):
        path = reference + (offset * np.exp(-5.0 * u))[:, None] * normal
        ax.plot(path[: k + 1, 0], path[: k + 1, 1], color=colour, lw=2.0, alpha=a, zorder=5)
        ax.add_patch(Circle(path[k], 0.08, color=colour, alpha=a, zorder=6))
        if offset == 0.95 and 0.01 < travel < 0.45:                          # the geodesic from the plan to the state
            ax.plot([reference[k, 0], path[k, 0]], [reference[k, 1], path[k, 1]], color=FG, lw=1.3, linestyle=":", zorder=6)
            label(ax, path[k, 0] + 0.12, (reference[k, 1] + path[k, 1]) / 2, r"geodesic $\gamma$", FG, 10.5)
    label(ax, 0.8, 3.85, "any start contracts to the plan at rate λ", MUTED, 11.5, ramp(s, 2.5, 3.5))

    x = 8.1
    label(ax, x, 6.95, r"$u\ =\ \hat{u}_d+k(\hat{x}_d,\hat{u}_d,x)$", FG, 19, ramp(s, 1.0, 1.8))
    label(ax, x, 6.0, r"$k=0$", FG, 18, ramp(s, 2.0, 2.8))
    label(ax, x + 1.5, 6.0, r"if $a_0\leq 0$,", MUTED, 14, ramp(s, 2.0, 2.8))
    label(ax, x + 3.3, 6.0, r"$k=-\frac{a_0}{\|b\|^2}\,b$", FG, 19, ramp(s, 2.4, 3.2))
    label(ax, x + 6.2, 6.0, r"if $a_0>0$", MUTED, 14, ramp(s, 2.4, 3.2))
    label(ax, x, 5.2, r"smallest correction with  $\dot{E}\leq-2\lambda E$,   $E=d(\hat{x}_d,x)^2$", FG, 14.5, ramp(s, 3.4, 4.2))
    label(ax, x, 4.55, "uses only the plan and the measured state — never the expert", TEAL, 12, ramp(s, 4.2, 5.0))

    # Distance curves: nominal contraction, then with disturbance.
    px, py, pw, ph = 1.3, 0.75, 5.8, 2.15
    a2 = ramp(s, 5.0, 5.8)
    ax.plot([px, px + pw], [py, py], color=DIM, lw=1.0, alpha=a2)
    ax.plot([px, px], [py, py + ph], color=DIM, lw=1.0, alpha=a2)
    label(ax, px - 0.15, py + ph, r"$d$", FG, 13, a2, ha="right")
    label(ax, px + pw, py - 0.25, "time", MUTED, 10.5, a2, ha="right")
    tt = np.linspace(0, 5, 200)
    f = ramp(s, 5.6, 8.0)
    count = max(2, int(f * 200))
    for d0, colour in zip((1.0, 0.7, 0.45), (BLUE, "#7fd3e6", "#3fa9c9")):
        ax.plot(px + tt[:count] / 5 * pw, py + d0 * np.exp(-tt[:count]) * ph * 0.95, color=colour, lw=2.0, alpha=a2)
    label(ax, px + 1.9, py + 2.0, r"no disturbance:  $d(t)\leq e^{-\lambda t}d(0)$", BLUE, 12, ramp(s, 6.4, 7.2))
    f2 = ramp(s, 8.6, 11.0)
    count2 = max(2, int(f2 * 200))
    if f2 > 0.01:
        ball = 0.22
        for d0, colour in zip((1.0, 0.45), (RED, "#fd9a90")):
            wiggle = 0.035 * np.sin(7 * tt + d0 * 5) * (1 - np.exp(-tt))
            ax.plot(px + tt[:count2] / 5 * pw, py + (d0 * np.exp(-tt[:count2]) + ball * (1 - np.exp(-tt[:count2])) + wiggle[:count2]) * ph * 0.95,
                    color=colour, lw=2.0)
        ax.plot([px, px + pw], [py + ball * ph * 0.95] * 2, color=RED, lw=1.0, linestyle=":", alpha=f2)
    label(ax, x, 2.55, r"with disturbance:", RED, 14, ramp(s, 8.6, 9.4), weight="bold")
    label(ax, x, 1.8, r"$\dot{d}\ \leq\ -\lambda d+\|\zeta\|_M\ \ \Rightarrow\ \ d(t)\leq e^{-\lambda t}d(0)+\frac{\bar{\zeta}}{\lambda}(1-e^{-\lambda t})$", FG, 15.5, ramp(s, 9.2, 10.0))
    label(ax, x, 1.0, "contraction to a ball — its size is what conformal prediction calibrates", MUTED, 12, ramp(s, 10.2, 11.0))


# =============================================================================== 8. score and tube
def scene_score(fig, s):
    ax = canvas(fig)
    heading(ax, "One residual score, one conformal tube", "conditional score: was the language preference satisfied?", ramp(s, 0, 0.6))
    x = 0.7
    label(ax, x, 7.0, r"$D(t)=d\left(x_r(t),\,x(t)\right)$", FG, 18, ramp(s, 0.6, 1.4))
    label(ax, x + 5.2, 7.0, "Riemannian distance from the reference to the flown state  (offline)", MUTED, 11.5, ramp(s, 0.6, 1.4))
    label(ax, x, 6.2, r"$\rho(t)=\left[\dot{D}(t)+\lambda D(t)\right]_+$", TEAL, 19, ramp(s, 1.6, 2.4))
    label(ax, x + 5.2, 6.2, "how much the contraction inequality is violated", MUTED, 11.5, ramp(s, 1.6, 2.4))

    # The two cases.
    a1, a2 = ramp(s, 3.0, 3.8), ramp(s, 4.4, 5.2)
    box(ax, 4.05, 4.75, 6.9, 1.25, "", GREEN, a1)
    label(ax, 0.85, 5.05, "preference satisfied  (same corridor as the expert)", GREEN, 12.5, a1, weight="bold")
    label(ax, 0.85, 4.5, r"reference $x_r$ = language-selected expert,    regret $L=0$", FG, 12.5, a1)
    box(ax, 11.6, 4.75, 7.6, 1.25, "", ORANGE, a2)
    label(ax, 8.05, 5.05, "not satisfied  (other side of an obstacle)", ORANGE, 12.5, a2, weight="bold")
    label(ax, 8.05, 4.5, r"reference $x_r$ = nearest HRRT* route,    regret $L>0$ [m]", FG, 12.5, a2)
    label(ax, 8.0, 3.4, r"$S\ =\ \max\left(\ \sup_t\,\rho(t)\ ,\ \ \beta\,\frac{\lambda}{\sqrt{w}}\,L\ \right)$", FG, 24, ramp(s, 5.8, 6.8), ha="center")

    # Histogram of the real calibration scores with the conformal quantile.
    a3 = ramp(s, 7.2, 8.0)
    px, py, pw, ph = 0.9, 0.75, 5.6, 1.75
    scores = np.asarray(DATA["calibration_scores"])
    q = CERT["quantile"]
    edges = np.linspace(0, 1.6, 33)
    counts, _ = np.histogram(np.clip(scores, 0, 1.6), edges)
    grow = ramp(s, 7.4, 8.6)
    for left, right, count in zip(edges[:-1], edges[1:], counts):
        colour = BLUE if right <= q + 1e-9 else RED
        ax.add_patch(Rectangle((px + left / 1.6 * pw, py), (right - left) / 1.6 * pw * 0.9, grow * count / counts.max() * ph,
                               facecolor=colour, alpha=0.8 * a3, edgecolor="none"))
    ax.plot([px, px + pw], [py, py], color=DIM, lw=1.0, alpha=a3)
    aq = ramp(s, 8.6, 9.4)
    ax.plot([px + q / 1.6 * pw] * 2, [py, py + ph * 1.08], color=FG, lw=1.8, linestyle="--", alpha=aq)
    label(ax, px + q / 1.6 * pw + 0.1, py + ph * 1.02, rf"$q={q:.2f}$", FG, 13, aq)
    label(ax, px, py - 0.3, rf"{CERT['calibration_size']} calibration scores  ·  $q$ = the $\lceil(n+1)(1-\alpha)\rceil$-th smallest", MUTED, 10.5, a3)

    af = ramp(s, 10.0, 11.0)
    box(ax, 11.35, 1.5, 8.9, 2.0, "", TEAL, af, glow=0.25 * pulse(s, 10.6, 12.2))
    label(ax, 11.35, 1.95, r"$\Pr\left[\ \|p(t)-p_r(t)\|\ \leq\ \sqrt{w}\left(e^{-\lambda t}D(0)+\frac{q}{\lambda}\left(1-e^{-\lambda t}\right)\right)\ \ \forall t\ \right]\ \geq\ 1-\alpha$",
          FG, 13.5, af, ha="center")
    label(ax, 11.35, 1.0, r"and the language regret is bounded:  $L\leq\sqrt{w}\,q/(\lambda\beta)$",
          MUTED, 12, af, ha="center")


# =============================================================================== 9. simulation
def make_simulation_scene(episode, title_note):
    world = WorldMap((0.6, 0.6, 9.6, 7.3), bounds_of(episode, 0.75))

    def draw(fig, s, duration=7.0):
        ax = canvas(fig)
        heading(ax, "Simulation", "trained transformer · HRRT* routes with the GPT selector · simulated 4D car with disturbance", ramp(s, 0, 0.5))
        a = ramp(s, 0.1, 0.7)
        world.frame(ax, a)
        for route in episode["family"]:
            draw_path(ax, world.pt(route), 1.0, DIM, 1.2, a, zorder=2)
        world.obstacles(ax, episode["obstacles"], a)
        world.endpoints(ax, episode["start"], episode["goal"], a)
        expert, plan, flown = world.pt(xy(episode["expert"])), world.pt(xy(episode["plan"])), world.pt(xy(episode["flown"]))
        radius = np.asarray(episode["radius"]) * world.scale
        f = float(np.clip((s - 0.9) / (duration - 2.2), 0.0, 1.0))
        k = int(f * (len(flown) - 1))
        draw_tube(ax, expert, radius, max(f, 0.02), alpha=0.18)
        draw_path(ax, expert, 1.0, GREEN, 2.6, a, zorder=4)
        draw_path(ax, plan, 1.0, YELLOW, 2.0, a, style=(0, (4, 2.5)), zorder=5)
        ax.plot(flown[: k + 1, 0], flown[: k + 1, 1], color=BLUE, lw=2.4, alpha=a, zorder=6)
        car_icon(ax, *flown[k], episode["flown"][k][2], 0.17, BLUE, a, zorder=7)
        ax.add_patch(Circle(expert[k], radius[k], facecolor="none", edgecolor=TEAL, lw=1.6, alpha=a, zorder=6))
        ax.add_patch(Circle(expert[k], 0.07, color=GREEN, alpha=a, zorder=7))

        x = 10.0
        label(ax, x, 6.9, title_note, TEAL, 13, a, weight="bold")
        words = episode["preference"].split()
        lines, line = [], ""
        for word in words:
            if len(line) + len(word) > 40:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        lines.append(line)
        label(ax, x, 6.35, "$L_2$:", GREEN, 13, a)
        for j, text in enumerate(lines):
            label(ax, x + 0.6, 6.35 - 0.42 * j, ("“" if j == 0 else "") + text + ("”" if j == len(lines) - 1 else ""), FG, 12.5, a)
        for j, (colour, name) in enumerate([(GREEN, "reference (HRRT* expert)"), (YELLOW, "transformer plan"),
                                            (BLUE, "simulated car, closed loop"), (TEAL, "certified tube, 90 %")]):
            ax.plot([x, x + 0.5], [4.7 - 0.5 * j] * 2, color=colour, lw=3.5, alpha=a, solid_capstyle="round")
            label(ax, x + 0.7, 4.7 - 0.5 * j, name, MUTED, 11.5, a)
        label(ax, x, 2.35, rf"$t={episode['time'][k]:.1f}$ s", FG, 15, a)
        label(ax, x, 1.8, rf"tube radius  $r(t)={episode['radius'][k]:.2f}$ m", TEAL, 14, a)
        label(ax, x, 1.25, rf"distance to reference  ${episode['distance'][k]:.2f}$ m", BLUE, 14, a)
    return draw


# =============================================================================== end card
def scene_end(fig, s):
    ax = canvas(fig)
    a = ramp(s, 0.1, 0.9)
    label(ax, 8, 5.2, "One score.  One quantile.  One tube around the language-conditioned expert.", FG, 17, a, ha="center", weight="bold")
    label(ax, 8, 4.1, r"$\Pr\left[\ \|p(t)-p_d(t)\|\leq\sqrt{w}\left(e^{-\lambda t}D(0)+\frac{q}{\lambda}(1-e^{-\lambda t})\right)\ \forall t\ \right]\geq 1-\alpha$",
          TEAL, 20, ramp(s, 0.6, 1.4), ha="center")
    label(ax, 8, 2.9, "distribution-free  ·  split conformal prediction  ·  contraction metric from SOS", MUTED, 13, ramp(s, 1.0, 1.8), ha="center")


def build(videos: Path, out: Path, simulation_out: Path | None = None) -> None:
    audit = Clip(videos / "human_audinting.mp4", 2.0, 8.0)
    hardware = Clip(videos / "starling2_rrt_tube_stage10.29-1.00.mp4", 50.0, 72.0, speed=2.2)
    sim_b = make_simulation_scene(EP_B, "preference satisfied → selected expert")
    sim_c = make_simulation_scene(EP_C, "wrong corridor → nearest HRRT* route")
    movie = Movie(out)
    for scene, seconds in [
            (scene_title, 3.5), (scene_problem, 14.5), (scene_expert, 13.5),
            (audit.scene("Human audit of the expert", "reviewers check the selected route against the language preference before it is used"), audit.duration),
            (scene_transformer, 13.5), (scene_trajectory, 10.5), (scene_dynamics, 12.5), (scene_controller, 12.5),
            (scene_score, 13.5), (sim_b, 7.0), (sim_c, 7.0),
            (hardware.scene("Hardware", "Starling 2: language prompt → transformer plan → calibrated tube → flight  (2.2× speed)"), hardware.duration),
            (scene_end, 4.0)]:
        movie.play(scene, seconds)
    movie.close()
    if simulation_out is not None:
        clip = Movie(simulation_out)
        clip.play(sim_b, 7.0)
        clip.play(sim_c, 7.0)
        clip.close()


def stills(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sim_b = make_simulation_scene(EP_B, "preference satisfied → selected expert")
    sim_c = make_simulation_scene(EP_C, "wrong corridor → nearest HRRT* route")
    for name, scene, times in [("0_title", scene_title, [3.0]), ("1_problem", scene_problem, [5.0, 13.5]),
                               ("2_expert", scene_expert, [2.8, 6.5, 9.3, 13.0]), ("4_transformer", scene_transformer, [4.0, 12.8]),
                               ("5_trajectory", scene_trajectory, [9.8]), ("6_dynamics", scene_dynamics, [3.0, 11.5]),
                               ("7_controller", scene_controller, [3.0, 11.8]), ("8_score", scene_score, [13.0]),
                               ("9_sim_b", sim_b, [4.0]), ("9_sim_c", sim_c, [6.5]), ("z_end", scene_end, [3.5])]:
        for time in times:
            still(scene, time, directory / f"{name}_{time:04.1f}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, default=HERE.parents[2] / "videos")
    parser.add_argument("--out", type=Path, default=HERE.parents[2] / "videos" / "framework_nonlinear.mp4")
    parser.add_argument("--simulation-out", type=Path, default=None)
    parser.add_argument("--stills", type=Path, default=None)
    args = parser.parse_args()
    if args.stills:
        stills(args.stills)
    else:
        build(args.videos, args.out, args.simulation_out)
