#!/usr/bin/env python3
"""Rebuild framework-film data from the saved shortest/balanced simulation run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLANNER = Path("/home/prachit/Desktop/starling_final_testing/src/llm_vision_planner")
sys.path.insert(0, str(PLANNER))

from simulation.config import SimulationConfig  # noqa: E402
from simulation.conformal.contraction_residual_score import ContractionResidualScore  # noqa: E402
from simulation.data.scene_dataset import SceneDataset  # noqa: E402
from simulation.paths import TEST_DATASET  # noqa: E402
from simulation.planning.car_reference_generator import CarReferenceGenerator  # noqa: E402
from simulation.planning.smooth_path import SmoothPath  # noqa: E402


RESULTS = PLANNER / "simulation/results"
CACHE = RESULTS / "cache"
RUN_CACHE = CACHE / "bd62da7645c6"
CERTIFICATE = RESULTS / "benchmark/certificate_caseA_gpt.json"
CONDITIONAL_CERTIFICATE = RESULTS / "benchmark/certificate_caseB_gpt.json"
OUTPUT = ROOT / "animations/framework_video_data.json"
SAMPLE_IDS = ("scene-000005-goal-00-pref-00", "scene-000052-goal-00-pref-02")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


def xy(point: dict) -> list[float]:
    return [float(point["x"]), float(point["y"])]


def route_xy(route: dict) -> list[list[float]]:
    return [xy(point) for point in route["waypoints"]]


def main() -> None:
    config = SimulationConfig()
    dataset = SceneDataset(TEST_DATASET)
    sample_to_row = {scene.sample_id: row for row, scene in enumerate(dataset)}
    transformer = load_npz(CACHE / "transformer_waypoints.npz")
    transformer_routes = {str(sample_id): route for sample_id, route in zip(transformer["ids"], transformer["routes"])}
    rollout = load_npz(RUN_CACHE / "rollout_gaussian.npz")
    signals = load_npz(RUN_CACHE / "signals_gaussian_gpt.npz")
    certificate = json.loads(CERTIFICATE.read_text())
    conditional_certificate = json.loads(CONDITIONAL_CERTIFICATE.read_text())

    residual_scores = ContractionResidualScore(
        float(signals["contraction_rate"]), float(signals["record_step"])
    ).score(signals["expert_distance"])
    calibration_rows = [sample_to_row[sample_id] for sample_id in certificate["calibration_sample_ids"]]
    calibration_scores = residual_scores[calibration_rows]

    generator = CarReferenceGenerator(config.nominal_speed, config.dt)
    q = float(certificate["quantile"])
    rate = float(certificate["contraction_rate"])
    w = float(certificate["position_block_w"])
    episodes = []
    for sample_id in SAMPLE_IDS:
        row = sample_to_row[sample_id]
        scene = dataset[row]
        route = transformer_routes[sample_id]
        plan = generator.reference(SmoothPath(route, config.corner_radius))
        expert = generator.reference(SmoothPath(scene.gpt_waypoints, config.corner_radius), plan.steps)
        count = int(rollout["records"][row])
        every = int(rollout["record_every"])
        time = rollout["time"][:count]
        initial_distance = float(signals["expert_distance"][row, 0])
        metric_radius = np.exp(-rate * time) * initial_distance + q / rate * (1.0 - np.exp(-rate * time))
        position_radius = np.sqrt(w) * metric_radius
        episodes.append({
            "sample_id": sample_id,
            "preference": scene.preference_text,
            "preference_type": scene.preference_type,
            "same_corridor": True,
            "start": xy(scene.start),
            "goal": xy(scene.goal),
            "workspace": [scene.workspace["x"], scene.workspace["y"]],
            "obstacles": [{"label": obstacle["label"], "lo": obstacle["min_corner"][:2],
                           "hi": obstacle["max_corner"][:2]} for obstacle in scene.obstacles],
            "family": [route_xy(candidate) for candidate in scene.candidate_routes],
            "selected": scene.gpt_waypoints.tolist(),
            "transformer_waypoints": route.tolist(),
            "time": np.round(time, 4).tolist(),
            "expert": np.round(expert.states[::every][:count], 4).tolist(),
            "plan": np.round(plan.states[::every][:count], 4).tolist(),
            "flown": np.round(rollout["states"][row, :count], 4).tolist(),
            "radius": np.round(position_radius, 4).tolist(),
            "distance": np.round(signals["position_error"][row, :count], 4).tolist(),
        })

    payload = {
        "certificate": {
            "quantile": q,
            "alpha": float(certificate["alpha"]),
            "calibration_size": int(certificate["calibration_size"]),
            "rank": int(certificate["rank"]),
            "contraction_rate": rate,
            "position_block_w": w,
            "steady_state_radius_m": float(certificate["steady_state_radius_m"]),
        },
        "calibration_scores": np.round(calibration_scores, 4).tolist(),
        "conditional_certificate": {
            "quantile": float(conditional_certificate["quantile"]),
            "alpha": float(conditional_certificate["alpha"]),
            "calibration_size": int(conditional_certificate["calibration_size"]),
            "rank": int(conditional_certificate["rank"]),
            "contraction_rate": float(conditional_certificate["contraction_rate"]),
            "position_block_w": float(conditional_certificate["position_block_w"]),
            "beta": float(conditional_certificate["beta"]),
            "steady_state_radius_m": float(conditional_certificate["steady_state_radius_m"]),
            "language_regret_bound_m": float(conditional_certificate["language_regret_bound_m"]),
        },
        "conditional_calibration_scores": [
            round(float(score), 4) for score in conditional_certificate["calibration_scores"]
        ],
        "episodes": episodes,
    }
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUTPUT} with {len(calibration_scores)} calibration scores and {len(episodes)} episodes")


if __name__ == "__main__":
    main()
