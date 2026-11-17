# Framework animations

Source for the three generated videos in `videos/`:

| Output | Command |
|---|---|
| `framework_nonlinear.mp4`, `simulation_rollouts.mp4` | `python video1.py --simulation-out ../../videos/simulation_rollouts.mp4` |
| `framework_hardware.mp4` | `python video2.py` |

Requirements: Python 3.10+, `numpy`, `matplotlib`, `opencv-python`, and `ffmpeg` on the PATH.
`python video1.py --stills <dir>` writes one PNG per scene for checking layouts.

`framework_video_data.json` holds the data drawn in video 1: the finalized conditional-score
certificate, the 200 calibration scores, and three held-out episodes (scene, HRRT* routes, GPT-selected
route, transformer waypoints, simulated closed-loop rollout, tube radius) exported from the
`simulation/` benchmark. The drawn scenes of video 2 are schematic illustrations, not measured data;
the embedded recordings are the real clips from `videos/`.
