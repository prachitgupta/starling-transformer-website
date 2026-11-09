# Add your recordings here

Use MP4 (H.264 recommended for browser compatibility), WebM, OGV, or M4V. Keep each file under 95 MiB. Videos are published publicly with this repository. Do not add sensitive recordings.

Descriptive names automatically select a section. For unambiguous placement, begin the filename with a slot followed by `_`:

| Example filename | Section |
| --- | --- |
| framework_overview.mp4 | Framework |
| mission_navigate_around_chair.mp4 | Interactive mission |
| transformer_waypoint_prediction.mp4 | Transformer prediction |
| hardware_flight.mp4 | Hardware execution |
| perception_vicon_corners.mp4 | Perception |
| control_trajectory_tracking.mp4 | Tracking |
| clearance_obstacle_avoidance.mp4 | Clearance |
| acp_adaptive_coverage.mp4 | Adaptive conformal prediction |
| expert_hrrt_trajectory.mp4 | HRRT expert |
| calibration_residual_quantile.mp4 | Calibration |

Multiple clips are supported per section. Subdirectories and filenames with spaces are supported. Without a slot prefix, keyword matches in `data/video_slots.json` determine placement; ties follow the order in that file. Unmatched names appear in Additional Recordings. The displayed caption comes from the filename.

For exceptions, create `videos/placements.json` mapping a filename relative to this directory to a slot, for example:

```json
{"my demonstration.mp4": "acp"}
```

Run `python scripts/publish.py --message "Add hardware demonstrations"` from the repository root. Alternatively upload videos here through GitHub; the deployment workflow indexes them automatically. The generated root `index.html` is refreshed by the local publish command; Pages always rebuilds from current files.
