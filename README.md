# Project website

Static page for *Probabilistically Robust Tracking of Language-Conditioned Oracle Motion Plans by Black-Box Imitation Learners*.
`index.html` is the whole site (Nerfies template); `static/` holds the CSS, JS, plots and the two framework videos.
Pushing to `main` deploys the repository root to GitHub Pages.

`animations/` holds the source of the framework videos:

```bash
cd animations
python video1.py          # static/videos/framework_nonlinear.mp4  (uses the recordings in static/videos/clips)
python video2.py          # static/videos/framework_hardware.mp4
```

Requirements: Python 3.10+, numpy, matplotlib, opencv-python, and ffmpeg on the PATH.
`framework_video_data.json` holds the certificate, calibration scores and held-out episodes drawn in video 1.

Result figures and the simulation GIF are rebuilt from the saved benchmark data and the two hardware
split-conformal certificates with:

```bash
python3 tools/build_result_assets.py
```

## Equations
Edit `tools/index_source.html` (TeX in `\( \)` / `\[ \]`), then pre-render with KaTeX:

    npm install katex
    node tools/render_math.js tools/index_source.html index.html
