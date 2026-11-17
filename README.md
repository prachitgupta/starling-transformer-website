# Project website

Static page for *Probabilistically Robust and Safe Tracking of Language-Conditioned Natural-Language Tasks*.
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

## Equations
Edit `tools/index_source.html` (TeX in `\( \)` / `\[ \]`), then pre-render with KaTeX:

    npm install katex
    node tools/render_math.js tools/index_source.html index.html
