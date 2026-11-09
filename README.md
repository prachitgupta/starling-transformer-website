# Language-Guided Transformer Planning — project website

Standalone static research website inspired by the layout of [GBDiff](https://anonymousicra2027.github.io/GBDiff/). It contains only website sources and exported results, not the ROS workspace, datasets, or model checkpoints.

- Site: https://prachitgupta.github.io/starling-transformer-website/
- Research code: https://github.com/prachitgupta/starling_testing_ws

## Preview

Requires Python 3.10+; no Python packages or Node build are needed.

```bash
python scripts/build_site.py
python -m http.server 8000 --directory dist
```

Open http://localhost:8000. Edit `templates/index.html`, `assets/css/site.css`, or the section rendering in `scripts/build_site.py`, then rebuild.

## Add videos and publish

Place videos in `videos/`, using names such as `framework_overview.mp4`, `hardware_flight.mp4`, or `acp_coverage.mp4`. See [video routing](videos/README.md). Empty sections retain visible placeholders. All recognized videos are included, even when their names do not match a section.

```bash
python scripts/publish.py --message "Add mission and ACP videos"
```

This validates video sizes and routing, builds the page, runs static-site tests, commits website changes, and pushes `main`. It uses your existing Git authentication (SSH or HTTPS credentials); never put a token into these files. The GitHub Actions workflow rebuilds and deploys Pages after every push. Direct GitHub uploads into `videos/` also trigger deployment. Public recordings must be under 95 MiB per file; larger videos need compression or an explicitly implemented external host.

## Refresh saved results

```bash
python scripts/extract_results.py --source /path/to/llm_vision_planner/fine_tuning
python scripts/publish.py --message "Refresh Transformer evaluation"
```

The extractor targets `outputs/final-transformer-vertical-teacher_only_80_10_10`, exports allowlisted metrics and all six route examples, copies three result figures, and records the source Git commit. It does not run inference, ROS, or training. If changing the evaluation run or architecture, also update the narrative and headline summaries in `scripts/build_site.py` to match the new data.

The archived checkpoint has unavailable (infinite) quantile/radius values; JSON exports represent these as `null`. The page does not claim hardware performance or empirically verified adaptive coverage. The abstract is a provisional description. No paper acceptance or author affiliations are asserted.

## Deployment and checks

GitHub Pages must use **GitHub Actions** as its source. `.github/workflows/pages.yml` builds `dist/` and deploys it. Only static site assets, result JSON, and videos enter the deployed artifact.

```bash
python -m unittest discover -s tests
python -m compileall -q scripts tests
python scripts/build_site.py
```

Styles follow the reference's centered hero, rounded links, narrow abstract/framework, two-column media grids, and full-width result figures. Original reference text and research media are not included. See [third-party notices](THIRD_PARTY_NOTICES.md).
