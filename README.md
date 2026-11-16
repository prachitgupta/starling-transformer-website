# STARLING Transformer project website

Static research website for probabilistically robust and safe tracking of language-conditioned natural-language tasks.

- Site: https://prachitgupta.github.io/starling-transformer-website/
- Research code: https://github.com/prachitgupta/starling_testing_ws

The simulation section intentionally contains only the finalized **Case B conditional score with the GPT expert selector**. It does not publish Case A or heuristic-selector results. Hardware recordings demonstrate the workflow; the environmental-uncertainty comparison remains visibly labeled as synthetic until measured calibration and test data exist.

## Preview

```bash
python scripts/build_site.py
python -m http.server 8000 --directory dist
```

Open http://localhost:8000.

## Refresh the finalized simulation evidence

The import copies the paper's original Case B/GPT histogram, scene, and literature-comparison figures. It regenerates Case B-only coverage and tightness summaries from the same benchmark JSON so the website never displays a mixed Case A/Case B panel.

```bash
python scripts/import_simulation_results.py \
  --benchmark-dir ~/Desktop/starling_final_testing/src/llm_vision_planner/simulation/results/benchmark \
  --report ~/Desktop/starling_final_testing/src/papers/report/simulation.pdf
```

Generated website evidence:

- `data/case_b_gpt.json`
- `assets/images/simulation/`
- `assets/papers/simulation.pdf`

## Video routing

Recordings in `videos/` are assigned through `videos/placements.json`. Each placement may specify `slot`, `start`, and `end`; the website seeks to the selected excerpt and returns to its start when the excerpt finishes.

## Validation and deployment

```bash
python -m unittest discover -s tests
python -m compileall -q scripts tests
python scripts/build_site.py
```

The GitHub Pages workflow rebuilds `dist/` and deploys on every push to `main`. Public videos must remain below GitHub's per-file limit enforced by the tests.
