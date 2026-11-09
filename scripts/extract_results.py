#!/usr/bin/env python3
"""Export only publication assets and allowlisted metrics; never model weights."""
import argparse, json, math, pathlib, shutil, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
def clean(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: clean(v) for k,v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    return value

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--source', type=pathlib.Path, required=True)
    source=parser.parse_args().source.resolve()
    run=source/'outputs/final-transformer-vertical-teacher_only_80_10_10'
    read=lambda path: json.loads((run/path).read_text())
    post=read('reports/postflight.json'); metrics=read('final_transformer_vertical_teacher_only_80_10_10_best/checkpoint/metrics.json')
    result={k:post[k] for k in ('validation','test','route_examples','gradient_norms','sensitivity')}
    result.update(checkpoint_metrics=metrics, history=read('model/training_history.json'), provenance={
        'repository':'https://github.com/prachitgupta/starling_testing_ws',
        'snapshot_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        'run':'fine_tuning/outputs/'+run.name,
        'note':'Archived offline evaluation. Unavailable infinite quantiles/radii are exported as null. This is not a hardware or adaptive-coverage evaluation.'})
    (ROOT/'data/results.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    for name in ('training_curves.png','route_examples.png','component_diagnostics.png'):
        shutil.copyfile(run/'dashboard'/name,ROOT/'assets/images'/name)
if __name__=='__main__': main()
