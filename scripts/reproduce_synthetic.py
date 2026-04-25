"""Run from the repository root after pip install -e ."""
import argparse
import json
from pathlib import Path
import numpy as np
from selective_multimodal.cli import load_config, reproduce, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/synthetic.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 11, 19])
    args = parser.parse_args()
    base = load_config(args.config)
    summaries = []
    for seed in args.seeds:
        c = {**base, 'seed': seed, 'run_dir': f'runs/seed-{seed}'}
        reproduce(c)
        results = json.loads((Path(c['run_dir'])/'metrics.json').read_text())
        for r in results:
            summaries.append({'seed': seed, **r})
    aggregate = []
    keys = sorted({(r['condition'], r['strategy'], r['budget']) for r in summaries})
    for condition, strategy, budget in keys:
        rows = [r for r in summaries if (r['condition'],r['strategy'],r['budget']) == (condition,strategy,budget)]
        measurements = {}
        for metric in ('accuracy','macro_f1','coverage','selective_error','normalized_cost'):
            values = [r['metrics'][metric] for r in rows if r['metrics'][metric] is not None]
            measurements[metric] = {'mean': float(np.mean(values)) if values else None, 'sample_std': float(np.std(values, ddof=1)) if len(values)>1 else None, 'defined_seeds': len(values)}
        aggregate.append({'condition': condition, 'strategy': strategy, 'budget': budget, 'metrics': measurements})
    write_json('runs/seed_summary.json', {'seeds': args.seeds, 'results': aggregate, 'uncertainty': 'Sample standard deviations across seeds, not confidence intervals'})


if __name__ == '__main__':
    main()
