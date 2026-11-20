"""Produce a careful summary of cost-scaling and data-scaling trends."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def report(path):
    path=Path(path)
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    name='nodes_budget' if 'node_sweep' in str(path) else 'train_n_requested'
    groups=defaultdict(list)
    for r in rows: groups[(r['dataset'],r['model'],r[name])].append(r)
    lines=[f'# Sweep: {name}', '', 'All figures measured; only complete available seeds included.',
           '', '| Dataset | Model | Budget | n | Mean test score | Mean estimated MACs/sample | Median batch ms |',
           '|---|---|---:|---:|---:|---:|---:|']
    for (d,m,b),g in sorted(groups.items()):
        lines.append(f'| {d} | {m} | {b} | {len(g)} | {np.mean([r["test_metric"] for r in g]):.4f} | {np.mean([r["estimated_macs_per_example"] for r in g]):.1f} | {np.median([r["batch_inference_ms"] for r in g]):.4f} |')
    lines+=['','**Not an established superlinear score/compute law.** For regression, lower is better. For digits, higher is better. MAC counts ignore memory/control; inference elapsed times are CPU measurements with fixed batch sizes. Selecting a favorable interval post hoc cannot establish a scaling exponent.']
    output=path.with_name('REPORT.md')
    output.write_text('\n'.join(lines)+'\n')
    print(f'Wrote {output}, {len(rows)} total training runs')


def main():
    p=argparse.ArgumentParser();p.add_argument('path');a=p.parse_args();report(a.path)
if __name__=='__main__':main()
