"""Predeclared budgets and sample sizes: don't choose based on test scores.

All results under separate folders to avoid accidentally pooling configs.
"""
import argparse
import json
from pathlib import Path
from .experiment import run_suite


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['nodes','samples'],required=True)
    p.add_argument('--out',default='results')
    p.add_argument('--epochs',type=int,default=180)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    a=p.parse_args()
    root=Path(a.out)/('node_sweep' if a.kind=='nodes' else 'sample_sweep')
    aggregate=[]
    if a.kind=='nodes':
        for nodes in [1,2,4,8,16]:
            rows=run_suite(['bilinear','quartic','sextic','switched','digits'],
                           ['learned_product','routed_product'],a.seeds,a.epochs,nodes,
                           root/f'nodes_{nodes}')
            aggregate.extend(rows)
    else:
        for n in [128,512,2048]:
            rows=run_suite(['bilinear','quartic','sextic','switched'],
                           ['mlp','learned_product','routed_product'],a.seeds,a.epochs,8,
                           root/f'train_{n}',train_n=n)
            aggregate.extend(rows)
    with (root/'all_runs.jsonl').open('w') as f:
        for r in aggregate:f.write(json.dumps(r,sort_keys=True)+'\n')
    print('SWEEP',a.kind,'DONE',len(aggregate),'runs',flush=True)

if __name__=='__main__':main()
