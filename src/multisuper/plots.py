"""Regenerate validation-independent plots from recorded measurement files."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--main',default='results/main/runs.jsonl')
    p.add_argument('--node',default='results/node_sweep/all_runs.jsonl')
    p.add_argument('--out',default='results/figures')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    main=[json.loads(line) for line in Path(a.main).read_text().splitlines()]
    dsets=['bilinear','quartic','sextic','switched','parity6']
    means=defaultdict(list)
    for r in main:
        if r['dataset'] in dsets and r['model'] in ['mlp','learned_product','routed_product']:
            means[(r['dataset'],r['model'])].append(r['test_metric'])
    fig,ax=plt.subplots(figsize=(10,4.5))
    x=np.arange(len(dsets));w=.25
    for j,m in enumerate(['mlp','learned_product','routed_product']):
        vals=[np.mean(means[(d,m)]) for d in dsets]
        errors=[np.std(means[(d,m)],ddof=1) for d in dsets]
        ax.bar(x+(j-1)*w,vals,width=w,yerr=errors,label=m,capsize=3)
    ax.set_xticks(x,dsets)
    ax.set_ylabel('Test normalized MSE (lower is better)')
    ax.set_title('Held-out synthetic tasks: 8 seeds, 180 epochs')
    ax.legend()
    fig.tight_layout();fig.savefig(out/'synthetic_comparison.png',dpi=160)
    plt.close(fig)
    if Path(a.node).exists():
        sweep=[json.loads(line) for line in Path(a.node).read_text().splitlines()]
        fig,ax=plt.subplots(figsize=(8,5))
        for ds in ['bilinear','quartic','sextic','switched']:
            s=[];score=[];dev=[]
            for n in [1,2,4,8,16]:
                vals=[r['test_metric'] for r in sweep if r['dataset']==ds and r['model']=='learned_product' and r['nodes_budget']==n]
                if vals:
                    s.append(n);score.append(np.mean(vals));dev.append(np.std(vals,ddof=1) if len(vals)>1 else 0)
            ax.errorbar(s,score,yerr=dev,marker='o',label=ds,capsize=3)
        ax.set_xscale('log',base=2)
        ax.set_xticks([1,2,4,8,16],[1,2,4,8,16])
        ax.set_xlabel('Learned multiplicative nodes budget')
        ax.set_ylabel('Test normalized MSE (lower is better)')
        ax.set_title('Scaling node count does not guarantee better learning')
        ax.legend()
        fig.tight_layout();fig.savefig(out/'node_scaling.png',dpi=160)
        plt.close(fig)
    print('Figures:',*[str(p) for p in out.glob('*.png')])

if __name__=='__main__': main()
