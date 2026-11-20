"""Generate a clear mixed-positive/negative report, with paired seed differences.

Uncorrected exploratory uncertainty intervals; not a statistical publication.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def paired(rows,reference="mlp",challenger="learned_product",bootstraps=10000):
    idx={(r["dataset"],r["model"],r["seed"]):r for r in rows}
    out=[]
    for ds in sorted({r["dataset"] for r in rows}):
        seeds=sorted({r["seed"] for r in rows if r["dataset"]==ds})
        diffs=[]; compare=[]
        for seed in seeds:
            a=idx.get((ds,reference,seed));b=idx.get((ds,challenger,seed))
            if a is None or b is None:continue
            sign=1 if a["metric"]=="accuracy" else -1
            diffs.append(sign*(b["test_metric"]-a["test_metric"]))
            compare.append((a,b))
        if not diffs:continue
        rng=np.random.default_rng(3121)
        d=np.asarray(diffs)
        draws=rng.choice(d,size=(bootstraps,len(d)),replace=True).mean(axis=1)
        lb,ub=np.percentile(draws,[2.5,97.5])
        out.append({"dataset":ds,"provenance":compare[0][0]["provenance"],
                    "n":len(d),"unit":"accuracy_points" if compare[0][0]["metric"]=="accuracy" else "normalized_mse_reduction",
                    "mean_advantage":float(d.mean()),"bootstrap95":(float(lb),float(ub)),
                    "wins":int((d>0).sum()),"mean_mlp":float(np.mean([a["test_metric"] for a,b in compare])),
                    "mean_competitor":float(np.mean([b["test_metric"] for a,b in compare])),
                    "mean_mlp_macs":float(np.mean([a["estimated_macs_per_example"] for a,b in compare])),
                    "mean_competitor_macs":float(np.mean([b["estimated_macs_per_example"] for a,b in compare]))})
    return out


def produce(directory):
    d=Path(directory)
    rows=[json.loads(s) for s in (d/"runs.jsonl").read_text().splitlines() if s.strip()]
    pairs=paired(rows)
    route=paired(rows,reference="learned_product",challenger="routed_product")
    def table(recs):
        lines=["| Dataset (source) | Mean advantage | Exploratory 95% bootstrap CI | Wins / seeds | MACs ref → candidate |",
               "|---|---:|---:|---:|---:|"]
        for r in recs:
            a,b=r["bootstrap95"]
            lines.append(f'| {r["dataset"]} ({r["provenance"]}) | {r["mean_advantage"]:+.4f} | [{a:+.4f}, {b:+.4f}] | {r["wins"]}/{r["n"]} | {r["mean_mlp_macs"]:.0f} → {r["mean_competitor_macs"]:.0f} |')
        return "\n".join(lines)
    lines=["# Empirical results (automatically generated)",
           "",f"**Source:** `{d}/runs.jsonl` (each row = an actually executed training run).", "",
           "**Interpretation:** Advantage is positive when candidate is better. Classification: accuracy-point difference; regression: reduction in normalized MSE. Scores are held-out, checkpoints selected exclusively by validation loss. All synthetic and real provenance is tagged per run. Confidence intervals are exploratory uncorrected bootstrap over seed-level paired runs; real splits overlap, and multiple dataset comparisons are not corrected.","",
           "## Learned multiplication vs MLP",table(pairs),"",
           "## Input-routed multiplication vs always-executed learned multiplication",table(route),"",
           "## What this does NOT prove", "",
           "- Degree doubling is algebraic, not a measured learning-efficiency gain.",
           "- Strong synthetic performance is not evidence of advantage on natural or deployment data.",
           "- A significant positive pairwise score does not mean superlinear capability relative to *actual* inference cost.",
           "- Node counts/MAC estimates omit memory, routing overhead, energy and device effects; CPU batch latency is measured separately in the raw runs.",
           "- The learned product network uses a bounded, ordered graph grammar; it is not unconstrained architecture discovery.",""]
    report='\n'.join(lines)
    (d/'REPORT.md').write_text(report)
    print(report)
    return pairs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',default='results/main',nargs='?')
    produce(p.parse_args().directory)

if __name__=='__main__':main()
