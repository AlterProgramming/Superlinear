"""CLI: held-out datasets, validation selection, genuine training, repeat seeds.

Does not use test or OOD scores for model selection. The unit of one run is
(model, data, seed); each produces reproducible JSONL record.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import SYNTHETIC, REAL, load_dataset
from .models import make_model
from .algebra import scaling_report

MODELS=("linear","mlp","fixed_random","learned_product","routed_product")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metric(model,x,y,task):
    model.eval()
    with torch.no_grad():
        p, stats = model(x,return_stats=True)
        if task=="classification":
            value=float((p.argmax(1)==y).float().mean())
            loss=float(F.cross_entropy(p,y))
        else:
            value=float(F.mse_loss(p.squeeze(1),y))
            loss=value
    return value,loss,stats


def run_one(name, model_name, seed, epochs=140, nodes=8, cost_penalty=.01, train_n=1024, checkpoint_path=None):
    set_seed(seed)
    data=load_dataset(name,seed=seed,train_n=train_n)
    d=data.X_train.shape[1]
    out= int(np.max(data.y_train))+1 if data.task=="classification" else 1
    model=make_model(model_name,d,out,nodes=nodes)
    par=sum(t.numel() for t in model.parameters() if t.requires_grad)
    xt=torch.from_numpy(data.X_train)
    yt=torch.from_numpy(data.y_train)
    xv=torch.from_numpy(data.X_val)
    yv=torch.from_numpy(data.y_val)
    xe=torch.from_numpy(data.X_test)
    ye=torch.from_numpy(data.y_test)
    lr=.015 if model_name in ("learned_product","routed_product") else .025
    optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    # Real, independent validation selection, not test-driven tuning.
    best=float('inf'); best_state=None; best_epoch=0
    tic=time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        pred=model(xt)
        loss=(F.cross_entropy(pred,yt) if data.task=="classification" else F.mse_loss(pred.squeeze(1),yt))
        if model_name=="routed_product": loss = loss + cost_penalty*model._last_cost
        if not bool(torch.isfinite(loss)):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optim.step()
        # Validation selection every 5 steps to avoid unequal tuning.
        if epoch%5==0 or epoch==epochs-1:
            _,vl,_=metric(model,xv,yv,data.task)
            if vl<best:
                best=vl; best_epoch=epoch
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    secs=time.perf_counter()-tic
    if best_state is None: raise RuntimeError("Training produced no finite checkpoint")
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        path=Path(checkpoint_path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"state_dict":best_state,"dataset":name,"model":model_name,"seed":seed,
                    "nodes":nodes,"train_n":train_n,"epochs":epochs,"best_epoch":best_epoch},path)
    test, test_loss,stats=metric(model,xe,ye,data.task)
    val_score,_,_=metric(model,xv,yv,data.task)
    # Median elapsed inference time, same held-out input batch on CPU for all models.
    with torch.no_grad():
        model.eval()
        for _ in range(3): model(xe)
        latency=[]
        for _ in range(9):
            st=time.perf_counter()
            model(xe)
            latency.append((time.perf_counter()-st)*1000)
    median_latency_ms=float(np.median(latency))
    row={"dataset":name,"provenance":data.provenance,"source":data.source,"task":data.task,
         "model":model_name,"seed":seed,"train_size":len(xt),"val_size":len(xv),"test_size":len(xe),
         "parameters":par,"epochs_budget":epochs,"best_epoch":best_epoch,"seconds_train":round(secs,5),
         "nodes_budget":nodes,"train_n_requested":train_n,"batch_inference_ms":median_latency_ms,
         "estimated_macs_per_example":stats["estimated_macs_per_example"],
         "validation_metric":val_score,"test_metric":test,"test_loss":test_loss,
         "metric":"accuracy" if data.task=="classification" else "normalized_mse",
         "mean_active_fraction":stats["mean_active"],"mean_products_executed_per_example":stats["executed_products"],
         "nonzero_route_penalty":cost_penalty if model_name=="routed_product" else 0}
    if data.X_ood is not None:
        xo=torch.from_numpy(data.X_ood)
        yo=torch.from_numpy(data.y_ood)
        ov,_,_=metric(model,xo,yo,data.task)
        if np.isfinite(ov):
            row["ood_normalized_mse"]=ov
        else:
            # Keep a machine-readable failure flag; strict JSON cannot store NaN/Inf.
            row["ood_normalized_mse"]=None
            row["ood_nonfinite"]=str(ov)
    return row


def summarize(rows):
    # Summary is for documentation only; no inferential claim from mean alone.
    groups={}
    for r in rows: groups.setdefault((r["dataset"],r["model"]),[]).append(r)
    out=[]
    for (ds,m),records in sorted(groups.items()):
        scores=[r["test_metric"] for r in records]
        out.append({"dataset":ds,"provenance":records[0]["provenance"],"model":m,
                    "metric":records[0]["metric"],"n_seeds":len(scores),
                    "test_mean":float(np.mean(scores)),"test_sd":float(np.std(scores,ddof=1)) if len(scores)>1 else None,
                    "mean_train_seconds":float(np.mean([r["seconds_train"] for r in records])),
                    "mean_products_per_example":float(np.mean([r["mean_products_executed_per_example"] for r in records])),
                    "parameters":records[0]["parameters"],
                    "mean_macs_per_example":float(np.mean([r["estimated_macs_per_example"] for r in records])),
                    "median_batch_inference_ms":float(np.median([r["batch_inference_ms"] for r in records]))})
    return out


def run_suite(datasets,models,seeds,epochs,nodes,out_dir,cost_penalty=.01,train_n=1024):
    os.environ.setdefault("OMP_NUM_THREADS","1")
    torch.set_num_threads(1)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    existing=[]
    filepath=out/"runs.jsonl"
    if filepath.exists():
        existing=[json.loads(s) for s in filepath.read_text().splitlines() if s.strip()]
    keys={(r["dataset"],r["model"],r["seed"],r["epochs_budget"],r.get("nodes_budget",8),r.get("train_n_requested",1024),r.get("nonzero_route_penalty",0)) for r in existing}
    rows=list(existing)
    for name in datasets:
        for model in models:
            for seed in seeds:
                key=(name,model,seed,epochs,nodes,train_n,cost_penalty if model=="routed_product" else 0)
                if key in keys: continue
                r=run_one(name,model,seed,epochs=epochs,nodes=nodes,cost_penalty=cost_penalty,train_n=train_n)
                rows.append(r)
                with filepath.open("a") as f: f.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
                print(f"{name:15} {model:18} seed={seed} {r['metric']}={r['test_metric']:.4f} products={r['mean_products_executed_per_example']:.2f} {r['seconds_train']:.2f}s",flush=True)
    summary=summarize(rows)
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys()));w.writeheader();w.writerows(summary)
    (out/"algebra.json").write_text(json.dumps(scaling_report(),indent=2))
    (out/"config.json").write_text(json.dumps({"datasets":datasets,"models":models,"seeds":seeds,"epochs":epochs,"nodes":nodes,"train_n":train_n,"cost_penalty":cost_penalty,
                      "torch":torch.__version__,"numpy":np.__version__},indent=2))
    print("\nWrote",len(rows),"runs to",str(out),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--datasets",nargs="+",default=list(SYNTHETIC)+list(REAL),choices=list(SYNTHETIC)+list(REAL))
    p.add_argument("--models",nargs="+",default=list(MODELS),choices=list(MODELS))
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--epochs",type=int,default=140)
    p.add_argument("--nodes",type=int,default=8)
    p.add_argument("--cost-penalty",type=float,default=.01)
    p.add_argument("--train-n",type=int,default=1024)
    p.add_argument("--out",default="results/pilot")
    a=p.parse_args()
    run_suite(a.datasets,a.models,a.seeds,a.epochs,a.nodes,a.out,a.cost_penalty,a.train_n)

if __name__=="__main__": main()
