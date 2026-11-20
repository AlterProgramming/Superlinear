"""Post-hoc optimization diagnostics after seeing high-order failures.

This is explicitly exploratory: do not describe any finding here as preregistered.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_dataset
from .experiment import set_seed
from .models import LearnedProductNet


def node_values(model,x):
    states=[x]
    for i in range(model.nodes):
        s=torch.cat(states,dim=1)
        states.append(((s @ model.left[i]+model.left_bias[i])*(s @ model.right[i]+model.right_bias[i])).unsqueeze(1))
    return states[1:]


def snapshot(model,x,y,epoch,data,seed):
    model.eval()
    model.zero_grad(set_to_none=True)
    pred=model(x)
    loss=F.mse_loss(pred.squeeze(),y)
    loss.backward()
    info=[]
    with torch.no_grad():
        nodes=node_values(model,x)
        for i in range(model.nodes):
            rawgrad=float(torch.cat([model.left[i].grad[:model.dim],model.right[i].grad[:model.dim]]).norm())
            reusegrad=float(torch.cat([model.left[i].grad[model.dim:],model.right[i].grad[model.dim:]]).norm()) if i else float('nan')
            info.append({'dataset':data,'seed':seed,'epoch':epoch,'node':i,
                         'train_mse':float(loss.detach()), 'output_sd':float(nodes[i].std()),
                         'raw_operand_grad_norm':rawgrad,'reuse_operand_grad_norm':reusegrad,
                         'readout_coef_abs':float(model.readout.weight[0,model.dim+i].abs())})
    return info


def trace(dataset,seed,epochs=360,nodes=8):
    set_seed(seed)
    data=load_dataset(dataset,seed)
    x=torch.from_numpy(data.X_train)
    y=torch.from_numpy(data.y_train)
    net=LearnedProductNet(x.shape[1],1,nodes=nodes)
    opt=torch.optim.Adam(net.parameters(),lr=.015,weight_decay=1e-5)
    checkpoints={0,5,20,50,100,180,260,360}
    records=[]
    for epoch in range(epochs+1):
        if epoch in checkpoints:
            records.extend(snapshot(net,x,y,epoch,dataset,seed))
        if epoch==epochs: break
        net.train()
        opt.zero_grad(set_to_none=True)
        loss=F.mse_loss(net(x).squeeze(),y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        opt.step()
    return records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',default=['bilinear','quartic','sextic','switched'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--epochs',type=int,default=360)
    p.add_argument('--out',default='results/diagnostics')
    args=p.parse_args()
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    allrows=[]
    for ds in args.datasets:
        for seed in args.seeds:
            allrows.extend(trace(ds,seed,args.epochs))
            print(ds,seed,'completed',flush=True)
    with (out/'gradient_trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0].keys()))
        writer.writeheader();writer.writerows(allrows)
    print('Saved',len(allrows),'node snapshots',flush=True)

if __name__=='__main__':main()
