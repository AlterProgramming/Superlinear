"""Reproduce a bad observed test result; keep evidence rather than hiding it."""
import json
from pathlib import Path
import torch
from .experiment import run_one,metric
from .data import load_dataset
from .models import make_model


def main():
    torch.set_num_threads(1)
    root=Path('results/incident_quartic16_seed2');root.mkdir(parents=True,exist_ok=True)
    filepath=root/'model.pt'
    result=run_one('quartic','routed_product',seed=2,epochs=180,nodes=16,checkpoint_path=filepath)
    obj=torch.load(filepath,weights_only=True)
    data=load_dataset('quartic',seed=2)
    net=make_model('routed_product',12,1,nodes=16)
    net.load_state_dict(obj['state_dict'])
    net.eval()
    distributions={}
    with torch.no_grad():
        for label,X,y in [('train',data.X_train,data.y_train),('val',data.X_val,data.y_val),('test',data.X_test,data.y_test)]:
            yp=net(torch.from_numpy(X)).squeeze().numpy()
            err=(yp-y)**2
            idx=err.argsort()[-5:][::-1]
            distributions[label]={"n":len(y),"mse":float(err.mean()),"max_abs_prediction":float(abs(yp).max()),
                                  "max_abs_target":float(abs(y).max()),
                                  "largest_errors":[{"index":int(i),"prediction":float(yp[i]),"target":float(y[i]),"squared_error":float(err[i])} for i in idx]}
    # Trace an anomalous held-out example through the learned sequential graph.
    with torch.no_grad():
        xx=torch.from_numpy(data.X_test[114:115])
        routes=(torch.sigmoid(net.route(xx))>.5).squeeze(0)
        states=[xx]
        trace=[]
        for i in range(net.nodes):
            s=torch.cat(states,dim=1)
            left=(s@net.left[i]+net.left_bias[i]).item()
            right=(s@net.right[i]+net.right_bias[i]).item()
            active=bool(routes[i])
            value=left*right if active else 0.0
            trace.append({'node':i,'active':active,'left':left,'right':right,'node_output':value})
            states.append(torch.tensor([[value]]))
    report={'run':result,'sets':distributions,'anomalous_example_114_trace':trace}
    (root/'incident.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
