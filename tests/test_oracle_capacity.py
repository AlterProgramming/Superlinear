"""Representability proof by explicit parameter assignment, NOT an SGD success.

The actual learned-operand architecture has a feasible exact solution for hidden
rotated higher-order products. Failure to train is not automatically incapacity.
"""
import numpy as np
import pytest
import torch
from multisuper.models import LearnedProductNet


def set_oracle(model, orthogonal, degree):
    """Set just 3 nodes for quartic or 5 for sextic using raw linear forms."""
    d=model.dim
    with torch.no_grad():
        for p in list(model.left)+list(model.right)+list(model.left_bias)+list(model.right_bias):
            p.zero_()
        model.readout.weight.zero_()
        model.readout.bias.zero_()
        for i in range(degree//2):
            model.left[i][:d].copy_(torch.from_numpy(orthogonal[:,2*i]).float())
            model.right[i][:d].copy_(torch.from_numpy(orthogonal[:,2*i+1]).float())
        if degree==4:
            model.left[2][d+0]=1
            model.right[2][d+1]=1
            model.readout.weight[0,d+2]=1
        elif degree==6:
            model.left[3][d+0]=1
            model.right[3][d+1]=1
            model.left[4][d+3]=1
            model.right[4][d+2]=1
            model.readout.weight[0,d+4]=1
        else: raise ValueError(degree)

@pytest.mark.parametrize('degree',[4,6])
@pytest.mark.parametrize('seed',[0,1,2,3])
def test_exact_rotated_polynomial_is_representable(degree,seed):
    rng=np.random.default_rng(seed)
    q,_=np.linalg.qr(rng.normal(size=(12,12)))
    x=rng.uniform(-1,1,size=(100,12)).astype(np.float32)
    model=LearnedProductNet(12,1,nodes=degree//2+(1 if degree==4 else 2))
    set_oracle(model,q,degree)
    model.eval()
    got=model(torch.from_numpy(x)).detach().numpy().squeeze(1)
    want=np.prod((x @ q)[:,:degree],axis=1)
    np.testing.assert_allclose(got,want,atol=2e-6,rtol=2e-5)
