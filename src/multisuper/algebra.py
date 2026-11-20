"""Algebraic, exact mathematical property vs experimentally learned behavior."""
import numpy as np


def homogeneous_product_stack(x, weights):
    """Every layer multiplies two homogeneous linear combinations of prior layer.

    x is shape [batch, dim]; weights[l] is tuple (left, right) each [dim, dim].
    No biases/normalization/activation: output has degree exactly 2**len(weights)
    unless degenerate coefficients cancel. This does NOT prove learnability.
    """
    v = x
    for a,b in weights:
        v = (v @ a) * (v @ b)
    return v


def scaling_report(seed=0,max_depth=6):
    rng=np.random.default_rng(seed)
    x = rng.normal(0,0.4,(64,4))
    layers=[(rng.normal(0,.2,(4,4)),rng.normal(0,.2,(4,4))) for _ in range(max_depth)]
    rows=[]
    for depth in range(1,max_depth+1):
        ws=layers[:depth]
        a=homogeneous_product_stack(x,ws)
        t=1.125
        b=homogeneous_product_stack(t*x,ws)
        expected=t**(2**depth)
        observed=float(np.linalg.norm(b)/np.linalg.norm(a)) if np.linalg.norm(a)>1e-290 else float('nan')
        rows.append({"depth":depth,"theoretical_degree":2**depth,
                     "expected_scaling":expected,"observed_scaling":observed,
                     "relative_error":abs(observed-expected)/expected})
    return rows
