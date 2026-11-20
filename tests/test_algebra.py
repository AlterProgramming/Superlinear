import numpy as np
import pytest
from multisuper.algebra import homogeneous_product_stack, scaling_report

@pytest.mark.parametrize('depth',range(1,7))
@pytest.mark.parametrize('scale',[0.5,1.125,1.7])
def test_exact_homogeneity(depth,scale):
    rng=np.random.default_rng(3)
    x=rng.normal(size=(50,4))*.3
    layers=[(rng.normal(0,.2,(4,4)),rng.normal(0,.2,(4,4))) for _ in range(depth)]
    y=homogeneous_product_stack(x,layers)
    yt=homogeneous_product_stack(scale*x,layers)
    np.testing.assert_allclose(yt,scale**(2**depth)*y,rtol=2e-11,atol=1e-280)


def test_nonadditivity_product():
    f=lambda x:x[0]*x[1]
    a=np.array([1.,0.]); b=np.array([0.,1.])
    assert f(a+b)!=f(a)+f(b)


def test_algebra_report():
    assert len(scaling_report())==6
    assert max(x['relative_error'] for x in scaling_report())<1e-9
