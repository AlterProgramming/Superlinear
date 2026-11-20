import pytest
import torch
from multisuper.models import LearnedProductNet,make_model

@pytest.mark.parametrize('kind',['linear','mlp','fixed_random','learned_product','routed_product'])
def test_shapes_and_gradients(kind):
    torch.manual_seed(0)
    net=make_model(kind,12,3)
    x=torch.randn(17,12)
    y,stats=net(x,return_stats=True)
    assert y.shape==(17,3)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert 0<=stats['executed_products']<=8


def test_learned_operands_get_gradient():
    net=LearnedProductNet(6,1,nodes=3)
    net(torch.randn(64,6)).square().mean().backward()
    assert all(w.grad is not None for w in net.left)
    assert all(w.grad is not None for w in net.right)


def test_hard_route_skips_operations():
    net=LearnedProductNet(4,1,nodes=5,gate='conditional')
    with torch.no_grad():
        net.route.weight.zero_()
        net.route.bias.fill_(-20)
    net.eval()
    _,stats=net(torch.randn(32,4),return_stats=True)
    assert stats['executed_products']==0
    with torch.no_grad(): net.route.bias.fill_(20)
    _,stats=net(torch.randn(32,4),return_stats=True)
    assert stats['executed_products']==5


def test_selective_conditional_execution():
    net=LearnedProductNet(2,1,nodes=1,gate='conditional')
    with torch.no_grad():
        net.route.weight[:] = torch.tensor([[5.,0.]])
        net.route.bias.zero_()
    net.eval()
    x=torch.tensor([[1.,2.],[-1.,3.]])
    _,stats=net(x,return_stats=True)
    assert stats['executed_products']==.5


def test_no_named_feature_pairs_in_learned_model():
    model=LearnedProductNet(9,1,nodes=5)
    assert all(model.left[i].numel()==9+i for i in range(5))
    assert all(model.right[i].numel()==9+i for i in range(5))


def test_reject_invalid_gate():
    with pytest.raises(ValueError): LearnedProductNet(4,1,gate='unknown')
