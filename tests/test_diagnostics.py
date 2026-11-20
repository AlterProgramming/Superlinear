import torch
from multisuper.diagnose import snapshot,node_values
from multisuper.models import LearnedProductNet


def test_diagnostic_sees_all_nodes_and_gradients():
    torch.manual_seed(3)
    net=LearnedProductNet(4,1,nodes=5)
    x=torch.randn(20,4)
    values=node_values(net,x)
    assert len(values)==5
    rows=snapshot(net,x,torch.randn(20),0,'synthetic_test',3)
    assert len(rows)==5
    assert all(row['raw_operand_grad_norm']>=0 for row in rows)
    assert all(row['output_sd']>=0 for row in rows)
