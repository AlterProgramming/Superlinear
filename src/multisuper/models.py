"""Trainable learned-operand multiplication and matched inductive-bias controls.

Nodes discover operands among raw inputs and all previously created nodes.
There are no manually selected variable pairs. This *isn't* an unconstrained
self-modifying graph: the maximum number of nodes and their sequential order
are architectural priors, and conditional skips are learned within that bound.
"""
import math
import torch
from torch import nn


class LinearNet(nn.Module):
    def __init__(self, dim, outputs):
        super().__init__()
        self.readout = nn.Linear(dim, outputs)

    def forward(self, x, return_stats=False):
        y = self.readout(x)
        return (y, {"mean_active": 0.0, "executed_products": 0.0, "estimated_macs_per_example": float(self.readout.in_features*self.readout.out_features)}) if return_stats else y


class MLPNet(nn.Module):
    def __init__(self, dim, outputs, hidden=24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, outputs))

    def forward(self, x, return_stats=False):
        y = self.net(x)
        return (y, {"mean_active": 0.0, "executed_products": 0.0, "estimated_macs_per_example": float(self.net[0].in_features*self.net[0].out_features+self.net[2].in_features*self.net[2].out_features)}) if return_stats else y


class FixedRandomProductNet(nn.Module):
    """Frozen random pairwise operand projections; only linear readout trained.

    This is an *easier fixed-topology proxy*, not our learned-graph proposal.
    """
    def __init__(self, dim, outputs, nodes=8):
        super().__init__()
        self.nodes = nodes
        for i in range(nodes):
            self.register_buffer(f"a{i}", torch.randn(dim+i)/math.sqrt(dim+i))
            self.register_buffer(f"b{i}", torch.randn(dim+i)/math.sqrt(dim+i))
        self.readout = nn.Linear(dim+nodes, outputs)

    def forward(self, x, return_stats=False):
        states = [x]
        for i in range(self.nodes):
            s = torch.cat(states, dim=-1)
            states.append(((s @ getattr(self, f"a{i}"))*(s @ getattr(self, f"b{i}"))).unsqueeze(-1))
        y = self.readout(torch.cat(states, dim=-1))
        return (y, {"mean_active": 1.0, "executed_products": float(self.nodes), "estimated_macs_per_example": float(sum(2*(self.readout.in_features-self.nodes+i) for i in range(self.nodes))+self.readout.in_features*self.readout.out_features)}) if return_stats else y


class LearnedProductNet(nn.Module):
    """Learns operands and optionally input-conditioned binary computation routes.

    gate='none': all computation is executed.
    gate='conditional': a straight-through Bernoulli threshold during training
      with sigmoid surrogate gradients; actual skipped computation at inference.
    Gate decisions depend on raw input and are not a predefined feature lookup.
    """
    def __init__(self, dim, outputs, nodes=8, gate="none"):
        super().__init__()
        if gate not in ("none", "conditional"):
            raise ValueError(gate)
        self.dim = dim
        self.nodes = nodes
        self.gate = gate
        self.left = nn.ParameterList()
        self.right = nn.ParameterList()
        self.left_bias = nn.ParameterList()
        self.right_bias = nn.ParameterList()
        for i in range(nodes):
            d = dim+i
            self.left.append(nn.Parameter(torch.randn(d)/math.sqrt(d)))
            self.right.append(nn.Parameter(torch.randn(d)/math.sqrt(d)))
            self.left_bias.append(nn.Parameter(torch.zeros(1)))
            self.right_bias.append(nn.Parameter(torch.zeros(1)))
        if gate == "conditional":
            self.route = nn.Linear(dim, nodes)
            nn.init.normal_(self.route.weight, std=0.02)
            nn.init.constant_(self.route.bias, 1.0)
        self.readout = nn.Linear(dim+nodes, outputs)
        self._last_cost = None

    def forward(self, x, return_stats=False):
        states = [x]
        batch = x.shape[0]
        probs = torch.sigmoid(self.route(x)) if self.gate == "conditional" else None
        actual_active = []
        for i in range(self.nodes):
            if probs is None:
                active = x.new_ones(batch)
            else:
                p = probs[:, i]
                hard = (p > .5).to(x.dtype)
                active = hard - p.detach() + p if self.training else hard
            if not self.training and bool((active == 0).all()):
                # Skip the two dot products entirely if nobody uses this node.
                node = x.new_zeros(batch, 1)
            elif not self.training and bool((active == 0).any()):
                # Compute only for active samples, to make conditional execution real.
                indices = torch.where(active > 0)[0]
                s = torch.cat(states, dim=-1)
                small = s.index_select(0, indices)
                product = (small @ self.left[i] + self.left_bias[i]) * (small @ self.right[i] + self.right_bias[i])
                node = x.new_zeros(batch, 1).index_copy(0, indices, product.unsqueeze(-1))
            else:
                s = torch.cat(states, dim=-1)
                node = (((s @ self.left[i] + self.left_bias[i]) * (s @ self.right[i] + self.right_bias[i])) * active).unsqueeze(-1)
            states.append(node)
            actual_active.append((active.detach() > .5).to(x.dtype).mean())
        y = self.readout(torch.cat(states, dim=-1))
        mean_active = torch.stack(actual_active).mean()
        # Differentiable expected cost for objective; measurable actual cost for eval.
        self._last_cost = probs.mean() if probs is not None else x.new_tensor(1.0)
        active_rates = [float(v.detach()) for v in actual_active]
        # Approximate scalar multiply-accumulate counts (MACs) per example.
        # Includes learned projections, classifier readout and route predictor;
        # excludes framework overhead and elementwise operations.
        macs = sum(rate*(2*(self.dim+i)) for i,rate in enumerate(active_rates))
        macs += (self.dim+self.nodes)*self.readout.out_features
        if self.gate == "conditional": macs += self.dim*self.nodes
        stats = {"mean_active": float(mean_active.detach()),
                 "executed_products": float((mean_active*self.nodes).detach()),
                 "active_by_node": active_rates,
                 "estimated_macs_per_example": float(macs)}
        return (y, stats) if return_stats else y


def make_model(name, dim, outputs, nodes=8):
    if name == "linear": return LinearNet(dim, outputs)
    if name == "mlp": return MLPNet(dim, outputs, hidden=24)
    if name == "fixed_random": return FixedRandomProductNet(dim, outputs, nodes=nodes)
    if name == "learned_product": return LearnedProductNet(dim, outputs, nodes=nodes)
    if name == "routed_product": return LearnedProductNet(dim, outputs, nodes=nodes, gate="conditional")
    raise ValueError(f"Unknown model: {name}")
