import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_softmax


class BehaviorConv(MessagePassing):
    """Behavior-aware graph convolution supporting GCN and target-aware normalization."""

    def __init__(self, dim, norm='gcn'):
        super().__init__(aggr='add')
        self.dim = dim
        self.norm = norm

    def _normalize_gcn(self, edge_index, n_nodes):
        ones = torch.ones(edge_index.size(1), device=edge_index.device)
        src, dst = edge_index[0], edge_index[1]
        deg = scatter_add(ones, dst, dim=0, dim_size=n_nodes)
        d_inv = deg.pow(-0.5).masked_fill(deg == 0, 0)
        return edge_index, d_inv[src] * ones * d_inv[dst]

    def _normalize_target(self, edge_index, x, target, n_nodes):
        src, dst = edge_index[0], edge_index[1]
        score = (target[src] * x[dst]).sum(dim=-1, keepdim=True)
        weight = scatter_softmax(score, src, dim=0, dim_size=n_nodes)
        return edge_index, weight

    def forward(self, x, edge_index, target_emb=None):
        n = x.size(0)
        if self.norm == 'target':
            edge_index, ew = self._normalize_target(edge_index, x, target_emb, n)
        elif self.norm == 'gcn':
            edge_index, ew = self._normalize_gcn(edge_index, n)
        else:
            raise ValueError(f'Unknown norm: {self.norm}')
        return self.propagate(edge_index, x=x, edge_weight=ew)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j
