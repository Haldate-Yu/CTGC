import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv.gcn_conv import gcn_norm


class SSGConv(MessagePassing):
    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, dropout: float = 0.05, **kwargs):
        super(SSGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.dropout = dropout

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

            alpha = 0.05
            output = alpha * x

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)

                output = output + (1 - alpha) * (1. / self.K) * x

            x = output
            if self.cached:
                self._cached_x = x
        else:
            x = cache.detach()

        return self.lin(x)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


class CTGConv(MessagePassing):
    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 alpha: float = 0.1, aggr_type: str = 'sgc', norm_type: str = 'arctan',
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, dropout: float = 0.05, **kwargs):
        super(CTGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K

        self.alpha = alpha
        self.aggr_type = aggr_type
        self.norm_type = norm_type

        self.cached = cached
        self.add_self_loops = add_self_loops
        self.dropout = dropout

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

            output = self.alpha * x

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if self.aggr_type == 'ssgc':
                    output = output + (1 - self.alpha) * (1. / self.K) * x
                elif self.aggr_type == 'ssgc_no_avg':
                    output = output + (1 - self.alpha) * x

            if self.aggr_type == 'sgc':
                output = output + (1 - self.alpha) * x

            x = output

            if self.cached:
                self._cached_x = x
        else:
            x = cache.detach()

        if self.norm_type == 'arctan':
            x = 2 * torch.arctan(x) / math.pi
        elif self.norm_type == 'f2':
            x = F.normalize(x, p=2, dim=1)

        return self.lin(x)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, K={}, alpha={}, aggr={}, norm={})'.format(self.__class__.__name__,
                                                                     self.in_channels, self.out_channels,
                                                                     self.K, self.alpha, self.aggr_type, self.norm_type)
