"""
MeshGraphNet-style GNN for per-node stress prediction.

Architecture follows Pfaff et al. 2021 (`pfaff2021meshgraphnets`):
encoder -> L-layer processor with residual connections -> decoder.

The skeleton is agnostic to input-feature dimensionality: the caller passes
`in_node_dim` and `in_edge_dim` and is responsible for producing features of
the matching width. `src/models/dataset.py` defines the specific feature
pack this project uses; other callers (e.g. Phase-2 ensemble members) can
reuse this architecture with the same signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


def _mlp(in_dim: int, hidden: int, out_dim: int,
         n_hidden_layers: int = 2, layer_norm: bool = True) -> nn.Module:
    """Standard MLP block: Linear -> ReLU repeated, optional trailing LayerNorm.

    Used for the encoder, processor edge/node updates, and decoder. The
    trailing LayerNorm stabilizes training at depth (MeshGraphNets default).
    """
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden_layers):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.ReLU())
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class _MeshProcessorLayer(MessagePassing):
    """One message-passing layer with edge + node updates and residual skip.

    Edge update:   e_ij' = MLP_e(cat(e_ij, n_i, n_j)) + e_ij
    Node update:   n_i'  = MLP_n(cat(n_i, sum_j e_ij')) + n_i
    """

    def __init__(self, hidden: int):
        super().__init__(aggr="sum", flow="source_to_target")
        self.edge_mlp = _mlp(3 * hidden, hidden, hidden)
        self.node_mlp = _mlp(2 * hidden, hidden, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        edge_input = torch.cat([edge_attr, x[src], x[dst]], dim=-1)
        edge_out = self.edge_mlp(edge_input) + edge_attr
        # Pin `size` so isolated nodes (no incoming edges) still get a row
        # of zeros in `aggr`, keeping it the same length as `x`.
        N = x.size(0)
        aggr = self.propagate(edge_index, x=x, edge_attr=edge_out,
                              size=(N, N))
        node_out = self.node_mlp(torch.cat([x, aggr], dim=-1)) + x
        return node_out, edge_out

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return edge_attr


@dataclass
class MeshGNNConfig:
    in_node_dim: int
    in_edge_dim: int
    hidden: int = 128
    num_layers: int = 5
    out_dim: int = 1


class MeshGNN(nn.Module):
    """MeshGraphNet-style GNN predicting a per-node scalar (von Mises).

    Inputs
    ------
    x           : (N, in_node_dim) node features.
    edge_index  : (2, E) directed edge list \u2014 we assume it already contains
                  both (i, j) and (j, i) entries (i.e. the graph is treated
                  as undirected via symmetric edges).
    edge_attr   : (E, in_edge_dim) edge features.

    Output
    ------
    y_pred      : (N, out_dim) per-node prediction.
    """

    def __init__(self, cfg: MeshGNNConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden
        self.node_encoder = _mlp(cfg.in_node_dim, H, H)
        self.edge_encoder = _mlp(cfg.in_edge_dim, H, H)
        self.processor = nn.ModuleList(
            _MeshProcessorLayer(H) for _ in range(cfg.num_layers)
        )
        self.decoder = _mlp(H, H, cfg.out_dim, layer_norm=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        n = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for layer in self.processor:
            n, e = layer(n, edge_index, e)
        return self.decoder(n)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x, edge_index, edge_attr)


def build_default_model(in_node_dim: int, in_edge_dim: int,
                        hidden: int = 128, num_layers: int = 5) -> MeshGNN:
    return MeshGNN(MeshGNNConfig(
        in_node_dim=in_node_dim, in_edge_dim=in_edge_dim,
        hidden=hidden, num_layers=num_layers, out_dim=1,
    ))
