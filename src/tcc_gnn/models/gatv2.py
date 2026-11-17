"""GATv2 sobre o grafo time-aware (T09, D-06): conversão do grafo para o PyG e o bloco GP do SLC-HELM (Eqs. 1–2)."""

from __future__ import annotations

import networkx as nx
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


def time_aware_edge_index(graph: nx.Graph) -> torch.Tensor:
    """`edge_index` `(2, 2|E|)` do PyG com cada aresta nos dois sentidos, ordenado; nós devem ser `0..n-1`, sem self-loops."""
    n = graph.number_of_nodes()
    if n == 0:
        raise ValueError("Grafo sem nós")
    if set(graph.nodes) != set(range(n)):
        raise ValueError("Os nós do grafo devem ser os inteiros 0..n-1 (índice do frame)")
    if nx.number_of_selfloops(graph):
        raise ValueError("O grafo tem self-loops; a Eq. 1 do SLC-HELM agrega só os vizinhos")
    pairs = sorted({(u, v) for a, b in graph.edges for u, v in ((a, b), (b, a))})
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


class GraphProcessingBlock(nn.Module):
    """Bloco GP: h⁰ = Linear(x); ĥ⁰ = GATv2 sobre os vizinhos (Eqs. 1–2); h¹ = BN(ĥ⁰ + h⁰); dropout na saída."""

    def __init__(self, in_dim: int, embedding_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embedding = nn.Linear(in_dim, embedding_size)
        self.conv = GATv2Conv(
            embedding_size,
            embedding_size,
            heads=heads,
            concat=False,
            dropout=0.0,
            add_self_loops=False,
            edge_dim=None,
            bias=False,
            share_weights=False,
        )
        self.norm = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """`(n, embedding_size)`; com `return_attention`, também `(edge_index, β)` da GATv2, `β` com forma `(|E|, heads)`."""
        if x.dim() != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"x deve ter forma (n, {self.in_dim}), veio {tuple(x.shape)}")
        if edge_index.numel() and int(edge_index.max()) >= x.shape[0]:
            raise ValueError("edge_index aponta para um nó que não existe em x")
        h0 = self.embedding(x)
        h_hat, attention = self.conv(h0, edge_index, return_attention_weights=True)
        h1 = self.dropout(self.norm(h_hat + h0))
        return (h1, attention) if return_attention else h1


def graph_processing_block(config: dict, in_dim: int) -> GraphProcessingBlock:
    """Bloco GP com `embedding_size`, `gat_heads` e `dropout` da seção `model` do yaml."""
    model = config["model"]
    return GraphProcessingBlock(in_dim, model["embedding_size"], model["gat_heads"], model["dropout"])
