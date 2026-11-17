"""Compara o grafo do frame.py com o grafo de Frame.calc_end/calculate_similarity do HieTaSumm-lib (T01)."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np

from tcc_gnn.graph.frame import build_time_aware_graph
from tcc_gnn.reference import reference_graph


def _edge_weights(graph: nx.Graph) -> dict[tuple[int, int], float]:
    """Peso por aresta `(menor, maior)`."""
    return {(min(u, v), max(u, v)): data["weight"] for u, v, data in graph.edges(data=True)}


def compare(leonardo_graph: nx.Graph, ours_graph: nx.Graph) -> None:
    """Imprime as arestas exclusivas de cada lado e a maior diferença de peso nas comuns."""
    leonardo_edges = _edge_weights(leonardo_graph)
    ours_edges = _edge_weights(ours_graph)

    only_leonardo = set(leonardo_edges) - set(ours_edges)
    only_ours = set(ours_edges) - set(leonardo_edges)
    common = set(leonardo_edges) & set(ours_edges)

    max_weight_diff = max(
        (abs(leonardo_edges[k] - ours_edges[k]) for k in common), default=0.0
    )

    print(f"Arestas só na referência (Leonardo): {len(only_leonardo)}")
    print(f"Arestas só no nosso grafo:           {len(only_ours)}")
    print(f"Arestas em comum:                    {len(common)}")
    print(f"Maior diferença de peso (comuns):    {max_weight_diff:.10f}")

    if only_leonardo or only_ours:
        print("\nFALHOU: os conjuntos de arestas não batem.")
    elif max_weight_diff > 1e-9:
        print("\nFALHOU: pesos não batem numericamente.")
    else:
        print("\nOK: grafos equivalentes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--delta-t", type=int, required=True)
    args = parser.parse_args()

    features = np.load(args.features)
    print(f"Features carregadas: shape {features.shape}")

    compare(reference_graph(features, args.delta_t), build_time_aware_graph(features, args.delta_t))
