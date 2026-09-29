"""Checagem qualitativa do grafo construído a partir de um .npy de features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tcc_gnn.graph.frame import build_time_aware_graph


def weight_by_distance(graph) -> dict[int, list[float]]:
    """Agrupa os pesos das arestas pela distância temporal |v1 - v2|."""
    by_distance: dict[int, list[float]] = {}
    for v1, v2, data in graph.edges(data=True):
        distance = abs(v2 - v1)
        if distance == 0:
            continue
        by_distance.setdefault(distance, []).append(data["weight"])
    return by_distance


def summarize(graph, features: np.ndarray) -> None:
    """Imprime estatísticas gerais e a média de peso por distância temporal."""
    weights = np.array([data["weight"] for _, _, data in graph.edges(data=True)])

    print(f"Nós: {graph.number_of_nodes()}  Arestas: {graph.number_of_edges()}")
    print(f"Peso mínimo:  {weights.min():.4f}")
    print(f"Peso máximo:  {weights.max():.4f}")
    print(f"Peso médio:   {weights.mean():.4f}")
    print(f"NaN/Inf?      {not np.isfinite(weights).all()}")

    by_distance = weight_by_distance(graph)
    distances = sorted(by_distance)
    means = [float(np.mean(by_distance[d])) for d in distances]

    print("\nPeso médio por distância temporal |v1 - v2|:")
    for d, m in zip(distances, means):
        print(f"  distância {d:>2}: peso médio {m:.4f}  (n={len(by_distance[d])})")

    if len(distances) >= 2:
        correlation = float(np.corrcoef(distances, means)[0, 1])
        print(f"\nCorrelação (distância x peso médio): {correlation:.4f}")
        print(
            "Esperado: próxima de +1 (peso cresce com a distância temporal, "
            "indicando que frames mais distantes tendem a ser mais dissimilares)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--delta-t", type=int, required=True)
    args = parser.parse_args()

    features = np.load(args.features)
    print(f"Features carregadas: shape {features.shape}")

    graph = build_time_aware_graph(features, args.delta_t)
    summarize(graph, features)