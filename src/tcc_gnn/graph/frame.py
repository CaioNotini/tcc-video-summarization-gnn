"""Construção do grafo time-aware (nós = frames, arestas = janela temporal)."""

from __future__ import annotations
import networkx as nx
import numpy as np


def calc_window_end(i: int, delta_t: int, n_frames: int) -> int:
    if (i + delta_t) > n_frames or delta_t < 0:
        return n_frames
    return i + delta_t


def edge_weight(feature_a: np.ndarray, feature_b: np.ndarray) -> float:
    return float(np.linalg.norm(feature_a - feature_b, ord=1) * 100 / len(feature_b))


def build_time_aware_graph(features: np.ndarray, delta_t: int) -> nx.Graph:
    n_frames = len(features)
    RG = nx.Graph()
    RG.add_nodes_from(range(n_frames))

    for v1 in range(n_frames):
        end = calc_window_end(v1, delta_t, n_frames)
        for v2 in range(v1, end):
            if v1 == v2:
                continue
            w = edge_weight(features[v1], features[v2])
            RG.add_edge(v1, v2, weight=w)

    return RG