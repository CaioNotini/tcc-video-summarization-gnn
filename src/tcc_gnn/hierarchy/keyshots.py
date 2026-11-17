"""Keyshots do baseline HieTaSumm, porta de `Summary.group_central_frames` (HieTaSumm-lib, D-02, D-23)."""

from __future__ import annotations

import networkx as nx


def initial_len_shot(n_frames: int, percent: float) -> int:
    """Orçamento total de keyshots em frames: `round(n_frames * percent/100) - 1` (`Summary.py:56`)."""
    # fiel ao original: com percent = 15, menos de 7 frames dá orçamento 0 ou -1
    return round(n_frames * (percent / 100)) - 1


def group_central_frames(graph: nx.Graph, leaflist: list[int], len_shot: int) -> list[int]:
    """Porta de `Summary.group_central_frames` (`Summary.py:167-192`) sem o `cp`: keyshots na ordem em que o original os grava."""
    S = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    len_shot = int(len_shot / len(S))  # fiel ao original: grafo sem nós divide por zero
    leaf_set = set(leaflist)
    keyshots: list[int] = []
    for c in range(len(S)):
        # fiel ao original: o "centro" é o da ordem de nós do networkx no subgrafo, não a ordem temporal
        comp_leaf_list = [node for node in list(S[c]) if node in leaf_set]
        len_leaf_list = len(comp_leaf_list)
        cn = int(len_leaf_list / 2)  # nó central da estratégia de keyframe
        if not (cn == 0):
            if len_leaf_list < len_shot:
                init_keyshots = 0
                end_keyshots = len_leaf_list - 1  # fiel ao original: o range exclusivo deixa a última folha de fora
            else:
                init_keyshots = cn - (int(len_shot / 2))
                end_keyshots = cn + (int(len_shot / 2))
            for k in range(init_keyshots, end_keyshots):
                keyshots.append(comp_leaf_list[k])
    return keyshots
