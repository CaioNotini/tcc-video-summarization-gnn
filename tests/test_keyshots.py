"""Testes para hierarchy/keyshots.py: casos à mão e igualdade com Summary.group_central_frames."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tcc_gnn.hierarchy.keyshots import group_central_frames, initial_len_shot
from tcc_gnn.reference import SUBMODULE, reference_group_central_frames


def _three_components() -> tuple[nx.Graph, list[int]]:
    """A: folhas 0..5 + nó interno 100; B: folha 6 sozinha; C: folhas 7, 8 + nó interno 101."""
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3, 4, 5, 100, 6, 7, 8, 101])
    g.add_edges_from([(0, 100), (1, 100), (2, 100), (3, 100), (4, 100), (5, 100), (7, 101), (8, 101)])
    return g, [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_initial_len_shot() -> None:
    assert initial_len_shot(152, 15) == 22  # round(22.8) - 1
    assert initial_len_shot(716, 15) == 106  # round(107.4) - 1


def test_ordem_das_folhas_vem_do_subgrafo() -> None:
    # comp_leaf_list segue a ordem de list(S[c]) do networkx, como no original: em C é [8, 101, 7]
    g, _ = _three_components()
    assert [list(g.subgraph(c).copy()) for c in nx.connected_components(g)] == [
        [0, 1, 2, 3, 4, 5, 100],
        [6],
        [8, 101, 7],
    ]


def test_tres_ramos_orcamento_3_por_componente() -> None:
    # len_shot = 3; A: cn = 3, k em [2, 4) -> 2, 3; B: cn = 0 -> nada; C: [8, 7], 2 < 3, k em [0, 1) -> 8
    g, leaves = _three_components()
    assert group_central_frames(g, leaves, 9) == [2, 3, 8]


def test_tres_ramos_orcamento_4_por_componente() -> None:
    # len_shot = int(12 / 3) = 4. A: cn = 3, k em [1, 5) -> 1..4. C: 2 < 4 -> 8.
    g, leaves = _three_components()
    assert group_central_frames(g, leaves, 12) == [1, 2, 3, 4, 8]


needs_submodule = pytest.mark.skipif(
    not (SUBMODULE / "HieTaSumm" / "Summary.py").exists(),
    reason="submódulo HieTaSumm-lib não baixado",
)


@needs_submodule
@pytest.mark.parametrize("len_shot", [9, 12, 3, 0])
def test_igual_a_referencia_casos_a_mao(len_shot: int) -> None:
    g, leaves = _three_components()
    assert group_central_frames(g, leaves, len_shot) == reference_group_central_frames(g, leaves, len_shot)


@needs_submodule
def test_igual_a_referencia_em_grafos_aleatorios() -> None:
    rng = np.random.default_rng(3)
    for _ in range(30):
        n_leaves = int(rng.integers(5, 80))
        g = nx.Graph()
        g.add_nodes_from(range(n_leaves))
        next_internal = n_leaves
        start = 0
        while start < n_leaves:  # componentes contíguas de tamanho aleatório, cada uma com um nó interno
            size = int(rng.integers(1, 15))
            members = range(start, min(start + size, n_leaves))
            g.add_edges_from((m, next_internal) for m in members)
            next_internal += 1
            start += size
        leaves = list(range(n_leaves))
        len_shot = int(rng.integers(0, n_leaves))
        assert group_central_frames(g, leaves, len_shot) == reference_group_central_frames(g, leaves, len_shot)
