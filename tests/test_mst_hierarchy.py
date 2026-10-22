"""Testes para hierarchy/mst.py, com partição de um grafo sintético calculada à mão.

O grafo sintético é um caminho de 4 frames, 0-1-2-3 (edges contíguas, como no grafo time-aware).
A árvore watershed-by-area e a partição do corte foram traçadas à mão a partir da lógica
ported de `Graph.compute_hierarchy`/`Graph.cut_graph` antes deste arquivo existir, e confirmadas
por uma chamada direta ao higra (não ao código deste módulo) num terminal, para servir de
oráculo independente. Ver o rastreio em docs/evidence/t06-mst-hierarchy.md.
"""

from __future__ import annotations

import networkx as nx
import pytest
from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph, mst_partition


def _path_graph(weights: list[float]) -> nx.Graph:
    """Caminho 0-1-2-3 com os pesos de aresta dados, na ordem (0,1), (1,2), (2,3)."""
    g = nx.Graph()
    g.add_nodes_from(range(4))
    g.add_edge(0, 1, weight=weights[0])
    g.add_edge(1, 2, weight=weights[1])
    g.add_edge(2, 3, weight=weights[2])
    return g


# Árvore esperada (calculada à mão): pesos [1, 3, 2] -> aresta (0,1) é a menor, depois (2,3),
# depois (1,2) liga os dois grupos. {0,1} mergeiam no nó 4, {2,3} no nó 5, raiz é o nó 6.
TRIPLES_ESPERADOS = [
    (0, 4, -1),
    (1, 4, -1),
    (2, 5, -1),
    (3, 5, -1),
    (4, 6, -2),
    (5, 6, -2),
    (6, 6, -2),
]
LEAF_LIST_ESPERADO = [0, 1, 2, 3]


def test_compute_hierarchy_bate_com_a_arvore_calculada_a_mao():
    g = _path_graph([1.0, 3.0, 2.0])

    triples, leaf_list, altitudes = compute_hierarchy(g)

    assert triples == TRIPLES_ESPERADOS
    assert leaf_list == LEAF_LIST_ESPERADO
    assert len(altitudes) == 7


def test_compute_hierarchy_e_arvore_binaria_estrita():
    g = _path_graph([1.0, 3.0, 2.0])
    triples, leaf_list, _altitudes = compute_hierarchy(g)

    assert len(leaf_list) == g.number_of_nodes()
    for n, _leaf in [(n, leaf) for n, _parent, leaf in triples if leaf == -1]:
        assert n in leaf_list

    children_count: dict[int, int] = {}
    root_candidates = set()
    for node, parent, _leaf in triples:
        if node != parent:
            children_count[parent] = children_count.get(parent, 0) + 1
        else:
            root_candidates.add(node)

    assert len(root_candidates) == 1, "deve haver exatamente uma raiz (parent(root) == root)"
    internal_nodes = {node for node, _parent, leaf in triples if leaf == -2}
    for internal in internal_nodes:
        assert children_count.get(internal, 0) == 2, (
            f"nó interno {internal} não tem exatamente 2 filhos (árvore binária estrita)"
        )


def test_compute_hierarchy_trunca_pesos_para_int_antes_do_higra():
    # pesos (0,1)=2.9, (1,2)=10.0, (2,3)=2.1: truncados para int -> 2, 10, 2 (empate entre as
    # pontas), então {0,1} mergeia primeiro (menor índice de aresta ganha o empate), não {2,3}
    # como aconteceria com os floats crus (2.1 < 2.9, {2,3} mergearia primeiro). Confirmado
    # chamando higra diretamente com os pesos truncados e com os pesos crus, fora deste código.
    g = _path_graph([2.9, 10.0, 2.1])

    triples, leaf_list, _altitudes = compute_hierarchy(g)

    assert triples == TRIPLES_ESPERADOS
    assert leaf_list == LEAF_LIST_ESPERADO


def test_compute_hierarchy_grafo_desconexo_levanta_erro():
    # Dois grupos fora da janela delta_t um do outro: o higra exige um grafo conexo e levanta
    # RuntimeError. Isso corrige a expectativa inicial da especificação da T04 (que previa uma
    # "floresta"); o comportamento real, verificado chamando o higra diretamente, é uma exceção.
    # Não bloqueia o uso real porque o grafo time-aware é sempre conexo para delta_t >= 1.
    g = nx.Graph()
    g.add_nodes_from(range(4))
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(2, 3, weight=1.0)

    with pytest.raises(RuntimeError):
        compute_hierarchy(g)


def test_compute_hierarchy_e_determinista():
    g = _path_graph([1.0, 3.0, 2.0])

    triples_1, leaf_list_1, altitudes_1 = compute_hierarchy(g)
    triples_2, leaf_list_2, altitudes_2 = compute_hierarchy(g)

    assert triples_1 == triples_2
    assert leaf_list_1 == leaf_list_2
    assert list(altitudes_1) == list(altitudes_2)


# Corte traçado à mão sobre TRIPLES_ESPERADOS (ver docstring do módulo e docs/evidence/t06-mst-hierarchy.md):
# cutNumber=0 e cutNumber=1 dão a mesma partição ({0,1}, {2,3}), porque a primeira linha
# processada é o próprio registro da raiz (v1=v2=6), que já está em cut_list e consome a
# primeira unidade de cutNumber sem abrir nenhum nó novo. cutNumber=2 abre o nó 5 de verdade,
# o que isola as folhas 2 e 3 uma da outra.
@pytest.mark.parametrize(
    ("cut_number", "particao_esperada"),
    [
        (0, {0: "A", 1: "A", 2: "B", 3: "B"}),
        (1, {0: "A", 1: "A", 2: "B", 3: "B"}),
        (2, {0: "A", 1: "A", 2: "B", 3: "C"}),
    ],
)
def test_cut_graph_particao_bate_com_o_corte_tracado_a_mao(cut_number, particao_esperada):
    def canonical(frame_to_cluster: dict[int, int]) -> dict[int, str]:
        by_min_frame = sorted(set(frame_to_cluster.values()), key=lambda c: min(
            f for f, cl in frame_to_cluster.items() if cl == c
        ))
        labels = {c: chr(ord("A") + i) for i, c in enumerate(by_min_frame)}
        return {f: labels[c] for f, c in frame_to_cluster.items()}

    g = _path_graph([1.0, 3.0, 2.0])

    partition = mst_partition(g, cutNumber=cut_number)

    assert canonical(partition) == particao_esperada


def test_cut_graph_aresta_final_guarda_o_leaf_flag_nao_um_peso():
    # Comportamento fiel ao original: a coluna "weight" do grafo de corte é o leaf_flag herdado
    # dos triples (-1 ou -2), não uma distância real.
    cut = cut_graph(TRIPLES_ESPERADOS, cutNumber=0)

    for _u, _v, data in cut.edges(data=True):
        assert data["weight"] in (-1.0, -2.0)
