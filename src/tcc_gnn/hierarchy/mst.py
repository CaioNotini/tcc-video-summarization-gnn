"""Hierarquia MST com higra (watershed-by-area), porta fiel de HieTaSumm-lib/HieTaSumm/Graph.py (D-02, D-05).

Produz uma única partição (o corte `cutNumber`), não uma hierarquia multi-nível: ver a decisão
de escopo da T04 em docs/evidence/t04-mst-hierarchy-spec.md.
"""

from __future__ import annotations

import higra as hg
import networkx as nx
import numpy as np


def compute_hierarchy(
    input_g: nx.Graph,
) -> tuple[list[tuple[int, int, int]], list[int], np.ndarray]:
    """Porta de `Graph.compute_hierarchy`: árvore watershed-by-area, binarizada (D-02).

    `input_g` é o grafo time-aware (nós = `0..n_frames-1`, aresta `weight` = distância L1).
    Cada peso de aresta é truncado para `int` antes de chamar o higra
    (`edge_weights[i] = int(...)`, `Graph.py:60-61`); é um comportamento de truncamento, não de
    arredondamento, mantido de propósito como parte da baseline HieTaSumm (ver "Fidelity detail"
    em docs/evidence/t04-mst-hierarchy-spec.md).

    Retorna `(triples, leaf_list, altitudes)`: `triples` é a lista `(node, parent, leaf_flag)` na
    ordem `leaves_to_root_iterator` (`leaf_flag = -1` para folha, `-2` para nó interno, igual ao
    original); `leaf_list` são os ids de nó que são folhas (coincidem com os índices de frame
    originais); `altitudes[node]` é a altitude higra do nó, calculada mas não persistida no
    arquivo pelo código original — exposta aqui só para inspeção/teste.
    """
    leaf_list: list[int] = []
    graph = hg.UndirectedGraph()
    graph.add_vertices(max(input_g.nodes) + 1)
    edge_list = list(input_g.edges())

    for v1, v2 in edge_list:
        graph.add_edge(v1, v2)

    edge_weights = np.empty(shape=len(edge_list))
    sources, targets = graph.edge_list()
    for i in range(len(sources)):
        edge_weights[i] = int(input_g.adj[sources[i]][targets[i]]["weight"])

    nb_tree, nb_altitudes = hg.watershed_hierarchy_by_area(graph, edge_weights)
    tree, node_map = hg.tree_2_binary_tree(nb_tree)
    altitudes = nb_altitudes[node_map]

    triples: list[tuple[int, int, int]] = []
    for n in tree.leaves_to_root_iterator():
        leaf = -2
        if tree.is_leaf(n):
            leaf = -1
            leaf_list.append(n)
        triples.append((n, tree.parent(n), leaf))

    return triples, leaf_list, altitudes


def cut_graph(triples: list[tuple[int, int, int]], cutNumber: int) -> nx.Graph:
    """Porta de `Graph.cut_graph`: corte top-down a partir da raiz, abrindo `cutNumber` nós (D-02).

    `triples` é a saída de `compute_hierarchy`, na mesma ordem (`leaves_to_root`). No grafo de
    saída, o atributo `weight` de cada aresta é na verdade o `leaf_flag` herdado dos `triples`,
    não uma distância real: é assim que o `Graph.py` original reaproveita a coluna de peso do
    arquivo ao gravar o corte, e é mantido fiel aqui (ver `cut_graph_file.save_graph_data` no
    original). Os componentes conexos do grafo retornado são os grupos finais.
    """
    RG = nx.Graph()
    cut_list: list[int] = []

    line = len(triples) - 1
    cut_list.append(triples[line][0])

    while line >= 0:
        v1, v2, w = triples[line]
        if v2 in cut_list:
            if cutNumber > 0:
                cutNumber -= 1
                cut_list.append(v1)
            else:
                RG.add_node(v1)
        else:
            RG.add_edge(v1, v2, weight=float(w))
        line -= 1

    return RG


def partition_from_cut(cut_graph_result: nx.Graph, leaf_list: list[int]) -> dict[int, int]:
    """Rotula cada frame original (folha) com o id do componente conexo após o corte."""
    leaf_set = set(leaf_list)
    frame_to_cluster: dict[int, int] = {}
    for cluster_id, component in enumerate(nx.connected_components(cut_graph_result)):
        for node in component:
            if node in leaf_set:
                frame_to_cluster[node] = cluster_id
    return frame_to_cluster


def mst_partition(input_g: nx.Graph, cutNumber: int) -> dict[int, int]:
    """Partição única do grafo time-aware (corte `cutNumber`), equivalente ao HieTaSumm baseline."""
    triples, leaf_list, _altitudes = compute_hierarchy(input_g)
    cut = cut_graph(triples, cutNumber)
    return partition_from_cut(cut, leaf_list)
