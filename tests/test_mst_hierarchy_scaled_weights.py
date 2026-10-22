"""Teste sem truncamento (T06, verificação adicional): pesos escalados por 1e6 antes de qualquer
conversão, pra eliminar os empates que o truncamento int() causa (mais extremo no v21, ver
docs/evidence/t06-additional-verification.md). Não altera src/: só escala o peso da aresta no
grafo de entrada antes de chamar compute_hierarchy/cut_graph normalmente.

Decisão do pesquisador: o truncamento int() faz parte da baseline (D-02) e é o critério de
aceitação em toda comparação com a referência (ver experiments/compare_mst_cases.py). Os testes
aqui são só diagnóstico de empate, não um critério de aceitação alternativo — por isso nenhum deles
substitui ou afrouxa as comparações com int() em tests/test_mst_hierarchy.py."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_H5_PATH = _REPO_ROOT / "references" / "HieTaSumm-lib" / "HieTaSumm" / "eccv16_dataset_summe_google_pool5.h5"
_VIDEOS = ["v21", "v29"]

sys.path.insert(0, str(_REPO_ROOT / "experiments"))


def _features_path(video: str) -> Path:
    return _REPO_ROOT / "data" / "processed" / "vsumm" / video / "features_teste.npy"


pytestmark = pytest.mark.skipif(
    not (_H5_PATH.exists() and all(_features_path(v).exists() for v in _VIDEOS)),
    reason="submódulo HieTaSumm-lib ou features de v21/v29 não disponíveis",
)


def _scaled_graph(video: str, scale: float) -> nx.Graph:
    from validate_mst_hierarchy import build_graph_via_frame_direct

    features = np.load(_features_path(video))
    graph = build_graph_via_frame_direct(features, delta_t=8)
    scaled = nx.Graph()
    scaled.add_nodes_from(graph.nodes)
    for u, v, data in graph.edges(data=True):
        scaled.add_edge(u, v, weight=data["weight"] * scale)
    return scaled


@pytest.mark.parametrize("video", _VIDEOS)
def test_sem_truncamento_bate_com_a_referencia(video):
    from validate_mst_hierarchy import canonical_partition, reference_compute_hierarchy, reference_cut_graph
    from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph, partition_from_cut

    graph = _scaled_graph(video, scale=1e6)

    triples_ours, leaf_list_ours, _altitudes = compute_hierarchy(graph)
    triples_ref, leaf_list_ref = reference_compute_hierarchy(graph)
    assert triples_ours == triples_ref
    assert leaf_list_ours == leaf_list_ref

    cut_number = 20
    cut_ours = cut_graph(triples_ours, cut_number)
    partition_ours = canonical_partition(partition_from_cut(cut_ours, leaf_list_ours))

    cut_ref = reference_cut_graph(triples_ref, cut_number)
    partition_ref = canonical_partition(partition_from_cut(cut_ref, leaf_list_ref))

    assert partition_ours == partition_ref


@pytest.mark.parametrize("video", _VIDEOS)
def test_sem_truncamento_muda_a_particao_em_relacao_ao_peso_original(video):
    # cutNumber=20: tanto no v21 (todo peso trunca pra 0) quanto no v29 (2048-d, sem esse
    # degenerado) a árvore sem empate (pesos x1e6) difere da árvore com int() — mais frames mudam
    # no v21 (107/110) do que no v29 (31/65), ver docs/evidence/t06-additional-verification.md,
    # mas em nenhum dos dois o truncamento é um no-op nesse cutNumber. Isso é só diagnóstico de
    # empate: não substitui o critério de aceitação com int() (D-02).
    from validate_mst_hierarchy import canonical_partition
    from tcc_gnn.hierarchy.mst import mst_partition

    original_graph = _scaled_graph(video, scale=1.0)
    scaled_graph = _scaled_graph(video, scale=1e6)

    cut_number = 20
    partition_original = canonical_partition(mst_partition(original_graph, cutNumber=cut_number))
    partition_scaled = canonical_partition(mst_partition(scaled_graph, cutNumber=cut_number))

    assert partition_original != partition_scaled
