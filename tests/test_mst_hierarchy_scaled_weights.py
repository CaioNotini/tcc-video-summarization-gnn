"""Diagnóstico de empate da T06: pesos x1e6 batem com a referência e mudam a partição; não substitui o critério com int() (D-02)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tcc_gnn.config import VSUMM_FEATURES_FILE, VSUMM_PROCESSED
from tcc_gnn.hierarchy.mst import (
    compute_hierarchy,
    cut_graph,
    mst_partition,
    partition_from_cut,
)
from tcc_gnn.reference import (
    SUBMODULE,
    canonical_partition,
    reference_compute_hierarchy,
    reference_cut_graph,
    reference_graph,
    scaled_weights,
)

_VIDEOS = ["v21", "v29"]



def _features_path(video: str) -> Path:
    """`features_teste.npy` do vídeo do VSUMM."""
    return VSUMM_PROCESSED / video / VSUMM_FEATURES_FILE


pytestmark = pytest.mark.skipif(
    not ((SUBMODULE / "HieTaSumm" / "Graph.py").exists() and all(_features_path(v).exists() for v in _VIDEOS)),
    reason="submódulo HieTaSumm-lib ou features de v21/v29 não disponíveis",
)


def _scaled_graph(video: str, scale: float) -> nx.Graph:
    """Grafo de referência (`delta_t = 8`) com os pesos multiplicados por `scale`."""
    features = np.load(_features_path(video))
    return scaled_weights(reference_graph(features, delta_t=8), scale)


@pytest.mark.parametrize("video", _VIDEOS)
def test_sem_truncamento_bate_com_a_referencia(video):
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
    # cutNumber=20: pesos x1e6 mudam a partição no v21 (107/110 frames) e no v29 (31/65); só diagnóstico, o critério segue int() (D-02)
    original_graph = _scaled_graph(video, scale=1.0)
    scaled_graph = _scaled_graph(video, scale=1e6)

    cut_number = 20
    partition_original = canonical_partition(mst_partition(original_graph, cutNumber=cut_number))
    partition_scaled = canonical_partition(mst_partition(scaled_graph, cutNumber=cut_number))

    assert partition_original != partition_scaled
