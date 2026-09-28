"""Testes para src/tcc_gnn/graph/frame.py -- construção do grafo time-aware.

Dois blocos de teste:
1. Dados falsos (np.random.rand) -- valida a lógica da janela temporal e
   do cálculo de peso, sem depender de nenhum dataset.
2. Um vídeo real do SumMe, lido direto do h5 já bundled no submódulo
   HieTaSumm-lib -- só garante que a função aguenta dimensões reais
   (300 amostras, 1024 features) sem quebrar. Pulado automaticamente se
   o submódulo não tiver sido baixado ainda.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tcc_gnn.graph.frame import build_time_aware_graph, calc_window_end, edge_weight


# --- Bloco 1: dados falsos --------------------------------------------------

def test_calc_window_end_dentro_dos_limites():
    assert calc_window_end(i=0, delta_t=2, n_frames=6) == 2
    assert calc_window_end(i=5, delta_t=2, n_frames=6) == 6  # truncado no fim da lista
    assert calc_window_end(i=0, delta_t=0, n_frames=6) == 6  # delta_t<=0 -> sem limite


def test_edge_weight_features_identicas_da_zero():
    f = np.array([1.0, 2.0, 3.0])
    assert edge_weight(f, f) == 0.0


def test_build_time_aware_graph_com_dados_falsos():
    features = np.random.rand(6, 8)  # 6 frames falsos, 8 dims cada
    delta_t = 2

    g = build_time_aware_graph(features, delta_t)

    # todos os nós devem existir, mesmo os que não tiverem aresta nenhuma
    assert set(g.nodes) == set(range(6))

    # nó 0 só deve conectar com nós dentro da janela (delta_t=2 -> só o nó 1)
    assert set(g.neighbors(0)) == {1}

    # frames fora da janela não podem estar conectados
    assert not g.has_edge(0, 5)

    # nenhum self-loop
    for v1, v2 in g.edges():
        assert v1 != v2

    # pesos devem ser não-negativos
    for _, _, data in g.edges(data=True):
        assert data["weight"] >= 0


# --- Bloco 2: vídeo real do SumMe (h5 bundled no submódulo) ----------------

H5_PATH = Path(
    "references/HieTaSumm-lib/HieTaSumm/eccv16_dataset_summe_google_pool5.h5"
)


@pytest.mark.skipif(
    not H5_PATH.exists(),
    reason="submódulo HieTaSumm-lib não baixado (rode: git submodule update --init --recursive)",
)
def test_build_time_aware_graph_com_video_real_do_summe():
    with h5py.File(H5_PATH, "r") as f:
        primeiro_video = list(f.keys())[0]
        features = f[primeiro_video]["features"][()]  # shape (n_amostras, 1024)

    delta_t = 10
    g = build_time_aware_graph(features, delta_t)

    assert g.number_of_nodes() == len(features)
    assert g.number_of_edges() > 0

    # frames muito distantes no tempo (fora da janela) não devem conectar
    ultimo = len(features) - 1
    assert not g.has_edge(0, ultimo)