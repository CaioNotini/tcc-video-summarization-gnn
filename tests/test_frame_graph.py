"""Testes para graph/frame.py."""

from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import pytest
from tcc_gnn.graph.frame import build_time_aware_graph, calc_window_end, edge_weight


def test_calc_window_end_dentro_dos_limites():
    assert calc_window_end(i=0, delta_t=2, n_frames=6) == 2
    assert calc_window_end(i=5, delta_t=2, n_frames=6) == 6
    assert calc_window_end(i=0, delta_t=-1, n_frames=6) == 6
    assert calc_window_end(i=2, delta_t=0, n_frames=6) == 2


def test_edge_weight_features_identicas_da_zero():
    f = np.array([1.0, 2.0, 3.0])
    assert edge_weight(f, f) == 0.0


def test_build_time_aware_graph_com_dados_falsos():
    features = np.random.rand(6, 8)
    delta_t = 2

    g = build_time_aware_graph(features, delta_t)

    assert set(g.nodes) == set(range(6))
    assert set(g.neighbors(0)) == {1}
    assert not g.has_edge(0, 5)

    for v1, v2 in g.edges():
        assert v1 != v2

    for _, _, data in g.edges(data=True):
        assert data["weight"] >= 0


H5_PATH = Path(
    "references/HieTaSumm-lib/HieTaSumm/eccv16_dataset_summe_google_pool5.h5"
)


@pytest.mark.skipif(not H5_PATH.exists(), reason="submódulo HieTaSumm-lib não baixado")
def test_build_time_aware_graph_com_video_real_do_summe():
    with h5py.File(H5_PATH, "r") as f:
        primeiro_video = list(f.keys())[0]
        features = f[primeiro_video]["features"][()]

    delta_t = 10
    g = build_time_aware_graph(features, delta_t)

    assert g.number_of_nodes() == len(features)
    assert g.number_of_edges() > 0

    ultimo = len(features) - 1
    assert not g.has_edge(0, ultimo)