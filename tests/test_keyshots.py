"""Testes para hierarchy/keyshots.py: casos à mão e igualdade com Summary.group_central_frames."""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tcc_gnn.hierarchy.keyshots import group_central_frames, initial_len_shot

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBMODULE = _REPO_ROOT / "references" / "HieTaSumm-lib"


def _three_components() -> tuple[nx.Graph, list[int]]:
    """A: folhas 0..5 + nó interno 100; B: folha 6 sozinha; C: folhas 7, 8 + nó interno 101."""
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3, 4, 5, 100, 6, 7, 8, 101])
    g.add_edges_from([(0, 100), (1, 100), (2, 100), (3, 100), (4, 100), (5, 100), (7, 101), (8, 101)])
    return g, [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_initial_len_shot() -> None:
    assert initial_len_shot(152) == 22  # round(22.8) - 1
    assert initial_len_shot(716) == 106  # round(107.4) - 1


def test_ordem_das_folhas_vem_do_subgrafo() -> None:
    # comp_leaf_list segue list(S[c]), a ordem do subgraph().copy() do networkx, como no original;
    # na componente C ela é [8, 101, 7], não a ordem de inserção.
    g, _ = _three_components()
    assert [list(g.subgraph(c).copy()) for c in nx.connected_components(g)] == [
        [0, 1, 2, 3, 4, 5, 100],
        [6],
        [8, 101, 7],
    ]


def test_tres_ramos_orcamento_3_por_componente() -> None:
    # len_shot = int(9 / 3) = 3. A: folhas [0..5], 6 >= 3, cn = 3, k em [2, 4) -> 2, 3.
    # B: 1 folha, cn = 0 -> nada. C: folhas [8, 7], 2 < 3 -> k em [0, 1) -> 8.
    g, leaves = _three_components()
    assert group_central_frames(g, leaves, 9) == [2, 3, 8]


def test_tres_ramos_orcamento_4_por_componente() -> None:
    # len_shot = int(12 / 3) = 4. A: cn = 3, k em [1, 5) -> 1..4. C: 2 < 4 -> 8.
    g, leaves = _three_components()
    assert group_central_frames(g, leaves, 12) == [1, 2, 3, 4, 8]


def _reference_group_central_frames(graph: nx.Graph, leaflist: list[int], len_shot: int) -> list[int]:
    """Roda Summary.group_central_frames do submódulo (cp desligado) e lê os keyshots gravados."""
    stub_names = [
        "tensorflow",
        "tensorflow.keras",
        "tensorflow.keras.applications",
        "tensorflow.keras.applications.vgg16",
        "tensorflow.keras.applications.resnet50",
        "tensorflow.keras.models",
        "tensorflow.keras.utils",
    ]
    stubs = {name: types.ModuleType(name) for name in stub_names}
    stubs["tensorflow.keras.applications.vgg16"].VGG16 = object
    stubs["tensorflow.keras.applications.vgg16"].preprocess_input = lambda x: x
    stubs["tensorflow.keras.applications.resnet50"].ResNet50 = object
    stubs["tensorflow.keras.applications.resnet50"].preprocess_input = lambda x: x
    stubs["tensorflow.keras.models"].Model = object
    stubs["tensorflow.keras"].utils = stubs["tensorflow.keras.utils"]
    stubs["tensorflow.keras"].applications = stubs["tensorflow.keras.applications"]
    stubs["tensorflow.keras"].models = stubs["tensorflow.keras.models"]
    stubs["tensorflow"].keras = stubs["tensorflow.keras"]
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)
    if str(_SUBMODULE) not in sys.path:
        sys.path.insert(0, str(_SUBMODULE))
    from HieTaSumm.Files import Files
    from HieTaSumm.Summary import Summary

    real_os_system = os.system
    os.system = lambda *a, **k: 0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            s = Summary.__new__(Summary)
            s.len_shot = len_shot
            s.summ_path = str(Path(tmp) / "summ") + os.sep
            s.summ_input = Files(str(Path(tmp) / "summ_input.txt"))
            s.video_file = tmp + os.sep
            s.group_central_frames(graph, leaflist)
            path = Path(s.summ_input.file)
            lines = path.read_text().split() if path.exists() else []
    finally:
        os.system = real_os_system
    return [int(line[:-4]) for line in lines]


needs_submodule = pytest.mark.skipif(
    not (_SUBMODULE / "HieTaSumm" / "Summary.py").exists(),
    reason="submódulo HieTaSumm-lib não baixado",
)


@needs_submodule
@pytest.mark.parametrize("len_shot", [9, 12, 3, 0])
def test_igual_a_referencia_casos_a_mao(len_shot: int) -> None:
    g, leaves = _three_components()
    assert group_central_frames(g, leaves, len_shot) == _reference_group_central_frames(g, leaves, len_shot)


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
        assert group_central_frames(g, leaves, len_shot) == _reference_group_central_frames(g, leaves, len_shot)
