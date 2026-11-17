"""Testes para reference.py: os stubs do tensorflow e o sys.path só existem dentro do contexto."""

from __future__ import annotations

import sys
import types

import networkx as nx
import numpy as np
import pytest

from tcc_gnn import reference
from tcc_gnn.graph.frame import build_time_aware_graph
from tcc_gnn.reference import (
    SUBMODULE,
    canonical_partition,
    hietasumm_reference,
    reference_compute_hierarchy,
    reference_cut_graph,
    reference_evaluation,
    reference_graph,
)

needs_submodule = pytest.mark.skipif(
    not (SUBMODULE / "HieTaSumm" / "__init__.py").exists(),
    reason="submódulo HieTaSumm-lib não baixado",
)


@needs_submodule
def test_stubs_e_sys_path_removidos_ao_sair() -> None:
    assert "tensorflow" not in sys.modules
    path_before = list(sys.path)
    bytecode_before = sys.dont_write_bytecode

    with hietasumm_reference() as ref:
        assert "tensorflow" in sys.modules
        assert str(SUBMODULE) in sys.path
        assert sys.dont_write_bytecode is True
        assert {"HieTaSumm", "Evaluation", "Files", "Frame", "Graph", "Summary"} <= set(vars(ref))

    assert not any(name.startswith("tensorflow") for name in sys.modules)
    assert sys.path == path_before
    assert sys.dont_write_bytecode == bytecode_before


@needs_submodule
def test_stubs_removidos_mesmo_com_excecao() -> None:
    path_before = list(sys.path)
    with pytest.raises(RuntimeError), hietasumm_reference():
        raise RuntimeError("falha dentro do contexto")
    assert "tensorflow" not in sys.modules
    assert sys.path == path_before


@needs_submodule
def test_modulo_pre_existente_nao_e_substituido_nem_removido(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("tensorflow")
    monkeypatch.setitem(sys.modules, "tensorflow", fake)
    with hietasumm_reference():
        assert sys.modules["tensorflow"] is fake
    assert sys.modules["tensorflow"] is fake


def test_submodulo_ausente_levanta_erro(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference, "SUBMODULE", tmp_path / "nao_existe")
    with pytest.raises(FileNotFoundError, match="não baixado"), hietasumm_reference():
        pass
    assert "tensorflow" not in sys.modules


@needs_submodule
def test_referencia_usavel_depois_do_contexto() -> None:
    evaluation = reference_evaluation()
    assert "tensorflow" not in sys.modules
    assert evaluation.f_score(np.array([1, 1, 0, 0]), np.array([[1, 0, 1, 0]]), "max") == pytest.approx(50.0)


@needs_submodule
def test_grafo_hierarquia_e_corte_de_referencia_num_caso_pequeno() -> None:
    features = np.random.default_rng(0).random((12, 8)) * 1000
    graph = reference_graph(features, delta_t=3)
    ours = build_time_aware_graph(features, delta_t=3)
    assert set(graph.edges) == set(ours.edges)
    assert all(graph.edges[e]["weight"] == ours.edges[e]["weight"] for e in graph.edges)

    triples, leaf_list = reference_compute_hierarchy(graph)
    assert sorted(leaf_list) == list(range(12))
    cut = reference_cut_graph(triples, 2)
    assert isinstance(cut, nx.Graph)
    assert set(range(12)) <= set(cut.nodes)


def test_canonical_partition_renumera_pelo_menor_frame() -> None:
    assert canonical_partition({0: 7, 1: 7, 2: 3, 3: 9, 4: 3}) == {0: 0, 1: 0, 2: 1, 3: 2, 4: 1}
