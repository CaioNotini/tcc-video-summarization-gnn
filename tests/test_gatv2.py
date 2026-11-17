"""Testes para models/gatv2.py: conversão do grafo e bloco GP (forma, finitude, gradiente, determinismo, localidade)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from tcc_gnn.config import load_config
from tcc_gnn.graph.frame import build_time_aware_graph
from tcc_gnn.models.gatv2 import (
    GraphProcessingBlock,
    graph_processing_block,
    time_aware_edge_index,
)


def test_edge_index_calculado_a_mao() -> None:
    # 4 frames, delta_t = 2: arestas (0,1), (1,2), (2,3), cada uma nos dois sentidos
    graph = build_time_aware_graph(np.arange(8, dtype=float).reshape(4, 2), delta_t=2)
    expected = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    assert torch.equal(time_aware_edge_index(graph), expected)


def test_mesmas_arestas_do_grafo_time_aware() -> None:
    features = np.random.default_rng(0).random((40, 16))
    graph = build_time_aware_graph(features, delta_t=5)

    edge_index = time_aware_edge_index(graph)

    pairs = set(map(tuple, edge_index.t().tolist()))
    assert edge_index.dtype == torch.long
    assert edge_index.shape == (2, 2 * graph.number_of_edges())
    assert pairs == {(u, v) for a, b in graph.edges for u, v in ((a, b), (b, a))}
    assert all(0 < abs(u - v) < 5 for u, v in pairs)  # só vizinhos temporais, sem o próprio nó


def test_um_frame_sem_arestas() -> None:
    graph = build_time_aware_graph(np.zeros((1, 3)), delta_t=4)
    assert time_aware_edge_index(graph).shape == (2, 0)


def test_grafo_vazio_levanta_erro() -> None:
    with pytest.raises(ValueError, match="sem nós"):
        time_aware_edge_index(nx.Graph())


def test_nos_fora_de_0_a_n_levantam_erro() -> None:
    graph = nx.Graph()
    graph.add_edge(1, 2)
    with pytest.raises(ValueError, match="0..n-1"):
        time_aware_edge_index(graph)


def test_self_loop_levanta_erro() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 1)])
    with pytest.raises(ValueError, match="self-loops"):
        time_aware_edge_index(graph)


def _block(seed: int = 0, in_dim: int = 16, heads: int = 4) -> GraphProcessingBlock:
    """Bloco pequeno com pesos de seed fixa."""
    torch.manual_seed(seed)
    return GraphProcessingBlock(in_dim, embedding_size=8, heads=heads, dropout=0.2)


def _inputs(n: int = 30, in_dim: int = 16, delta_t: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Features aleatórias de seed fixa e o `edge_index` do grafo time-aware delas."""
    features = np.random.default_rng(1).random((n, in_dim))
    edge_index = time_aware_edge_index(build_time_aware_graph(features, delta_t))
    return torch.tensor(features, dtype=torch.float32), edge_index


def test_configuracao_fiel_a_especificacao() -> None:
    conv = _block().conv
    assert (conv.heads, conv.concat, conv.add_self_loops, conv.bias, conv.lin_edge, conv.share_weights) == (
        4, False, False, None, None, False,
    )
    assert conv.dropout == 0.0


def test_forma_e_valores_finitos() -> None:
    x, edge_index = _inputs()
    block = _block().eval()
    with torch.no_grad():
        out = block(x, edge_index)
    assert out.shape == (30, 8)
    assert torch.isfinite(out).all()


def test_concat_false_mantem_a_dimensao_com_qualquer_numero_de_cabecas() -> None:
    x, edge_index = _inputs()
    for heads in (1, 2, 4):
        with torch.no_grad():
            assert _block(heads=heads).eval()(x, edge_index).shape == (30, 8)


def test_gradiente_chega_a_todos_os_parametros() -> None:
    x, edge_index = _inputs()
    block = _block().train()
    (block(x, edge_index) ** 2).mean().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_deterministico_com_seed_fixa_em_eval() -> None:
    x, edge_index = _inputs()
    with torch.no_grad():
        a = _block(seed=7).eval()(x, edge_index)
        b = _block(seed=7).eval()(x, edge_index)
        c = _block(seed=8).eval()(x, edge_index)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_atencao_so_sobre_vizinhos_e_soma_um() -> None:
    x, edge_index = _inputs()
    with torch.no_grad():
        _, (att_index, beta) = _block().eval()(x, edge_index, return_attention=True)
    src, dst = att_index
    assert not (src == dst).any()  # Eq. 1: sem o próprio nó
    assert set(map(tuple, att_index.t().tolist())) == set(map(tuple, edge_index.t().tolist()))
    sums = torch.zeros(30, beta.shape[1]).index_add_(0, dst, beta)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_saida_de_um_no_so_depende_da_sua_janela() -> None:
    x, edge_index = _inputs(delta_t=3)
    changed = x.clone()
    changed[20] += 5.0
    block = _block().eval()
    with torch.no_grad():
        before, after = block(x, edge_index), block(changed, edge_index)
    differs = (before - after).abs().amax(dim=1) > 0
    assert differs.nonzero().flatten().tolist() == [18, 19, 20, 21, 22]  # o nó 20 e os vizinhos a menos de 3


def test_frame_isolado_fica_so_com_o_residual() -> None:
    x = torch.randn(1, 16)
    block = _block().eval()
    with torch.no_grad():
        out = block(x, torch.empty((2, 0), dtype=torch.long))
        expected = block.norm(block.embedding(x))
    assert torch.allclose(out, expected)


def test_entradas_invalidas() -> None:
    x, edge_index = _inputs()
    block = _block().eval()
    with pytest.raises(ValueError, match="forma"):
        block(torch.randn(30, 5), edge_index)
    with pytest.raises(ValueError, match="não existe"):
        block(x[:10], edge_index)


def test_bloco_lido_do_yaml() -> None:
    block = graph_processing_block(load_config(), in_dim=2048)
    assert block.embedding.in_features == 2048
    assert block.embedding.out_features == 64
    assert block.conv.heads == 4
    assert block.dropout.p == 0.2
