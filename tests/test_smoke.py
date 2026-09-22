"""Teste de fumaça — confirma que o pacote importa e o ambiente está configurado.

Serve como ponto de partida: cada módulo novo em src/tcc_gnn/ deve ganhar
seu próprio arquivo de teste aqui (ex.: test_graph.py para src/tcc_gnn/graph/).
"""

import tcc_gnn


def test_package_imports():
    assert tcc_gnn is not None
