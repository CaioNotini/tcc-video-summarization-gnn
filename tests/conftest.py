"""Configuração do pytest: os testes que importam o submódulo HieTaSumm-lib não gravam __pycache__ nele."""

import sys

sys.dont_write_bytecode = True
