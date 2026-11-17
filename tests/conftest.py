"""Configuração do pytest: temporários em .cache/tmp/pytest no D: (regra 17 do CLAUDE.md) e sem __pycache__ no submódulo."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.dont_write_bytecode = True

_REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config: pytest.Config) -> None:
    """Falha se TEMP não está no drive do repositório e fixa o basetemp do pytest em .cache/tmp/pytest."""
    temp_drive = Path(tempfile.gettempdir()).resolve().drive.upper()
    if temp_drive != _REPO_ROOT.drive.upper():
        raise pytest.UsageError(
            f"TEMP está em {tempfile.gettempdir()}, fora de {_REPO_ROOT.drive}: ative a venv (.venv/Scripts/Activate.ps1)"
        )
    if config.option.basetemp is None:
        config.option.basetemp = str(_REPO_ROOT / ".cache" / "tmp" / "pytest")
