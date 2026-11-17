"""Testes para config.py: leitura do default.yaml e resolução do tracking do mlflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcc_gnn.config import load_config, tracking_config


def test_default_yaml_tem_os_parametros_da_baseline() -> None:
    cfg = load_config()
    assert cfg["graph"]["delta_t"] == 32  # D-24
    assert cfg["graph"]["rate"] == 4  # D-25
    assert cfg["hierarchy"]["alpha"] == 75  # D-20
    assert cfg["hierarchy"]["percent"] == 15  # D-21
    assert cfg["training"]["unsupervised_objective"] is None  # D-17 em aberto


def test_default_yaml_sem_campos_mortos() -> None:
    cfg = load_config()
    assert "data" not in cfg
    assert cfg["evaluation"]["metrics"] == ["f_score"]


def test_tracking_config_resolve_relativo_a_raiz(tmp_path: Path) -> None:
    cfg = {"experiment_tracking": {"tracking_uri": "sqlite:///mlruns/mlflow.db", "artifact_root": "./mlruns"}}
    uri, artifacts = tracking_config(cfg, repo_root=tmp_path)
    assert uri == "sqlite:///" + (tmp_path / "mlruns" / "mlflow.db").resolve().as_posix()
    assert artifacts == (tmp_path / "mlruns").resolve().as_uri()
    assert (tmp_path / "mlruns").is_dir()


def test_tracking_config_recusa_backend_de_arquivo(tmp_path: Path) -> None:
    cfg = {"experiment_tracking": {"tracking_uri": "./mlruns", "artifact_root": "./mlruns"}}
    with pytest.raises(ValueError, match="sqlite"):
        tracking_config(cfg, repo_root=tmp_path)
