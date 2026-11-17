"""Testes para features/tf_resnet_extractor.py (só validação, D-04); pulados sem o extra tf-validation."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")

from tcc_gnn.features.tf_resnet_extractor import _AVAILABLE_MODELS, _build_backbone  # noqa: E402


def test_camadas_do_original() -> None:
    assert _AVAILABLE_MODELS["resnet50"]["layer"] == "predictions"
    assert _AVAILABLE_MODELS["vgg16"]["layer"] == "fc2"


def test_modelo_desconhecido_levanta_erro() -> None:
    with pytest.raises(ValueError, match="não suportado"):
        _build_backbone("inception")
