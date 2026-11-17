"""Testes para features/features.py: forma, valores finitos, independência do batch e cache do .npy."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from tcc_gnn.features.features import cached_features, extract_features

CPU = torch.device("cpu")


def _write_frames(frames_dir: Path, n_frames: int = 5) -> None:
    """Grava `n_frames` jpgs 48x40 aleatórios com seed fixa."""
    frames_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(n_frames):
        Image.fromarray(rng.integers(0, 256, size=(40, 48, 3), dtype=np.uint8)).save(frames_dir / f"{i:06d}.jpg")


def test_forma_finitos_e_independente_do_batch(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")

    batch_2 = extract_features(tmp_path / "frames", CPU, batch_size=2)
    batch_16 = extract_features(tmp_path / "frames", CPU, batch_size=16)

    assert batch_2.shape == (5, 2048)
    assert np.isfinite(batch_2).all()
    np.testing.assert_allclose(batch_2, batch_16, rtol=1e-5, atol=1e-6)


def test_igual_ao_extrator_frame_a_frame(tmp_path: Path) -> None:
    # o extrator frame a frame que gerou os features_teste.npy do VSUMM (features.py até o commit 5b73d21)
    _write_frames(tmp_path / "frames", n_frames=3)
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    with torch.no_grad():
        expected = np.stack([
            model(weights.transforms()(Image.open(p).convert("RGB")).unsqueeze(0)).squeeze(0).numpy()
            for p in sorted((tmp_path / "frames").glob("*.jpg"))
        ])

    np.testing.assert_allclose(extract_features(tmp_path / "frames", CPU), expected, rtol=1e-5, atol=1e-6)


def test_pasta_sem_jpg_levanta_erro(tmp_path: Path) -> None:
    (tmp_path / "frames").mkdir()
    with pytest.raises(ValueError, match="Nenhum frame"):
        extract_features(tmp_path / "frames", CPU)


def test_pasta_com_arquivo_fora_do_padrao_levanta_erro(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames", n_frames=2)
    (tmp_path / "frames" / "nota.txt").write_text("intruso")
    with pytest.raises(ValueError, match="fora do padrão"):
        extract_features(tmp_path / "frames", CPU)


def test_cached_features_le_o_npy_sem_extrair(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames", n_frames=3)
    cache = tmp_path / "features.npy"
    np.save(cache, np.arange(6.0).reshape(3, 2))
    np.testing.assert_array_equal(cached_features(tmp_path / "frames", cache, CPU), np.arange(6.0).reshape(3, 2))


def test_cached_features_contagem_diferente_levanta_erro(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames", n_frames=3)
    cache = tmp_path / "features.npy"
    np.save(cache, np.zeros((2, 2048)))
    with pytest.raises(ValueError, match="2 features para 3 frames"):
        cached_features(tmp_path / "frames", cache, CPU)


def test_cached_features_mais_antigo_que_os_frames_levanta_erro(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames", n_frames=2)
    cache = tmp_path / "features.npy"
    np.save(cache, np.zeros((2, 2048)))
    frame = tmp_path / "frames" / "000001.jpg"
    later = cache.stat().st_mtime + 10
    os.utime(frame, (later, later))  # simula frames reextraídos depois do .npy
    with pytest.raises(ValueError, match="mais antigo"):
        cached_features(tmp_path / "frames", cache, CPU)


def test_cached_features_extrai_e_grava_quando_falta(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames", n_frames=2)
    cache = tmp_path / "features.npy"

    features = cached_features(tmp_path / "frames", cache, CPU)

    assert features.shape == (2, 2048)
    np.testing.assert_array_equal(np.load(cache), features)
