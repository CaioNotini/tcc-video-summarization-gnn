"""Testes para hierarchy/best_cut.py: equivalência com Summary.bestCutNumber num caso pequeno.

O caso pequeno são 6 jpgs sintéticos 16x12 com conteúdo determinístico (seed fixa), gravados em
tmp_path. A referência é o submódulo HieTaSumm-lib real, chamado como em
experiments/validate_mst_hierarchy.py (Summary.__new__, só com frames_path e delta_t), com stubs
de tensorflow/h5py para o import do pacote.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcc_gnn.hierarchy.best_cut import best_cut_number, cut_number_from_best

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBMODULE = _REPO_ROOT / "references" / "HieTaSumm-lib"

pytestmark = pytest.mark.skipif(
    not (_SUBMODULE / "HieTaSumm" / "Summary.py").exists(),
    reason="submódulo HieTaSumm-lib não baixado",
)


def _reference_best_cut_number(frames_dir: Path, delta_t: int) -> float:
    """Chama Summary.bestCutNumber do submódulo, com os mesmos stubs de validate_mst_hierarchy.py."""
    stub_names = [
        "tensorflow",
        "tensorflow.keras",
        "tensorflow.keras.applications",
        "tensorflow.keras.applications.vgg16",
        "tensorflow.keras.applications.resnet50",
        "tensorflow.keras.models",
        "tensorflow.keras.utils",
        "h5py",
    ]
    stubs = {name: types.ModuleType(name) for name in stub_names}
    stubs["tensorflow.keras.applications.vgg16"].VGG16 = object
    stubs["tensorflow.keras.applications.vgg16"].preprocess_input = lambda x: x
    stubs["tensorflow.keras.applications.resnet50"].ResNet50 = object
    stubs["tensorflow.keras.applications.resnet50"].preprocess_input = lambda x: x
    stubs["tensorflow.keras.models"].Model = object
    stubs["h5py"].File = object
    stubs["tensorflow.keras"].utils = stubs["tensorflow.keras.utils"]
    stubs["tensorflow.keras"].applications = stubs["tensorflow.keras.applications"]
    stubs["tensorflow.keras"].models = stubs["tensorflow.keras.models"]
    stubs["tensorflow"].keras = stubs["tensorflow.keras"]
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)
    if str(_SUBMODULE) not in sys.path:
        sys.path.insert(0, str(_SUBMODULE))

    from HieTaSumm.Summary import Summary

    import os

    summary = Summary.__new__(Summary)
    summary.frames_path = str(frames_dir) + os.sep
    summary.delta_t = delta_t
    return float(summary.bestCutNumber())


def _write_synthetic_frames(frames_dir: Path, n_frames: int = 6) -> None:
    frames_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        pixels = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(frames_dir / f"{i:06d}.jpg")


@pytest.mark.parametrize("delta_t", [3, 8])
def test_best_cut_number_identico_a_referencia(tmp_path: Path, delta_t: int) -> None:
    frames_dir = tmp_path / "frames"
    _write_synthetic_frames(frames_dir)

    esperado = _reference_best_cut_number(frames_dir, delta_t)
    obtido = best_cut_number(frames_dir, delta_t)

    assert obtido == esperado


def test_best_cut_number_janela_maior_que_o_video(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_synthetic_frames(frames_dir, n_frames=4)

    esperado = _reference_best_cut_number(frames_dir, delta_t=32)
    obtido = best_cut_number(frames_dir, delta_t=32)

    assert obtido == esperado


def test_best_cut_number_pasta_sem_jpg(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "nota.txt").write_text("sem frames")

    with pytest.raises(ValueError, match="Nenhum frame"):
        best_cut_number(frames_dir, delta_t=8)


def test_best_cut_number_desvio_zero_levanta_erro(tmp_path: Path) -> None:
    # Um único frame: só existe o par (0, 0), então std([1.0]) = 0 exato; o original
    # entraria em loop infinito no `while cut <= 1`.
    frames_dir = tmp_path / "frames"
    _write_synthetic_frames(frames_dir, n_frames=1)

    with pytest.raises(ValueError, match="Desvio-padrão zero"):
        best_cut_number(frames_dir, delta_t=8)


def test_best_cut_number_delta_t_invalido(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_synthetic_frames(frames_dir, n_frames=2)

    with pytest.raises(ValueError, match="delta_t deve ser positivo"):
        best_cut_number(frames_dir, delta_t=0)


def test_cut_number_from_best_piso_e_escala() -> None:
    assert cut_number_from_best(8.0, alpha=75) == 6
    assert cut_number_from_best(4.0, alpha=75) == 3  # int(3.0) = 3
    assert cut_number_from_best(2.0, alpha=75) == 3  # int(1.5) = 1 -> piso 3
    assert cut_number_from_best(0.0, alpha=75) == 3
