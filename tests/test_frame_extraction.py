"""Testes para features/frame_extraction.py: um frame a cada 0,25 s, como HieTaSumm.frame_extractor."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from tcc_gnn.features.frame_extraction import extract_frames

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBMODULE = _REPO_ROOT / "references" / "HieTaSumm-lib"

# Índice nativo esperado para t = 0, 0.25, ..., 2.75 s: o menor c com c / fps >= t, à mão.
EXPECTED_INDICES = {
    25.0: [0, 7, 13, 19, 25, 32, 38, 44, 50, 57, 63, 69],
    29.97: [0, 8, 15, 23, 30, 38, 45, 53, 60, 68, 75, 83],
    30.0: [0, 8, 15, 23, 30, 38, 45, 53, 60, 68, 75, 83],
}


def _write_video(path: Path, fps: float, n_frames: int) -> None:
    """Vídeo MJPG 32x32 em que o frame i tem cor uniforme 2*i, para identificar o frame salvo."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 32))
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((32, 32, 3), 2 * i, dtype=np.uint8))
    writer.release()


def _frame_index(jpg: Path) -> int:
    return int(round(cv2.imread(str(jpg)).mean() / 2))


@pytest.mark.parametrize("fps", [25.0, 29.97, 30.0])
def test_um_frame_a_cada_quarto_de_segundo(tmp_path: Path, fps: float) -> None:
    video = tmp_path / "video.avi"
    _write_video(video, fps, n_frames=90 if fps != 25.0 else 75)  # 3 s de vídeo

    saved = extract_frames(video, tmp_path / "frames", fps=4)

    assert [p.name for p in saved] == [f"{i:06d}.jpg" for i in range(12)]
    assert [_frame_index(p) for p in saved] == EXPECTED_INDICES[fps]


def test_sem_fps_grava_todos_os_frames(tmp_path: Path) -> None:
    video = tmp_path / "video.avi"
    _write_video(video, 25.0, n_frames=10)

    saved = extract_frames(video, tmp_path / "frames")

    assert [_frame_index(p) for p in saved] == list(range(10))


def test_video_inexistente_levanta_erro(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Não consegui abrir"):
        extract_frames(tmp_path / "nao_existe.avi", tmp_path / "frames", fps=4)


def _reference_frame_extractor(video: Path, rate: int, frames_root: Path) -> list[Path]:
    """Chama HieTaSumm.frame_extractor do submódulo, com stubs de tensorflow/h5py para o import."""
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

    from HieTaSumm import HieTaSumm

    extractor = HieTaSumm.__new__(HieTaSumm)
    extractor.frame_extractor(video.as_posix(), rate, frames_root.as_posix())
    return sorted((frames_root / video.stem / "frames").glob("*.jpg"))


@pytest.mark.skipif(
    not (_SUBMODULE / "HieTaSumm" / "__init__.py").exists(),
    reason="submódulo HieTaSumm-lib não baixado",
)
@pytest.mark.parametrize("fps", [25.0, 29.97, 30.0])
def test_identico_ao_frame_extractor_de_referencia(tmp_path: Path, fps: float) -> None:
    video = tmp_path / "video.avi"
    _write_video(video, fps, n_frames=90 if fps != 25.0 else 75)
    ref_root = tmp_path / "ref"
    ref_root.mkdir()

    ref = _reference_frame_extractor(video, 4, ref_root)
    ours = extract_frames(video, tmp_path / "ours", fps=4)

    assert len(ours) == len(ref)
    assert [p.read_bytes() for p in ours] == [p.read_bytes() for p in ref]
