"""Testes para features/frame_extraction.py: um frame a cada 0,25 s, como HieTaSumm.frame_extractor."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tcc_gnn.features.frame_extraction import (
    expected_frame_count,
    extract_frames,
    video_metadata,
)
from tcc_gnn.reference import SUBMODULE, reference_frame_extractor

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
    """Índice nativo do frame gravado, pela cor uniforme 2*i."""
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


@pytest.mark.skipif(
    not (SUBMODULE / "HieTaSumm" / "__init__.py").exists(),
    reason="submódulo HieTaSumm-lib não baixado",
)
@pytest.mark.parametrize("fps", [25.0, 29.97, 30.0])
def test_identico_ao_frame_extractor_de_referencia(tmp_path: Path, fps: float) -> None:
    video = tmp_path / "video.avi"
    _write_video(video, fps, n_frames=90 if fps != 25.0 else 75)
    ref_root = tmp_path / "ref"
    ref_root.mkdir()

    ref = reference_frame_extractor(video, 4, ref_root)
    ours = extract_frames(video, tmp_path / "ours", fps=4)

    assert len(ours) == len(ref)
    assert [p.read_bytes() for p in ours] == [p.read_bytes() for p in ref]


def test_pasta_de_saida_nao_vazia_levanta_erro_sem_gravar(tmp_path: Path) -> None:
    video = tmp_path / "video.avi"
    _write_video(video, 25.0, n_frames=10)
    out = tmp_path / "frames"
    out.mkdir()
    (out / "000000.jpg").write_bytes(b"antigo")

    with pytest.raises(ValueError, match="não está vazia"):
        extract_frames(video, out, fps=4)
    assert [p.name for p in out.iterdir()] == ["000000.jpg"]
    assert (out / "000000.jpg").read_bytes() == b"antigo"


def test_pasta_de_saida_vazia_existente_e_aceita(tmp_path: Path) -> None:
    video = tmp_path / "video.avi"
    _write_video(video, 25.0, n_frames=10)
    out = tmp_path / "frames"
    out.mkdir()

    assert len(extract_frames(video, out)) == 10


def test_video_metadata(tmp_path: Path) -> None:
    video = tmp_path / "video.avi"
    _write_video(video, 25.0, n_frames=75)

    fps, n_frames = video_metadata(video)

    assert fps == pytest.approx(25.0)
    assert n_frames == 75


def test_video_metadata_video_inexistente(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Não consegui abrir"):
        video_metadata(tmp_path / "nao_existe.avi")


@pytest.mark.parametrize("fps", [25.0, 29.97, 30.0])
def test_expected_frame_count_bate_com_o_que_extract_frames_grava(tmp_path: Path, fps: float) -> None:
    video = tmp_path / "video.avi"
    _write_video(video, fps, n_frames=100)  # duração fracionária: a fração final não é amostrada

    saved = extract_frames(video, tmp_path / "frames", fps=4)

    assert expected_frame_count(video, 4) == len(saved) == 4 * int(100 / fps)
