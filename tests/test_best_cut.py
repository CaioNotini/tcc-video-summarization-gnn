"""Testes para hierarchy/best_cut.py: equivalência com Summary.bestCutNumber em 6 jpgs sintéticos 16x12 (seed fixa)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcc_gnn.hierarchy.best_cut import best_cut_number, cut_number_from_best
from tcc_gnn.reference import SUBMODULE, reference_best_cut_number

pytestmark = pytest.mark.skipif(
    not (SUBMODULE / "HieTaSumm" / "Summary.py").exists(),
    reason="submódulo HieTaSumm-lib não baixado",
)


def _reference_best_cut_number(frames_dir: Path, delta_t: int) -> float:
    """Valor bruto de `Summary.bestCutNumber` do submódulo."""
    return reference_best_cut_number(frames_dir, delta_t, alpha=75)["best_cut_number_raw"]


def _write_synthetic_frames(frames_dir: Path, n_frames: int = 6) -> None:
    """Grava `n_frames` jpgs 16x12 aleatórios com seed fixa."""
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
    # um frame só tem o par (0, 0), std([1.0]) = 0, e o original ficaria em loop no `while cut <= 1`
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


def test_frame_preto_levanta_erro_como_o_original_falha(tmp_path: Path) -> None:
    # divergência 3: a porta levanta ValueError; o original chega a NaN e falha no int() do cutNumber
    frames_dir = tmp_path / "frames"
    _write_synthetic_frames(frames_dir, n_frames=4)
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(frames_dir / "000002.jpg")

    with pytest.raises(ValueError, match="norma zero"):
        best_cut_number(frames_dir, delta_t=3)
    with pytest.raises(ValueError), np.errstate(invalid="ignore", divide="ignore"):
        reference_best_cut_number(frames_dir, delta_t=3, alpha=75)
