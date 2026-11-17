"""Testes para evaluation/fscore.py: casos calculados à mão e igualdade com Evaluation.f_score."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tcc_gnn.evaluation.fscore import f_score, sampled_to_native

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBMODULE = _REPO_ROOT / "references" / "HieTaSumm-lib"


def test_meio_a_meio() -> None:
    # sobreposição 1; P = 1/2, R = 1/2, F = 50
    assert f_score(np.array([1, 1, 0, 0]), np.array([[1, 0, 1, 0]])) == pytest.approx(50.0)


def test_precisao_meia_recall_total() -> None:
    # sobreposição 2; P = 2/4, R = 2/2, F = 2 * 0.5 * 1 / 1.5 * 100 = 66.67
    S = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    G = np.array([[1, 1, 0, 0, 0, 0, 0, 0]])
    assert f_score(S, G) == pytest.approx(200 / 3)


def test_maximo_e_media_entre_anotadores() -> None:
    # anotador 1: P = 1, R = 1/2, F = 66.67; anotador 2: sem sobreposição, F = 0
    S = np.array([1, 0, 0, 0])
    G = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    assert f_score(S, G, "max") == pytest.approx(200 / 3)
    assert f_score(S, G, "avg") == pytest.approx(100 / 3)


def test_sem_sobreposicao_vale_zero() -> None:
    assert f_score(np.array([1, 1, 0, 0]), np.array([[0, 0, 1, 1]])) == 0.0


def test_comprimentos_diferentes_completam_com_zeros() -> None:
    # S = [1, 1, 1, 0, 0]; P = 3/3, R = 3/5, F = 2 * 1 * 0.6 / 1.6 * 100 = 75
    assert f_score(np.array([1, 1, 1]), np.array([[1, 1, 1, 1, 1]])) == pytest.approx(75.0)


def test_resumo_vazio_vale_zero() -> None:
    assert f_score(np.zeros(4, dtype=int), np.array([[1, 0, 1, 0]])) == 0.0


def test_entradas_invalidas() -> None:
    with pytest.raises(ValueError, match="user_summary"):
        f_score(np.array([1, 0]), np.array([1, 0]))
    with pytest.raises(ValueError, match="predicted_summary"):
        f_score(np.array([[1, 0]]), np.array([[1, 0]]))
    with pytest.raises(ValueError, match="vazio"):
        f_score(np.array([1, 0]), np.array([[1, 0], [0, 0]]))


def _marked(fps: float, sampled: list[int], n_frames: int = 200) -> list[int]:
    return np.flatnonzero(sampled_to_native(sampled, fps, n_frames)).tolist()


@pytest.mark.parametrize(
    ("fps", "sampled", "expected"),
    [
        # 25 fps: k=0 -> c/25 < 0.25 -> c 0..6; k=1 -> 7..12; k=3 -> [0.75, 1.0) -> 19..24
        (25.0, [0], list(range(0, 7))),
        (25.0, [1], list(range(7, 13))),
        (25.0, [3], list(range(19, 25))),
        # 30 fps: k=0 -> c < 7.5 -> 0..7; k=1 -> [7.5, 15) -> 8..14
        (30.0, [0], list(range(0, 8))),
        (30.0, [1], list(range(8, 15))),
        # 29.97 fps: k=0 -> c < 7.4925 -> 0..7; k=1 -> [7.4925, 14.985) -> 8..14; k=4 -> [29.97, 37.4625) -> 30..37
        (29.97, [0], list(range(0, 8))),
        (29.97, [1], list(range(8, 15))),
        (29.97, [4], list(range(30, 38))),
        # 15 fps: k=0 -> c < 3.75 -> 0..3; k=1 -> [3.75, 7.5) -> 4..7; k=2 -> [7.5, 11.25) -> 8..11
        (15.0, [0, 1, 2], list(range(0, 12))),
        (15.0, [2], list(range(8, 12))),
    ],
)
def test_sampled_to_native_calculado_a_mao(fps: float, sampled: list[int], expected: list[int]) -> None:
    assert _marked(fps, sampled) == expected


def test_sampled_to_native_corta_no_fim_e_ignora_repetidos() -> None:
    # 25 fps com 10 frames nativos: k=1 cobriria 7..12, sobra 7..9; k=5 fica todo fora
    assert _marked(25.0, [1, 1, 5], n_frames=10) == [7, 8, 9]
    assert len(sampled_to_native([1], 25.0, 10)) == 10


def test_sampled_to_native_vazio_e_invalidos() -> None:
    assert sampled_to_native([], 25.0, 5).tolist() == [0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="fps"):
        sampled_to_native([0], 0.0, 5)
    with pytest.raises(ValueError, match="n_frames"):
        sampled_to_native([0], 25.0, 0)
    with pytest.raises(ValueError, match=">= 0"):
        sampled_to_native([-1], 25.0, 5)


@pytest.mark.parametrize("fps", [25.0, 29.97, 30.0])
def test_primeiro_frame_do_intervalo_e_o_frame_que_o_extrator_grava(fps: float) -> None:
    # frame_extraction grava, para o instante k, o primeiro c com c / fps >= 0.25 k (test_frame_extraction.py)
    saved_by_extractor = {
        25.0: [0, 7, 13, 19, 25, 32, 38, 44, 50, 57, 63, 69],
        29.97: [0, 8, 15, 23, 30, 38, 45, 53, 60, 68, 75, 83],
        30.0: [0, 8, 15, 23, 30, 38, 45, 53, 60, 68, 75, 83],
    }[fps]
    assert [_marked(fps, [k])[0] for k in range(12)] == saved_by_extractor


def _reference_evaluation():
    """Classe Evaluation do submódulo, com stubs de tensorflow para o import do pacote."""
    stub_names = [
        "tensorflow",
        "tensorflow.keras",
        "tensorflow.keras.applications",
        "tensorflow.keras.applications.vgg16",
        "tensorflow.keras.applications.resnet50",
        "tensorflow.keras.models",
        "tensorflow.keras.utils",
    ]
    stubs = {name: types.ModuleType(name) for name in stub_names}
    stubs["tensorflow.keras.applications.vgg16"].VGG16 = object
    stubs["tensorflow.keras.applications.vgg16"].preprocess_input = lambda x: x
    stubs["tensorflow.keras.applications.resnet50"].ResNet50 = object
    stubs["tensorflow.keras.applications.resnet50"].preprocess_input = lambda x: x
    stubs["tensorflow.keras.models"].Model = object
    stubs["tensorflow.keras"].utils = stubs["tensorflow.keras.utils"]
    stubs["tensorflow.keras"].applications = stubs["tensorflow.keras.applications"]
    stubs["tensorflow.keras"].models = stubs["tensorflow.keras.models"]
    stubs["tensorflow"].keras = stubs["tensorflow.keras"]
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)
    if str(_SUBMODULE) not in sys.path:
        sys.path.insert(0, str(_SUBMODULE))
    from HieTaSumm.Evaluation import Evaluation

    return Evaluation.__new__(Evaluation)


@pytest.mark.skipif(
    not (_SUBMODULE / "HieTaSumm" / "Evaluation.py").exists(),
    reason="submódulo HieTaSumm-lib não baixado",
)
@pytest.mark.parametrize("eval_method", ["max", "avg"])
def test_igual_ao_evaluation_f_score_de_referencia(eval_method: str) -> None:
    reference = _reference_evaluation()
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(5, 300))
        S = rng.integers(0, 2, size=int(rng.integers(1, n + 20)))
        S[0] = 1  # o original divide por zero com resumo vazio
        G = rng.integers(0, 2, size=(int(rng.integers(1, 6)), n))
        G[:, 0] = 1
        assert f_score(S, G, eval_method) == reference.f_score(S, G, eval_method)
