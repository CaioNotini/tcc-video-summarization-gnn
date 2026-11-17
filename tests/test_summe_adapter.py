"""Testes para data/summe.py."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from tcc_gnn.data.summe import DEFAULT_DATASET_PATH, SumMeAdapter

pytestmark = pytest.mark.skipif(
    not DEFAULT_DATASET_PATH.exists(), reason="submódulo HieTaSumm-lib não baixado"
)


def test_list_videos_retorna_os_25_videos_em_ordem_numerica():
    adapter = SumMeAdapter()
    videos = adapter.list_videos()

    assert len(videos) == 25
    assert videos[0] == "video_1"
    assert videos[-1] == "video_25"
    assert videos == sorted(videos, key=lambda v: int(v.split("_")[1]))


def test_load_ground_truth_video_1_bate_com_o_arquivo_do_submodulo():
    with h5py.File(DEFAULT_DATASET_PATH, "r") as f:
        v = f["video_1"]
        n_frames_esperado = int(v["n_frames"][()])
        keyframes_esperado = np.asarray(v["user_summary"][()], dtype=np.int8)
        scores_values_esperado = np.asarray(v["gtscore"][()], dtype=np.float32)
        picks_esperado = np.asarray(v["picks"][()], dtype=np.int64)
        change_points_esperado = np.asarray(v["change_points"][()], dtype=np.int64)

    adapter = SumMeAdapter()
    gt = adapter.load_ground_truth("video_1")

    assert gt.video_id == "video_1"
    assert gt.n_frames == n_frames_esperado == 4494
    np.testing.assert_array_equal(gt.keyframes, keyframes_esperado)
    assert gt.keyframes.shape == (15, 4494)
    assert set(np.unique(gt.keyframes).tolist()) <= {0, 1}

    assert gt.scores is not None
    np.testing.assert_array_equal(gt.scores.values, scores_values_esperado)
    np.testing.assert_array_equal(gt.scores.frame_indices, picks_esperado)
    assert gt.scores.values.shape == gt.scores.frame_indices.shape == (300,)

    np.testing.assert_array_equal(gt.change_points, change_points_esperado)
    assert gt.change_points.shape == (30, 2)


NOMES_COM_ESPACO = {
    "Base jumping",
    "Bike Polo",
    "Eiffel Tower",
    "Excavators river crossing",
    "Fire Domino",
    "Saving dolphines",
    "St Maarten Landing",
    "Statue of Liberty",
}
_VIDEOS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "summe" / "SumMe" / "videos"


def test_video_names_25_nomes_unicos_em_ordem():
    names = SumMeAdapter().video_names()

    assert list(names) == SumMeAdapter().list_videos()
    assert names["video_1"] == "Air_Force_One"
    assert names["video_12"] == "Jumps"
    assert len(set(names.values())) == 25
    assert {n for n in names.values() if " " in n} == NOMES_COM_ESPACO


@pytest.mark.skipif(not _VIDEOS_DIR.is_dir(), reason="vídeos do SumMe não estão em data/raw")
def test_video_names_cada_nome_tem_um_mp4():
    names = set(SumMeAdapter().video_names().values())
    mp4 = {p.stem for p in _VIDEOS_DIR.glob("*.mp4")}

    assert names == mp4


def test_load_ground_truth_video_inexistente_levanta_erro():
    adapter = SumMeAdapter()

    with pytest.raises(KeyError):
        adapter.load_ground_truth("video_999")
