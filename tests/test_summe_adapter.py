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


def test_load_ground_truth_video_inexistente_levanta_erro():
    adapter = SumMeAdapter()

    with pytest.raises(KeyError):
        adapter.load_ground_truth("video_999")
