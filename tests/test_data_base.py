"""Testes para data/base.py: o formato comum de ground truth (D-15) e a interface dos adaptadores."""

from __future__ import annotations

import inspect

import numpy as np

from tcc_gnn.data.base import DatasetAdapter, GroundTruth, ScoreSeries
from tcc_gnn.data.summe import SumMeAdapter


def test_ground_truth_campos_opcionais_valem_none() -> None:
    gt = GroundTruth(video_id="v", n_frames=3, keyframes=np.zeros((2, 3), dtype=np.int8))
    assert gt.scores is None
    assert gt.change_points is None


def test_ground_truth_com_score_e_change_points() -> None:
    scores = ScoreSeries(values=np.array([0.1, 0.9]), frame_indices=np.array([0, 15]))
    gt = GroundTruth(
        video_id="v",
        n_frames=30,
        keyframes=np.zeros((1, 30), dtype=np.int8),
        scores=scores,
        change_points=np.array([[0, 14], [15, 29]]),
    )
    assert gt.scores is scores
    assert gt.change_points.shape == (2, 2)


def test_summe_implementa_a_interface_do_adaptador() -> None:
    for name in ("list_videos", "load_ground_truth"):
        protocol_sig = inspect.signature(getattr(DatasetAdapter, name))
        adapter_sig = inspect.signature(getattr(SumMeAdapter, name))
        assert list(protocol_sig.parameters) == list(adapter_sig.parameters)
