"""Formato comum de ground truth (D-15) e interface que cada adaptador de dataset implementa."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class ScoreSeries:
    """Score de importância contínuo, definido só num subconjunto dos frames do vídeo."""

    values: np.ndarray  # shape (n_scored,), float
    frame_indices: np.ndarray  # shape (n_scored,), int, índice no vídeo original


@dataclass
class GroundTruth:
    """Ground truth de um vídeo no formato comum entre datasets (D-15)."""

    video_id: str
    n_frames: int
    keyframes: np.ndarray  # shape (n_annotators, n_frames), binário {0, 1}
    scores: ScoreSeries | None = None
    change_points: np.ndarray | None = None  # shape (n_segments, 2), int


class DatasetAdapter(Protocol):
    """Interface que cada adaptador de dataset implementa para expor seu ground truth."""

    def list_videos(self) -> list[str]: ...

    def load_ground_truth(self, video_id: str) -> GroundTruth: ...
