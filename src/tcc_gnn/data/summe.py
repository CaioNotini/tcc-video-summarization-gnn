"""Adaptador do SumMe (D-15): lista os vídeos e lê o ground truth no formato comum."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tcc_gnn.data.base import GroundTruth, ScoreSeries

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[3]
    / "references"
    / "HieTaSumm-lib"
    / "HieTaSumm"
    / "eccv16_dataset_summe_google_pool5.h5"
)


class SumMeAdapter:
    """Lê `eccv16_dataset_summe_google_pool5.h5`, do submódulo HieTaSumm-lib."""

    def __init__(self, dataset_path: Path = DEFAULT_DATASET_PATH) -> None:
        self.dataset_path = dataset_path

    def list_videos(self) -> list[str]:
        with h5py.File(self.dataset_path, "r") as f:
            return sorted(f.keys(), key=lambda video_id: int(video_id.split("_")[1]))

    def load_ground_truth(self, video_id: str) -> GroundTruth:
        with h5py.File(self.dataset_path, "r") as f:
            video = f[video_id]
            n_frames = int(video["n_frames"][()])
            keyframes = np.asarray(video["user_summary"][()], dtype=np.int8)
            scores = ScoreSeries(
                values=np.asarray(video["gtscore"][()], dtype=np.float32),
                frame_indices=np.asarray(video["picks"][()], dtype=np.int64),
            )
            change_points = np.asarray(video["change_points"][()], dtype=np.int64)

        return GroundTruth(
            video_id=video_id,
            n_frames=n_frames,
            keyframes=keyframes,
            scores=scores,
            change_points=change_points,
        )
