"""Extração de frames de vídeo bruto, com a regra de amostragem de HieTaSumm.frame_extractor (D-02)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def video_metadata(video_path: Path) -> tuple[float, int]:
    """`(fps nativo, nº de frames)` do OpenCV; levanta `ValueError` se o vídeo não abre ou os metadados são inválidos."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Não consegui abrir o vídeo: {video_path}")
    fps, n_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or n_frames <= 0:
        raise ValueError(f"Metadados inválidos (fps={fps}, frames={n_frames}): {video_path}")
    return fps, n_frames


def saving_frames_durations(native_fps: float, frame_count: float, saving_period: float) -> list[float]:
    """Instantes (s) a gravar: `np.arange(0, int(frame_count / fps), período)` (`__init__.py:134-142`)."""
    # fiel ao original: o int() da duração deixa a fração final do vídeo sem amostra
    clip_duration = int(frame_count / native_fps)
    return list(np.arange(0, clip_duration, saving_period))


def expected_frame_count(video_path: Path, fps: float) -> int:
    """Número de frames que `extract_frames(video_path, ..., fps)` grava."""
    native_fps, frame_count = video_metadata(video_path)
    return len(saving_frames_durations(native_fps, frame_count, min(native_fps, 1 / fps)))


def extract_frames(video_path: Path, output_dir: Path, fps: float | None = None) -> list[Path]:
    """Grava `000000.jpg`, `000001.jpg`, ... (0-based, o índice do nó) em `output_dir` vazia; com `fps`, segue `__init__.py:86-131`."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Pasta de saída não está vazia: {output_dir}")

    durations: list[float] | None = None
    if fps is not None and fps > 0:
        native_fps, frame_count = video_metadata(video_path)
        # fiel ao original: min() compara fps nativo (frames/s) com o período 1/fps (s)
        durations = saving_frames_durations(native_fps, frame_count, min(native_fps, 1 / fps))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Não consegui abrir o vídeo: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if durations is None:
            save = True
        elif not durations:
            break
        else:
            save = count / native_fps >= durations[0]
            if save:
                durations.pop(0)
        if save:
            frame_path = output_dir / f"{len(saved_paths):06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_paths.append(frame_path)
        count += 1

    cap.release()
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path, help="Caminho do vídeo de entrada")
    parser.add_argument("output_dir", type=Path, help="Pasta de saída dos frames (vazia ou inexistente)")
    parser.add_argument("--fps", type=float, default=None, help="Frames por segundo (padrão: todos os frames)")
    args = parser.parse_args()

    frames = extract_frames(args.video_path, args.output_dir, fps=args.fps)
    print(f"{len(frames)} frames salvos em {args.output_dir}")
