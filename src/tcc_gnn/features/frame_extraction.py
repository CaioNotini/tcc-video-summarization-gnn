"""Extração de frames de vídeo bruto, com a regra de amostragem de HieTaSumm.frame_extractor (D-02).
"""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def saving_frames_durations(native_fps: float, frame_count: float, saving_period: float) -> list[float]:
    """Instantes (s) a gravar: `np.arange(0, int(frame_count / fps), período)` (`__init__.py:134-142`)."""
    clip_duration = int(frame_count / native_fps)
    return list(np.arange(0, clip_duration, saving_period))


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float | None = None,
) -> list[Path]:
    """Grava frames como `000000.jpg`, `000001.jpg`, ...; com `fps`, um frame a cada `1/fps` s.

    Com `fps`, segue `HieTaSumm.frame_extractor` (`__init__.py:86-131`): período
    `min(fps_nativo, 1/fps)`, instantes de 0 até `int(duração)` exclusive, e grava o primeiro frame
    nativo com `índice / fps_nativo >= instante` pendente, um instante por frame. A numeração
    começa em 0, não em 1 como no original, para coincidir com o índice do nó no grafo.
    Sem `fps` (ou `fps <= 0`), grava todos os frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Não consegui abrir o vídeo: {video_path}")

    durations: list[float] | None = None
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is not None and fps > 0:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if native_fps <= 0 or frame_count <= 0:
            cap.release()
            raise ValueError(
                f"Metadados inválidos (fps={native_fps}, frames={frame_count}): {video_path}"
            )
        durations = saving_frames_durations(native_fps, frame_count, min(native_fps, 1 / fps))

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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path, help="Caminho do vídeo de entrada")
    parser.add_argument("output_dir", type=Path, help="Pasta de saída dos frames")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Taxa de amostragem em frames/segundo (padrão: extrai todos os frames)",
    )
    args = parser.parse_args()

    frames = extract_frames(args.video_path, args.output_dir, fps=args.fps)
    print(f"{len(frames)} frames salvos em {args.output_dir}")
