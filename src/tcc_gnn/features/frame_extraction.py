"""Extração de frames de vídeo bruto para imagens estáticas.
"""

from __future__ import annotations
from pathlib import Path
import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Não consegui abrir o vídeo: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        frame_interval = 1
    else:
        frame_interval = max(1, round(native_fps / fps))

    saved_paths: list[Path] = []
    frame_index = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_interval == 0:
            frame_path = output_dir / f"{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_paths.append(frame_path)
            saved_count += 1

        frame_index += 1

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