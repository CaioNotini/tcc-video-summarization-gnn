"""Mede, por vídeo do SumMe, a duração do resumo keyshot da baseline HieTaSumm vs. o orçamento de 15% (Q-10).

Pipeline por vídeo: um frame a cada 1/rate = 0,25 s (frame_extraction.py, regra de
HieTaSumm.frame_extractor), com a contagem conferida contra os instantes esperados; cada frame
representa 0,25 s, então a duração do resumo é n_keyshot * 0,25 s, comparada à duração do vídeo
(frames nativos / fps). Features ResNet50/avgpool (mesmos
pesos e preprocessamento de features.py, com batch e GPU se houver), grafo time-aware (frame.py),
cutNumber = int(bestCutNumber() * alpha/100) com piso 3 (Summary.py:57-59), via a porta best_cut.py
(equivalente ao Summary.bestCutNumber do submódulo, validada em experiments/validate_best_cut.py e
validate_best_cut_keyshots.py),
hierarquia e corte portados (mst.py), e contagem dos frames de keyshot reproduzindo exatamente a
aritmética de Summary.group_central_frames (Summary.py:167-192): len_shot total =
round(n_disk * 15/100) - 1 (Summary.py:56), dividido por len(S) componentes (Summary.py:170);
componente com 1 folha contribui 0, com len < len_shot contribui len-1, senão 2*int(len_shot/2).
Nada em src/ é alterado; a cópia de jpgs do original é irrelevante para a duração e não é feita.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from tcc_gnn.hierarchy.best_cut import best_cut_number, cut_number_from_best  # noqa: E402
from tcc_gnn.features.frame_extraction import extract_frames, saving_frames_durations  # noqa: E402
from tcc_gnn.graph.frame import build_time_aware_graph  # noqa: E402
from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph  # noqa: E402

RATE_FPS = 4.0  # options.json: "rate": 4
PERCENT = 15.0  # options.json: "percent": 15
ALPHA = 75  # options.json: "alpha": 75

CSV_FIELDNAMES = [
    "video",
    "video_duration_s",
    "n_frames_extracted",
    "delta_t",
    "best_cut_number_raw",
    "cut_number",
    "n_components",
    "len_shot_total",
    "n_keyshot_frames",
    "summary_duration_s",
    "percent_of_duration",
    "exceeds_15",
]


def video_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Não consegui abrir o vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or n <= 0:
        raise ValueError(f"Metadados inválidos (fps={fps}, frames={n}): {video_path}")
    return n / fps


def expected_frame_count(video_path: Path) -> int:
    """Número de instantes que frame_extraction grava para o vídeo (um a cada 1/rate s)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return len(saving_frames_durations(fps, n, min(fps, 1 / RATE_FPS)))


def extract_features_batched(frame_dir: Path, device: torch.device, batch_size: int = 16) -> np.ndarray:
    """Mesmos pesos e preprocessamento de features.extract_features_torch, com batch e device."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preprocess = weights.transforms()

    frame_paths = sorted(p for p in Path(frame_dir).iterdir() if p.suffix == ".jpg")
    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[start : start + batch_size]
            batch = torch.stack(
                [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            ).to(device)
            out = model(batch)
            features.append(out.cpu().numpy())
    return np.concatenate(features)


def count_keyshot_frames(cut_graph_result: nx.Graph, leaf_list: list[int], n_disk: int) -> tuple[int, int, int]:
    """Conta os frames de keyshot de group_central_frames (Summary.py:167-192) sem copiar arquivos.

    Retorna (n_keyshot_frames, n_components, len_shot_total). len(S) inclui componentes sem folha,
    como no original (Summary.py:168-170 usa todos os componentes conexos do grafo de corte).
    """
    leaf_set = set(leaf_list)
    components = list(nx.connected_components(cut_graph_result))
    len_shot_total = round(n_disk * (PERCENT / 100)) - 1  # Summary.py:56
    len_shot = int(len_shot_total / len(components))  # Summary.py:170

    total = 0
    for component in components:
        len_leaf_list = sum(1 for node in component if node in leaf_set)
        cn = int(len_leaf_list / 2)  # Summary.py:177
        if cn == 0:  # Summary.py:178
            continue
        if len_leaf_list < len_shot:  # Summary.py:180-182
            total += len_leaf_list - 1
        else:  # Summary.py:184-185
            total += 2 * int(len_shot / 2)
    return total, len(components), len_shot_total


def run_video(video_path: Path, frames_root: Path, delta_ts: list[int], device: torch.device) -> list[dict]:
    name = video_path.stem
    frames_dir = frames_root / name / "frames"

    if not frames_dir.is_dir() or not any(frames_dir.glob("*.jpg")):
        extract_frames(video_path, frames_dir, fps=RATE_FPS)
    n_disk = sum(1 for _ in frames_dir.glob("*.jpg"))

    duration_s = video_duration_seconds(video_path)
    n_expected = expected_frame_count(video_path)
    if n_disk != n_expected:
        raise ValueError(f"{name}: {n_disk} frames em disco, {n_expected} esperados (extração incompleta)")

    features_path = frames_root / name / "features_resnet50_avgpool.npy"
    if features_path.exists():
        features = np.load(features_path)
    else:
        features = extract_features_batched(frames_dir, device)
        np.save(features_path, features)
    if len(features) != n_disk:
        raise ValueError(f"{name}: {len(features)} features para {n_disk} frames")

    rows = []
    for delta_t in delta_ts:
        best_cut_number_raw = best_cut_number(frames_dir, delta_t)
        cut_number = cut_number_from_best(best_cut_number_raw, alpha=ALPHA)
        graph = build_time_aware_graph(features, delta_t)
        triples, leaf_list, _altitudes = compute_hierarchy(graph)
        cut = cut_graph(triples, cut_number)
        n_keyshot, n_components, len_shot_total = count_keyshot_frames(cut, leaf_list, n_disk)

        summary_s = n_keyshot * (1 / RATE_FPS)  # cada frame amostrado cobre 1/rate = 0,25 s
        percent = 100.0 * summary_s / duration_s
        rows.append(
            {
                "video": name,
                "video_duration_s": round(duration_s, 2),
                "n_frames_extracted": n_disk,
                "delta_t": delta_t,
                "best_cut_number_raw": best_cut_number_raw,
                "cut_number": cut_number,
                "n_components": n_components,
                "len_shot_total": len_shot_total,
                "n_keyshot_frames": n_keyshot,
                "summary_duration_s": round(summary_s, 2),
                "percent_of_duration": round(percent, 2),
                "exceeds_15": percent > 15.0,
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, default=_REPO_ROOT / "data" / "processed" / "summe")
    parser.add_argument("--delta-t", type=int, nargs="+", default=[32, 8],
                        help="32 = rate*time do options.json (Summary.py:15); 8 = configs/default.yaml")
    parser.add_argument("--csv-out", type=Path,
                        default=_REPO_ROOT / "docs" / "evidence" / "sheets" / "q10-baseline-budget.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    video_paths = sorted(
        p for p in args.videos_dir.iterdir() if p.suffix.lower() in {".mp4", ".webm", ".avi", ".mov"}
    )
    print(f"{len(video_paths)} vídeos em {args.videos_dir}")

    all_rows: list[dict] = []
    for i, video_path in enumerate(video_paths, 1):
        t0 = time.time()
        rows = run_video(video_path, args.frames_root, args.delta_t, device)
        all_rows.extend(rows)
        for r in rows:
            print(
                f"[{i}/{len(video_paths)}] {r['video']}: dur={r['video_duration_s']}s "
                f"delta_t={r['delta_t']} cut={r['cut_number']} comps={r['n_components']} "
                f"keyshot={r['n_keyshot_frames']}f = {r['summary_duration_s']}s "
                f"({r['percent_of_duration']}%) exceeds={r['exceeds_15']} "
                f"[{time.time() - t0:.0f}s]"
            )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV salvo em {args.csv_out}")

    for delta_t in args.delta_t:
        sub = [r for r in all_rows if r["delta_t"] == delta_t]
        over = [r for r in sub if r["exceeds_15"]]
        pcts = [r["percent_of_duration"] for r in sub]
        print(
            f"delta_t={delta_t}: {len(over)}/{len(sub)} vídeos acima de 15%; "
            f"média={np.mean(pcts):.2f}% máx={np.max(pcts):.2f}%"
        )
