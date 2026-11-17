"""Q-10: duração do resumo keyshot da baseline HieTaSumm (n_keyshots * 1/rate s) contra o orçamento de 15%, por vídeo do SumMe."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from tcc_gnn.config import REPO_ROOT, load_config
from tcc_gnn.features.features import cached_features
from tcc_gnn.features.frame_extraction import (
    expected_frame_count,
    extract_frames,
    video_metadata,
)
from tcc_gnn.graph.frame import build_time_aware_graph
from tcc_gnn.hierarchy.best_cut import best_cut_number, cut_number_from_best
from tcc_gnn.hierarchy.keyshots import group_central_frames, initial_len_shot
from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph

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


def run_video(video_path: Path, frames_root: Path, delta_ts: list[int], cfg: dict, device: torch.device) -> list[dict]:
    """Uma linha do CSV por `delta_t`; extrai frames e features só se faltarem em disco."""
    rate, alpha, percent = cfg["graph"]["rate"], cfg["hierarchy"]["alpha"], cfg["hierarchy"]["percent"]
    name = video_path.stem
    frames_dir = frames_root / name / "frames"

    if not frames_dir.is_dir() or not any(frames_dir.glob("*.jpg")):
        extract_frames(video_path, frames_dir, fps=rate)
    n_disk = sum(1 for _ in frames_dir.glob("*.jpg"))

    native_fps, n_native = video_metadata(video_path)
    duration_s = n_native / native_fps
    n_expected = expected_frame_count(video_path, rate)
    if n_disk != n_expected:
        raise ValueError(f"{name}: {n_disk} frames em disco, {n_expected} esperados (extração incompleta)")

    features = cached_features(frames_dir, frames_root / name / "features_resnet50_avgpool.npy", device)
    if len(features) != n_disk:
        raise ValueError(f"{name}: {len(features)} features para {n_disk} frames")

    rows = []
    for delta_t in delta_ts:
        best_cut_number_raw = best_cut_number(frames_dir, delta_t)
        cut_number = cut_number_from_best(best_cut_number_raw, alpha=alpha)
        triples, leaf_list, _altitudes = compute_hierarchy(build_time_aware_graph(features, delta_t))
        cut = cut_graph(triples, cut_number)
        len_shot_total = initial_len_shot(n_disk, percent)
        n_keyshot = len(group_central_frames(cut, leaf_list, len_shot_total))

        summary_s = n_keyshot * (1 / rate)  # cada frame amostrado cobre 1/rate = 0,25 s
        pct = 100.0 * summary_s / duration_s
        rows.append(
            {
                "video": name,
                "video_duration_s": round(duration_s, 2),
                "n_frames_extracted": n_disk,
                "delta_t": delta_t,
                "best_cut_number_raw": best_cut_number_raw,
                "cut_number": cut_number,
                "n_components": nx.number_connected_components(cut),
                "len_shot_total": len_shot_total,
                "n_keyshot_frames": n_keyshot,
                "summary_duration_s": round(summary_s, 2),
                "percent_of_duration": round(pct, 2),
                "exceeds_15": pct > percent,
            }
        )
    return rows


if __name__ == "__main__":
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, default=REPO_ROOT / "data" / "processed" / "summe")
    parser.add_argument("--delta-t", type=int, nargs="+", default=[cfg["graph"]["delta_t"]], help="D-24")
    parser.add_argument("--csv-out", type=Path,
                        default=REPO_ROOT / "docs" / "evidence" / "sheets" / "q10-baseline-budget.csv")
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
        rows = run_video(video_path, args.frames_root, args.delta_t, cfg, device)
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
