"""T07: baseline HieTaSumm de ponta a ponta no SumMe, com F oficial (D-26), F suplementar pela regra original e tempo/memória por etapa."""

from __future__ import annotations

import argparse
import csv
import platform
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import mlflow
import networkx as nx
import numpy as np
import psutil
import torch
from scipy import io

from tcc_gnn.config import REPO_ROOT as _REPO_ROOT
from tcc_gnn.config import load_config, tracking_config
from tcc_gnn.data.summe import SumMeAdapter
from tcc_gnn.evaluation.fscore import f_score, sampled_to_native
from tcc_gnn.features.features import extract_features
from tcc_gnn.features.frame_extraction import extract_frames, video_metadata
from tcc_gnn.graph.frame import build_time_aware_graph
from tcc_gnn.hierarchy.best_cut import best_cut_number, cut_number_from_best
from tcc_gnn.hierarchy.keyshots import group_central_frames, initial_len_shot
from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph
from tcc_gnn.reference import reference_evaluation

_CFG = load_config()
RATE_FPS = float(_CFG["graph"]["rate"])  # D-25
PERCENT = float(_CFG["hierarchy"]["percent"])  # D-21
ALPHA = _CFG["hierarchy"]["alpha"]  # D-20
VIDEOS_DIR = _REPO_ROOT / "data" / "raw" / "summe" / "SumMe" / "videos"
FRAMES_ROOT = _REPO_ROOT / "data" / "processed" / "summe"
MAT_DIR = _REPO_ROOT / "references" / "HieTaSumm-lib" / "HieTaSumm" / "mat-files"
TMP_ROOT = _REPO_ROOT / ".cache" / "tmp" / "t07_extraction"
STAGES = [
    "extraction",
    "features",
    "best_cut",
    "graph",
    "hierarchy_cut",
    "keyshots",
    "eval_official",
    "eval_original_rule",
]


@contextmanager
def measure(stage: str, row: dict, interval_s: float = 0.02):
    """Grava em `row` o tempo de parede e o pico de RSS do processo (amostrado a cada 20 ms) da etapa."""
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample() -> None:
        """Atualiza o pico de RSS até a etapa terminar."""
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.wait(interval_s)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        row[f"time_{stage}_s"] = round(time.perf_counter() - t0, 3)
        stop.set()
        sampler.join()
        peak[0] = max(peak[0], process.memory_info().rss)
        row[f"cpu_peak_mb_{stage}"] = round(peak[0] / 2**20, 1)


def frames_identical(dir_a: Path, dir_b: Path) -> bool:
    """Mesmos nomes e mesmos bytes nos jpgs das duas pastas."""
    a = sorted(dir_a.glob("*.jpg"))
    b = sorted(dir_b.glob("*.jpg"))
    return [p.name for p in a] == [p.name for p in b] and all(
        pa.read_bytes() == pb.read_bytes() for pa, pb in zip(a, b)
    )


def run_video(video_id: str, name: str, delta_t: int, device: torch.device, adapter: SumMeAdapter, reference) -> dict:
    """Linha do CSV de um vídeo: reprodutibilidade de frames e features, pipeline, F oficial e suplementar, tempo e memória."""
    video_path = VIDEOS_DIR / f"{name}.mp4"
    frames_dir = FRAMES_ROOT / name / "frames"
    features_path = FRAMES_ROOT / name / "features_resnet50_avgpool.npy"
    fps, n_frames_opencv = video_metadata(video_path)
    gt = adapter.load_ground_truth(video_id)
    mat = io.loadmat(MAT_DIR / f"{name.replace(' ', '_')}.mat")
    row: dict = {
        "video_id": video_id,
        "video": name,
        "fps": round(fps, 3),
        "n_frames_h5": gt.n_frames,
        "n_frames_opencv": n_frames_opencv,
        "duration_opencv_s": round(n_frames_opencv / fps, 2),
        "duration_mat_s": round(float(mat["video_duration"][0][0]), 2),
    }

    tmp_dir = TMP_ROOT / name
    shutil.rmtree(tmp_dir, ignore_errors=True)
    with measure("extraction", row):
        extract_frames(video_path, tmp_dir, fps=RATE_FPS)
    row["extraction_identical"] = frames_identical(tmp_dir, frames_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    cached = np.load(features_path)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with measure("features", row):
        recomputed = extract_features(frames_dir, device)
    row["gpu_peak_mb_features"] = (
        round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if device.type == "cuda" else ""
    )
    row["features_allclose"] = bool(recomputed.shape == cached.shape and np.allclose(recomputed, cached))
    row["features_max_abs_diff"] = (
        float(np.max(np.abs(recomputed - cached))) if recomputed.shape == cached.shape else ""
    )
    features = cached
    n_disk = sum(1 for _ in frames_dir.glob("*.jpg"))
    row["n_frames_sampled"] = n_disk
    if len(features) != n_disk:
        raise ValueError(f"{name}: {len(features)} features para {n_disk} frames")

    with measure("best_cut", row):
        raw = best_cut_number(frames_dir, delta_t)
        cut_number = cut_number_from_best(raw, alpha=ALPHA)
    with measure("graph", row):
        graph = build_time_aware_graph(features, delta_t)
    with measure("hierarchy_cut", row):
        triples, leaf_list, _altitudes = compute_hierarchy(graph)
        cut = cut_graph(triples, cut_number)
    with measure("keyshots", row):
        keyshots = group_central_frames(cut, leaf_list, initial_len_shot(n_disk, PERCENT))
    with measure("eval_official", row):
        summary = sampled_to_native(keyshots, fps, gt.n_frames)
        f_official = f_score(summary, gt.keyframes, "max")
    with measure("eval_original_rule", row):
        # passa o id da folha (0-based) como o original, embora o jpg dele seja 1-based: característica do original
        sb_summary = reference.find_corresponding_frames(gt.n_frames, float(mat["video_duration"][0][0]), list(keyshots))
        f_original = float(reference.f_score(sb_summary, gt.keyframes, "max"))

    row.update(
        {
            "best_cut_number_raw": raw,
            "cut_number": cut_number,
            "n_components": nx.number_connected_components(cut),
            "n_keyshots": len(keyshots),
            "summary_pct_duration": round(100 * len(keyshots) * (1 / RATE_FPS) / (n_frames_opencv / fps), 2),
            "summary_pct_native": round(100 * int(summary.sum()) / gt.n_frames, 2),
            "summary_pct_original_rule": round(100 * int(np.sum(sb_summary)) / gt.n_frames, 2),
            "f_score_official": round(f_official, 4),
            "f_score_original_rule": round(f_original, 4),
            "keyshots": " ".join(str(k) for k in sorted(keyshots)),
        }
    )
    return row


def git_commit() -> str:
    """Commit curto do HEAD, com `-dirty` se há mudanças locais."""
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=_REPO_ROOT, capture_output=True, text=True)
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="*", help="nomes dos vídeos (padrão: os 25 do SumMe)")
    parser.add_argument("--delta-t", type=int, default=_CFG["graph"]["delta_t"], help="D-24")
    parser.add_argument("--csv-out", type=Path,
                        default=_REPO_ROOT / "docs" / "evidence" / "sheets" / "t07-baseline-summe.csv")
    parser.add_argument("--experiment", default="T07-hietasumm-baseline-summe")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = SumMeAdapter()
    names = adapter.video_names()
    selected = [(vid, n) for vid, n in names.items() if not args.videos or n in args.videos]
    unknown = set(args.videos or []) - {n for _, n in selected}
    if unknown:
        raise SystemExit(f"Vídeos desconhecidos: {sorted(unknown)}")
    reference = reference_evaluation()

    uri, artifact_root = tracking_config(_CFG)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(args.experiment) is None:
        mlflow.create_experiment(args.experiment, artifact_location=f"{artifact_root}/{args.experiment}")
    mlflow.set_experiment(args.experiment)
    rows: list[dict] = []
    t_start = time.perf_counter()
    with mlflow.start_run():
        mlflow.log_params({
            "delta_t": args.delta_t, "alpha": ALPHA, "percent": PERCENT, "rate": RATE_FPS,
            "extractor": "torchvision ResNet50 IMAGENET1K_V2 avgpool, batch 16",
            "frame_sampling": "every 0.25 s, HieTaSumm-lib rule (D-25)",
            "official_mapping": "D-26", "commit": git_commit(), "n_videos": len(selected),
        })
        mlflow.set_tags({
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "none",
            "cuda": str(torch.version.cuda), "torch": torch.__version__,
            "cpu": platform.processor(), "os": platform.platform(), "python": platform.python_version(),
        })
        mlflow.set_tag("mlflow_version", mlflow.__version__)
        print(f"Device: {device} | {len(selected)} vídeos | tracking {uri} | artefatos {artifact_root}")
        for i, (video_id, name) in enumerate(selected):
            row = run_video(video_id, name, args.delta_t, device, adapter, reference)
            rows.append(row)
            for key in ["f_score_official", "f_score_original_rule", "summary_pct_duration", "cut_number"]:
                mlflow.log_metric(key, row[key], step=i)
            total = sum(row[f"time_{s}_s"] for s in STAGES)
            print(
                f"[{i + 1}/{len(selected)}] {name}: cut={row['cut_number']} keyshots={row['n_keyshots']} "
                f"pct={row['summary_pct_duration']}% F_oficial={row['f_score_official']} "
                f"F_regra_original={row['f_score_original_rule']} frames_iguais={row['extraction_identical']} "
                f"features_allclose={row['features_allclose']} [{total:.0f}s]"
            )
        wall = time.perf_counter() - t_start
        mean = {k: float(np.mean([r[k] for r in rows])) for k in
                ["f_score_official", "f_score_original_rule", "summary_pct_duration"]}
        mlflow.log_metrics({f"mean_{k}": v for k, v in mean.items()} | {"wall_time_s": wall})

        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        mlflow.log_artifact(str(args.csv_out))

    print(f"\nCSV salvo em {args.csv_out}")
    print(
        f"médias: F_oficial={mean['f_score_official']:.2f} F_regra_original={mean['f_score_original_rule']:.2f} "
        f"pct={mean['summary_pct_duration']:.2f}% | frames idênticos {sum(r['extraction_identical'] for r in rows)}/"
        f"{len(rows)} | features allclose {sum(r['features_allclose'] for r in rows)}/{len(rows)} | "
        f"tempo total {wall:.0f}s"
    )
