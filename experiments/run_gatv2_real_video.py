"""T09: bloco GP (Linear + GATv2 + residual + BN) sem treino, em eval(), sobre o grafo time-aware de um vídeo real do SumMe."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric

from tcc_gnn.config import REPO_ROOT, SUMME_FEATURES_FILE, SUMME_PROCESSED, load_config
from tcc_gnn.features.frame_extraction import frame_paths
from tcc_gnn.graph.frame import build_time_aware_graph
from tcc_gnn.models.gatv2 import (
    GraphProcessingBlock,
    graph_processing_block,
    time_aware_edge_index,
)

WARMUP_FORWARDS = 2
TIMED_FORWARDS = 10


def git_commit() -> str:
    """Commit curto do HEAD, com `-dirty` se há mudanças locais."""
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def probe_diffs(block: GraphProcessingBlock, x: torch.Tensor, edge_index: torch.Tensor, probe_frame: int) -> np.ndarray:
    """Maior mudança absoluta por nó quando as features de `probe_frame` recebem +1."""
    changed = x.clone()
    changed[probe_frame] += 1.0
    with torch.no_grad():
        return (block(changed, edge_index) - block(x, edge_index)).abs().amax(dim=1).cpu().numpy()


def repeat_noise(block: GraphProcessingBlock, x: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Maior diferença absoluta entre duas chamadas com a mesma entrada."""
    with torch.no_grad():
        return float((block(x, edge_index) - block(x, edge_index)).abs().max())


def run(name: str, seed: int, probe_frame: int) -> dict:
    """Checagens exatas na CPU (localidade, determinismo) e medição na GPU (forma, finitude, atenção, tempo, memória)."""
    cfg = load_config()
    delta_t = cfg["graph"]["delta_t"]
    frames_dir = SUMME_PROCESSED / name / "frames"
    features = np.load(SUMME_PROCESSED / name / SUMME_FEATURES_FILE)
    n_frames_on_disk = len(frame_paths(frames_dir))
    if len(features) != n_frames_on_disk:
        raise ValueError(f"{name}: {len(features)} features para {n_frames_on_disk} frames")
    graph = build_time_aware_graph(features, delta_t)
    edge_index = time_aware_edge_index(graph)
    x = torch.tensor(features, dtype=torch.float32)
    n = x.shape[0]
    window = np.array([abs(i - probe_frame) < delta_t for i in range(n)])

    torch.manual_seed(seed)
    cpu_block = graph_processing_block(cfg, in_dim=x.shape[1]).eval()
    torch.manual_seed(seed)
    rebuilt = graph_processing_block(cfg, in_dim=x.shape[1]).eval()
    with torch.no_grad():
        cpu_out = cpu_block(x, edge_index)
        rebuilt_out = rebuilt(x, edge_index)
    cpu_diffs = probe_diffs(cpu_block, x, edge_index, probe_frame)
    cpu = {
        "output_shape": list(cpu_out.shape),
        "output_finite": bool(torch.isfinite(cpu_out).all()),
        "same_seed_rebuilt_identical": bool(torch.equal(cpu_out, rebuilt_out)),
        "repeat_noise": repeat_noise(cpu_block, x, edge_index),
        "probe_outside_window_max_change": float(cpu_diffs[~window].max()),
        "probe_outside_window_nodes_changed": int((cpu_diffs[~window] > 0).sum()),
        "probe_inside_window_min_change": float(cpu_diffs[window].min()),
        "probe_changed_nodes_equal_window": bool(((cpu_diffs > 0) == window).all()),
    }

    gpu = None
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        gpu_block = graph_processing_block(cfg, in_dim=x.shape[1])
        gpu_block.load_state_dict(cpu_block.state_dict())
        gpu_block = gpu_block.to(dev).eval()
        xg, eg = x.to(dev), edge_index.to(dev)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        with torch.no_grad():
            t0 = time.perf_counter()
            gpu_out, (att_index, beta) = gpu_block(xg, eg, return_attention=True)
            torch.cuda.synchronize(dev)
            first_call_s = time.perf_counter() - t0
            for _ in range(WARMUP_FORWARDS):
                gpu_block(xg, eg)
            times = []
            for _ in range(TIMED_FORWARDS):
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                gpu_block(xg, eg)
                torch.cuda.synchronize(dev)
                times.append(time.perf_counter() - t0)
        src, dst = att_index
        sums = torch.zeros(n, beta.shape[1], device=dev).index_add_(0, dst, beta)
        gpu_diffs = probe_diffs(gpu_block, xg, eg, probe_frame)
        gpu_np = gpu_out.cpu().numpy()
        gpu = {
            "output_shape": list(gpu_out.shape),
            "output_finite": bool(torch.isfinite(gpu_out).all()),
            "output_mean": float(gpu_np.mean()),
            "output_std": float(gpu_np.std()),
            "output_min": float(gpu_np.min()),
            "output_max": float(gpu_np.max()),
            "max_abs_diff_vs_cpu": float((gpu_out.cpu() - cpu_out).abs().max()),
            "attention_shape": list(beta.shape),
            "attention_self_pairs": int((src == dst).sum()),
            "attention_min": float(beta.min()),
            "attention_max": float(beta.max()),
            "attention_sum_max_abs_dev_from_1": float((sums - 1).abs().max()),
            "repeat_noise": repeat_noise(gpu_block, xg, eg),
            "probe_outside_window_max_change": float(gpu_diffs[~window].max()),
            "probe_outside_window_nodes_changed": int((gpu_diffs[~window] > 0).sum()),
            "probe_inside_window_min_change": float(gpu_diffs[window].min()),
            "first_call_s": round(first_call_s, 4),
            "warmup_forwards": WARMUP_FORWARDS,
            "timed_forwards": TIMED_FORWARDS,
            "forward_mean_ms": round(1000 * float(np.mean(times)), 3),
            "forward_std_ms": round(1000 * float(np.std(times)), 3),
            "peak_memory_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1),
            "gpu": torch.cuda.get_device_name(dev),
        }

    return {
        "video": name,
        "n_frames": n,
        "delta_t": delta_t,
        "n_undirected_edges": graph.number_of_edges(),
        "edge_index_shape": list(edge_index.shape),
        "input_shape": list(x.shape),
        "n_parameters": sum(p.numel() for p in cpu_block.parameters()),
        "seed": seed,
        "probe_frame": probe_frame,
        "window_size": int(window.sum()),
        "mode": "eval, torch.no_grad, untrained weights",
        "config_model": cfg["model"],
        "cpu": cpu,
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__,
        "python": platform.python_version(),
        "processor": platform.processor(),
        "os": platform.platform(),
        "commit": git_commit(),
    }


def checks_pass(result: dict) -> bool:
    """Critério da T09: CPU exata (forma, finitude, localidade, mesma seed) e GPU com forma, finitude e atenção somando 1."""
    expected_shape = [result["n_frames"], result["config_model"]["embedding_size"]]
    cpu = result["cpu"]
    ok = (
        cpu["output_shape"] == expected_shape
        and cpu["output_finite"]
        and cpu["same_seed_rebuilt_identical"]
        and cpu["probe_outside_window_max_change"] == 0.0
        and cpu["probe_changed_nodes_equal_window"]
    )
    gpu = result["gpu"]
    if gpu is not None:
        ok = ok and (
            gpu["output_shape"] == expected_shape
            and gpu["output_finite"]
            and gpu["attention_self_pairs"] == 0
            and gpu["attention_sum_max_abs_dev_from_1"] < 1e-5
        )
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="Jumps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-frame", type=int, default=76, help="frame perturbado na checagem de localidade")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.video, args.seed, args.probe_frame)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    ok = checks_pass(result)
    print("T09 CHECAGENS:", "OK" if ok else "FALHOU")
    raise SystemExit(0 if ok else 1)
