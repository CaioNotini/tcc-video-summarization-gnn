"""Valida a porta (best_cut.py, mst.py, keyshots.py) contra o HieTaSumm-lib em vídeos do SumMe: bestCutNumber, cutNumber e keyshots."""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from tcc_gnn.config import REPO_ROOT, load_config
from tcc_gnn.features.features import cached_features
from tcc_gnn.graph.frame import build_time_aware_graph
from tcc_gnn.hierarchy.best_cut import best_cut_number, cut_number_from_best
from tcc_gnn.hierarchy.keyshots import group_central_frames, initial_len_shot
from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph
from tcc_gnn.reference import (
    reference_best_cut_number,
    reference_compute_hierarchy,
    reference_cut_graph,
    reference_graph,
    reference_group_central_frames,
)

FRAMES_ROOT = REPO_ROOT / "data" / "processed" / "summe"
_CACHE = REPO_ROOT / ".cache" / "reference_best_cut.json"


def _ref_load() -> dict:
    """Cache do bestCutNumber de referência, por `vídeo|delta_t`."""
    if _CACHE.exists():
        with open(_CACHE) as f:
            return json.load(f)
    return {}


def _ref_put(name: str, delta_t: int, ref: dict) -> None:
    """Grava no cache o bestCutNumber de referência de `(name, delta_t)`."""
    data = _ref_load()
    data[f"{name}|{delta_t}"] = {"best_cut_number_raw": ref["best_cut_number_raw"], "cut_number_ref": ref["cut_number_ref"]}
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE, "w") as f:
        json.dump(data, f, indent=2)


def run_video(name: str, delta_t: int, alpha: int, percent: float, device: torch.device, use_cache: bool) -> bool:
    """Compara porta e referência num vídeo; imprime o resultado e devolve se tudo bateu."""
    frames_dir = FRAMES_ROOT / name / "frames"
    n_disk = sum(1 for _ in frames_dir.glob("*.jpg"))
    len_shot_total = initial_len_shot(n_disk, percent)
    features = cached_features(frames_dir, FRAMES_ROOT / name / "features_resnet50_avgpool.npy", device)
    if len(features) != n_disk:
        raise ValueError(f"{name}: {len(features)} features para {n_disk} frames")

    t0 = time.time()
    best_raw_port = best_cut_number(frames_dir, delta_t)
    t_best_port = time.time() - t0
    cut_number_port = cut_number_from_best(best_raw_port, alpha)
    triples_port, leaf_port, _ = compute_hierarchy(build_time_aware_graph(features, delta_t))
    port_keyshots = sorted(group_central_frames(cut_graph(triples_port, cut_number_port), leaf_port, len_shot_total))
    t_port = time.time() - t0

    t0 = time.time()
    cached = _ref_load().get(f"{name}|{delta_t}") if use_cache else None
    if cached:
        ref, ref_src = cached, "cache"
    else:
        ref, ref_src = reference_best_cut_number(frames_dir, delta_t, alpha=alpha), "computed"
        _ref_put(name, delta_t, ref)
    t_best_ref = time.time() - t0
    triples_ref, leaf_ref = reference_compute_hierarchy(reference_graph(features, delta_t))
    cut_ref = reference_cut_graph(triples_ref, ref["cut_number_ref"])
    ref_keyshots = sorted(group_central_frames(cut_ref, leaf_ref, len_shot_total))
    actual_ref = sorted(reference_group_central_frames(cut_ref, leaf_ref, len_shot_total))
    t_ref = time.time() - t0

    cut_equal = cut_number_port == ref["cut_number_ref"] and best_raw_port == ref["best_cut_number_raw"]
    keyshots_equal = port_keyshots == ref_keyshots
    repro_equal = ref_keyshots == actual_ref

    print(f"{name} delta_t={delta_t} [ref_bestCut={ref_src}]:")
    print(
        f"  cutNumber: port raw={best_raw_port} cut={cut_number_port} [{t_best_port:.1f}s] | "
        f"ref raw={ref['best_cut_number_raw']} cut={ref['cut_number_ref']} [{t_best_ref:.1f}s] | "
        f"{'IDENTICO' if cut_equal else 'DIFERENTE'}"
    )
    print(f"  keyshots: port={len(port_keyshots)} ref={len(ref_keyshots)} | {'IDENTICO' if keyshots_equal else 'DIFERENTE'}")
    print(
        f"  reproducao vs metodo real: {'OK' if repro_equal else 'DIFERENTE'} "
        f"(repro={len(ref_keyshots)} actual={len(actual_ref)})"
    )
    if not keyshots_equal:
        ps, rs = set(port_keyshots), set(ref_keyshots)
        print(f"    so na porta: {sorted(ps - rs)}")
        print(f"    so na ref : {sorted(rs - ps)}")
    print(f"  tempos: port={t_port:.1f}s ref={t_ref:.1f}s")
    return cut_equal and keyshots_equal and repro_equal


if __name__ == "__main__":
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", default=["Jumps", "Air_Force_One"])
    parser.add_argument("--delta-t", type=int, nargs="+", default=[cfg["graph"]["delta_t"]])
    parser.add_argument("--no-cache", action="store_true", help="recalcula o bestCutNumber de referência")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    all_ok = True
    for name in args.videos:
        for delta_t in args.delta_t:
            ok = run_video(name, delta_t, cfg["hierarchy"]["alpha"], cfg["hierarchy"]["percent"], device, not args.no_cache)
            all_ok = ok and all_ok
    sys.exit(0 if all_ok else 1)
