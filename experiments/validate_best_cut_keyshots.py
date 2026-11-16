"""Valida a equivalência de ponta a ponta (keyshots) entre a porta (best_cut.py + mst.py) e a
referência HieTaSumm-lib, em vídeos reais do SumMe, para um delta_t.

Para cada vídeo: a porta (best_cut_number -> cut_number_from_best -> build_time_aware_graph ->
compute_hierarchy -> cut_graph -> reprodução de group_central_frames) e a referência
(reference_best_cut_number -> build_graph_via_frame_direct -> reference_compute_hierarchy ->
reference_cut_graph -> mesma reprodução de group_central_frames); compara as listas de keyshots.
Confere ainda que a reprodução de group_central_frames bate com o método real
Summary.group_central_frames (submódulo) no lado da referência. O bestCutNumber de referência é
cacheado em .cache/reference_best_cut.json por (vídeo, delta_t) para reuso. Qualquer diferença
encerra com código 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from measure_q10_budget import extract_features_batched  # noqa: E402  (extrator canônico)
from validate_mst_hierarchy import (  # noqa: E402
    build_graph_via_frame_direct,
    reference_best_cut_number,
    reference_compute_hierarchy,
    reference_cut_graph,
)
from tcc_gnn.graph.frame import build_time_aware_graph  # noqa: E402
from tcc_gnn.hierarchy.best_cut import best_cut_number, cut_number_from_best  # noqa: E402
from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph  # noqa: E402

PERCENT = 15.0
ALPHA = 75
FRAMES_ROOT = _REPO_ROOT / "data" / "processed" / "summe"
_CACHE = _REPO_ROOT / ".cache" / "reference_best_cut.json"


def _ref_load() -> dict:
    if _CACHE.exists():
        with open(_CACHE) as f:
            return json.load(f)
    return {}


def _ref_get(name: str, delta_t: int) -> dict | None:
    return _ref_load().get(f"{name}|{delta_t}")


def _ref_put(name: str, delta_t: int, raw: float, cut: int) -> None:
    data = _ref_load()
    data[f"{name}|{delta_t}"] = {"best_cut_number_raw": raw, "cut_number_ref": cut}
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE, "w") as f:
        json.dump(data, f, indent=2)


def collect_keyshots(cut_graph_result: nx.Graph, leaf_list: list[int], len_shot_total: int) -> list[int]:
    """Reprodução fiel de Summary.group_central_frames (Summary.py:167-192): retorna a lista
    ordenada dos índices de frame keyshot, sem copiar arquivos. A divisão de len_shot por |S|
    é feita uma vez antes do loop, como no original (Summary.py:170)."""
    leaf_set = set(leaf_list)
    components = list(nx.connected_components(cut_graph_result))
    len_shot = int(len_shot_total / len(components))
    keyshots: list[int] = []
    for component in components:
        sub = cut_graph_result.subgraph(component).copy()
        comp_leaf_list = [node for node in list(sub) if node in leaf_set]
        len_leaf_list = len(comp_leaf_list)
        cn = int(len_leaf_list / 2)
        if cn == 0:
            continue
        if len_leaf_list < len_shot:
            init_keyshots, end_keyshots = 0, len_leaf_list - 1
        else:
            init_keyshots = cn - int(len_shot / 2)
            end_keyshots = cn + int(len_shot / 2)
        for k in range(init_keyshots, end_keyshots):
            keyshots.append(comp_leaf_list[k])
    return sorted(keyshots)


def actual_reference_keyshots(
    cut_ref: nx.Graph, leaf_ref: list[int], len_shot_total: int, frames_dir: Path
) -> list[int]:
    """Roda o método real Summary.group_central_frames (submódulo) sobre o corte de referência,
    coletando os keyshots gravados via Files.save_graph_data. os.system (cp do resumo em jpg) é
    inibido para não copiar arquivos nem gerar ruído no Windows. Retorna a lista ordenada de
    índices de frame."""
    sys.path.insert(0, str(_REPO_ROOT / "references" / "HieTaSumm-lib"))
    from HieTaSumm.Files import Files  # noqa: E402
    from HieTaSumm.Summary import Summary  # noqa: E402

    parent = str(frames_dir.parent) + os.sep
    real_os_system = os.system
    os.system = lambda *a, **k: 0  # inibe os cp (resumo em jpg)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            summ_input_file = str(Path(tmp) / "summ_input.txt")
            s = Summary.__new__(Summary)
            s.len_shot = len_shot_total
            s.summ_path = str(Path(tmp) / "summ") + os.sep
            Path(s.summ_path).mkdir(parents=True, exist_ok=True)
            s.summ_input = Files(summ_input_file)
            s.video_file = parent
            s.group_central_frames(cut_ref, leaf_ref)
            frames: list[int] = []
            with open(summ_input_file) as f:
                for line in f:
                    line = line.strip()
                    if line.endswith(".jpg"):
                        frames.append(int(line[:-4]))
    finally:
        os.system = real_os_system
    return sorted(frames)


def run_video(name: str, delta_t: int, device: torch.device) -> bool:
    frames_dir = FRAMES_ROOT / name / "frames"
    features_path = FRAMES_ROOT / name / "features_resnet50_avgpool.npy"
    n_disk = sum(1 for _ in frames_dir.glob("*.jpg"))
    len_shot_total = round(n_disk * (PERCENT / 100)) - 1

    if features_path.exists():
        features = np.load(features_path)
    else:
        features = extract_features_batched(frames_dir, device)
        np.save(features_path, features)
    if len(features) != n_disk:
        raise ValueError(f"{name}: {len(features)} features para {n_disk} frames")

    # ---- PORTA ----
    t0 = time.time()
    best_raw_port = best_cut_number(frames_dir, delta_t)
    cut_number_port = cut_number_from_best(best_raw_port, ALPHA)
    graph_port = build_time_aware_graph(features, delta_t)
    triples_port, leaf_port, _ = compute_hierarchy(graph_port)
    cut_port = cut_graph(triples_port, cut_number_port)
    port_keyshots = collect_keyshots(cut_port, leaf_port, len_shot_total)
    t_port = time.time() - t0

    # ---- REFERÊNCIA ----
    t0 = time.time()
    cached = _ref_get(name, delta_t)
    if cached:
        best_raw_ref = cached["best_cut_number_raw"]
        cut_number_ref = cached["cut_number_ref"]
        ref_src = "cache"
    else:
        ref = reference_best_cut_number(frames_dir, delta_t, alpha=ALPHA)
        best_raw_ref = ref["best_cut_number_raw"]
        cut_number_ref = ref["cut_number_ref"]
        _ref_put(name, delta_t, best_raw_ref, cut_number_ref)
        ref_src = "computed"
    graph_ref = build_graph_via_frame_direct(features, delta_t)
    triples_ref, leaf_ref = reference_compute_hierarchy(graph_ref)
    cut_ref = reference_cut_graph(triples_ref, cut_number_ref)
    ref_keyshots = collect_keyshots(cut_ref, leaf_ref, len_shot_total)
    actual_ref = actual_reference_keyshots(cut_ref, leaf_ref, len_shot_total, frames_dir)
    t_ref = time.time() - t0

    cut_equal = cut_number_port == cut_number_ref and best_raw_port == best_raw_ref
    keyshots_equal = port_keyshots == ref_keyshots
    repro_equal = ref_keyshots == actual_ref

    print(f"{name} delta_t={delta_t} [ref_bestCut={ref_src}]:")
    print(
        f"  cutNumber: port raw={best_raw_port} cut={cut_number_port} | "
        f"ref raw={best_raw_ref} cut={cut_number_ref} | "
        f"{'IDENTICO' if cut_equal else 'DIFERENTE'}"
    )
    print(
        f"  keyshots: port={len(port_keyshots)} ref={len(ref_keyshots)} | "
        f"{'IDENTICO' if keyshots_equal else 'DIFERENTE'}"
    )
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", default=["Jumps", "Air_Force_One"])
    parser.add_argument("--delta-t", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    all_ok = True
    for name in args.videos:
        all_ok = run_video(name, args.delta_t, device) and all_ok
    sys.exit(0 if all_ok else 1)
