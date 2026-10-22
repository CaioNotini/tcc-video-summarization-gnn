"""Compara implementação vs. HieTaSumm-lib para vários (vídeo, cutNumber) (T06, verificação adicional).

Critério de aceitação da baseline é sempre a comparação com int() (D-02): a decisão do pesquisador
é que o truncamento faz parte da baseline e deve ser mantido em toda comparação com a referência.
A comparação com pesos escalados (x1e6, sem empate de truncamento) é só diagnóstico de empate,
reportada nas mesmas linhas mas não usada como critério de aceitação.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from validate_mst_hierarchy import (  # noqa: E402
    build_graph_via_frame_direct,
    canonical_partition,
    features_path_for_video,
    merge_csv_rows,
    reference_compute_hierarchy,
    reference_cut_graph,
    segment_sizes,
)
from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph, mst_partition, partition_from_cut  # noqa: E402

CSV_FIELDNAMES = [
    "video",
    "delta_t",
    "cut_number",
    "cutnumber_fonte",
    "n_frames",
    "tree_matches",
    "n_frames_diferentes",
    "n_clusters_implementacao",
    "n_clusters_referencia",
    "particoes_identicas",
    "tamanhos_segmentos",
    "frames_diferentes_escalado_vs_referencia_escalada",
    "frames_diferentes_escalado_vs_int",
]


def _scaled(graph: nx.Graph, scale: float) -> nx.Graph:
    scaled = nx.Graph()
    scaled.add_nodes_from(graph.nodes)
    for u, v, data in graph.edges(data=True):
        scaled.add_edge(u, v, weight=data["weight"] * scale)
    return scaled


def run_case(features_path: Path, delta_t: int, cut_number: int, cutnumber_fonte: str = "mao") -> dict:
    features = np.load(features_path)
    graph = build_graph_via_frame_direct(features, delta_t)
    n_frames = graph.number_of_nodes()

    # Critério de aceitação: comparação com int() (D-02, decisão do pesquisador).
    triples_ours, leaf_list_ours, _altitudes = compute_hierarchy(graph)
    triples_ref, leaf_list_ref = reference_compute_hierarchy(graph)
    tree_matches = triples_ours == triples_ref and leaf_list_ours == leaf_list_ref

    cut_ours = cut_graph(triples_ours, cut_number)
    partition_ours = canonical_partition(partition_from_cut(cut_ours, leaf_list_ours))

    cut_ref = reference_cut_graph(triples_ref, cut_number)
    partition_ref = canonical_partition(partition_from_cut(cut_ref, leaf_list_ref))

    n_diff = sum(1 for f in range(n_frames) if partition_ours.get(f) != partition_ref.get(f))

    # Diagnóstico de empate (não é critério de aceitação): pesos x1e6, sem truncamento degenerado.
    scaled_graph = _scaled(graph, scale=1e6)
    triples_scaled_ours, leaf_list_scaled_ours, _alt = compute_hierarchy(scaled_graph)
    triples_scaled_ref, leaf_list_scaled_ref = reference_compute_hierarchy(scaled_graph)

    cut_scaled_ours = cut_graph(triples_scaled_ours, cut_number)
    partition_scaled_ours = canonical_partition(partition_from_cut(cut_scaled_ours, leaf_list_scaled_ours))

    cut_scaled_ref = reference_cut_graph(triples_scaled_ref, cut_number)
    partition_scaled_ref = canonical_partition(partition_from_cut(cut_scaled_ref, leaf_list_scaled_ref))

    n_diff_scaled_vs_ref_scaled = sum(
        1 for f in range(n_frames) if partition_scaled_ours.get(f) != partition_scaled_ref.get(f)
    )
    n_diff_scaled_vs_int = sum(
        1 for f in range(n_frames) if partition_scaled_ours.get(f) != partition_ours.get(f)
    )

    return {
        "video": features_path.parent.name,
        "delta_t": delta_t,
        "cut_number": cut_number,
        "cutnumber_fonte": cutnumber_fonte,
        "n_frames": n_frames,
        "tree_matches": tree_matches,
        "n_frames_diferentes": n_diff,
        "n_clusters_implementacao": len(set(partition_ours.values())),
        "n_clusters_referencia": len(set(partition_ref.values())),
        "particoes_identicas": n_diff == 0,
        "tamanhos_segmentos": ";".join(str(s) for s in segment_sizes(partition_ours)),
        "frames_diferentes_escalado_vs_referencia_escalada": n_diff_scaled_vs_ref_scaled,
        "frames_diferentes_escalado_vs_int": n_diff_scaled_vs_int,
    }


DEFAULT_CASES = [
    ("v21", 8, 5, "mao"),
    ("v21", 8, 50, "mao"),
    ("v29", 8, 20, "mao"),
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="roda só este vídeo (ex.: v29); omitido roda a bateria padrão")
    parser.add_argument("--delta-t", type=int, default=8)
    parser.add_argument("--cut-number", type=int)
    parser.add_argument("--cutnumber-fonte", default="mao", help="rótulo da origem do cutNumber (ex.: mao, bestCutNumber(alpha=75))")
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=_REPO_ROOT / "docs" / "evidence" / "sheets" / "t06-comparison-cases.csv",
    )
    args = parser.parse_args()

    if args.video:
        if args.cut_number is None:
            parser.error("--cut-number é obrigatório junto com --video")
        cases = [(args.video, args.delta_t, args.cut_number, args.cutnumber_fonte)]
    else:
        cases = DEFAULT_CASES

    results = [
        run_case(features_path_for_video(video), delta_t, cut_number, cutnumber_fonte)
        for video, delta_t, cut_number, cutnumber_fonte in cases
    ]

    for r in results:
        print(r)

    merge_csv_rows(args.csv_out, results, CSV_FIELDNAMES, key_fields=["video", "cut_number", "cutnumber_fonte"])
    print(f"\nCSV salvo em {args.csv_out}")
