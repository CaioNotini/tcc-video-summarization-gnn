"""Arestas cortadas pelo cutNumber e empates no peso truncado (T06, verificação adicional; método em t06-additional-verification.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from validate_mst_hierarchy import features_path_for_video, merge_csv_rows

from tcc_gnn.hierarchy.mst import compute_hierarchy, mst_partition
from tcc_gnn.reference import reference_graph

CSV_FIELDNAMES = [
    "video",
    "parent_node_severed",
    "node_u",
    "node_v",
    "weight_float",
    "weight_int",
    "higra_rank",
    "n_arestas_candidatas",
    "cluster_u",
    "cluster_v",
    "mesmo_cluster",
]


def trace_severed_nodes(triples: list[tuple[int, int, int]], cutNumber: int) -> list[int]:
    """Repete o laço top-down de `cut_graph` só para listar os nós que ele isola (`RG.add_node`)."""
    cut_list: list[int] = []
    severed: list[int] = []

    line = len(triples) - 1
    cut_list.append(triples[line][0])

    while line >= 0:
        v1, v2, w = triples[line]
        if v2 in cut_list:
            if cutNumber > 0:
                cutNumber -= 1
                cut_list.append(v1)
            else:
                severed.append(v1)
        line -= 1

    return severed


def build_children_and_leaf_sets(
    triples: list[tuple[int, int, int]], leaf_set: set[int]
) -> tuple[dict[int, frozenset[int]], dict[int, list[int]]]:
    """Mapas nó -> folhas descendentes e nó -> filhos, de baixo para cima sobre `triples` (ordem leaves_to_root)."""
    children: dict[int, list[int]] = {}
    for n, p, _leaf in triples:
        if n != p:
            children.setdefault(p, []).append(n)

    leaves_under: dict[int, frozenset[int]] = {}
    for n, _p, _leaf in triples:
        if n in leaf_set:
            leaves_under[n] = frozenset({n})
        else:
            leaves_under[n] = frozenset().union(*(leaves_under[c] for c in children[n]))

    return leaves_under, children


def analyze(features_path: Path, delta_t: int, cut_number: int, video: str, csv_out: Path) -> dict:
    """Grava no CSV a aresta de cada corte e devolve o resumo dos empates."""
    features = np.load(features_path)
    graph = reference_graph(features, delta_t)
    n_frames = graph.number_of_nodes()

    triples, leaf_list, _altitudes = compute_hierarchy(graph)
    leaf_set = set(leaf_list)
    parent_of = {n: p for n, p, _leaf in triples}
    severed = trace_severed_nodes(triples, cut_number)
    distinct_parents = sorted(set(parent_of[x] for x in severed))

    leaves_under, children = build_children_and_leaf_sets(triples, leaf_set)

    edge_weights_int: dict[tuple[int, int], float] = {}
    for u, v, data in graph.edges(data=True):
        key = (min(u, v), max(u, v))
        edge_weights_int[key] = int(data["weight"])
    edge_keys = list(edge_weights_int.keys())
    weights_array = np.array([edge_weights_int[k] for k in edge_keys])
    order = np.argsort(weights_array, kind="stable")
    rank_of_key = {edge_keys[order[i]]: i for i in range(len(order))}

    partition = mst_partition(graph, cutNumber=cut_number)

    rows = []
    for parent in distinct_parents:
        child_a, child_b = children[parent]
        leaves_a, leaves_b = leaves_under[child_a], leaves_under[child_b]
        candidates = []
        for u in leaves_a:
            for v in leaves_b:
                key = (min(u, v), max(u, v))
                if key in edge_weights_int:
                    candidates.append((rank_of_key[key], key[0], key[1]))
        candidates.sort()
        _rank, u, v = candidates[0]
        key = (u, v)

        w_float = graph.adj[u][v]["weight"]
        w_int = edge_weights_int[key]
        pos = rank_of_key[key]
        cu, cv = partition.get(u), partition.get(v)
        rows.append(
            {
                "video": video,
                "parent_node_severed": parent,
                "node_u": u,
                "node_v": v,
                "weight_float": w_float,
                "weight_int": w_int,
                "higra_rank": pos,
                "n_arestas_candidatas": len(candidates),
                "cluster_u": cu,
                "cluster_v": cv,
                "mesmo_cluster": cu == cv,
            }
        )

    merge_csv_rows(csv_out, rows, CSV_FIELDNAMES, key_fields=["video"])

    last_row = max(rows, key=lambda r: r["higra_rank"])
    w_int_last = last_row["weight_int"]
    edges_same_weight = int(np.sum(weights_array == w_int_last))
    cut_same_weight = sum(1 for r in rows if r["weight_int"] == w_int_last)
    stable_order_is_identity = bool(np.array_equal(order, np.arange(len(order))))
    edges_truncating_to_zero = int(np.sum(weights_array == 0))

    return {
        "video": video,
        "n_frames": n_frames,
        "n_edges": len(edge_keys),
        "edges_truncating_to_zero": edges_truncating_to_zero,
        "n_severed_nodes": len(severed),
        "n_distinct_cut_edges": len(rows),
        "n_clusters": len(set(partition.values())),
        "any_cu_eq_cv": any(r["mesmo_cluster"] for r in rows),
        "last_cut_weight_int": w_int_last,
        "last_cut_rank": last_row["higra_rank"],
        "edges_with_last_cut_weight": edges_same_weight,
        "of_those_cut": cut_same_weight,
        "stable_order_is_identity": stable_order_is_identity,
        "csv_out": str(csv_out),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="nome do vídeo em data/processed/vsumm/ (ex.: v21, v29)")
    parser.add_argument("--delta-t", type=int, required=True)
    parser.add_argument("--cut-number", type=int, required=True)
    parser.add_argument(
        "--csv-out",
        type=Path,
        required=True,
        help="mesmo arquivo pra todos os vídeos (coluna 'video' distingue as linhas)",
    )
    args = parser.parse_args()

    features = features_path_for_video(args.video)
    result = analyze(features, args.delta_t, args.cut_number, args.video, args.csv_out)
    for key, value in result.items():
        print(f"{key}: {value}")

    if result["n_distinct_cut_edges"] != args.cut_number:
        print(
            f"\nATENCAO: {result['n_distinct_cut_edges']} arestas distintas cortadas, "
            f"diferente de cutNumber={args.cut_number} "
            f"({result['n_severed_nodes']} nos da arvore severados; uma aresta pode severar "
            "os dois filhos do mesmo no de uma vez, reduzindo o numero de arestas distintas)."
        )

    if result["edges_with_last_cut_weight"] > result["of_those_cut"]:
        print(
            f"\nEMPATE NO PONTO DE CORTE: {result['edges_with_last_cut_weight']} arestas do grafo "
            f"tem o peso truncado {result['last_cut_weight_int']} (o mesmo do ultimo corte), mas "
            f"so {result['of_those_cut']} foram cortadas. A particao depende da ordem de "
            "desempate (ordem de insercao das arestas), nao so do peso."
        )
