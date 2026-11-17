"""Compara a hierarquia MST de hierarchy/mst.py com HieTaSumm.Graph num vídeo real (T06)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph, partition_from_cut
from tcc_gnn.reference import (
    canonical_partition,
    reference_compute_hierarchy,
    reference_cut_graph,
    reference_graph,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def features_path_for_video(video: str) -> Path:
    """Caminho de `features_teste.npy` de um vídeo do VSUMM (v21, v29)."""
    return _REPO_ROOT / "data" / "processed" / "vsumm" / video / "features_teste.npy"


def segment_sizes(partition: dict[int, int]) -> list[int]:
    """Tamanho de cada cluster (nº de frames), do maior para o menor."""
    sizes: dict[int, int] = {}
    for cluster in partition.values():
        sizes[cluster] = sizes.get(cluster, 0) + 1
    return sorted(sizes.values(), reverse=True)


def merge_csv_rows(
    csv_path: Path,
    new_rows: list[dict],
    fieldnames: list[str],
    key_fields: list[str],
) -> None:
    """Grava `new_rows` em `csv_path`, trocando só as linhas antigas com as mesmas `key_fields` (reexecução idempotente)."""
    new_keys = {tuple(str(row[k]) for k in key_fields) for row in new_rows}

    existing_rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)

    kept_rows = [
        row for row in existing_rows if tuple(str(row.get(k, "")) for k in key_fields) not in new_keys
    ]
    combined = kept_rows + new_rows

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--delta-t", type=int, required=True)
    parser.add_argument("--cut-number", type=int, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    features = np.load(args.features)
    print(f"Features carregadas: shape {features.shape}")

    graph = reference_graph(features, args.delta_t)
    n_frames = graph.number_of_nodes()
    print(f"Grafo construido (Frame.calc_end/calculate_similarity diretos): {n_frames} frames")

    triples_ours, leaf_list_ours, _altitudes_ours = compute_hierarchy(graph)
    triples_ref, leaf_list_ref = reference_compute_hierarchy(graph)

    tree_matches = triples_ours == triples_ref and leaf_list_ours == leaf_list_ref
    print(f"Arvore (triples + leaf_list) identica a referencia: {tree_matches}")
    if not tree_matches:
        print(f"  len(triples) nosso={len(triples_ours)} referencia={len(triples_ref)}")
        print(f"  leaf_list igual: {leaf_list_ours == leaf_list_ref}")

    cut_ours = cut_graph(triples_ours, args.cut_number)
    partition_ours = canonical_partition(partition_from_cut(cut_ours, leaf_list_ours))

    cut_ref = reference_cut_graph(triples_ref, args.cut_number)
    partition_ref = canonical_partition(partition_from_cut(cut_ref, leaf_list_ref))

    partition_matches = partition_ours == partition_ref
    n_clusters_ours = len(set(partition_ours.values()))
    n_clusters_ref = len(set(partition_ref.values()))
    print(f"cutNumber={args.cut_number} (escolhido a mao, nao via bestCutNumber/Summary.py - fora do escopo de T06)")
    print(f"Particao identica a referencia: {partition_matches}")
    print(f"Clusters: nosso={n_clusters_ours} referencia={n_clusters_ref}")

    diffs = [f for f in range(n_frames) if partition_ours.get(f) != partition_ref.get(f)]
    print(f"Frames com cluster diferente: {len(diffs)}")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "cluster_implementacao", "cluster_referencia", "difere"])
        for frame in range(n_frames):
            c_ours = partition_ours.get(frame)
            c_ref = partition_ref.get(frame)
            writer.writerow([frame, c_ours, c_ref, c_ours != c_ref])
    print(f"CSV salvo em {args.csv_out}")

    if tree_matches and partition_matches:
        print("\nOK: arvore e particao equivalentes a referencia.")
    else:
        print("\nFALHOU: arvore ou particao nao batem com a referencia.")
