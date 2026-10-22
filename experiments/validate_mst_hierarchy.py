"""Compara a hierarquia MST de hierarchy/mst.py com HieTaSumm.Graph num vídeo real (T06)."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "references" / "HieTaSumm-lib"))

_stub_names = [
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.applications",
    "tensorflow.keras.applications.vgg16",
    "tensorflow.keras.applications.resnet50",
    "tensorflow.keras.models",
    "tensorflow.keras.utils",
    "h5py",
]
_stubs = {name: types.ModuleType(name) for name in _stub_names}
_stubs["tensorflow.keras.applications.vgg16"].VGG16 = object
_stubs["tensorflow.keras.applications.vgg16"].preprocess_input = lambda x: x
_stubs["tensorflow.keras.applications.resnet50"].ResNet50 = object
_stubs["tensorflow.keras.applications.resnet50"].preprocess_input = lambda x: x
_stubs["tensorflow.keras.models"].Model = object
_stubs["h5py"].File = object
_stubs["tensorflow.keras"].utils = _stubs["tensorflow.keras.utils"]
_stubs["tensorflow.keras"].applications = _stubs["tensorflow.keras.applications"]
_stubs["tensorflow.keras"].models = _stubs["tensorflow.keras.models"]
_stubs["tensorflow"].keras = _stubs["tensorflow.keras"]
for _name, _mod in _stubs.items():
    sys.modules.setdefault(_name, _mod)
# higra NÃO é stubado: é o pacote real (D-05), precisamos dele calculando de verdade.

from HieTaSumm.Frame import Frame  # noqa: E402
from HieTaSumm.Graph import Graph  # noqa: E402
from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph, partition_from_cut  # noqa: E402


class _TripleCollector:
    """Substitui Files como `input_higra`/`cut_graph_file`: guarda os triples em memória."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, int, float | int]] = []

    def save_graph_data(self, v1, v2, weight) -> None:
        self.rows.append((v1, v2, weight))


def build_graph_via_frame_direct(features: np.ndarray, delta_t: int):
    """Constrói o grafo integer-indexed chamando Frame.calc_end/calculate_similarity direto,
    a mesma convenção estabelecida em experiments/validate_graph.py (T01), contornando o bug de
    rótulo de Frame.load documentado em docs/evidence/t04-mst-hierarchy-spec.md."""
    import networkx as nx

    frame = Frame(model=None)
    n = len(features)
    g = nx.Graph()
    g.add_nodes_from(range(n))

    for v1 in range(n):
        end = frame.calc_end(v1, delta_t, n)
        for v2 in range(v1, end):
            if v1 == v2:
                continue
            w = frame.calculate_similarity(features[v1], features[v2])
            g.add_edge(v1, v2, weight=float(w))

    return g


def reference_compute_hierarchy(graph) -> tuple[list[tuple[int, int, int]], list[int]]:
    ref_graph = Graph(is_binary=True, hierarchy="watershed_hierarchy_by_area")
    collector = _TripleCollector()
    leaf_list = ref_graph.compute_hierarchy(graph, collector)
    triples = [(v1, v2, int(w)) for v1, v2, w in collector.rows]
    return triples, leaf_list


def reference_cut_graph(triples: list[tuple[int, int, int]], cut_number: int):
    ref_graph = Graph(is_binary=True, hierarchy="watershed_hierarchy_by_area")
    with tempfile.TemporaryDirectory() as tmp_dir:
        higra_path = os.path.join(tmp_dir, "higra.txt")
        with open(higra_path, "w") as f:
            for v1, v2, w in triples:
                f.write(f"{v1}, {v2}, {w}\n")

        class _FileLike:
            file = higra_path

        collector = _TripleCollector()
        cut_result = ref_graph.cut_graph(_FileLike(), collector, cut_number)
    return cut_result


def canonical_partition(frame_to_cluster: dict[int, int]) -> dict[int, int]:
    """Renomeia os ids de cluster por ordem do menor frame, pra comparar partições sem depender
    da numeração arbitrária dos componentes conexos."""
    by_min_frame = sorted(
        set(frame_to_cluster.values()),
        key=lambda c: min(f for f, cl in frame_to_cluster.items() if cl == c),
    )
    relabel = {c: i for i, c in enumerate(by_min_frame)}
    return {f: relabel[c] for f, c in frame_to_cluster.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--delta-t", type=int, required=True)
    parser.add_argument("--cut-number", type=int, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    features = np.load(args.features)
    print(f"Features carregadas: shape {features.shape}")

    graph = build_graph_via_frame_direct(features, args.delta_t)
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
