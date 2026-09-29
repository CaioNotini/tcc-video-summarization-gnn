"""Compara o grafo do frame.py com a saída real do Frame.py do Leonardo."""

from __future__ import annotations

import argparse
import sys
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
    "higra",
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

from HieTaSumm.Frame import Frame  # noqa: E402
from tcc_gnn.graph.frame import build_time_aware_graph  # noqa: E402


def build_leonardo_reference_graph(
    features: np.ndarray, delta_t: int
) -> dict[tuple[int, int], float]:
    frame = Frame(model=None)
    n = len(features)
    edges: dict[tuple[int, int], float] = {}

    for v1 in range(n):
        end = frame.calc_end(v1, delta_t, n)
        for v2 in range(v1, end):
            if v1 == v2:
                continue
            w = frame.calculate_similarity(features[v1], features[v2])
            edges[(v1, v2)] = float(w)

    return edges


def compare(leonardo_edges: dict[tuple[int, int], float], ours_graph) -> None:
    ours_edges = {
        (min(u, v), max(u, v)): data["weight"]
        for u, v, data in ours_graph.edges(data=True)
    }

    only_leonardo = set(leonardo_edges) - set(ours_edges)
    only_ours = set(ours_edges) - set(leonardo_edges)
    common = set(leonardo_edges) & set(ours_edges)

    max_weight_diff = max(
        (abs(leonardo_edges[k] - ours_edges[k]) for k in common), default=0.0
    )

    print(f"Arestas só na referência (Leonardo): {len(only_leonardo)}")
    print(f"Arestas só no nosso grafo:           {len(only_ours)}")
    print(f"Arestas em comum:                    {len(common)}")
    print(f"Maior diferença de peso (comuns):    {max_weight_diff:.10f}")

    if only_leonardo or only_ours:
        print("\nFALHOU: os conjuntos de arestas não batem.")
    elif max_weight_diff > 1e-9:
        print("\nFALHOU: pesos não batem numericamente.")
    else:
        print("\nOK: grafos equivalentes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--delta-t", type=int, required=True)
    args = parser.parse_args()

    features = np.load(args.features)
    print(f"Features carregadas: shape {features.shape}")

    leonardo_edges = build_leonardo_reference_graph(features, args.delta_t)
    ours_graph = build_time_aware_graph(features, args.delta_t)

    compare(leonardo_edges, ours_graph)