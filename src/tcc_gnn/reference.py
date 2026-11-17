"""Acesso ao submódulo HieTaSumm-lib, só para validação: stubs do tensorflow e chamadas à referência."""

from __future__ import annotations

import os
import sys
import tempfile
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import networkx as nx
import numpy as np

SUBMODULE = Path(__file__).resolve().parents[2] / "references" / "HieTaSumm-lib"

_TF_MODULES = [
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.applications",
    "tensorflow.keras.applications.vgg16",
    "tensorflow.keras.applications.resnet50",
    "tensorflow.keras.models",
    "tensorflow.keras.utils",
]


def _tensorflow_stubs() -> dict[str, types.ModuleType]:
    """Módulos vazios com os nomes que `HieTaSumm/Models.py` importa do tensorflow."""
    stubs = {name: types.ModuleType(name) for name in _TF_MODULES}
    stubs["tensorflow.keras.applications.vgg16"].VGG16 = object
    stubs["tensorflow.keras.applications.vgg16"].preprocess_input = lambda x: x
    stubs["tensorflow.keras.applications.resnet50"].ResNet50 = object
    stubs["tensorflow.keras.applications.resnet50"].preprocess_input = lambda x: x
    stubs["tensorflow.keras.models"].Model = object
    stubs["tensorflow.keras"].utils = stubs["tensorflow.keras.utils"]
    stubs["tensorflow.keras"].applications = stubs["tensorflow.keras.applications"]
    stubs["tensorflow.keras"].models = stubs["tensorflow.keras.models"]
    stubs["tensorflow"].keras = stubs["tensorflow.keras"]
    return stubs


@contextmanager
def hietasumm_reference() -> Iterator[types.SimpleNamespace]:
    """Importa o pacote HieTaSumm com stubs do tensorflow; ao sair, remove os stubs e o submódulo do sys.path."""
    if not (SUBMODULE / "HieTaSumm" / "__init__.py").exists():
        raise FileNotFoundError(f"Submódulo HieTaSumm-lib não baixado: {SUBMODULE}")
    stubs = _tensorflow_stubs()
    added = [name for name in stubs if name not in sys.modules]
    for name in added:
        sys.modules[name] = stubs[name]
    path_added = str(SUBMODULE) not in sys.path
    if path_added:
        sys.path.insert(0, str(SUBMODULE))
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # não gravar __pycache__ no submódulo somente leitura
    try:
        import HieTaSumm
        from HieTaSumm.Evaluation import Evaluation
        from HieTaSumm.Files import Files
        from HieTaSumm.Frame import Frame
        from HieTaSumm.Graph import Graph
        from HieTaSumm.Summary import Summary

        yield types.SimpleNamespace(
            HieTaSumm=HieTaSumm.HieTaSumm,
            Evaluation=Evaluation,
            Files=Files,
            Frame=Frame,
            Graph=Graph,
            Summary=Summary,
        )
    finally:
        for name in added:
            sys.modules.pop(name, None)
        if path_added:
            sys.path.remove(str(SUBMODULE))
        sys.dont_write_bytecode = previous_bytecode


class _TripleCollector:
    """Substitui `Files` como `input_higra`/`cut_graph_file`: guarda os triples em memória."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, int, float | int]] = []

    def save_graph_data(self, v1, v2, weight) -> None:
        self.rows.append((v1, v2, weight))


def reference_graph(features: np.ndarray, delta_t: int) -> nx.Graph:
    """Grafo de índices inteiros via `Frame.calc_end`/`calculate_similarity` (contorna o rótulo de `Frame.load`, T04)."""
    with hietasumm_reference() as ref:
        frame = ref.Frame(model=None)
        n = len(features)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for v1 in range(n):
            end = frame.calc_end(v1, delta_t, n)
            for v2 in range(v1, end):
                if v1 == v2:
                    continue
                graph.add_edge(v1, v2, weight=float(frame.calculate_similarity(features[v1], features[v2])))
    return graph


def reference_compute_hierarchy(graph: nx.Graph) -> tuple[list[tuple[int, int, int]], list[int]]:
    """`Graph.compute_hierarchy` do submódulo: devolve `(triples, leaf_list)`."""
    with hietasumm_reference() as ref:
        ref_graph = ref.Graph(is_binary=True, hierarchy="watershed_hierarchy_by_area")
        collector = _TripleCollector()
        leaf_list = ref_graph.compute_hierarchy(graph, collector)
    return [(v1, v2, int(w)) for v1, v2, w in collector.rows], leaf_list


def reference_cut_graph(triples: list[tuple[int, int, int]], cut_number: int) -> nx.Graph:
    """`Graph.cut_graph` do submódulo, lendo os triples de um arquivo temporário como o original."""
    with hietasumm_reference() as ref, tempfile.TemporaryDirectory() as tmp_dir:
        higra_path = os.path.join(tmp_dir, "higra.txt")
        with open(higra_path, "w") as f:
            for v1, v2, w in triples:
                f.write(f"{v1}, {v2}, {w}\n")

        class _FileLike:
            file = higra_path

        ref_graph = ref.Graph(is_binary=True, hierarchy="watershed_hierarchy_by_area")
        return ref_graph.cut_graph(_FileLike(), _TripleCollector(), cut_number)


def reference_best_cut_number(frames_dir: Path, delta_t: int, alpha: int = 75) -> dict:
    """`Summary.bestCutNumber` sem o `__init__` do Summary, e o `cutNumber` com piso 3 de `Summary.py:57-59`."""
    with hietasumm_reference() as ref:
        summary = ref.Summary.__new__(ref.Summary)
        summary.frames_path = str(frames_dir) + os.sep
        summary.delta_t = delta_t
        best_cut_number_raw = float(summary.bestCutNumber())
    cut_number_ref = int(best_cut_number_raw * (alpha / 100))
    if cut_number_ref <= 2:
        cut_number_ref = 3
    return {"best_cut_number_raw": best_cut_number_raw, "alpha": alpha, "cut_number_ref": cut_number_ref}


def reference_group_central_frames(graph: nx.Graph, leaflist: list[int], len_shot: int) -> list[int]:
    """`Summary.group_central_frames` do submódulo, com o `cp` desligado; devolve os keyshots na ordem gravada."""
    real_os_system = os.system
    os.system = lambda *a, **k: 0
    try:
        with hietasumm_reference() as ref, tempfile.TemporaryDirectory() as tmp:
            summary = ref.Summary.__new__(ref.Summary)
            summary.len_shot = len_shot
            summary.summ_path = str(Path(tmp) / "summ") + os.sep
            summary.summ_input = ref.Files(str(Path(tmp) / "summ_input.txt"))
            summary.video_file = tmp + os.sep
            summary.group_central_frames(graph, leaflist)
            path = Path(summary.summ_input.file)
            lines = path.read_text().split() if path.exists() else []
    finally:
        os.system = real_os_system
    return [int(line[:-4]) for line in lines]


def reference_evaluation():
    """Instância de `Evaluation` do submódulo, sem `__init__` (só os métodos)."""
    with hietasumm_reference() as ref:
        return ref.Evaluation.__new__(ref.Evaluation)


def reference_frame_extractor(video: Path, rate: int, frames_root: Path) -> list[Path]:
    """`HieTaSumm.frame_extractor` do submódulo; devolve os jpgs gravados em `frames_root/<vídeo>/frames`."""
    with hietasumm_reference() as ref:
        extractor = ref.HieTaSumm.__new__(ref.HieTaSumm)
        extractor.frame_extractor(video.as_posix(), rate, frames_root.as_posix())
    return sorted((frames_root / video.stem / "frames").glob("*.jpg"))


def canonical_partition(frame_to_cluster: dict[int, int]) -> dict[int, int]:
    """Renumera os clusters pela ordem do menor frame, para comparar partições sem depender da numeração dos componentes."""
    by_min_frame = sorted(
        set(frame_to_cluster.values()),
        key=lambda c: min(f for f, cl in frame_to_cluster.items() if cl == c),
    )
    relabel = {c: i for i, c in enumerate(by_min_frame)}
    return {f: relabel[c] for f, c in frame_to_cluster.items()}
