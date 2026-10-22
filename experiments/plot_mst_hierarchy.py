"""Gera as duas figuras PNG de verificação visual da T06, com PIL (matplotlib não é dependência).

(a) docs/evidence/t06-synthetic-graph.png: grafo sintético, nós coloridos pelo cluster que o
    código atribuiu (mst_partition), com o cluster esperado (calculado à mão) marcado ao lado —
    uma divergência de cor entre o preenchimento e o marcador indicaria um erro.
(b) docs/evidence/t06-real-video-partition.png: a partição do vídeo real (docs/evidence/
    t06-real-video-partition.csv) como duas faixas de frames, implementação vs. referência,
    com os frames divergentes marcados.
"""

from __future__ import annotations

import colorsys
import csv
from pathlib import Path

from PIL import Image, ImageDraw

import networkx as nx
from tcc_gnn.hierarchy.mst import mst_partition

EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "docs" / "evidence"


def cluster_color(cluster_id: int, n_clusters: int) -> tuple[int, int, int]:
    """Cor estável por id de cluster, numa roda de matizes igualmente espaçada."""
    hue = (cluster_id % max(n_clusters, 1)) / max(n_clusters, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def plot_synthetic_graph(output_path: Path) -> None:
    g = nx.Graph()
    g.add_nodes_from(range(4))
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=3.0)
    g.add_edge(2, 3, weight=2.0)

    cut_number = 2
    raw_assigned = mst_partition(g, cutNumber=cut_number)  # saída real do código
    # Partição esperada, calculada à mão sobre a árvore watershed-by-area deste grafo (ver
    # docs/evidence/t06-mst-hierarchy.md e tests/test_mst_hierarchy.py): cutNumber=2 isola as
    # folhas 2 e 3 uma da outra.
    expected = {0: "A", 1: "A", 2: "B", 3: "C"}

    def canonicalize(partition: dict[int, int | str]) -> dict[int, int]:
        """Reindexa os clusters por ordem do menor frame, pra que a MESMA cor represente o
        mesmo agrupamento em assigned e expected quando as partições coincidem — só uma
        divergência real de agrupamento produz cores diferentes."""
        by_min_frame = sorted(
            set(partition.values()), key=lambda c: min(f for f, cl in partition.items() if cl == c)
        )
        relabel = {c: i for i, c in enumerate(by_min_frame)}
        return {f: relabel[c] for f, c in partition.items()}

    assigned = canonicalize(raw_assigned)
    expected_ids = canonicalize(expected)

    n_clusters = max(len(set(assigned.values())), len(set(expected_ids.values())))

    width, height = 760, 320
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    xs = [140 + i * 160 for i in range(4)]
    y = 140
    radius = 36

    for i in range(3):
        draw.line([(xs[i], y), (xs[i + 1], y)], fill=(60, 60, 60), width=3)

    mismatches = []
    for frame in range(4):
        x = xs[frame]
        fill = cluster_color(assigned[frame], n_clusters)
        expected_color = cluster_color(expected_ids[frame], n_clusters)
        # Compara por partição canônica (mesmo agrupamento), não por id bruto: dois clusters
        # equivalentes podem ter ids diferentes por numeração arbitrária.
        same_group_assigned = {f for f, c in assigned.items() if c == assigned[frame]}
        same_group_expected = {f for f, c in expected_ids.items() if c == expected_ids[frame]}
        is_mismatch = same_group_assigned != same_group_expected
        mismatches.append(is_mismatch)

        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=fill,
            outline=(0, 0, 0),
            width=3,
        )
        draw.text((x - 6, y - 8), str(frame), fill=(0, 0, 0))

        marker_y = y + radius + 24
        marker_color = expected_color if not is_mismatch else (220, 20, 20)
        draw.rectangle(
            [x - 14, marker_y - 14, x + 14, marker_y + 14],
            fill=marker_color,
            outline=(0, 0, 0),
            width=2 if not is_mismatch else 4,
        )
        draw.text((x - 55, marker_y + 20), f"esperado: {expected[frame]}", fill=(0, 0, 0))

    draw.text((20, 20), "Grafo sintetico: corte (cutNumber=2) do codigo vs. esperado a mao", fill=(0, 0, 0))
    draw.text((20, 44), "Circulo = cluster atribuido pelo codigo | quadrado = cluster esperado (vermelho = divergencia)", fill=(0, 0, 0))

    if any(mismatches):
        draw.text((20, height - 30), "DIVERGENCIA ENCONTRADA", fill=(220, 20, 20))
    else:
        draw.text((20, height - 30), "OK: nenhuma divergencia", fill=(20, 140, 20))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Salvo: {output_path} (divergencia: {any(mismatches)})")


def plot_real_video_partition(csv_path: Path, output_path: Path) -> None:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "ours": int(row["cluster_implementacao"]),
                    "ref": int(row["cluster_referencia"]),
                    "differs": row["difere"] == "True",
                }
            )

    n_frames = len(rows)
    n_clusters = max(max(r["ours"] for r in rows), max(r["ref"] for r in rows)) + 1

    cell_w = 6
    row_h = 60
    margin_left = 160
    margin_top = 60
    width = margin_left + n_frames * cell_w + 20
    height = margin_top + 2 * row_h + 80

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw.text((20, 15), "Particao do video real (v21, delta_t=8, cutNumber=20): implementacao vs. HieTaSumm-lib", fill=(0, 0, 0))
    draw.text((20, margin_top + row_h // 2 - 7), "Implementacao", fill=(0, 0, 0))
    draw.text((20, margin_top + row_h + row_h // 2 - 7), "Referencia", fill=(0, 0, 0))

    n_diffs = 0
    for row in rows:
        x0 = margin_left + row["frame"] * cell_w
        x1 = x0 + cell_w
        draw.rectangle(
            [x0, margin_top, x1, margin_top + row_h],
            fill=cluster_color(row["ours"], n_clusters),
        )
        draw.rectangle(
            [x0, margin_top + row_h, x1, margin_top + 2 * row_h],
            fill=cluster_color(row["ref"], n_clusters),
        )
        if row["differs"]:
            n_diffs += 1
            draw.line(
                [(x0, margin_top - 6), (x0, margin_top + 2 * row_h + 6)],
                fill=(220, 20, 20),
                width=2,
            )

    draw.rectangle(
        [margin_left, margin_top, margin_left + n_frames * cell_w, margin_top + 2 * row_h],
        outline=(0, 0, 0),
        width=2,
    )

    summary = (
        f"{n_diffs} de {n_frames} frames com cluster diferente"
        if n_diffs
        else f"0 de {n_frames} frames diferem: particoes identicas"
    )
    draw.text((20, margin_top + 2 * row_h + 20), summary, fill=(20, 140, 20) if n_diffs == 0 else (220, 20, 20))
    draw.text(
        (20, margin_top + 2 * row_h + 44),
        "Linhas vermelhas verticais marcariam frames divergentes (nenhuma neste run)."
        if n_diffs == 0
        else "Linhas vermelhas verticais marcam os frames divergentes.",
        fill=(60, 60, 60),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Salvo: {output_path} ({n_diffs} divergencias de {n_frames} frames)")


if __name__ == "__main__":
    plot_synthetic_graph(EVIDENCE_DIR / "t06-synthetic-graph.png")
    plot_real_video_partition(
        EVIDENCE_DIR / "t06-real-video-partition.csv",
        EVIDENCE_DIR / "t06-real-video-partition.png",
    )
