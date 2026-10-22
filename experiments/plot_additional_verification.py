"""Figuras da verificação adicional da T06 (docs/evidence/images/), com Pillow (matplotlib não é
dependência).

(a) t06-cut-edges-<video>.png: faixa de frames do vídeo (cutNumber=20) com os pontos de corte
    marcados, e uma nota sobre o empate de peso truncado no ponto de corte.
(b) t06-truncation-comparison-<video>.png: a partição do vídeo com os pesos originais (int())
    comparada com a dos mesmos pesos escalados por 1e6 antes do truncamento (sem empate).
(c) t06-comparison-cases.png: nº de clusters e de frames divergentes por caso, lido de
    docs/evidence/sheets/t06-comparison-cases.csv (todos os vídeos/cutNumber juntos).
"""

from __future__ import annotations

import colorsys
import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from validate_mst_hierarchy import build_graph_via_frame_direct, canonical_partition, features_path_for_video  # noqa: E402
from tcc_gnn.hierarchy.mst import mst_partition  # noqa: E402

IMAGES_DIR = _REPO_ROOT / "docs" / "evidence" / "images"
SHEETS_DIR = _REPO_ROOT / "docs" / "evidence" / "sheets"


def cluster_color(cluster_id: int, n_clusters: int) -> tuple[int, int, int]:
    hue = (cluster_id % max(n_clusters, 1)) / max(n_clusters, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def plot_cut_edges(video: str, delta_t: int, cut_number: int, cut_edges_csv: Path, output_path: Path) -> None:
    features = np.load(features_path_for_video(video))
    graph = build_graph_via_frame_direct(features, delta_t)
    n_frames = graph.number_of_nodes()

    partition = canonical_partition(mst_partition(graph, cutNumber=cut_number))
    n_clusters = len(set(partition.values()))

    cut_rows = []
    with open(cut_edges_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["video"] == video:
                cut_rows.append((int(row["node_u"]), int(row["node_v"])))

    cell_w = max(2, min(10, 1100 // n_frames))
    row_h = 60
    margin_left = 40
    margin_top = 70
    width = margin_left + n_frames * cell_w + 20
    height = margin_top + row_h + 100

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw.text((20, 15), f"{video}, delta_t={delta_t}, cutNumber={cut_number}: faixa de clusters e pontos de corte", fill=(0, 0, 0))
    draw.text((20, 36), f"{len(cut_rows)} arestas distintas cortadas (ver docs/evidence/sheets/t06-cut-edges-v21.csv, coluna video)", fill=(80, 80, 80))

    for frame in range(n_frames):
        x0 = margin_left + frame * cell_w
        x1 = x0 + cell_w
        draw.rectangle(
            [x0, margin_top, x1, margin_top + row_h],
            fill=cluster_color(partition[frame], n_clusters),
        )

    for u, v in cut_rows:
        x = margin_left + max(u, v) * cell_w
        draw.line([(x, margin_top - 6), (x, margin_top + row_h + 6)], fill=(0, 0, 0), width=2)

    draw.rectangle(
        [margin_left, margin_top, margin_left + n_frames * cell_w, margin_top + row_h],
        outline=(0, 0, 0),
        width=2,
    )
    draw.text((20, margin_top + row_h + 20), f"{n_clusters} clusters", fill=(0, 0, 0))
    draw.text((20, margin_top + row_h + 42), "Linhas pretas verticais = arestas cortadas", fill=(60, 60, 60))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Salvo: {output_path}")


def plot_truncation_comparison(video: str, delta_t: int, cut_number: int, output_path: Path) -> None:
    features = np.load(features_path_for_video(video))
    graph = build_graph_via_frame_direct(features, delta_t)
    n_frames = graph.number_of_nodes()

    scaled_graph = nx.Graph()
    scaled_graph.add_nodes_from(graph.nodes)
    for u, v, data in graph.edges(data=True):
        scaled_graph.add_edge(u, v, weight=data["weight"] * 1e6)

    partition_original = canonical_partition(mst_partition(graph, cutNumber=cut_number))
    partition_scaled = canonical_partition(mst_partition(scaled_graph, cutNumber=cut_number))

    n_clusters = max(len(set(partition_original.values())), len(set(partition_scaled.values())))

    cell_w = max(2, min(10, 1100 // n_frames))
    row_h = 55
    margin_left = 220
    margin_top = 60
    width = margin_left + n_frames * cell_w + 20
    height = margin_top + 2 * row_h + 90

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw.text((20, 15), f"{video}, cutNumber={cut_number}: particao com peso original (int()) vs. peso x1e6 (sem empate)", fill=(0, 0, 0))
    draw.text((20, margin_top + row_h // 2 - 7), "Original (int())", fill=(0, 0, 0))
    draw.text((20, margin_top + row_h + row_h // 2 - 7), "Escalado (x1e6)", fill=(0, 0, 0))

    n_diff = 0
    for frame in range(n_frames):
        x0 = margin_left + frame * cell_w
        x1 = x0 + cell_w
        draw.rectangle(
            [x0, margin_top, x1, margin_top + row_h],
            fill=cluster_color(partition_original[frame], n_clusters),
        )
        draw.rectangle(
            [x0, margin_top + row_h, x1, margin_top + 2 * row_h],
            fill=cluster_color(partition_scaled[frame], n_clusters),
        )
        if partition_original[frame] != partition_scaled[frame]:
            n_diff += 1
            draw.line(
                [(x0, margin_top - 6), (x0, margin_top + 2 * row_h + 6)],
                fill=(220, 20, 20),
                width=1,
            )

    draw.rectangle(
        [margin_left, margin_top, margin_left + n_frames * cell_w, margin_top + 2 * row_h],
        outline=(0, 0, 0),
        width=2,
    )
    draw.text(
        (20, margin_top + 2 * row_h + 20),
        f"{n_diff} de {n_frames} frames mudam de cluster quando o empate de truncamento e removido (so diagnostico, nao e criterio de aceitacao)",
        fill=(150, 0, 0) if n_diff else (20, 140, 20),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Salvo: {output_path} ({n_diff} frames diferentes)")


def plot_comparison_cases(csv_path: Path, output_path: Path) -> None:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    bar_w = 150
    gap = 30
    margin_left = 60
    margin_top = 60
    chart_h = 220
    width = margin_left + len(rows) * (bar_w + gap) + 40
    height = margin_top + chart_h + 160

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 15), "Casos adicionais: clusters e frames divergentes vs. referencia (docs/evidence/sheets/t06-comparison-cases.csv)", fill=(0, 0, 0))

    max_clusters = max(int(r["n_clusters_implementacao"]) for r in rows)
    baseline_y = margin_top + chart_h

    for i, row in enumerate(rows):
        x0 = margin_left + i * (bar_w + gap)
        n_clusters = int(row["n_clusters_implementacao"])
        bar_h = int(chart_h * n_clusters / max_clusters)
        color = (80, 140, 220) if row["particoes_identicas"] == "True" else (220, 60, 60)
        draw.rectangle([x0, baseline_y - bar_h, x0 + bar_w, baseline_y], fill=color, outline=(0, 0, 0))
        draw.text((x0, baseline_y - bar_h - 20), f"{n_clusters} clusters", fill=(0, 0, 0))
        label = f"{row['video']}\ncutNumber={row['cut_number']}\n({row['cutnumber_fonte']})"
        for j, line in enumerate(label.split("\n")):
            draw.text((x0, baseline_y + 10 + j * 16), line, fill=(0, 0, 0))
        status = "identica a referencia" if row["particoes_identicas"] == "True" else f"{row['n_frames_diferentes']} frames diferem"
        draw.text((x0, baseline_y + 64), status, fill=(20, 110, 20) if row["particoes_identicas"] == "True" else (150, 0, 0))

    draw.line([(margin_left - 10, baseline_y), (width - 20, baseline_y)], fill=(0, 0, 0), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Salvo: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="gera só as figuras por-video deste vídeo; omitido gera v21 e v29")
    parser.add_argument("--delta-t", type=int, default=8)
    parser.add_argument("--cut-number", type=int, default=20)
    args = parser.parse_args()

    videos = [args.video] if args.video else ["v21", "v29"]
    for video in videos:
        plot_cut_edges(
            video,
            args.delta_t,
            args.cut_number,
            SHEETS_DIR / "t06-cut-edges-v21.csv",
            IMAGES_DIR / f"t06-cut-edges-{video}.png",
        )
        plot_truncation_comparison(
            video, args.delta_t, args.cut_number, IMAGES_DIR / f"t06-truncation-comparison-{video}.png"
        )

    plot_comparison_cases(SHEETS_DIR / "t06-comparison-cases.csv", IMAGES_DIR / "t06-comparison-cases.png")
