"""Porta vetorizada de `Summary.bestCutNumber` (HieTaSumm-lib), a estatística de corte da baseline (D-02).

Reimplementação de desempenho, não de algoritmo: mesma meia-resolução do `rgbSim`
(`Summary.py:127-133`), mesmo cosseno float64 na fórmula do scipy (`uv / sqrt(uu * vv)`), mesmo
clamp em 20 do `spatialSim` (`Summary.py:135-139`), mesma janela `calc_end` (`Summary.py:256-260`),
mesma ordem dos pares no `np.std` e o mesmo `np.round(cut) - 1` (`Summary.py:104-125`). A única
divergência intencional: desvio-padrão zero levanta `ValueError` em vez do loop infinito
`while cut <= 1` do original. Validada por equivalência em experiments/validate_best_cut.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_BLOCK_ROWS = 64


def _rgb_features_uint8(frames_path: Path) -> np.ndarray:
    """Carrega os frames jpg como no `rgbSim`: meia resolução, achatados; mantém uint8 (o /255
    para float64 é aplicado por blocos em `best_cut_number`, elemento a elemento, sem mudar valor)."""
    frame_list = sorted(p.name for p in frames_path.iterdir())
    rows: list[np.ndarray] = []
    for frames in frame_list:
        if frames.endswith("jpg"):
            frame = Image.open(frames_path / frames)
            frame_reshape = frame.resize(
                (round(frame.size[0] * 0.5), round(frame.size[1] * 0.5))
            )
            rows.append(np.asarray(frame_reshape).reshape(-1))
    if not rows:
        raise ValueError(f"Nenhum frame .jpg em {frames_path}")
    return np.stack(rows)


def best_cut_number(frames_path: Path, delta_t: int) -> float:
    """Equivalente a `Summary.bestCutNumber` para uma pasta de frames e um `delta_t` positivo.

    `delta_t <= 0` levanta `ValueError`: no original, `delta_t < 0` cai no ramo janela-completa
    de `calc_end`, que o pipeline nunca usa e esta porta não reproduz.
    """
    if delta_t <= 0:
        raise ValueError(f"delta_t deve ser positivo, veio {delta_t}")
    pixels = _rgb_features_uint8(frames_path)
    n = len(pixels)

    norms_sq = np.empty(n, dtype=np.float64)
    for start in range(0, n, _BLOCK_ROWS):
        block = pixels[start : start + _BLOCK_ROWS].astype(np.float64) / 255
        norms_sq[start : start + _BLOCK_ROWS] = np.einsum("ij,ij->i", block, block)

    # weight_matrix[v1, d] = peso do par (v1, v1 + d); NaN onde calc_end exclui o par.
    weight_matrix = np.full((n, delta_t), np.nan, dtype=np.float64)
    for start in range(0, n, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, n)
        window = pixels[start : min(stop + delta_t - 1, n)].astype(np.float64) / 255
        for d in range(delta_t):
            last_v1 = min(stop, n - d)
            if last_v1 <= start:
                break
            a = window[: last_v1 - start]
            b = window[d : d + (last_v1 - start)]
            uv = np.einsum("ij,ij->i", a, b)
            dist = 1.0 - uv / np.sqrt(norms_sq[start:last_v1] * norms_sq[start + d : last_v1 + d])
            # scipy.spatial.distance.cosine recorta a distância para [0, 2]; replicado aqui.
            sim = 100 * (1.0 - np.clip(dist, 0.0, 2.0))
            weight_matrix[start:last_v1, d] = np.maximum(sim, 20.0) / 100

    weight_list = weight_matrix.reshape(-1)
    weight_list = weight_list[~np.isnan(weight_list)]

    cut = np.std(weight_list)
    if cut <= 0:
        raise ValueError(
            "Desvio-padrão zero dos pesos: o `while cut <= 1` do original não terminaria"
        )
    while cut <= 1:
        cut = cut * 10
    return float(np.round(cut) - 1)


def cut_number_from_best(best_cut_number_raw: float, alpha: int = 75) -> int:
    """`cutNumber = int(bestCutNumber * alpha/100)`, com piso 3 (`Summary.py:57-59`)."""
    cut_number = int(best_cut_number_raw * (alpha / 100))
    if cut_number <= 2:
        cut_number = 3
    return cut_number
