"""Porta vetorizada de `Summary.bestCutNumber` (D-02); só diverge em `delta_t <= 0`, frame de norma zero e std = 0, que levantam `ValueError`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_BLOCK_ROWS = 64


def _rgb_features_uint8(frames_path: Path) -> np.ndarray:
    """Frames jpg em meia resolução e achatados, como no `rgbSim` (`Summary.py:127-133`), ainda em uint8."""
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
    """`Summary.bestCutNumber` (`Summary.py:104-125`) para uma pasta de frames; as três divergências levantam `ValueError`."""
    if delta_t <= 0:  # divergência 1: no original, delta_t < 0 cai no ramo janela-completa de calc_end
        raise ValueError(f"delta_t deve ser positivo, veio {delta_t}")
    pixels = _rgb_features_uint8(frames_path)
    n = len(pixels)

    norms_sq = np.empty(n, dtype=np.float64)
    for start in range(0, n, _BLOCK_ROWS):
        block = pixels[start : start + _BLOCK_ROWS].astype(np.float64) / 255
        norms_sq[start : start + _BLOCK_ROWS] = np.einsum("ij,ij->i", block, block)
    if (norms_sq == 0).any():  # divergência 3: no original, o cosseno de um frame preto é NaN e o int(nan) do cutNumber falha
        raise ValueError(f"Frame todo preto (norma zero) em {frames_path}: índices {np.flatnonzero(norms_sq == 0).tolist()[:5]}")

    # weight_matrix[v1, d] = peso do par (v1, v1 + d); NaN só onde calc_end exclui o par (não há norma zero).
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
    if cut <= 0:  # divergência 2: no original, std = 0 deixa o `while cut <= 1` em loop infinito
        raise ValueError(
            "Desvio-padrão zero dos pesos: o `while cut <= 1` do original não terminaria"
        )
    while cut <= 1:
        cut = cut * 10
    return float(np.round(cut) - 1)


def cut_number_from_best(best_cut_number_raw: float, alpha: int) -> int:
    """`cutNumber = int(bestCutNumber * alpha/100)`, com piso 3 (`Summary.py:57-59`)."""
    cut_number = int(best_cut_number_raw * (alpha / 100))
    if cut_number <= 2:
        cut_number = 3
    return cut_number
