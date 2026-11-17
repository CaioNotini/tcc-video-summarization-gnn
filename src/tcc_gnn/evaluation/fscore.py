"""F-score de resumos, porta de HieTaSumm-lib `Evaluation.f_score` (D-02), sob o protocolo da T19."""

from __future__ import annotations

import numpy as np

SAMPLING_PERIOD_S = 0.25  # um frame amostrado a cada 1/rate s, rate = 4 (D-25)


def sampled_to_native(sampled_frames: list[int] | np.ndarray, fps: float, n_frames: int) -> np.ndarray:
    """Resumo 0/1 com `n_frames` posições: o frame amostrado `k` marca os nativos `c` com `0.25 k <= c / fps < 0.25 (k + 1)` (D-26)."""
    if fps <= 0:
        raise ValueError(f"fps deve ser positivo, veio {fps}")
    if n_frames <= 0:
        raise ValueError(f"n_frames deve ser positivo, veio {n_frames}")
    sampled = np.asarray(sampled_frames, dtype=np.int64).reshape(-1)
    if (sampled < 0).any():
        raise ValueError("índices de frame amostrado devem ser >= 0")
    # c / fps / 0.25 = 4 * c / fps: multiplicar por 4 é exato, então floor() dá o k da desigualdade acima
    interval = np.floor((np.arange(n_frames) / fps) / SAMPLING_PERIOD_S).astype(np.int64)
    return np.isin(interval, sampled).astype(np.int8)


def f_score(predicted_summary: np.ndarray, user_summary: np.ndarray, eval_method: str = "max") -> float:
    """Porta de `Evaluation.f_score` (`Evaluation.py:132-155`): F (0-100), máximo (`"max"`) ou média entre anotadores; resumo vazio vale 0."""
    predicted_summary = np.asarray(predicted_summary)
    user_summary = np.asarray(user_summary)
    if predicted_summary.ndim != 1:
        raise ValueError(f"predicted_summary deve ser 1-D, veio shape {predicted_summary.shape}")
    if user_summary.ndim != 2 or user_summary.shape[0] == 0:
        raise ValueError(f"user_summary deve ser (n_anotadores, n_frames), veio shape {user_summary.shape}")

    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[: len(predicted_summary)] = predicted_summary
    if S.sum() == 0:
        return 0.0

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[: user_summary.shape[1]] = user_summary[user]
        if G.sum() == 0:
            raise ValueError(f"Anotador {user} tem resumo vazio")
        overlapped = S & G

        precision = overlapped.sum() / S.sum()
        recall = overlapped.sum() / G.sum()
        if precision + recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))

    # mantém os np.float64 até o fim: o sum() do Python 3.12 compensa floats nativos e mudaria o último bit
    if eval_method == "max":
        return float(max(f_scores))
    return float(sum(f_scores) / len(f_scores))
