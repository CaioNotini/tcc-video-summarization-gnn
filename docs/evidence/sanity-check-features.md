```
Role: Evidence for the feature-graph sanity check. Saved run of the correlation between edge weight and temporal distance on VSUMM video v29 with torchvision features.
Entry: ../status.md
Related to: [status.md](../status.md), [architecture.md](../architecture.md)
```

# Sanity check: edge weight grows with temporal distance (v29)

Run on 2026-11-16, repository commit `02652a5` (working tree with uncommitted documentation
changes; `src/` and `experiments/validate_graph_torchvision.py` unmodified).

This saves the run behind the "correlation of 0.97" previously listed in
[status.md](../status.md) under "Reported, not yet evidence". The script is
`experiments/validate_graph_torchvision.py`; there is no `sanity_check_features.py` in the
repository, and this is the script that computes this check.

## Parameters

- Script: `experiments/validate_graph_torchvision.py`
- Features: `data/processed/vsumm/v29/features_teste.npy`, shape (65, 2048) — ResNet50
  avgpool output dimension, produced earlier with the torchvision extractor (as recorded in
  the Local state section of [status.md](../status.md))
- Graph: `src/tcc_gnn/graph/frame.py`, `build_time_aware_graph`
- `delta_t = 8`, the value in `configs/default.yaml` used in the T01/T06 validations

## Output (verbatim, mojibake from the Windows console encoding)

```
Features carregadas: shape (65, 2048)
N�s: 65  Arestas: 427
Peso m�nimo:  2.1525
Peso m�ximo:  20.0302
Peso m�dio:   9.8694
NaN/Inf?      False

Peso m�dio por dist�ncia temporal |v1 - v2|:
  dist�ncia  1: peso m�dio 6.4691  (n=64)
  dist�ncia  2: peso m�dio 8.1563  (n=63)
  dist�ncia  3: peso m�dio 9.4702  (n=62)
  dist�ncia  4: peso m�dio 10.4396  (n=61)
  dist�ncia  5: peso m�dio 11.0975  (n=60)
  dist�ncia  6: peso m�dio 11.6412  (n=59)
  dist�ncia  7: peso m�dio 12.2366  (n=58)

Correla��o (dist�ncia x peso m�dio): 0.9749
Esperado: pr�xima de +1 (peso cresce com a dist�ncia temporal, indicando que frames mais
distantes tendem a ser mais dissimilares).
```

## Result

Correlation between temporal distance and mean edge weight: **0.9749**, consistent with the
0.97 previously reported without evidence. Mean edge weight (feature distance) grows
monotonically with temporal distance, with no NaN or Inf, which is the expected behaviour of
the time-aware graph on real features.
