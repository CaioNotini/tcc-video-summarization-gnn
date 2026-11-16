```
Role: Evidence for the feature-graph sanity check. Saved runs of the correlation between edge weight and temporal distance on VSUMM video v29 with torchvision features, with delta_t = 8 and delta_t = 32.
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

## Second run: `delta_t = 32`

Run on 2026-11-16, repository commit `c71a4ed` with uncommitted changes; `src/` and
`experiments/validate_graph_torchvision.py` unmodified. Same script and same features file as
above. `delta_t = 32` is `rate * time` of the HieTaSumm-lib `options.json` (`Summary.py:15`), the
value used for the Q-10 measurement. The run above, with `delta_t = 8`, is kept as history.

Command: `python experiments/validate_graph_torchvision.py --features
data/processed/vsumm/v29/features_teste.npy --delta-t 32`

Output (verbatim, mojibake from the Windows console encoding):

```
Features carregadas: shape (65, 2048)
N�s: 65  Arestas: 1519
Peso m�nimo:  2.1525
Peso m�ximo:  23.9451
Peso m�dio:   12.7634
NaN/Inf?      False
Peso m�dio por dist�ncia temporal |v1 - v2|:
  dist�ncia  1: peso m�dio 6.4691  (n=64)
  dist�ncia  2: peso m�dio 8.1563  (n=63)
  dist�ncia  3: peso m�dio 9.4702  (n=62)
  dist�ncia  4: peso m�dio 10.4396  (n=61)
  dist�ncia  5: peso m�dio 11.0975  (n=60)
  dist�ncia  6: peso m�dio 11.6412  (n=59)
  dist�ncia  7: peso m�dio 12.2366  (n=58)
  dist�ncia  8: peso m�dio 12.7971  (n=57)
  dist�ncia  9: peso m�dio 13.1106  (n=56)
  dist�ncia 10: peso m�dio 13.1833  (n=55)
  dist�ncia 11: peso m�dio 13.4099  (n=54)
  dist�ncia 12: peso m�dio 13.5969  (n=53)
  dist�ncia 13: peso m�dio 13.8012  (n=52)
  dist�ncia 14: peso m�dio 13.8808  (n=51)
  dist�ncia 15: peso m�dio 14.0079  (n=50)
  dist�ncia 16: peso m�dio 14.0604  (n=49)
  dist�ncia 17: peso m�dio 14.0438  (n=48)
  dist�ncia 18: peso m�dio 14.0609  (n=47)
  dist�ncia 19: peso m�dio 14.0349  (n=46)
  dist�ncia 20: peso m�dio 13.9924  (n=45)
  dist�ncia 21: peso m�dio 13.8933  (n=44)
  dist�ncia 22: peso m�dio 13.9155  (n=43)
  dist�ncia 23: peso m�dio 14.0325  (n=42)
  dist�ncia 24: peso m�dio 14.1727  (n=41)
  dist�ncia 25: peso m�dio 14.3876  (n=40)
  dist�ncia 26: peso m�dio 14.4607  (n=39)
  dist�ncia 27: peso m�dio 14.5274  (n=38)
  dist�ncia 28: peso m�dio 14.5254  (n=37)
  dist�ncia 29: peso m�dio 14.4099  (n=36)
  dist�ncia 30: peso m�dio 14.3809  (n=35)
  dist�ncia 31: peso m�dio 14.2214  (n=34)
Correla��o (dist�ncia x peso m�dio): 0.7937
Esperado: pr�xima de +1 (peso cresce com a dist�ncia temporal, indicando que frames mais
distantes tendem a ser mais dissimilares).
```

Result: correlation **0.7937**, no NaN or Inf. Distances 1 to 7 give the same means as the
`delta_t = 8` run, as expected, since the edges are the same. The mean weight keeps growing up to
about distance 12 and then stays between 13.6 and 14.5 up to distance 31. The lower correlation
comes from this plateau over the longer window, not from a change in the short-range edges. The
plateau is an observation on one VSUMM video of 65 frames, not a measured property of the graph.
