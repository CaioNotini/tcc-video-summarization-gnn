```
Role: Evidence for the feature-graph sanity check. Saved runs of the Pearson and Spearman correlations between edge weight and temporal distance on VSUMM video v29 with torchvision features, with delta_t = 8 and delta_t = 32.
Entry: ../status.md
Related to: [status.md](../status.md), [architecture.md](../architecture.md)
```

# Sanity check: edge weight grows with temporal distance (v29)

This saves the run behind the "correlation of 0.97" previously listed in
[status.md](../status.md) under "Reported, not yet evidence". The script is
`experiments/validate_graph_torchvision.py`; there is no `sanity_check_features.py` in the
repository, and this is the script that computes this check.

## History

- 2026-11-16, commit `02652a5`: first run, `delta_t = 8`, Pearson only: **0.9749**.
- 2026-11-16, commit `c71a4ed` with uncommitted changes: second run, `delta_t = 32`, Pearson
  only: **0.7937**.
- 2026-11-16, commit `c71a4ed` with uncommitted changes: the script gained a Spearman line
  (`scipy.stats.spearmanr` over the same distance and mean-weight lists), and both values of
  `delta_t` were run again. The outputs below are from this run. The Pearson values and every
  per-distance mean are the same as in the two earlier runs.

## Parameters

- Script: `experiments/validate_graph_torchvision.py`
- Features: `data/processed/vsumm/v29/features_teste.npy`, shape (65, 2048), ResNet50 avgpool
  output produced earlier with the torchvision extractor (Local state in
  [status.md](../status.md)).
- Frames behind these features: 65 frames for a 64.8 s video at 29.97 fps, so about one frame per
  second (one every 30 native frames, 1.001 s). They were extracted before the extractor fix of
  2026-11-16 and were not re-extracted.
- Graph: `src/tcc_gnn/graph/frame.py`, `build_time_aware_graph`.
- `delta_t = 8`: the value in `configs/default.yaml`, used in the T01 and T06 validations.
- `delta_t = 32`: `rate * time` of the HieTaSumm-lib `options.json` (`Summary.py:15`), used for
  the Q-10 measurement.

## Output, `delta_t = 8` (verbatim, mojibake from the Windows console encoding)

Command: `python experiments/validate_graph_torchvision.py --features
data/processed/vsumm/v29/features_teste.npy --delta-t 8`

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
Spearman (dist�ncia x peso m�dio): 1.0000
Esperado: pr�xima de +1 (peso cresce com a dist�ncia temporal, indicando que frames mais distantes tendem a ser mais dissimilares).
```

## Output, `delta_t = 32` (verbatim)

Command: `python experiments/validate_graph_torchvision.py --features
data/processed/vsumm/v29/features_teste.npy --delta-t 32`

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
Spearman (dist�ncia x peso m�dio): 0.9456
Esperado: pr�xima de +1 (peso cresce com a dist�ncia temporal, indicando que frames mais distantes tendem a ser mais dissimilares).
```

## Result

| `delta_t` | Edges | Pearson | Spearman | NaN/Inf |
| --------- | ----- | ------- | -------- | ------- |
| 8         | 427   | 0.9749  | 1.0000   | no      |
| 32        | 1519  | 0.7937  | 0.9456   | no      |

With `delta_t = 8` the mean weight rises at every distance from 1 to 7, so the rank correlation
is exactly 1. The value 0.9749 previously reported as 0.97 is confirmed.

**Plateau.** With `delta_t = 32` the mean weight rises steadily up to about distance 12. From
distance 12 to 31 it stays between 13.60 and 14.53, with small rises and falls (for example
14.06 at 18, 13.89 at 21, 14.53 at 27). Pearson drops to 0.79 because it measures a straight-line
trend, and the curve bends. Spearman stays at 0.95 because the order is still mostly increasing;
it is not 1 because of the small falls inside the plateau. The short-range means (distances 1 to 7) are the same for both values of `delta_t`, since those edges are the same.

**Distance in seconds.** These v29 frames are about one per second, so distance 12 is about 12 s
of video, not 3 s. 12 frames would be 3 s only for frames sampled every 0.25 s, as in the SumMe
frames of the Q-10 measurement, and that case was not measured here.

This is one VSUMM video of 65 frames. The plateau is an observation on it, not a measured
property of the graph.
