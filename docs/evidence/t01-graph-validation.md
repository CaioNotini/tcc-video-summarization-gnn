```
Role: Evidence record for T01. Saves the real output of the test suite and of experiments/validate_graph.py.
Entry: docs/status.md
Related to: [status.md](../status.md), [phases.md](../phases.md)
```

# T01 evidence: time-aware graph validation

Run on 2026-10-20, repository commit `3d1f6acbd56951b35703f80f1a34fd0cba4ca140`.

Environment: Python 3.12.6, networkx 3.3, numpy 1.26.2, Windows 11.

Command order: `pytest`, then `experiments/validate_graph.py` on the v21 and v29 feature
files in `data/processed/vsumm/`, with `--delta-t 8` (the value recorded in
`configs/default.yaml` for the temporal window used by `Frame.py`).

## 1. `pytest -v`

```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-7.4.3, pluggy-1.6.0
rootdir: C:\Estudos\Faculdade\Trabalhos\TCC\tcc-video-summarization-gnn
configfile: pyproject.toml
testpaths: tests
plugins: anyio-3.7.1, asyncio-0.21.1, cov-4.1.0
asyncio: mode=Mode.STRICT
collecting ... collected 5 items

tests/test_frame_graph.py::test_calc_window_end_dentro_dos_limites PASSED [ 20%]
tests/test_frame_graph.py::test_edge_weight_features_identicas_da_zero PASSED [ 40%]
tests/test_frame_graph.py::test_build_time_aware_graph_com_dados_falsos PASSED [ 60%]
tests/test_frame_graph.py::test_build_time_aware_graph_com_video_real_do_summe PASSED [ 80%]
tests/test_smoke.py::test_package_imports PASSED                         [100%]

============================== 5 passed in 4.49s ==============================
```

## 2. `experiments/validate_graph.py` on v21

Command: `python experiments/validate_graph.py --features data/processed/vsumm/v21/features_teste.npy --delta-t 8`

```
Features carregadas: shape (110, 1000)
Arestas só na referência (Leonardo): 0
Arestas só no nosso grafo:           0
Arestas em comum:                    742
Maior diferença de peso (comuns):    0.0000000000

OK: grafos equivalentes.
```

## 3. `experiments/validate_graph.py` on v29

Command: `python experiments/validate_graph.py --features data/processed/vsumm/v29/features_teste.npy --delta-t 8`

```
Features carregadas: shape (65, 2048)
Arestas só na referência (Leonardo): 0
Arestas só no nosso grafo:           0
Arestas em comum:                    427
Maior diferença de peso (comuns):    0.0000000000

OK: grafos equivalentes.
```

## Result

Both videos: the edge sets of our graph and Leonardo's reference graph match exactly
(zero edges only on one side), and the maximum weight difference on the common edges
is `0.0`. The five tests pass, including the graph tests.

## Discrepancy with a previously reported, non-evidenced number

[status.md](../status.md), under "Reported, not yet evidence", recorded a prior claim of
"945 matching edges" for the Leonardo equivalence check, with no saved output and no
stated `delta_t` or video. That number does not match this run (742 edges for v21, 427
for v29, with `delta_t=8`). Since the earlier claim was never saved as evidence, and its
`delta_t` is unknown, this file's numbers are the ones that count as validated, per
principle 3. The earlier entry should be removed from "Reported, not yet evidence" once
this file is linked from status.md.

## Note on self-loops in the original `Frame.load` (added 2026-11-16)

The original loop in `Frame.load` (`references/HieTaSumm-lib/HieTaSumm/Frame.py`,
`for vertex2 in range(vertex1, self.calc_end(...))`) includes the case `vertex2 == vertex1`
and therefore adds one zero-weight self-loop per frame to its graph. The port
(`src/tcc_gnn/graph/frame.py`) skips `v1 == v2` and produces no self-loops, and
the reference graph used by `experiments/validate_graph.py` (`build_leonardo_reference_graph` up to commit `5b73d21`, now `reference_graph` in `src/tcc_gnn/reference.py`) skips the same case, so the
edge-set comparison above covers the proper edges only. The discard is deliberate: the
self-loops carry weight 0 and no information, and `Frame.load` itself cannot feed
`Graph.compute_hierarchy` because of the node-label mismatch documented in
[t04-mst-hierarchy-spec.md](./t04-mst-hierarchy-spec.md), so the integer-indexed,
self-loop-free convention established here is the one every later comparison uses. This note
records the deviation; it does not change the result above.

## Re-run after the cleanup of 2026-11-17

`src/tcc_gnn/graph/frame.py` received one-line docstrings and no logic change. The validation was
run again on the same features, with the reference graph now built by
`reference_graph` (`src/tcc_gnn/reference.py`). Output, verbatim:

`python experiments/validate_graph.py --features data/processed/vsumm/v21/features_teste.npy --delta-t 8`

```
Features carregadas: shape (110, 1000)
Arestas só na referência (Leonardo): 0
Arestas só no nosso grafo:           0
Arestas em comum:                    742
Maior diferença de peso (comuns):    0.0000000000

OK: grafos equivalentes.
```

`python experiments/validate_graph.py --features data/processed/vsumm/v29/features_teste.npy --delta-t 8`

```
Features carregadas: shape (65, 2048)
Arestas só na referência (Leonardo): 0
Arestas só no nosso grafo:           0
Arestas em comum:                    427
Maior diferença de peso (comuns):    0.0000000000

OK: grafos equivalentes.
```
