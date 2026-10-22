```
Role: Evidence record for T06. Saves the real output of the test suite, the real-video comparison against HieTaSumm-lib, and the verification figures.
Entry: docs/status.md
Related to: [status.md](../status.md), [phases.md](../phases.md), [decisions.md](../decisions.md), [t04-mst-hierarchy-spec.md](./t04-mst-hierarchy-spec.md)
```

# T06 evidence: MST hierarchy (higra), compared against HieTaSumm-lib

Run on 2026-10-22, repository commit `da3158840992aa3b32c2bbc09ced56055eff8aeb`.

Environment: Python 3.12.6, networkx 3.3, numpy 1.26.2, higra (installed, no `__version__`
attribute exposed by the package), Windows 11.

Scope: only `src/tcc_gnn/hierarchy/mst.py` (`compute_hierarchy`, `cut_graph`,
`partition_from_cut`, `mst_partition`), ported from `HieTaSumm-lib/HieTaSumm/Graph.py` per the
approved T04 specification ([t04-mst-hierarchy-spec.md](./t04-mst-hierarchy-spec.md)). `cutNumber`
is an explicit input parameter, as in the original `Graph.cut_graph`; `Summary.bestCutNumber`
(which derives `cutNumber` from raw-pixel similarity) belongs to `Summary.py`, i.e. to T07
(HieTaSumm baseline end to end), not to T06. `references/` was not modified.

## 1. `pytest -v`

```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-7.4.3, pluggy-1.6.0
rootdir: C:\Estudos\Faculdade\Trabalhos\TCC\tcc-video-summarization-gnn
configfile: pyproject.toml
testpaths: tests
plugins: anyio-3.7.1, asyncio-0.21.1, cov-4.1.0
asyncio: mode=Mode.STRICT
collecting ... collected 17 items

tests/test_frame_graph.py::test_calc_window_end_dentro_dos_limites PASSED [  5%]
tests/test_frame_graph.py::test_edge_weight_features_identicas_da_zero PASSED [ 11%]
tests/test_frame_graph.py::test_build_time_aware_graph_com_dados_falsos PASSED [ 17%]
tests/test_frame_graph.py::test_build_time_aware_graph_com_video_real_do_summe PASSED [ 23%]
tests/test_mst_hierarchy.py::test_compute_hierarchy_bate_com_a_arvore_calculada_a_mao PASSED [ 29%]
tests/test_mst_hierarchy.py::test_compute_hierarchy_e_arvore_binaria_estrita PASSED [ 35%]
tests/test_mst_hierarchy.py::test_compute_hierarchy_trunca_pesos_para_int_antes_do_higra PASSED [ 41%]
tests/test_mst_hierarchy.py::test_compute_hierarchy_grafo_desconexo_levanta_erro PASSED [ 47%]
tests/test_mst_hierarchy.py::test_compute_hierarchy_e_determinista PASSED [ 52%]
tests/test_mst_hierarchy.py::test_cut_graph_particao_bate_com_o_corte_tracado_a_mao[0-particao_esperada0] PASSED [ 58%]
tests/test_mst_hierarchy.py::test_cut_graph_particao_bate_com_o_corte_tracado_a_mao[1-particao_esperada1] PASSED [ 64%]
tests/test_mst_hierarchy.py::test_cut_graph_particao_bate_com_o_corte_tracado_a_mao[2-particao_esperada2] PASSED [ 70%]
tests/test_mst_hierarchy.py::test_cut_graph_aresta_final_guarda_o_leaf_flag_nao_um_peso PASSED [ 76%]
tests/test_smoke.py::test_package_imports PASSED                         [ 82%]
tests/test_summe_adapter.py::test_list_videos_retorna_os_25_videos_em_ordem_numerica PASSED [ 88%]
tests/test_summe_adapter.py::test_load_ground_truth_video_1_bate_com_o_arquivo_do_submodulo PASSED [ 94%]
tests/test_summe_adapter.py::test_load_ground_truth_video_inexistente_levanta_erro PASSED [100%]

============================== warnings summary ===============================
..\..\..\..\..\Users\caion\AppData\Local\Programs\Python\Python312\Lib\site-packages\dateutil\tz\tz.py:37
  ...DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated...

======================== 17 passed, 1 warning in 1.17s =========================
```

8 tests in `tests/test_mst_hierarchy.py`, 9 pre-existing tests unaffected (no regression).

## 2. Synthetic graph: partition calculated by hand

`tests/test_mst_hierarchy.py` uses a 4-frame path graph (`0-1-2-3`, weights `[1, 3, 2]`). The
watershed-by-area tree and the `cutNumber` cut were traced by hand from the ported logic before
the test file was written, then cross-checked with a standalone, direct call to `higra`
(`hg.watershed_hierarchy_by_area` + `hg.tree_2_binary_tree`), outside `mst.py`, as an independent
oracle:

- Tree: leaves `0,1` merge into node `4`; leaves `2,3` merge into node `5`; node `6` is the root.
  `triples = [(0,4,-1),(1,4,-1),(2,5,-1),(3,5,-1),(4,6,-2),(5,6,-2),(6,6,-2)]`, `leaf_list = [0,1,2,3]`.
- Cut, hand-traced through `cut_graph`'s top-down walk: `cutNumber=0` and `cutNumber=1` both give
  the partition `{0,1}, {2,3}` (the first line processed is the root's own record, which consumes
  the first unit of `cutNumber` without opening a real node); `cutNumber=2` opens node `5` for
  real, which isolates leaves `2` and `3` from each other, giving `{0,1}, {2}, {3}`.

All of this is reproduced exactly by `compute_hierarchy`/`cut_graph`/`mst_partition`.

**Int() truncation (fidelity detail).** A second synthetic case uses weights `(0,1)=2.9`,
`(1,2)=10.0`, `(2,3)=2.1`. Truncated with `int()`, `(0,1)` and `(2,3)` tie at weight `2`, and the
tie is broken by edge order, so `{0,1}` merges first. Verified directly with `higra`: using the
raw floats instead (no truncation) gives the _opposite_ result — `{2,3}` merges first, because
`2.1 < 2.9` without truncation. `compute_hierarchy` reproduces the truncated (tied) result, not
the raw-float one, confirming the `int()` cast from `Graph.py:60-61` is actually exercised, not a
no-op in this test.

**Correction to the T04 spec's test case 3.** The spec expected a disconnected synthetic graph to
produce "a forest, not a single tree," as an open, undemonstrated expectation. Verified directly
with `higra`: a disconnected graph raises `RuntimeError: ... Input graph must be connected`, not a
forest. `test_compute_hierarchy_grafo_desconexo_levanta_erro` asserts this real behaviour. This
does not block anything: the real time-aware graph is always connected for `delta_t >= 1`
(every frame links at least to the next one), so the case never triggers in practice. Recording
this correction here per principle 1; `docs/phases.md` and the T04 spec evidence file were not
edited, per the researcher's instructions for this task.

## 3. Real video: comparison against `HieTaSumm.Graph` directly

Command:
`python experiments/validate_mst_hierarchy.py --features data/processed/vsumm/v21/features_teste.npy --delta-t 8 --cut-number 20 --csv-out docs/evidence/t06-real-video-partition.csv`

The input graph was built by calling `Frame.calc_end`/`Frame.calculate_similarity` directly on the
raw feature array — the same convention `experiments/validate_graph.py` established in T01 to
work around the node-label mismatch in `Frame.load` (documented in the T04 spec). `cutNumber=20`
was picked by hand for this comparison; it is not derived from `Summary.bestCutNumber`, which
depends on raw-pixel image similarity and belongs to `Summary.py` (T07's scope), not to T06.

```
Features carregadas: shape (110, 1000)
Grafo construido (Frame.calc_end/calculate_similarity diretos): 110 frames
Arvore (triples + leaf_list) identica a referencia: True
cutNumber=20 (escolhido a mao, nao via bestCutNumber/Summary.py - fora do escopo de T06)
Particao identica a referencia: True
Clusters: nosso=21 referencia=21
Frames com cluster diferente: 0
CSV salvo em docs\evidence\t06-real-video-partition.csv

OK: arvore e particao equivalentes a referencia.
```

The full per-frame comparison is in
[t06-real-video-partition.csv](./t06-real-video-partition.csv): for all 110 frames of `v21`, the
cluster assigned by `mst_partition` matches the cluster assigned by `HieTaSumm.Graph.compute_hierarchy`

- `HieTaSumm.Graph.cut_graph`, called directly on the same graph. Every cluster in both partitions
  covers a contiguous range of frame indices (the input graph only has temporal edges), consistent
  with the T04 spec's output invariant.

## 4. Verification figures

Generated by `python experiments/plot_mst_hierarchy.py`. `matplotlib` is not a dependency in
`pyproject.toml`; both figures were drawn with `Pillow` (`pillow`, already a dependency), per the
researcher's instruction not to add `matplotlib` without asking first.

- [docs\evidence\images\t06-synthetic-graph.png](./docs\evidence\images\t06-synthetic-graph.png): the synthetic graph (section 2), nodes
  coloured by the cluster `mst_partition` assigned (circle) next to the hand-computed expected
  cluster (square). Both colours come from the same canonical cluster numbering (ordered by each
  cluster's smallest frame index), so a real disagreement between code and hand-computed
  expectation would show as a colour mismatch between circle and square, flagged in red. Current
  result: all four colours match, "OK: nenhuma divergencia". The mismatch-detection path itself
  was exercised manually with a deliberately wrong expected partition (not saved — a throwaway
  check), confirming it does flag disagreeing frames.
- [docs\evidence\images\t06-synthetic-graph.png](./docs\evidence\images\t06-synthetic-graph.png): the real-video partition
  (section 3) as two coloured strips along the 110 frame indices, implementation vs. reference,
  with vertical red lines reserved for marking any differing frame. Both strips render identically
  (0 of 110 frames differ), consistent with the CSV and the console output above.

## Result

The MST hierarchy (`compute_hierarchy`, `cut_graph`, `mst_partition`) matches the synthetic,
hand-computed partition exactly, reproduces the `int()` truncation fidelity detail required by the
T04 spec, and matches `HieTaSumm.Graph` node-for-node (tree) and frame-for-frame (partition) on a
real video (`v21`, `delta_t=8`, `cutNumber=20`). One correction to the T04 spec's test case 3 is
recorded above (disconnected input raises, rather than producing a forest); it does not affect any
other part of the specification or any validated task.
