```
Role: Evidence record for T06's additional verification (cut-edge table, tie count, no-truncation diagnostic, v29 verification, bestCutNumber, sensitivity test, reference independence). Extends t06-mst-hierarchy.md, does not replace it.
Entry: docs/status.md
Related to: [status.md](../status.md), [phases.md](../phases.md), [t04-mst-hierarchy-spec.md](./t04-mst-hierarchy-spec.md), [t06-mst-hierarchy.md](./t06-mst-hierarchy.md)
```

# T06 evidence: additional verification (cut-edge weights, ties, v29, bestCutNumber)

Run on 2026-10-22, repository commit `9f4016352f2a8bf10358811b8487debcf4ae49e4` (base commit for
this session; this task adds no commit of its own). Sections 1–9 below are from the first pass of
this additional verification (`v21` only); the "V29 verification" and "Link audit" sections and
the methodology correction note were added in a follow-up pass that also reorganised the evidence
folder structure and reran the `v21` numbers with the corrected method.

Environment: Python 3.12.6, networkx 3.3, numpy 1.26.2, higra (installed), Pillow, Windows 11.

Scope: no change to `src/` is kept (one temporary, reverted edit — section 5). New/changed files
are in `tests/`, `experiments/`, `docs/evidence/sheets/` and `docs/evidence/images/` only. No
`docs/images/` folder and no loose CSV exist under `docs/evidence/` anymore — see "Evidence folder
structure" below. `references/`, `docs/decisions.md`, `docs/architecture.md` and `docs/status.md`
were not touched.

## Evidence folder structure

Figures live in [images/](./images/) and CSVs in [sheets/](./sheets/), both under
`docs/evidence/`. `docs/images/` (used by the first pass of this task) no longer exists; its three
files were moved into `docs/evidence/images/`, next to the T06 figures already there. The loose
CSVs that used to sit directly in `docs/evidence/` (`t06-real-video-partition.csv`,
`t06-cut-edges-v21.csv`, `t06-comparison-cases.csv`) were moved into `docs/evidence/sheets/`. No
file was duplicated; `git status` after the move showed the old paths as deleted and the new ones
as untracked, confirming plain moves.

## Methodology correction: `mst_edge_map` does not identify the cut edge on every video

**[CORRIGIDO]** Esta seção, a tabela de arestas do v21 (seção 1) e a tabela de arestas do v29 (em
"V29 verification") documentam o `cutNumber`/corte usando o método corrigido abaixo. A versão
anterior usava `bpt_canonical`/`tree.mst_edge_map` para achar a aresta de cada corte; foi
substituída pelo método que lê a árvore real retornada por `compute_hierarchy`, pelos motivos
explicados a seguir.

The first pass of this task (section 1 below, `v21`-only) found "the edge that built a severed
tree node" by calling `hg.bpt_canonical` on the same graph and reading `tree.mst_edge_map`, after
confirming its tree had the same parent-links as `compute_hierarchy`'s actual tree (built by
`hg.watershed_hierarchy_by_area`) — true for `v21`. **That topology match does not hold in
general**: checked directly for `v29`, `hg.bpt_canonical`'s tree and `compute_hierarchy`'s own
tree have different parent-links (`parents_bpt == parents_ours` is `False`), because
`watershed_hierarchy_by_area`'s "by area" tie-break, when a plateau of equal-weight edges exists,
does not always pick the same minimum spanning tree as `bpt_canonical`'s plain stable-sort
tie-break — and higra does not expose an `mst_edge_map` equivalent for the `by_area` tree
(`nb_tree.mst_edge_map` raises `AttributeError`).

`experiments/analyze_mst_cut_edges.py` was rewritten to stop depending on `bpt_canonical` or
`mst_edge_map` entirely. For a severed node `X` with parent `P` and sibling `S` (`P`'s other
child), it now computes the full leaf set under `X` and under `S` directly from
`compute_hierarchy`'s own `triples` (bottom-up DP, so it only uses the tree the port actually
built), finds every original graph edge connecting a leaf of `X`'s side to a leaf of `S`'s side,
and reports the one with the smallest `(truncated weight, higra_rank)` — the same tie-break higra
itself uses when sorting edges. Re-run against `v21`, this reproduces the exact same 20 edges the
`mst_edge_map` method reported (checked row by row); it is now also correct for `v29`. The new CSV
column `n_arestas_candidatas` records how many original graph edges connected the two sides at
each cut point, since on a dense `delta_t=8` graph there is usually more than one (up to 28, for
`v29`'s largest boundary) — the reported edge is a defensible minimum-weight representative of the
cut, not provably the single internal edge higra's C++ implementation used, since that detail
isn't exposed by the library for the `by_area` variant.

## 1. Table of cut edge weights (v21, `delta_t=8`, `cutNumber=20`) — [CORRIGIDA]

**Esta tabela e a sua documentação do `cutNumber` foram corrigidas nesta revisão.** A versão
anterior identificava a aresta de cada corte via `hg.bpt_canonical`/`tree.mst_edge_map`; essa
dependência foi removida e os números abaixo foram recalculados pelo método que lê a árvore real
de `compute_hierarchy` — ver "Methodology correction" acima para o porquê e a prova de que os
valores batem com os da versão anterior.

Saved in [sheets/t06-cut-edges-v21.csv](./sheets/t06-cut-edges-v21.csv) (rows with `video=v21`),
generated by `experiments/analyze_mst_cut_edges.py --video v21 --delta-t 8 --cut-number 20`, with
the corrected method above. `cut_graph`'s loop severs **21 tree nodes** for `cutNumber=20` on this
video, not 20 — but those 21 severances trace back to exactly **20 distinct original graph edges**.
One internal node has *both* of its children severed at once (its cut edge therefore accounts for
two severed tree nodes), which is why the node count (21) and the edge count (20) differ. 20
distinct edges is consistent with `cutNumber=20`: removing 20 edges from a tree always yields 21
components, matching the 21 clusters already reported in
[t06-mst-hierarchy.md](./t06-mst-hierarchy.md).

Each CSV row is one distinct cut edge: `parent_node_severed` (the tree node whose severed
children trace back to this edge), `node_u`/`node_v` (the original frame indices of the edge),
`weight_float` (the real L1 feature-distance weight), `weight_int` (after `int()` truncation),
`higra_rank` (the edge's position in the stable sort of truncated weights), `n_arestas_candidatas`
(how many graph edges connected the same two sides — see the methodology note above),
`cluster_u`/`cluster_v` (the final cluster id of each endpoint — always different, confirmed for
all 20 rows: `mesmo_cluster` is `False` everywhere).

## 2. Tie count at the cut boundary (v21)

**All 742 edges of the v21 graph (`delta_t=8`) have a truncated weight of exactly `0`.** The real
L1 feature-distance weights range from `0.0167` to `0.199` — every one of them is below `1.0`, so
`int()` truncates the entire graph to a single tied value. Confirmed directly:

```
n_edges: 742
edges_truncating_to_zero: 742
last_cut_weight_int: 0
last_cut_rank: 720
edges_with_last_cut_weight: 742
of_those_cut: 20
stable_order_is_identity: True

EMPATE NO PONTO DE CORTE: 742 arestas do grafo tem o peso truncado 0 (o mesmo do ultimo corte),
mas so 20 foram cortadas. A particao depende da ordem de desempate (ordem de insercao das arestas),
nao so do peso.
```

**The partition for v21 at this `delta_t` depends entirely on tie-breaking (edge insertion order),
not on the feature-distance weights.** `stable_order_is_identity: True` confirms the stable sort
left the 742 tied edges in their original `graph.edges()` order — the "ordering higra used" carries
no information from the actual distances here. This is a real property of this video's graph with
this extractor, not a bug in the port: `v29` does **not** have this problem (section "V29
verification" below) — its stored `features_teste.npy` is 1000-d for `v21` vs. 2048-d avgpool
(D-03) for `v29` (per [t01-graph-validation.md](./t01-graph-validation.md)), and the 1000-d
features produce much smaller L1 distances once divided by vector length. Noted here as context,
not as a new decision; no code was changed because of it.

## 3. Test without truncation (scaled weights, diagnostic only)

**Researcher's decision carried into this and every later section:** `int()` truncation is part of
the baseline (D-02) and stays the acceptance criterion in every comparison with the reference
(`experiments/compare_mst_cases.py`). The scaled-weight tests below are a tie diagnostic only —
they do not replace or loosen that criterion.

`tests/test_mst_hierarchy_scaled_weights.py`, now parametrized over `v21` and `v29` (4 tests
total), skippable if the submodule or either video's features file is missing. Production code is
unchanged: the input graph's edge weights are multiplied by `1e6` *before* being passed to the
normal, unmodified `compute_hierarchy`/`cut_graph`/`mst_partition`.

- `test_sem_truncamento_bate_com_a_referencia[v21]` / `[v29]`: with scaled weights, the
  implementation still matches `HieTaSumm.Graph` exactly on both videos — the port's fidelity holds
  independently of the tie.
- `test_sem_truncamento_muda_a_particao_em_relacao_ao_peso_original[v21]` / `[v29]`: the
  scaled-weight partition is **different** from the `int()` partition on both videos at
  `cutNumber=20` — **107 of 110 frames** change cluster on `v21`, **31 of 65** on `v29` (see
  [images/t06-truncation-comparison-v21.png](./images/t06-truncation-comparison-v21.png) and
  [images/t06-truncation-comparison-v29.png](./images/t06-truncation-comparison-v29.png)). `v21`'s
  change is far larger because of the all-zero tie in section 2; `v29`'s smaller-but-nonzero change
  confirms truncation still loses some precision even without a degenerate plateau — expected for
  any `int()` cast, not specific to `v21`. Per the researcher's decision, **neither number changes
  which partition counts as validated**: the `int()` partition is still the one compared against
  the reference, and it matches exactly on both videos (section 1 and "V29 verification").

```
tests/test_mst_hierarchy_scaled_weights.py::test_sem_truncamento_bate_com_a_referencia[v21] PASSED
tests/test_mst_hierarchy_scaled_weights.py::test_sem_truncamento_bate_com_a_referencia[v29] PASSED
tests/test_mst_hierarchy_scaled_weights.py::test_sem_truncamento_muda_a_particao_em_relacao_ao_peso_original[v21] PASSED
tests/test_mst_hierarchy_scaled_weights.py::test_sem_truncamento_muda_a_particao_em_relacao_ao_peso_original[v29] PASSED
```

## 4. Other v21 cases

Saved in [sheets/t06-comparison-cases.csv](./sheets/t06-comparison-cases.csv) (rows with
`video=v21`), generated by `experiments/compare_mst_cases.py`. Both extra `cutNumber` cases match
`HieTaSumm.Graph` exactly, with int() as the acceptance criterion:

| video | delta_t | cutNumber | fonte do cutNumber | frames | tree matches | frames diferentes | clusters (impl.) | clusters (ref.) | tamanhos dos segmentos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v21 | 8 | 5 | mao | 110 | True | 0 | 6 | 6 | 105;1;1;1;1;1 |
| v21 | 8 | 50 | mao | 110 | True | 0 | 51 | 51 | 60;1×50 |

`n_clusters = cutNumber + 1` in every case, consistent with removing `cutNumber` edges from a tree
(section 1). `v21`'s segment sizes are always one big remainder plus singletons — a direct
consequence of the all-edges-tied finding in section 2: with no real weight signal, each extra cut
just peels one more frame off the end of the sequence, rather than splitting the video into
balanced segments. `v29`'s segment sizes (section below) look qualitatively different. See
[images/t06-comparison-cases.png](./images/t06-comparison-cases.png) (now includes all four cases,
`v21` and `v29`).

## V29 verification (2048-d, production extractor)

Esta seção, incluindo a tabela de arestas do v29 abaixo, já foi gerada com o método corrigido (ver
"Methodology correction" acima) — não usou `bpt_canonical`/`mst_edge_map` em nenhum momento.

All of the following were produced with
`experiments/analyze_mst_cut_edges.py --video v29 --delta-t 8 --cut-number 20` and
`experiments/compare_mst_cases.py --video v29 ...`, the same scripts used for `v21` above, just
with `--video v29` (no new script was written for this).

**Cut-edge table.** Saved in the same file as `v21`'s,
[sheets/t06-cut-edges-v21.csv](./sheets/t06-cut-edges-v21.csv), rows with `video=v29`, per the
researcher's instruction to reuse one file with a `video` column. `cutNumber=20` severs 21 tree
nodes again, but this time they trace back to only **15 distinct edges** (more children share a
parent than on `v21`, since `v29`'s tree isn't a simple caterpillar near the root) — removing 15
edges from a tree gives 16 components, which does not match the 21 clusters reported below; see
"Discrepancy" paragraph right after this list.

**Zero-truncation count and tie count.** Confirmed directly:

```
n_edges: 427
edges_truncating_to_zero: 0
n_severed_nodes: 21
n_distinct_cut_edges: 15
n_clusters: 21
last_cut_weight_int: 10
last_cut_rank: 267
edges_with_last_cut_weight: 54
of_those_cut: 1
stable_order_is_identity: False

EMPATE NO PONTO DE CORTE: 54 arestas do grafo tem o peso truncado 10 (o mesmo do ultimo corte),
mas so 1 foram cortadas. A particao depende da ordem de desempate (ordem de insercao das arestas),
nao so do peso.
```

**0 of 427 edges truncate to `0`** — confirms `v29` does not have `v21`'s degenerate all-tied
problem (section 2). There is still a tie at the specific boundary weight of the 20th cut (`10`):
54 edges share that truncated weight, and only 1 of them was actually cut, so (as instructed) this
is stated explicitly: **that one specific cut (and only that one, not the other 14) depends on
tie-breaking order, not on the weight alone.**

**Discrepancy (21 clusters, but only 15 distinct cut edges — not 16).** `cut_graph` severs 21 tree
nodes for `cutNumber=20`, same as `v21`, but they collapse into 15 distinct edges because several
parents here have *both* children severed (not just one, like `v21`'s single such case) — each
such parent accounts for 2 severed nodes from only 1 edge. With 21 severed nodes and 20 clusters
of "edge budget" spent (`cutNumber=20`), the final cluster count is still 21 regardless of how many
*distinct* edges that maps to; the distinct-edge count is a property of the tree's shape at the cut
boundary, not of `cutNumber` directly. This generalises the same observation already made for
`v21` (21 severed nodes → 20 distinct edges there, because of one double-severed parent).

**Comparison with int() — acceptance criterion.** From
[sheets/t06-comparison-cases.csv](./sheets/t06-comparison-cases.csv), row `video=v29,
cut_number=20, cutnumber_fonte=mao`:

| frames | tree matches | frames diferentes | clusters (impl.) | clusters (ref.) | tamanhos dos segmentos |
| --- | --- | --- | --- | --- | --- |
| 65 | True | 0 | 21 | 21 | 12;7;7;5;5;5;4;3;3;3;1;1;1;1;1;1;1;1;1;1;1 |

Matches the reference exactly. Unlike `v21`, the segments aren't one big remainder plus
singletons — there's a spread of mid-sized groups (12, 7, 7, 5, 5, 5, 4, 3, 3, 3) before the
singletons start, consistent with `v29` having real, non-degenerate weight information to cut
along (section above).

**Comparison with scaled weights — diagnostic only, not acceptance.** Same row:
`frames_diferentes_escalado_vs_referencia_escalada = 0` (the port still matches the reference
exactly once ties are removed — fidelity holds regardless of scale, as on `v21`) and
`frames_diferentes_escalado_vs_int = 31` (31 of 65 frames move to a different cluster once the
truncation tie at weight `10` — and whatever smaller ties exist below it — is removed). Both
numbers are diagnostic; neither changes the accepted `int()` result above.

## bestCutNumber (v29)

`Summary.bestCutNumber` (`HieTaSumm-lib/HieTaSumm/Summary.py:104-125`) was called directly on
`v29`'s own frame images (`data/processed/vsumm/v29/frames/`, 65 `.jpg` files, already present
locally) via `experiments/validate_mst_hierarchy.py`'s new `reference_best_cut_number` helper. It
instantiates `Summary` with `Summary.__new__(Summary)` — bypassing `__init__`, which would extract
frames, build file paths for every intermediate artifact and load a TensorFlow model via `Models`,
none of which `bestCutNumber` itself needs — and sets only the two attributes it actually reads
(`frames_path`, `delta_t`). No function was renamed or found under a different name; the call is to
`Summary.bestCutNumber` itself, as named in the task.

```
best_cut_number_raw: 7.0
alpha: 75  (options.json default, Summary.py:52-59: cutNumber = int(bestCutNumber() * alpha / 100), floored to 3)
cutNumber (reference's own choice): int(7.0 * 75 / 100) = 5
```

**5 is different from the `cutNumber=20` used throughout this and the previous evidence file.**
Per the task's instruction, the implementation-vs-reference comparison was re-run with the
reference's own value too, reported separately from the hand-picked `cutNumber=20` above (same
[sheets/t06-comparison-cases.csv](./sheets/t06-comparison-cases.csv), row `video=v29, cut_number=5,
cutnumber_fonte=bestCutNumber(alpha=75)`):

| cutNumber | fonte | frames | tree matches | frames diferentes | clusters (impl.) | clusters (ref.) | tamanhos dos segmentos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | mao | 65 | True | 0 | 21 | 21 | 12;7;7;5;5;5;4;3;3;3;1;1;1;1;1;1;1;1;1;1;1 |
| 5 | bestCutNumber(alpha=75) | 65 | True | 0 | 6 | 6 | 18;16;9;8;7;7 |

Both match the reference exactly — `cutNumber`'s source does not affect fidelity, only how finely
the video gets cut. At `cutNumber=5`, `frames_diferentes_escalado_vs_int = 0`: at this coarser cut
the truncation tie near weight `10` never becomes a cut boundary, so (for this one case) truncation
happens not to change the result — still diagnostic only, not relied on for acceptance.

## 5. Sensitivity test

(Unchanged from the first pass of this task; not re-run with the now-larger, 21-test suite because
the task did not ask for it again. The specific "19/21 collected" and failing-test-name counts
below are from that earlier run and are kept as the historical record of that check, not as a
claim about the current full suite — section 9 below has the current, unrelated-to-this-mutation
count.)

`src/tcc_gnn/hierarchy/mst.py`'s `cut_graph` was temporarily edited to add
`cutNumber = cutNumber + 1` as its first line (a deliberate one-unit shift), to confirm the test
suite actually constrains `cutNumber` precisely rather than passing vacuously. Reverted with
`git checkout -- src/tcc_gnn/hierarchy/mst.py` immediately after capturing the failing run.

**With the mutation** (`pytest -v`, 19 collected at the time): **3 failed, 16 passed**.

```
tests/test_mst_hierarchy.py::test_cut_graph_particao_bate_com_o_corte_tracado_a_mao[1-particao_esperada1] FAILED
tests/test_mst_hierarchy.py::test_cut_graph_particao_bate_com_o_corte_tracado_a_mao[2-particao_esperada2] FAILED
tests/test_mst_hierarchy_scaled_weights.py::test_sem_truncamento_bate_com_a_referencia FAILED
=================== 3 failed, 16 passed, 1 warning in 1.85s ===================
```

**After revert**: pytest returned to **19 passed** (the suite size at that time, before this
task's `v29` parametrization added two more test cases). `git status --short src/` and
`git diff src/` were both empty immediately after the revert, confirmed before continuing.

## 6. Independence from the ported code

Read directly in the submodule (not inferred): `references/HieTaSumm-lib/HieTaSumm/Graph.py`
imports only `networkx`, `numpy` and `higra` — no import of `tcc_gnn` or anything under `src/`.
`experiments/validate_mst_hierarchy.py`'s `reference_compute_hierarchy` and `reference_cut_graph`
(used by every comparison in this file and in `t06-mst-hierarchy.md`) instantiate
`HieTaSumm.Graph.Graph` and call its `compute_hierarchy`/`cut_graph` methods directly; the
module's only `tcc_gnn` import (`from tcc_gnn.hierarchy.mst import compute_hierarchy, cut_graph,
partition_from_cut`) is used exclusively for *our* side of each comparison, in separate code paths
from the `reference_*` functions.

**Update from the bestCutNumber section above:** unlike the first pass of this task (which
reported `Summary.bestCutNumber` as read but never called), this task's `v29` section does call it,
via the new `reference_best_cut_number` helper. Checked the same way: `Summary.py`'s own imports
(`Files`, `Frame`, `Models`, `Graph`, `Evaluation`, `os`, `PIL.Image`, `numpy`, `scipy.spatial`,
`networkx`, `cv2`) include no `tcc_gnn` import, and `reference_best_cut_number` only calls
`summary.bestCutNumber()` on a bare `Summary.__new__(Summary)` instance — no `src/` code is called
from the reference side at any point in this file or in `t06-mst-hierarchy.md`.

## 7. Figures

`matplotlib` is still not a `pyproject.toml` dependency; all figures here were drawn with Pillow,
same as the original T06 figures. All of the following now live in
[images/](./images/) (`docs/evidence/images/`); none of the pre-existing T06 figures
(`t06-synthetic-graph.png`, `t06-real-video-partition.png`) were moved, renamed or removed.

- [images/t06-cut-edges-v21.png](./images/t06-cut-edges-v21.png) and
  [images/t06-cut-edges-v29.png](./images/t06-cut-edges-v29.png): each video's partition
  (`cutNumber=20`) as a frame strip, with its distinct cut points marked.
- [images/t06-truncation-comparison-v21.png](./images/t06-truncation-comparison-v21.png) and
  [images/t06-truncation-comparison-v29.png](./images/t06-truncation-comparison-v29.png): each
  video, `int()` partition vs. `x1e6`-scaled partition, with differing frames marked. (The `v21`
  file is a rename of this task's first-pass `t06-truncation-comparison.png`, now disambiguated
  from the new `v29` one — same image content, regenerated identically.)
- [images/t06-comparison-cases.png](./images/t06-comparison-cases.png): cluster counts and
  match/mismatch status for all four cases in `sheets/t06-comparison-cases.csv` (`v21`×2, `v29`×2,
  including both `cutNumber` sources for `v29`).

## Link audit (all `docs/evidence/t06-*.md` files)

Every link in `t06-mst-hierarchy.md` and `t06-additional-verification.md` (this file), checked
after the folder reorganisation above, resolved relative to each file's own location:

| file | link | target | exists |
| --- | --- | --- | --- |
| t06-mst-hierarchy.md | `images/t06-synthetic-graph.png` | `docs/evidence/images/t06-synthetic-graph.png` | yes |
| t06-mst-hierarchy.md | `images/t06-real-video-partition.png` | `docs/evidence/images/t06-real-video-partition.png` | yes |
| t06-mst-hierarchy.md | `sheets/t06-real-video-partition.csv` | `docs/evidence/sheets/t06-real-video-partition.csv` | yes |
| t06-additional-verification.md | `images/` | `docs/evidence/images/` | yes |
| t06-additional-verification.md | `sheets/` | `docs/evidence/sheets/` | yes |
| t06-additional-verification.md | `sheets/t06-cut-edges-v21.csv` | `docs/evidence/sheets/t06-cut-edges-v21.csv` | yes |
| t06-additional-verification.md | `sheets/t06-comparison-cases.csv` | `docs/evidence/sheets/t06-comparison-cases.csv` | yes |
| t06-additional-verification.md | `images/t06-truncation-comparison-v21.png` | `docs/evidence/images/t06-truncation-comparison-v21.png` | yes |
| t06-additional-verification.md | `images/t06-truncation-comparison-v29.png` | `docs/evidence/images/t06-truncation-comparison-v29.png` | yes |
| t06-additional-verification.md | `images/t06-comparison-cases.png` | `docs/evidence/images/t06-comparison-cases.png` | yes |
| t06-additional-verification.md | `images/t06-cut-edges-v21.png` | `docs/evidence/images/t06-cut-edges-v21.png` | yes |
| t06-additional-verification.md | `images/t06-cut-edges-v29.png` | `docs/evidence/images/t06-cut-edges-v29.png` | yes |
| t06-additional-verification.md | `t01-graph-validation.md` | `docs/evidence/t01-graph-validation.md` | yes |
| t06-additional-verification.md | `t04-mst-hierarchy-spec.md` | `docs/evidence/t04-mst-hierarchy-spec.md` | yes |
| t06-additional-verification.md | `t06-mst-hierarchy.md` | `docs/evidence/t06-mst-hierarchy.md` | yes |
| both files | `../status.md`, `../phases.md` | `docs/status.md`, `docs/phases.md` | yes |

No broken link found in this audit (the two broken image links found and fixed in the first pass
of this task, and the one stale CSV link fixed just above in "Evidence folder structure", are the
only corrections this task made to `t06-mst-hierarchy.md`).

## 8. `pytest -v` (full suite, src/ unmodified)

```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-7.4.3, pluggy-1.6.0
rootdir: C:\Estudos\Faculdade\Trabalhos\TCC\tcc-video-summarization-gnn
configfile: pyproject.toml
testpaths: tests
plugins: anyio-3.7.1, asyncio-0.21.1, cov-4.1.0
asyncio: mode=Mode.STRICT
collecting ... collected 21 items

tests/test_frame_graph.py::test_calc_window_end_dentro_dos_limites PASSED [  4%]
tests/test_frame_graph.py::test_edge_weight_features_identicas_da_zero PASSED [  9%]
tests/test_frame_graph.py::test_build_time_aware_graph_com_dados_falsos PASSED [ 14%]
tests/test_frame_graph.py::test_build_time_aware_graph_com_video_real_do_summe PASSED [ 19%]
tests/test_mst_hierarchy.py::test_compute_hierarchy_bate_com_a_arvore_calculada_a_mao PASSED [ 23%]
tests/test_mst_hierarchy.py::test_compute_hierarchy_e_arvore_binaria_estrita PASSED [ 28%]
tests/test_mst_hierarchy.py::test_compute_hierarchy_trunca_pesos_para_int_antes_do_higra PASSED [ 33%]
tests/test_mst_hierarchy.py::test_compute_hierarchy_grafo_desconexo_levanta_erro PASSED [ 38%]
tests/test_mst_hierarchy.py::test_compute_hierarchy_e_determinista PASSED [ 42%]
tests/test_mst_hierarchy.py::test_cut_graph_particao_bate_com_o_corte_tracado_a_mao[0-particao_esperada0] PASSED [ 47%]
tests/test_mst_hierarchy.py::test_cut_graph_particao_bate_com_o_corte_tracado_a_mao[1-particao_esperada1] PASSED [ 52%]
tests/test_mst_hierarchy.py::test_cut_graph_particao_bate_com_o_corte_tracado_a_mao[2-particao_esperada2] PASSED [ 57%]
tests/test_mst_hierarchy.py::test_cut_graph_aresta_final_guarda_o_leaf_flag_nao_um_peso PASSED [ 61%]
tests/test_mst_hierarchy_scaled_weights.py::test_sem_truncamento_bate_com_a_referencia[v21] PASSED [ 66%]
tests/test_mst_hierarchy_scaled_weights.py::test_sem_truncamento_bate_com_a_referencia[v29] PASSED [ 71%]
tests/test_mst_hierarchy_scaled_weights.py::test_sem_truncamento_muda_a_particao_em_relacao_ao_peso_original[v21] PASSED [ 76%]
tests/test_mst_hierarchy_scaled_weights.py::test_sem_truncamento_muda_a_particao_em_relacao_ao_peso_original[v29] PASSED [ 80%]
tests/test_smoke.py::test_package_imports PASSED                         [ 85%]
tests/test_summe_adapter.py::test_list_videos_retorna_os_25_videos_em_ordem_numerica PASSED [ 90%]
tests/test_summe_adapter.py::test_load_ground_truth_video_1_bate_com_o_arquivo_do_submodulo PASSED [ 95%]
tests/test_summe_adapter.py::test_load_ground_truth_video_inexistente_levanta_erro PASSED [100%]

============================== warnings summary ===============================
..\..\..\..\..\Users\caion\AppData\Local\Programs\Python\Python312\Lib\site-packages\dateutil\tz\tz.py:37
  ...DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated...

======================== 21 passed, 1 warning in 1.72s =========================
```

**21 tests total** (19 from the first pass of this task + 2 new, from parametrizing
`tests/test_mst_hierarchy_scaled_weights.py` over `v21` and `v29` instead of `v21` only — no new
test file was created, per the file-reuse rule). No production code change was kept: the sensitivity
test's mutation (section 5) was reverted before this run.

## Files changed and created by this (second) pass

**New** (none of these duplicate an existing script; all reuse of existing files is listed below
under "Changed"):
- `docs/evidence/t06-additional-verification.md` (this file, previously created, now rewritten)

No other new files were created. `docs/evidence/sheets/` and `docs/evidence/images/` are new
*directories*, not new scripts, created by moving already-existing CSVs/PNGs into them (see
"Evidence folder structure").

**Changed** (existing scripts, extended with a `--video` parameter and the bestCutNumber helper,
per the file-reuse rule — none renamed):
- `experiments/validate_mst_hierarchy.py`: added `features_path_for_video`, `frames_dir_for_video`,
  `segment_sizes`, `reference_best_cut_number`, `merge_csv_rows` (generalises the previous
  `merge_csv_rows_by_video`).
- `experiments/analyze_mst_cut_edges.py`: added `--video`; replaced the `mst_edge_map`-based cut
  edge lookup with the corrected, general method (see "Methodology correction" above); writes to
  `docs/evidence/sheets/t06-cut-edges-v21.csv` with a `video` column merged in, not overwritten.
- `experiments/compare_mst_cases.py`: added `--video`/`--cut-number`/`--cutnumber-fonte`;
  added segment-size and scaled-weight diagnostic columns; writes to
  `docs/evidence/sheets/t06-comparison-cases.csv`.
- `experiments/plot_additional_verification.py`: added `--video`; per-video figures now named with
  the video in the filename and saved to `docs/evidence/images/`.
- `tests/test_mst_hierarchy_scaled_weights.py`: parametrized over `v21`/`v29` instead of `v21` only.
- `docs/evidence/t06-mst-hierarchy.md`: one link fixed (`t06-real-video-partition.csv` →
  `sheets/t06-real-video-partition.csv`).

**Moved** (filesystem moves into the corrected evidence structure, content unchanged, confirmed via
line counts before/after):
- `docs/evidence/t06-real-video-partition.csv` → `docs/evidence/sheets/t06-real-video-partition.csv`
- `docs/evidence/t06-cut-edges-v21.csv` → `docs/evidence/sheets/t06-cut-edges-v21.csv` (then
  regenerated in place with the corrected method and the `v29` rows added)
- `docs/evidence/t06-comparison-cases.csv` → `docs/evidence/sheets/t06-comparison-cases.csv` (then
  regenerated in place with the new columns and the `v29` rows added)
- `docs/images/t06-cut-edges-v21.png` → `docs/evidence/images/t06-cut-edges-v21.png`
- `docs/images/t06-truncation-comparison.png` → `docs/evidence/images/t06-truncation-comparison-v21.png`
  (renamed on move, to disambiguate from the new `-v29` figure)
- `docs/images/t06-comparison-cases.png` → `docs/evidence/images/t06-comparison-cases.png` (then
  regenerated in place to include the `v29` cases)
- `docs/images/` removed (empty after the moves above)

**Generated** (new figures, written directly to their final location, nothing to move):
- `docs/evidence/images/t06-cut-edges-v29.png`
- `docs/evidence/images/t06-truncation-comparison-v29.png`

**Temporarily edited and reverted:** `src/tcc_gnn/hierarchy/mst.py` — only in the first pass of
this task (section 5); not touched again in this pass. Confirmed via `git status --short src/` and
`git diff src/` immediately before finishing this pass too (both empty).

## Result

The `v29` verification confirms the T06 port matches `HieTaSumm.Graph` exactly on the production
feature extractor (2048-d), both at the `cutNumber=20` used throughout this evidence and at the
reference's own `bestCutNumber`-derived value (`cutNumber=5`, `alpha=75`) — two independent
acceptance checks, both passing, reported separately per the task's instruction. The scaled-weight
diagnostic (section 3) confirms `v29` does not share `v21`'s all-edges-tied degeneracy (0 of 427
edges truncate to `0`, vs. 742 of 742 on `v21`), though a smaller tie still exists at its own cut
boundary (54 edges tied at the 20th cut's weight, only 1 cut) — consistent with `int()` truncation
losing some precision on any video, just far less destructively than on `v21`. Per the researcher's
decision, none of this changes the baseline's acceptance criterion: every `int()`-truncated
comparison in this file and in `t06-mst-hierarchy.md` matches the reference exactly, on both
videos, at every `cutNumber` tried. The most consequential finding of this pass is methodological,
not about the baseline: the original cut-edge table's method (`bpt_canonical`/`mst_edge_map`)
silently assumed a topology match that only happens to hold on `v21`; the corrected, tree-native
method in `experiments/analyze_mst_cut_edges.py` reproduces the same `v21` results and is now also
correct on `v29`.
