```
Role: T04 deliverable. Specifies the input, output, cut criteria and test cases of the MST hierarchy, checked against the HieTaSumm-lib submodule.
Entry: ../status.md
Related to: [status.md](../status.md), [phases.md](../phases.md)
```

# T04: MST hierarchy specification

Written on 2026-10-21, repository commit `6fcb61b`, submodule `references/HieTaSumm-lib` pinned at
`a554ed9` (`heads/main`), already initialized and populated. No submodule command was run; the
existing checkout was read only, for analysis. `references/` itself was not modified.

This specifies the MST hierarchy exactly as computed by `HieTaSumm-lib`, Leonardo's reference
implementation, per T04's goal in [phases.md](../phases.md): the input, the output, the cut
criteria and the test cases, before any code is written.

**Scope decision (researcher, 2026-10-21):** T06 implements only HieTaSumm's single final cut
(the MST hierarchy built with higra, cut the same way `cutNumber` cuts it in `HieTaSumm-lib`),
producing one partition. It does not implement a multi-level output. The "set of levels" shared
interface in [architecture.md](../architecture.md) applies only to SLC pooling (T12/T13), not to
the MST hierarchy.

## Files read in the submodule

- `HieTaSumm/Graph.py` (90 lines): `compute_hierarchy`, `cut_graph`.
- `HieTaSumm/Summary.py` (260 lines): `__init__`, `bestCutNumber`, `rgbSim`, `spatialSim`.
- `HieTaSumm/Frame.py` (65 lines): `load`, `compute_similarity_img`, `calculate_similarity`,
  `calc_end`.
- `HieTaSumm/Files.py` (59 lines): `save_graph_data`, `read_graph_file`.
- `HieTaSumm/__init__.py`: `HieTaSumm.__init__`, `hierarchical_summarization`.
- `HieTaSumm/options.json`: the library's default parameters.
- `higra.watershed_hierarchy_by_area`'s docstring, read from the installed package
  (`python -c "import higra as hg; help(hg.watershed_hierarchy_by_area)"`), to confirm the
  algorithm it implements (Cousty and Najman, ISMM 2011; Najman, Cousty and Perret, ISMM 2013) and
  that it defaults to a uniform `vertex_area` when none is given.

## Input

- The time-aware graph produced by `build_time_aware_graph`
  ([`src/tcc_gnn/graph/frame.py`](../../src/tcc_gnn/graph/frame.py), validated in T01): a
  `networkx.Graph` whose nodes are the integers `0..n_frames-1` (frame indices, in temporal order)
  and whose edges carry a `weight` float attribute (the L1 feature distance). In `HieTaSumm-lib`,
  the equivalent object is `Frame.load`'s output (`Frame.py`), consumed as `tree = RG` in
  `Summary.__init__` (`Summary.py:64`) and passed straight into `Graph.compute_hierarchy`.
- `hierarchy = 'watershed_hierarchy_by_area'`: the only mode ever configured, in
  `options.json` (`"hierarchy": "watershed_hierarchy_by_area"`), and the one the HieTaSumm paper
  and this project's documents describe as "a minimum spanning tree cut by area."
- `is_binary = True`: `options.json`'s default (`"is_binary": true`).
- No `vertex_area` is passed to higra: `Graph.compute_hierarchy` calls
  `hg.watershed_hierarchy_by_area(graph, edge_weights)` with no `vertex_area` argument, so higra
  uses the default, uniform leaf area (one unit per original frame; a merged node's area is the
  number of frames it contains).

**Note: label mismatch in `Frame.load`.** `Frame.load` labels graph nodes with full file paths
(`frame1 = os.path.join(video_file, frame_list[vertex1])`, `Frame.py:23-24`), not integer indices.
`Graph.compute_hierarchy` assumes integer node labels (`graph.add_vertices(max(input_g._node)+1)`,
`Graph.py:51`): `max(...)+1` cannot add `1` to a string. So, as written, `HieTaSumm-lib` cannot run
`compute_hierarchy` on the graph `Frame.load` actually builds; this is a pre-existing inconsistency
in the submodule, not something introduced here. `experiments/validate_graph.py` (T01) already hit
the same inconsistency and worked around it by calling `Frame.calc_end` and
`Frame.calculate_similarity` directly on integer-indexed feature arrays, never `Frame.load`. T06
must use the same, already-established convention: the integer-indexed graph from
`build_time_aware_graph`, and any direct comparison against the reference must call
`Graph.compute_hierarchy` directly, not through `Frame.load`.

## Output

One partition of the frames into groups, each original frame linked to the id of the cluster it
ends up in after the single cut, matching `HieTaSumm-lib`'s own final result exactly:

1. **Tree construction (intermediate step).** `Graph.compute_hierarchy` (`Graph.py:48-90`) builds a
   higra tree over `n_frames` leaves (the original frames) plus internal nodes (merged regions),
   with an altitude per node from the watershed-by-area computation. Because `is_binary = True`,
   the canonized tree is converted to a strict binary tree with `hg.tree_2_binary_tree`
   (`Graph.py:76-81`). Every tree node, visited bottom-up
   (`tree.leaves_to_root_iterator()`), is recorded as a triple `(node, parent, leaf_flag)`
   (`Graph.py:83-88`): `leaf_flag = -1` for a leaf (an original frame), `-2` for an internal node.
   `compute_hierarchy` also returns `leaf_list`, the ids of the leaf nodes. T06 must build this same
   tree; it is not the final deliverable, only the structure the single cut is read from.
2. **Single cut (the deliverable).** `Graph.cut_graph` (`Graph.py:18-46`) reads that tree top-down,
   from the root, opening `cutNumber` merges before stopping (see "Cut rule" below), and returns a
   graph whose connected components are the final groups. T06's output is this partition: for every
   original frame (filtered against `leaf_list`, so merged/internal nodes are excluded), the id of
   the connected component it belongs to. Two frames share a cluster id if and only if they are in
   the same connected component after the cut.
3. Each cluster is a contiguous range of frame indices: this follows only from the input graph
   having exclusively temporal edges, independent of the hierarchy algorithm.

**Fidelity detail, registered as baseline behaviour.** `Graph.compute_hierarchy` (`Graph.py:60-61`)
casts every edge weight to `int` before building the higra graph:
`edge_weights[i] = int(input_g.adj[...]["weight"])`. This is a truncation, not a rounding, and
changes the watershed result compared with using the float weight directly. T06 reproduces this
truncation, because it is part of the HieTaSumm baseline's logic and D-02 requires ported logic to
stay faithful to the original, names included (`v1`, `v2`, `leaf_list`, `cutNumber`, `cut_graph`,
`compute_hierarchy`).

## Cut rule (`cutNumber`)

`Graph.cut_graph(file, cut_graph_file, cutNumber)` (`Graph.py:18-46`) walks the tree from the root
(the last entry written by `compute_hierarchy`) back toward the leaves, keeping a `cut_list` of
nodes already "opened" starting from the root itself. At each node: if its parent is already in
`cut_list` and `cutNumber > 0`, the node is opened too and `cutNumber` is decremented; once
`cutNumber` reaches `0`, the node instead becomes an isolated node in the output graph (a cut
boundary), and everywhere else the node is reconnected to its parent as an edge. The connected
components of the resulting graph are the final groups.

**Source of the `cutNumber` value**, from `Summary.__init__` (`Summary.py:52-59`):
- If `gen_summary_method = 'n_fixed_keyframes'`: `cutNumber = n_keyframes - 1`, where
  `n_keyframes` is a parameter of `gen_summary_method` (a user-supplied target keyframe count).
- If `gen_summary_method` is `'group_central_frames'` or `'percent_spaced'` (HieTaSumm's own
  default, per `options.json`): `cutNumber = int(bestCutNumber() * alpha / 100)`, floored to a
  minimum of 3. `bestCutNumber` (`Summary.py:104-125`) computes, for every pair of frames inside
  the temporal window, a raw-pixel RGB cosine similarity (`rgbSim` + `spatialSim`; distinct from
  the ResNet-feature edge weights used to build the graph), takes the standard deviation of that
  similarity distribution across all pairs, scales it by 10 repeatedly until it is at least 1, and
  rounds it down by one.
- `options.json` defaults, used when no parameter is supplied: `percent = 15`, `alpha = 75`,
  `gen_summary_method = {"method": "group_central_frames", "n_keyframes": 7}`.

T06's `cutNumber` must come from the same source: either a `n_keyframes`-style parameter, or
`bestCutNumber() * alpha / 100` with the `options.json` defaults, so the baseline's single cut
matches Leonardo's method exactly, not a new rule.

## Test cases

1. Synthetic chain graph (6-8 integer-indexed frames, all pairwise edges inside a small `delta_t`,
   hand-picked weights) with a hand-computed watershed-by-area tree: verify leaf count equals
   `n_frames`, every non-root node has exactly one parent, leaves are flagged `-1` and internal
   nodes `-2`, and the tree is a strict binary tree (every internal node has exactly two children).
2. The same synthetic graph with float weights that truncate differently than they round (e.g.
   `2.9`, `3.4`): verify the computed tree matches truncating to `int` before calling higra, not
   rounding.
3. A disconnected synthetic graph (two groups of frames outside each other's `delta_t` window):
   document the expected behaviour (a forest, not a single tree). This case only needs a documented
   expectation, since the real graph is always connected for `delta_t >= 1` (every frame links to
   the next one).
4. Determinism: running the hierarchy twice on the same graph gives an identical tree and
   altitudes, and the single cut gives an identical partition for the same `cutNumber`.
5. Single cut on a small synthetic tree: for a hand-picked `cutNumber`, hand-trace
   `cut_graph`'s top-down walk and verify the resulting connected components (the partition) match.
6. Real video: on one of the videos already used in T01 (`v21` or `v29`), compute the hierarchy
   with higra's `watershed_hierarchy_by_area` on the integer-indexed graph from
   `build_time_aware_graph`, truncating weights to `int`, and compare node-for-node (parent links,
   altitudes, leaf flags) against `HieTaSumm.Graph.compute_hierarchy` called directly on the same
   integer-indexed graph (not through `Frame.load`, for the reason noted under "Input" above).
7. Full cut, on the same real video: compute `cutNumber` from `bestCutNumber() * alpha / 100` with
   the `options.json` defaults (`alpha = 75`), run `Graph.cut_graph`, and compare the resulting
   partition against the same computation run through the reference `Graph.cut_graph` directly.
8. Contiguity: for the real-video partition, verify that the frames in every cluster form a
   contiguous range of frame indices.

## Validation against HieTaSumm-lib

Each section above was re-checked line by line against the submodule:

- **Input:** matches `compute_hierarchy`'s assumptions and `options.json`'s fixed `hierarchy` and
  `is_binary` values.
- **Output:** the tree, altitudes, binary conversion, leaf/internal flags and the single-cut
  partition match `Graph.py:48-90` exactly; the output is explicitly one partition, not several
  levels, per the researcher's scope decision above.
- **Fidelity detail:** the `int()` truncation at `Graph.py:60-61` is quoted verbatim and kept as
  required baseline behaviour.
- **Cut rule:** `cutNumber`'s two sources, their file/line references, and the default parameter
  values (`percent=15`, `alpha=75`, `n_keyframes=7`, default method `group_central_frames`) match
  `options.json` and `Summary.py:52-59, 104-125` exactly.
- **Test cases 6-7 (real-video comparison):** confirmed runnable — `higra` is installed in the
  project environment (`python -c "import higra"` succeeds) — and consistent with how T01's own
  `experiments/validate_graph.py` already bypasses `Frame.load` for the same node-labelling reason.

No contradiction was found between this specification and the submodule's code.

## Result

This file is the T04 deliverable. It specifies the input, the output (one partition, each frame
linked to its cluster), the cut rule (`cutNumber`, with its source) and the test cases of the MST
hierarchy, checked against `HieTaSumm-lib` with no contradiction found. The scope point raised in
the previous version of this file (whether T06 needs a multi-level output) is resolved by the
researcher's decision above: T06 implements only the single final cut. `status.md`,
`decisions.md` and `architecture.md` are left unchanged, pending the researcher's approval of this
specification.
