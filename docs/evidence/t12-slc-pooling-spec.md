```
Role: Specification of T12: the single-link clustering (SLC) pooling of SLC-HELM adapted to the time-aware graph, with the transcription of Section III-C, the contiguity rule, the edge cases, the test cases for T13 and the open points.
Entry: ../status.md
Related to: [status.md](../status.md), [phases.md](../phases.md), [decisions.md](../decisions.md), [references.md](../references.md), [architecture.md](../architecture.md), [t09-gatv2-spec.md](./t09-gatv2-spec.md)
```

# T12 specification: SLC pooling on the time-aware graph

Written on 2026-11-17, repository commit `0b44754`. Approved by the researcher on 2026-11-17. The
consecutive-edge rule (D-27) and the threshold (Q-02) are pending the advisor. The goal of T12 is in
[phases.md](../phases.md): the specification of SLC pooling, its test cases and how the threshold is
chosen (Q-02). It specifies T13 and does not implement anything.

## Source

SLC-HELM (Batisteli et al., IEEE GRSL 2026, [references.md](../references.md)), Sections III-B,
III-C and IV-A. The PDF was given by the researcher on 2026-11-17 and is not in the repository,
so the text and the equations are transcribed here. Notation: R is the node set, E^S the spatial
edges, E^H the hierarchical edges, G^R_j the graph of scale j.

**Section III-C (full text, equations in plain notation).**

> At each hierarchy-expansion step, the SLC operator is applied to the current (i.e., most
> recent) graph scale G^R_{j−1} = (R_{j−1}, E^S_{j−1}), or the base RAG when j = 1, to generate a
> coarser representation. The operator groups nodes based on pairwise feature distances evaluated
> along existing edges, using a data-driven percentile threshold. Following the single-link
> criterion, two nodes r_i, r_k ∈ R_{j−1} are assigned to the same cluster if there exists a path
> between them in G^R_{j−1} such that all edges along the path have feature distances below the
> defined threshold δ. Figure 3 illustrates the application of two successive pooling steps to a
> given graph.
>
> Cluster assignment is determined by the connected components of the subgraph induced by those
> short edges, producing a partition P_j of R_{j−1}. Each cluster r_k ∈ R_j is represented by the
> mean of the feature vectors of its constituent nodes:
>
> **(6)** h^{(j)}_{r_k} = 1 / |{ r_i ∈ R_{j−1} | ϕ_j(r_i) = r_k }| · Σ_{r_i ∈ R_{j−1}, ϕ_j(r_i) = r_k} h^{(j−1)}_{r_i},
>
> where ϕ_j : R_{j−1} → R_j is the cluster assignment map induced by P_j.
>
> After each pooling step, both edge sets are updated to incorporate the new scale. The spatial
> edge set is extended by remapping each edge {r_i, r_k} ∈ E^S_{j−1} to its cluster-level
> counterpart, an edge {ϕ_j(r_i), ϕ_j(r_k)} is added to E^S_j whenever ϕ_j(r_i) ≠ ϕ_j(r_k). The
> accumulated spatial edge set covering all L scales is then E^S = ⋃_{j=0}^{L} E^S_j.
>
> The hierarchical edge set is augmented with bidirectional edges connecting every node
> r_i ∈ R_{j−1} to its parent ϕ_j(r_i) ∈ R_j:
>
> **(7)** E^H_j = { {r_i, ϕ_j(r_i)} | r_i ∈ R_{j−1} },
>
> so the full hierarchical edge set becomes E^H = ⋃_{j=1}^{L} E^H_j.

**Fig. 3.** Two pooling steps on a graph of 7 nodes: G^R_1 has the clusters {1,2}, {3}, {4–7};
G^R_2 has {1–3}, {4–7}.

**Section III-B (the parts the operator depends on).** "At each hierarchy-expansion step, a
Single-Link Clustering (SLC) pooling operator (Section III-C) is applied to the current
highest-resolution graph scale G^R_{j−1}, producing a coarser graph G^R_j whose nodes (R_j),
spatial edges (E^S_j), and hierarchical edges (E^H_j) are appended to the accumulated node and edge
sets." "Rather than replacing the previous scale, each pooling step appends the new cluster nodes
to the existing node set." The accumulated sets are R^ℓ = ⋃_{j=0}^{ℓ} R_j,
E^{S,ℓ} = ⋃_{j=0}^{ℓ} E^S_j and E^{H,ℓ} = ⋃_{j=1}^{ℓ} E^H_j. Fig. 2 gives the sequence
GP → SLC → MGP → SLC → MGP → SLC → RGR → CL.

**Section IV-A (the parts the operator depends on).** "The SLC-HELM model used L = 3 hierarchical
levels beyond the base graph […]. At each pooling step, edges were filtered using a threshold δ set
to the 50th percentile of pairwise Euclidean distances at the current scale, which controls node
merging and avoids excessive early coarsening, leading to a gradual reduction in graph size."
"Node and edge features were embedded into a 64-dimensional space."

## Adaptation to the temporal graph: clustering on consecutive edges (researcher, 2026-11-17)

**Rule.** At level 0 the clustering thresholds only the edges between consecutive frames
(i, i+1). At level j ≥ 1 it thresholds only the edges between consecutive clusters (k, k+1). The
window edges of `delta_t = 32` (D-24) stay as the spatial edges E^S that GATv2 (T09) and the MGP
(T15) use. They are not used by the clustering.

**Justification (researcher).** In SLC-HELM the edges of the base graph (the RAG) join adjacent
regions, so each node has few neighbours. In a video, the equivalent of "adjacent" is the
consecutive frame (i, i+1). The window of `delta_t` is a choice of HieTaSumm for the context seen
by the attention layer. So this rule translates the idea of the paper to the temporal domain. It
keeps contiguity by construction and should give the gradual coarsening that the paper describes.
Recorded as D-27 in [decisions.md](../decisions.md), open until the advisor confirms it.

**Why the window edges do not give contiguity.** `build_time_aware_graph` joins every pair with
0 < j − i < `delta_t` ([frame.py](../../src/tcc_gnn/graph/frame.py)), not only consecutive
frames. A short edge can skip frames: if the edge 0–5 is below δ and every edge from 1–4 to 0 or 5
is above it, then {0, 5} is one cluster and {1–4} lies inside it. The single-link rule of the paper
applied to the window graph therefore does not produce contiguous clusters in general. Test case 6
below gives the numbers.

**Contiguity holds by construction under the rule.** The nodes of level j − 1 are ordered in time,
and the clustering graph is the path 0 – 1 – … – (n_{j−1} − 1). The connected components of a
subgraph of a path are runs of consecutive nodes. So ϕ_j is non-decreasing, with ϕ_j(0) = 0 and
ϕ_j(k+1) − ϕ_j(k) ∈ {0, 1}. Clusters are numbered in temporal order. By induction over the levels,
every node of every level covers a contiguous interval of frames, and the intervals of one level are
disjoint and ordered. The remapped path of level j − 1 is the path of level j, so the rule needs no
extra edge set.

T13 must still check contiguity, as its acceptance criterion requires, on synthetic inputs and on a
real video. A test is needed because the property depends on the implementation using the path and
numbering the clusters in order.

## The operator

**Input of step j (j = 1, …, L):**

| Name | Shape | Content |
| ---- | ----- | ------- |
| `h` | (n_{j−1}, d), float | features of the nodes of the most recent level, rows in temporal order; d = 64 |
| `edge_index` | (2, 2·\|E^S_{j−1}\|), int64 | window edges of level j − 1, local indices, both directions (PyTorch Geometric) |

The path edges are not an input: they are (k, k+1) for k = 0, …, n_{j−1} − 2.

**Computation:**

1. d_k = ‖h_k − h_{k+1}‖₂ for every consecutive pair (Euclidean, as in IV-A).
2. δ_j = threshold of the current level, from the d_k (rule open, Q-02; the paper's rule is the
   50th percentile).
3. Short edge: d_k < δ_j (comparison open, see the open points).
4. ϕ_j(0) = 0 and ϕ_j(k+1) = ϕ_j(k) + [edge k is not short]. n_j = ϕ_j(n_{j−1} − 1) + 1.
5. H_j by eq. 6: the mean of the rows of `h` with the same ϕ_j. It is the mean of the level j − 1
   nodes, not of the frames they cover.
6. E^S_j: each window edge (a, b) of level j − 1 becomes (ϕ_j(a), ϕ_j(b)). Self-loops are removed
   and duplicates are merged.
7. E^H_j: the pairs (k, ϕ_j(k)) for k = 0, …, n_{j−1} − 1, one per node, so a singleton cluster also
   gets a hierarchical edge to its copy.

**Output of step j:**

| Name | Shape | Content |
| ---- | ----- | ------- |
| `h_new` | (n_j, d) | cluster features, eq. 6 |
| `assignment` | (n_{j−1},), int64 | ϕ_j, non-decreasing |
| `edge_index_new` | (2, 2·\|E^S_j\|), int64 | window edges of level j, local indices, both directions |
| `hier_index` | (2, 2·n_{j−1}), int64 | E^H_j in accumulated indices, both directions |

**Accumulation (III-B).** Level j occupies the rows o_j, …, o_j + n_j − 1 of the accumulated feature
matrix (N_ℓ, d), with o_0 = 0, o_j = n_0 + … + n_{j−1} and N_ℓ = n_0 + … + n_ℓ. A local index k of
level j becomes o_j + k. E^H_j joins o_{j−1} + k to o_j + ϕ_j(k). Accumulated spatial edges are the
union of the E^S_j shifted by o_j. One video is one graph, so δ_j is computed per video and per level.

**Gradient.** Steps 1 to 4 decide a discrete partition and carry no gradient. The gradient reaches
`h` only through the mean of eq. 6. The paper does not discuss this (open point 6).

## Position in the network

The input of step 1 is h¹ = GP(x), the output (n_0, 64) of the GP block of T09
([t09-gatv2-spec.md](./t09-gatv2-spec.md)), with n_0 the number of sampled frames. It comes with the
window `edge_index` of T09. In the full network (Fig. 2), steps 2 and 3 receive the MGP output (T15)
for the nodes of the most recent level. The MGP does not exist in T13, so T13 chains the three steps
directly on the GP output. That chain is a test of the operator, not the method. L = 3, as in the
paper; with the rule above and δ = median, the levels keep about 1/2, 1/4 and 1/8 of the frames
(first probe below).

## Edge cases

| Case | Under the consecutive-edge rule | Behaviour specified |
| ---- | ------------------------------- | ------------------- |
| Disconnected graph | The clustering path is always connected when n ≥ 2. A window graph with `delta_t` ≥ 2 always contains the path. With `delta_t` ≤ 1, `build_time_aware_graph` makes no edges at all. | The operator does not read the path from `edge_index`, so it does not depend on it. A short-edge subgraph with several components is the normal case: each component is a cluster. |
| All nodes in one cluster | With `<` and the median, impossible for n ≥ 2: at most ⌊(n−1)/2⌋ of the n − 1 edges are below their median. With `≤` it happens when all distances are equal (test case 2). | Valid output: n_j = 1, a single level-j node, every level j − 1 node with a hierarchical edge to it, E^S_j empty. |
| No pair below the threshold | With `<`, happens when all distances are equal or when n = 2 (test cases 2 and 3). | Valid output: n_j = n_{j−1}, ϕ_j the identity, H_j = h, E^S_j = E^S_{j−1}, one hierarchical edge per node to its copy. |
| Isolated frame | No frame is isolated in the path for n ≥ 2. A frame whose two consecutive edges are both long becomes a singleton cluster. | Valid output: its mean is its own vector and it has one hierarchical edge to its copy. |
| Fewer than 2 nodes | No consecutive edge exists, so δ is undefined. | Proposed: `ValueError`. The alternative is to treat it as "no pair below the threshold". Open for approval; it cannot arise from a level of 2 or more nodes under `<`. |
| Invalid input | `h` not 2-D, non-finite values, `edge_index` with indices outside 0..n−1 or with self-loops | `ValueError`. |

## Test cases for T13

Values computed by hand and checked with numpy (`.cache/scratch/t12_hand_check.py`). Features have
d = 1 so the arithmetic is visible. T13 also runs them with d = 64 by repeating the column, which
multiplies each Euclidean distance by √64 = 8. The partitions are expected to stay the same, and
T13 checks it. The decimals below are not exact in float (for example 5.1 − 1.25 =
3.8499999999999996). With an odd count, the median is one of the distances, so the median edge
equals δ bit for bit and `<` excludes it. Tests compare against the computed distances, not
against literal decimals.

**1. Three levels, `<`, median.** h = [0, 1, 1.5, 5, 5.2, 9].
- Level 1: d = [1, 0.5, 3.5, 0.2, 3.8], median 1.0 (odd count, so every percentile method agrees).
  Short edges: 1–2 and 3–4. ϕ₁ = [0, 1, 1, 2, 2, 3], H₁ = [0, 1.25, 5.1, 9].
- Level 2: d = [1.25, 3.85, 3.9], median 3.85. Short: 0–1. ϕ₂ = [0, 0, 1, 2], H₂ = [0.625, 5.1, 9].
  In frames: {0, 1, 2}, {3, 4}, {5}.
- Level 3: d = [4.475, 3.9]. With the linear percentile (the numpy default) the median is 4.1875
  and the edge 1–2 is short: ϕ₃ = [0, 1, 1], H₃ = [0.625, 7.05]. With the "lower" method the median
  is 3.9, nothing is short and H₃ = H₂.
- H₃[1] = 7.05 is the mean of 5.1 and 9 (eq. 6 over the level 2 nodes). The mean over frames 3, 4
  and 5 would be 6.4. The test separates the two.
- With `≤` at level 1, edge 0–1 (d = 1 = δ) is short as well: ϕ₁ = [0, 0, 0, 1, 1, 2],
  H₁ = [0.8333…, 5.1, 9].

**2. Equal distances.** h = [0, 0, 0, 0] (and h = [0, 1, 2, 3]). Every d is equal to the median.
With `<`: n₁ = 4, the identity. With `≤`: n₁ = 1. A static video never merges under `<`
(open point 2).

**3. Two nodes.** h = [0, 1]: one edge, median 1. With `<`: no merge. With `≤`: one cluster.

**4. Edges of level 1, from case 1.** Six frames with `delta_t = 4`, so 0 < j − i ≤ 3: E^S₀ has 12
edges. With ϕ₁ = [0, 1, 1, 2, 2, 3], E^S₁ = {(0,1), (0,2), (1,2), (1,3), (2,3)}, 5 edges. (1,2) and
(3,4) collapse to self-loops and are removed; (1,3), (1,4), (2,3) and (2,4) all become (1,2). The
path of level 1, {(0,1), (1,2), (2,3)}, is contained in E^S₁.

**5. Hierarchical edges with accumulated indices, from case 1.** o₁ = 6, o₂ = 10. E^H₁ = {(0,6),
(1,7), (2,7), (3,8), (4,8), (5,9)}, so `hier_index` has shape (2, 12). E^H₂ = {(6,10), (7,10),
(8,11), (9,12)}.

**6. Skipping edge (why the rule is needed).** h = [0, 10, 10, 10, 10, 0.1], with the window graph
of `delta_t = 32` (15 edges).
- The literal rule of the paper over the window edges: distances are 0 six times (inside 1–4),
  0.1 once (0–5), 9.9 four times and 10 four times. The median is 9.9, so the short edges are the
  six 0s and 0.1. Clusters: {1, 2, 3, 4} and {0, 5}, which is not contiguous.
- The consecutive-edge rule: d = [10, 0, 0, 0, 9.9], median 0. With `<`: nothing merges. With
  `≤`: {0}, {1–4}, {5}.
- In both comparisons, frames 0 and 5 are never in the same cluster.

**7. Contiguity property.** For random features (several seeds and sizes, including n = 2 and n = 3)
and both comparisons, at every level: ϕ_j(0) = 0, steps in {0, 1}, every node covers an interval of
frames, and the intervals of a level are disjoint, ordered and cover 0..n₀−1.

**8. Gradient.** In case 1, with `h` requiring gradient and the loss Σ H₁, the gradient of h is
[1, 0.5, 0.5, 0.5, 0.5, 1] (1/|cluster|). The partition does not change when `h` changes inside the
same order of distances.

**9. Determinism.** The same input gives the same `assignment`, `edge_index_new` and `hier_index`
in repeated calls on the CPU. The GPU check of `h_new` uses the tolerance of the T09 noise.

**10. Invalid inputs.** Fewer than 2 nodes (if `ValueError` is approved), `h` not 2-D, NaN or inf in
`h`, `edge_index` outside 0..n−1 or with self-loops.

**11. Real video (T13 acceptance).** GP output on SumMe `Jumps`, three steps: the number of nodes
per level, 0 non-contiguous clusters checked against the frames, and finite `h_new`. Output saved in
`evidence/`.

## Probes run while writing this specification (not evidence)

Read-only runs in `.cache/scratch/`, on the raw ResNet50 features of the 25 SumMe videos (not the
GP output), `delta_t = 32`, `<`. They are not results of the method. Their only role is to show the
size of the effects described in the open points.

| Probe | Script | Level 1 clusters (total over 25 videos) | Non-contiguous | Videos affected |
| ----- | ------ | --------------------------------------- | -------------- | --------------- |
| Paper's rule on window edges, δ = median of window edges | `slc_contiguity_probe.py` | 748 | 52 | 23 of 25 |
| Paper's rule on window edges, δ = median of all pairs | same | 79 | 15 | 12 of 25 |

On the window edges a single step collapses the video: `Air_Force_One` goes from 716 frames to 4
clusters (median of window edges) or to 1 (median of all pairs).

Consecutive-edge rule, δ = median of the consecutive edges at each level
(`slc_option_a_probe.py`):

| Level | Mean nodes | Min | Max | Non-contiguous clusters |
| ----- | ---------- | --- | --- | ----------------------- |
| 0 (frames) | 636.2 | 152 | 1552 | n/a |
| 1 | 319.1 | 77 | 777 | 0 |
| 2 | 160.0 | 39 | 389 | 0 |
| 3 | 80.7 | 20 | 195 | 0 |

At 4 frames per second (D-25), the mean cluster lasts about 0.5 s at level 1, 1 s at level 2 and
2 s at level 3.

## Open points (to be decided before T13)

1. **Threshold δ (Q-02, D-19).** Not decided here. What the paper says:
   - δ is the 50th percentile of the pairwise Euclidean distances at the current scale.
   - It is recomputed at each pooling step, so it is data-driven and not learned.
   - It is computed per graph, which is one image in the paper.
   - It is a fixed hyperparameter, with no ablation reported.
   - The paper does not say whether "pairwise" means the edges or every pair of nodes.

   Plausible options for this thesis:

   | Option | For | Against |
   | ------ | --- | ------- |
   | a. 50th percentile of the consecutive edges, per video and per level | Literal translation of the paper to the consecutive-edge rule. Gradual reduction (probe above). No parameter fitted on data. | With distinct distances, n_j = n_{j−1} − ⌊(n_{j−1}−1)/2⌋, about half, whatever the content: the data decide where the cuts fall, not how many. Example: 716 → 359 = 716 − 357. |
   | b. Another fixed percentile p of the consecutive edges | Same simplicity. p sets the reduction rate (about a fraction p of the edges merge). | The count is still fixed by n and p. p must be chosen before any result (D-13), and on which data? |
   | c. Percentile of the window edges or of all pairs, applied to the consecutive edges | The count of clusters depends on the content. Closer to the "pairwise" wording. | Those medians are expected to be larger than the consecutive median, because distances tend to grow with the gap, so more edges would merge. Not measured under this rule. |
   | d. One global δ, fixed from a training set | The count depends on the content, and every video uses the same scale. | The features change during training, so a fixed δ drifts. It needs a fitting set and a schedule. It departs from the paper. |
   | e. Learned δ | Fits the "learned hierarchy" claim. | The assignment is not differentiable, so it needs a relaxation that the paper does not have. Out of reach for TCC1. |

2. **`<` or `≤`.** The paper says "below". The two differ when distances tie (test cases 1 to 3). A
   static stretch has many distances equal to the median. Under `<` it never merges when every
   distance equals δ. Under `≤` it merges completely.
3. **Percentile method.** Not given in the paper. It changes δ when the count is even (test case 1,
   level 3: 4.1875 with linear, 3.9 with lower).
4. **Which level is pooled.** III-B says "highest-resolution graph scale G^R_{j−1}", the
   introduction says "the coarsest graph level", and III-C says "the current (i.e., most recent)
   graph scale". Reading: the most recent level, which is the coarsest so far. This is consistent
   with Fig. 2 and Fig. 3.
5. **Distance.** Euclidean, on the current features of the level (GP or MGP output for those
   nodes, eq. 6 means at higher levels). It is not the L1 weight of the HieTaSumm graph
   (`edge_weight` in [frame.py](../../src/tcc_gnn/graph/frame.py)), which stays only in the baseline.
6. **Gradient.** The assignment is not differentiable. The gradient passes only through the mean of
   eq. 6. The hierarchy is "learned" only in the sense that the features that define it are learned.
   This bears on the objective (Q-03) and on how the thesis text describes the hierarchy.
7. **Edge features at the new levels.** Not defined by the paper. IV-A says that node and edge
   features are embedded into 64 dimensions, but the GATv2 of T09 uses none (`edge_dim=None`, D-06
   note). This specification creates no edge features. It is linked to the T15 note in
   [status.md](../status.md) and to D-08 (under review).
8. **Fewer than 2 nodes.** `ValueError` or the identity (edge cases above).

## T13 dependencies still open

| Item | Owner | What it blocks in T13 |
| ---- | ----- | --------------------- |
| Q-02 / D-19, with open points 2 and 3 | researcher | The threshold of the layer: without it the operator has no δ, and test cases 1 to 3 cannot fix a single expected answer. |
| Q-03 / D-17 | advisor | Training. [phases.md](../phases.md) makes T13 depend on an answer or a recorded decision. Its acceptance criterion (synthetic tests, contiguity on a real video) can be met untrained, as in T09. With open point 6, the objective decides whether the hierarchy changes during training at all. |
| Q-04 | advisor | Whether the layer is checked against the SLC-HELM code (the paper gives github.com/IMScience-PPGINF-PucMinas/SLC-HELM, not confirmed). If it exists, points 2, 3 and 4 can be read from it; if not, T13 implements from this specification. |
| Consecutive-edge rule (D-27, open) | researcher with the advisor | The rule of this specification. It also requires the corrections to the documents that state contiguity. |
| GPU determinism (status blockers) | researcher | Reproducible training on the GPU. Not needed for the untrained tests. |

Q-11 (the SLC level used as the segmentation) does not block T13. It is affected by the cluster
durations above: about 0.5 s at level 1 and 2 s at level 3.

## Out of scope

The implementation (T13), training and the loss (D-17), the MGP (T15), the integration (T16), the
readout and keyframe selection (T17), VSUMM (T10), any change to `src/tcc_gnn/graph/` (D-02), and the
choice of δ (Q-02).
