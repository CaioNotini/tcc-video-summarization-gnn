```
Role: Specification of T09: the graph processing (GP) block of SLC-HELM, a GATv2 layer on the time-aware graph, with the transcription of eqs. 1–2, the approved interpretations and the checks.
Entry: ../status.md
Related to: [status.md](../status.md), [phases.md](../phases.md), [decisions.md](../decisions.md), [references.md](../references.md), [architecture.md](../architecture.md), [t09-gatv2-real-video.md](./t09-gatv2-real-video.md)
```

# T09 specification: GATv2 on the time-aware graph

Approved by the researcher on 2026-11-17. The acceptance criterion is in [phases.md](../phases.md),
T09: tests check the shape and the finiteness of the outputs, and the layer runs on one real
video, with the output saved in `evidence/`. CLAUDE.md adds gradient flow and fixed-seed
determinism for model layers.

## Source

SLC-HELM (Batisteli et al., IEEE GRSL 2026, [references.md](../references.md)), Section III-B and
Section IV-A. The PDF was given by the researcher on 2026-11-17 and is not in the repository, so
the equations are transcribed here.

**Embedding (III-B).** "Initial node features h̃ᵢ are first linearly embedded into a d-dimensional
space, giving the initial feature vectors h⁰ᵢ." Section IV-A: "Node and edge features were
embedded into a 64-dimensional space, and a dropout rate of 0.2 was applied throughout the
network."

**Eq. 1.** ĥ⁰ᵢ = Σ_{j ∈ N^{S,0}ᵢ} β⁰ᵢⱼ W⁰ h⁰ⱼ, with N^{S,0}ᵢ = { j | {i, j} ∈ E^S₀ }, where W⁰ is a
learnable linear transformation.

**Eq. 2.** β⁰ᵢⱼ = exp(a⁰ᵀ LeakyReLU(W⁰₁ h⁰ᵢ + W⁰₂ h⁰ⱼ)) / Σ_{j′ ∈ N^{S,0}ᵢ} exp(a⁰ᵀ LeakyReLU(W⁰₁ h⁰ᵢ + W⁰₂ h⁰ⱼ′)),
where a⁰, W⁰₁ and W⁰₂ are learnable.

**After eq. 2.** "A residual connection and batch normalization are applied, giving
h¹ᵢ = BN(ĥ⁰ᵢ + h⁰ᵢ)." No activation is mentioned in the GP block.

**Fig. 2.** The sequence Linear → BN → Graph Conv → BN → Linear → BN is drawn inside the box
labelled "Multigraph Processing Block", so it belongs to the MGP (T15), not to the GP block. The
GP block follows the text above.

## Interpretations (approved by the researcher)

1. **Base graph.** E^S₀ is the time-aware graph of `build_time_aware_graph` with `delta_t = 32`
   (D-24), each undirected edge passed to PyTorch Geometric in both directions.
2. **No self-loops.** N^{S,0}ᵢ contains only the neighbours joined by an edge, so the layer is
   built with `add_self_loops=False`; the node itself enters through the residual.
3. **W⁰ ≡ W⁰₂.** `GATv2Conv` of PyTorch Geometric 2.8.1 uses two matrices: `lin_r` on hᵢ and
   `lin_l` on hⱼ inside the attention, and `lin_l(hⱼ)` as the message. Eq. 1 writes a third matrix
   W⁰ for the message. This is taken as a difference of notation, by the researcher's reading:
   in the original GATv2 (Brody, Alon and Yahav, 2022) the message uses the same matrix that is
   applied to hⱼ in the attention. The GATv2 paper was not checked here.
   `GATv2Conv` is used as it is (D-06), with `share_weights=False`.
4. **No bias.** `bias=False`, literal with eq. 1; the batch normalization that follows would
   cancel a constant bias anyway.
5. **No edge attributes.** `edge_dim=None`: attention uses node features only, as the note of
   D-08 says of the SLC-HELM GATv2 branches. Recorded as a note in D-06.
6. **Heads.** The paper does not give the number of heads. `heads = 4` from `model.gat_heads` of
   `configs/default.yaml`, with `concat=False` (mean over heads), so the output keeps d = 64.
7. **Dropout.** The paper does not say where the dropout of 0.2 is applied. The mHELMNet code of
   the same laboratory (submodule `references/mHELMNet`) was checked and shows no usable
   convention: it has no GATv2 and no heads (its graph layer is `GatedGCNLayer`,
   `references/mHELMNet/model.py:46-50`); its only dropout is inside that layer, after batch
   normalization and activation and before the residual (`references/mHELMNet/gatedgcn.py:80-84`),
   and the layer is built with `dropout=0.` (`references/mHELMNet/model.py:48`). So the dropout of
   0.2 is applied to the block output h¹ (`nn.Dropout`), and the attention dropout of `GATv2Conv`
   is 0. In `eval()` the dropout has no effect.
8. **No training.** The training objective is open (D-17), so T09 runs the block with untrained
   weights from a fixed seed, in `eval()` and without gradients. Its output is not a result of the
   method, and no mlflow run is made (D-12 note).

## Block

`src/tcc_gnn/models/gatv2.py`, `GraphProcessingBlock(in_dim, embedding_size, heads, dropout)`:

```
h⁰ = Linear(in_dim → 64)(x)
ĥ⁰ = GATv2Conv(64 → 64, heads=4, concat=False, add_self_loops=False, bias=False,
               edge_dim=None, share_weights=False, dropout=0)(h⁰, edge_index)
h¹ = Dropout(0.2)(BatchNorm1d(64)(ĥ⁰ + h⁰))
```

`graph_processing_block(config, in_dim)` reads `embedding_size`, `gat_heads` and `dropout` from
the `model` section of the yaml. `time_aware_edge_index(graph)` converts the networkx graph into
the `edge_index` of PyTorch Geometric and refuses an empty graph, nodes other than `0..n-1` and
self-loops.

## Checks

- **Unit tests** (`tests/test_gatv2.py`, CPU): `edge_index` computed by hand and equal to the
  edges of the time-aware graph; the layer configuration above; output shape and finiteness;
  `concat=False` keeps d with 1, 2 and 4 heads; non-zero gradient in every parameter; identical
  output for the same seed in `eval()`; attention only over neighbours and summing to 1 per node;
  the output of a node changes only when a frame inside its window changes; an isolated frame
  keeps only the residual; invalid inputs raise `ValueError`; the block read from the yaml.
- **Real video** (`experiments/run_gatv2_real_video.py`, SumMe `Jumps`): locality and
  determinism on the CPU with an exact criterion (change = 0 outside the window, identical output
  for the same seed); on the GPU only the measurement forward (shape, finiteness, attention sums,
  time with warm-up, peak memory).
- **Why the exact checks run on the CPU.** On the GPU the scatter sums of PyTorch Geometric use
  atomic operations, so the same input gives outputs that differ by about 6e-8 between calls; an
  exact "change = 0" criterion fails there for that reason only. The measurement that shows it is
  in [t09-gatv2-real-video.md](./t09-gatv2-real-video.md).

## Out of scope

Training and loss (D-17), SLC pooling (T13), the MGP (T15), the readout (T17), VSUMM (T10) and any
change to `src/tcc_gnn/graph/` (D-02).

## Pending for T13

Reproducibility of training on the GPU (CLAUDE.md rule 10): test whether
`torch.use_deterministic_algorithms(True)` works with the PyTorch Geometric operations used, and
how much it costs. Not tested in T09.
