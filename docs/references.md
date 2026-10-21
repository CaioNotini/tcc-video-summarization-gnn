```
Role: Lists the works the thesis relies on, grouped by role, and explains where each one enters the project and which task uses it.
Entry: docs/status.md
Related to: [status.md](./status.md), [phases.md](./phases.md)
```

# References and their role in the work

This document explains how the literature enters the project. For each work, it says what the thesis takes from it, which component it justifies, and which task (T01 to T19, as defined in [phases.md](./phases.md)) uses it. Read it when you need to know why a component exists, or where a design choice came from.

Works are grouped into three roles. Base works supply the method itself: the thesis either implements them or adapts their code, so they must be read in full before the related task starts. Comparison works supply baselines, metrics or design ideas that the thesis measures itself against or borrows from, without being the foundation. Auxiliary works are read and may be used, but the thesis does not depend on them. Each entry below notes whether the work is already cited in the Overleaf document, where the bibliography is maintained. The numbering [1] to [16] refers to the citation order in that document.

The citation marks below were last checked against the Overleaf bibliography on 2026-10-21, and this date must be updated whenever they are checked again. At that check, sixteen works had bibliography entries, and all sixteen are cited in the thesis body text with a matching `\cite{}`: HieTaSumm [1], HELMNet [2], HRG/HRGNet [3], mHELMNet [4], SLC-HELM [5], VideoSAGE [6], VOGNet [7], SumGraph [8], RSGN [9], DMoN [10], DMoN-DPR [11], DiffPool [12], MinCutPool [13], the video summarisation survey [14], GATv2 [15] and ASAP [16]. [6] through [16] were added on 2026-10-20 and 2026-10-21: the bibliography entries first, under the scope restriction in [CLAUDE.md](../CLAUDE.md) rule 15 (references section only), then the `\cite{}` markers in the body paragraphs once the researcher explicitly extended the authorisation to that section too. T02's acceptance criterion ("at least ten works are cited in the thesis text") is met: sixteen works are cited.

## Base works

### HieTaSumm

Cardoso, Gomes, Guimarães and Patrocínio Júnior, BRACIS 2023. Cited as [1].

This is the foundation of the whole thesis. It provides two things. The first is the time-aware sparse graph, in which each video frame is a node and only temporally close frames are connected. The second is the fixed hierarchy, built as a minimum spanning tree and cut by area, which is used only for comparison: the original HieTaSumm method is the baseline. Its reference implementation, the `HieTaSumm-lib` submodule, is the ground truth for validation: the graph built in this repository must reproduce its output, and that is the purpose of T01. The same library is used for the SumMe data in T05. The method is evaluated on VSUMM with F-score, which is the metric the thesis keeps as its primary result.

Used in: T01, T05, T06, T07, T19.

### SLC-HELM

Batisteli, Kenmochi, Bougleux, Lézoray, Guimarães and Patrocínio Júnior, IEEE GRSL 2026. Cited as [5].

This is the source of the main method. It introduces single-link clustering pooling, which learns the hierarchy during training instead of computing it beforehand. The thesis adopts this idea, and the rule it follows is simple to state: two frames belong to the same cluster when a path of low-distance edges connects them, and because edges only join temporally close frames, every cluster is a contiguous segment of the video. The paper reports about three times faster preprocessing and inference than the pre-computed approach. The paper states that the code is available at github.com/IMScience-PPGINF-PucMinas/SLC-HELM. That link could not be confirmed from the project environment, so until it is checked the layer is treated as implemented from the paper's description. Confirming the link is an open item (Q-04).

The paper specifies the following, and the layer follows it. The threshold is the 50th percentile of pairwise distances at the current scale, which is a proposed rule for T12 and is not yet approved (Q-02). Each cluster is represented by the mean of its members' features (eq. 6). Edges are remapped to the new level, and each node is linked to its parent cluster (eq. 7). New levels are appended to the existing ones. Three levels were used. The first graph processing block applies one GATv2 (eqs. 1–2), and each multigraph module applies two GATv2 branches, one per relation type (eqs. 3–4). The only reported evaluation is on EuroSAT.

Used in: T12, T13, and as the reference for the comparison in T18.

### HELMNet

Batisteli, Passat, Guimarães and Patrocínio Júnior, Applied Soft Computing 2025. Cited as [2].

This work defines the multigraph design that the thesis uses. A hierarchical multigraph has two kinds of edges, temporal edges within one level and hierarchical edges between a node and its parent at the next level up. The two kinds are processed in separate branches and then merged. The paper also introduces the region graph readout (RGR), which weights the levels of the hierarchy by importance. The thesis uses this design in the multigraph modules (MGP) and in the readout that chooses keyframes. The hierarchy in HELMNet comes from watershed by area. The proposal notes that this algorithm is, at its core, a hierarchical minimum spanning forest, the same family as the MST in HieTaSumm. This is the reason the two lines of work connect naturally, and it is worth stating in the methodology.

Used in: T15 (MGP), T17 (RGR readout), and in the justification of the hierarchy in T14.

### mHELMNet

Batisteli, Guimarães and Patrocínio Júnior, IEEE GRSL 2025. Cited as [4].

This is an adaptation of HELMNet to remote sensing, and it is the closest thing the project has to an implementation template. It fixes the stack used in this repository: PyTorch, PyTorch Geometric, higra, and mlflow for tracking. Its building blocks (`Linear`, `BatchNorm`, `GraphSizeNorm`) and its gated graph convolution layer are the style reference for the new code. It also contributes one simplification that the thesis adopts: the edge features are kept fixed after the initial embedding instead of being updated at each layer. The paper reports that this is more stable and does not lose accuracy. The first version of the MGP modules in T15 follows this choice.

Used in: T15, T16, and as the reference for code style and dependencies across the project.

### HRG and HRGNet

Lopes, Batisteli, Guimarães and Patrocínio Júnior, CIARP 2025. Cited as [3].

This work is a supporting base rather than a core one. It shows that the hierarchical multigraph framework applies to 3D shapes as well as images, which supports the claim that the design is not specific to one domain. More importantly for the thesis, it establishes a comparison pattern: a model using only the base graph is compared against the full hierarchical model. The thesis reuses the idea of comparing the full model against reduced or fixed alternatives in the comparison of results (T18), and later in the ablation planned for the second phase.

Used in: T18, as the experimental design pattern.

### GATv2

Brody, S., Alon, U., Yahav, E. (2022) "How Attentive are Graph Attention Networks?", ICLR 2022, arXiv:2105.14491. Cited as [15]. Verified against the arXiv record and the ICLR proceedings on 2026-10-21.

This is the attention layer used after the graph and inside each multigraph branch. SLC-HELM applies it in its first graph processing block (eqs. 1–2) and in the two relation branches of each MGP module (eqs. 3–4). It is available as `GATv2Conv` in PyTorch Geometric (D-06). The original GAT is not used, because its attention ranking does not depend on the query node, which GATv2 corrects.

Used in: T09, T15.

## Comparison works

These works are not the foundation of the method, but the thesis compares against them or borrows specific ideas from them. None of them is implemented as the main method.

### VideoSAGE

Chaves, J. M. R., Tripathi, S. (2024) ``VideoSAGE: Video Summarization with Graph Representation Learning'', CVPR 2024 Workshops (SG2RL), p. 2527–2534, also arXiv:2404.10539. Cited as [6]. The published venue is the SG2RL workshop at CVPR 2024, not only the arXiv preprint as previously recorded here; corrected 2026-10-20.

VideoSAGE is the closest prior work that builds a graph in the same way: frames are nodes, edges join temporally close frames, and a lightweight graph neural network is applied. It is the main reference for the rank correlation metrics, Kendall's tau and Spearman's rho, which the thesis plans to report alongside the F-score. This matters because VSUMM does not give a per-frame importance score, so how these metrics are computed on VSUMM is an open decision (T11). VideoSAGE is also the natural comparison point in the discussion of results in T18.

Used in: T10, T11, T19, and the discussion in T18.

### VOGNet

Zhang, J., Wu, G., Song, S. (2023) ``Video Summarization Generation Based on Graph Structure Reconstruction'', _Electronics_, v. 12, n. 23, 4757. Cited as [7]. The author list (Zhang, Wu, Song) was verified against the published MDPI article on 2026-10-20 and is correct; the earlier caveat about checking it is resolved.

VOGNet shows that the graph can be learned without labels, by reconstructing its structure with a variational graph autoencoder. It also scores frames with graph attention. The thesis takes two ideas from it. The first is the unsupervised objective based on graph reconstruction, which is one of the candidates for the training loss and must be decided before T13 is finished. The second is the frame-level attention used inside each hierarchy level to pick the keyframe, implemented in T17.

Used in: T13 (training objective, as a decision), T17 (frame attention).

### SumGraph

Park, J., Lee, J., Kim, I.-J., Sohn, K. (2020) ``SumGraph: Video Summarization via Recursive Graph Modeling'', ECCV 2020. Cited as [8].

SumGraph refines the graph recursively to decide keyframes, and it works both in a supervised mode and in an unsupervised mode. It is useful mainly as related work. It shows that graph-based summarisation can be trained either way, which helps frame the thesis's choice of unsupervised training. The thesis does not implement it.

Used in: T02 (related work), T14 (methodology discussion).

### RSGN

Zhao, B., Li, H., Lu, X., Li, X. (2022) ``Reconstructive Sequence-Graph Network for Video Summarization'', _IEEE TPAMI_, v. 44, n. 5, p. 2793–2801 (early access 2021), arXiv:2105.04066. Cited as [9]. The publication year is 2022 (it was online/early-access in 2021); corrected 2026-10-20, verified against the IEEE Xplore record.

RSGN is the closest precedent for three features at once: it is hierarchical (it works at the frame level and the shot level), it uses a graph, and it is trained without labels, using a reconstruction reward. Its existence in the literature is the strongest support for the claim that a hierarchical graph model can be trained unsupervised. It informs the training objective decision in T13 and the way the methodology is justified in T14.

Used in: T02, T13 (training objective), T14.

## Auxiliary works

### DMoN and DMoN-DPR

Tsitsulin, A., Palowitch, J., Perozzi, B., Müller, E. (2023) `Graph Clustering with Graph Neural Networks'', *JMLR*, v. 24, n. 127, p. 1–21. Cited as [10]. DMoN-DPR is Salehi, Y., Giannacopoulos, D. (2025) `Deep Modularity Networks with Diversity-Preserving Regularization'', NeurIPS 2025 Workshop on New Perspectives in Graph Machine Learning, arXiv:2501.13451. Cited as [11]. DMoN-DPR's author list was missing here and was filled in on 2026-10-20.

DMoN is a graph clustering method based on a GNN. The thesis does not use it to choose clusters, because that is the job of the SLC pooling. Its regularisation idea is what matters here. DMoN-DPR adds a term that discourages redundancy between the representatives of different clusters. If the unsupervised objective includes diversity regularisation, this term can be added so that keyframes from different clusters are not near-duplicates. This is optional and depends on the decision for T13.

Used in: T13, only if the objective includes diversity regularisation.

### DiffPool and MinCutPool

Ying, R., You, J., Morris, C., Ren, X., Hamilton, W. L., Leskovec, J. (2018) `Hierarchical Graph Representation Learning with Differentiable Pooling'', NeurIPS 2018, p. 4800–4810. Cited as [12]. Bianchi, F. M., Grattarola, D., Alippi, C. (2020) `Spectral Clustering with Graph Neural Networks for Graph Pooling'', ICML 2020, PMLR v. 119, p. 874–883. Cited as [13].

These are the two standard learned pooling methods for graphs, and they are the natural alternatives to SLC pooling. DiffPool learns a dense, differentiable soft cluster-assignment matrix at each layer, mapping every node to every cluster of a fixed, pre-specified cluster count, stacked with a separate embedding GNN; this costs $O(N^2)$ in the assignment and needs link-prediction and entropy regularisation terms to stay well-behaved, and the number of levels and cluster sizes are hyperparameters, not discovered from the data. MinCutPool reformulates spectral (normalised min-cut) clustering as a GNN layer, with an MLP producing a soft assignment optimised by a relaxed min-cut loss plus an orthogonality regulariser; it avoids the eigendecomposition of classical spectral clustering but still fixes the target number of clusters ahead of time, and produces one flat partition per layer. SLC pooling differs from both: it grows the hierarchy bottom-up through single-link merges along the learned pairwise affinities, so the number of clusters and the merge order emerge from the graph itself, it needs no dense assignment matrix and no auxiliary cut or orthogonality loss, and because the graph only has temporal edges, every resulting cluster is a contiguous video segment. This contrast is the argument for choosing SLC pooling over DiffPool or MinCutPool, and it belongs in T12's specification and the related-work discussion in T14.

Used in: T02 (read, this summary), T12 (justification of the choice), T14 (related work).

### Video summarisation survey

Apostolidis, E., Adamantidou, E., Metsai, A. I., Mezaris, V., Patras, I. (2021) ``Video Summarization Using Deep Neural Networks: A Survey'', _Proceedings of the IEEE_, v. 109, n. 11, p. 1838–1863, also arXiv:2101.06072. Cited as [14]. Added 2026-10-20 to close the T02 gap for a survey that places the thesis in the field.

This is the standard survey of deep-learning-based video summarisation, covering supervised, unsupervised and reinforcement-learning approaches and their evaluation protocols. It is the reference used to situate the thesis's unsupervised, graph-based method within the broader field in the related-work discussion (T02, T14).

Used in: T02 (related work), T14 (methodology discussion).

### ASAP

Ranjan, E., Sanyal, S., Talukdar, P. (2020) ``ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations'', _Proceedings of the AAAI Conference on Artificial Intelligence_, v. 34, p. 5470–5477. Cited as [16]. Added 2026-10-20 to close the T02 gap for hierarchical-readout literature.

ASAP learns a sparse cluster assignment together with a self-attention-based cluster-fitness score, used both to pool nodes and to form the layer's readout: an explicit, hierarchical treatment of what to read out at each level, rather than a single final pooling step. It is the closest published analogue to this thesis's learned, scale-aware readout stage (adapted from HELMNet's region graph readout) that follows SLC pooling, and it is cited to justify that adaptation.

Used in: T02 (related work), T17 (justification of the readout design).

## Gaps to close in T02

T02 closes when the items below are done, and it is needed for PC3.

1. **Done, 2026-10-20.** A survey of video summarisation, to place the work in the field: added (video summarisation survey [14]).
2. **Done, 2026-10-20.** Literature on hierarchical readout in graph neural networks, to justify the adaptation of RGR: added (ASAP [15]).
3. **Done, 2026-10-20.** The details of VOGNet, RSGN and DMoN-DPR were checked against the primary sources. VOGNet's author list was confirmed correct. RSGN's publication year was corrected (2022, not 2021). DMoN-DPR's author list, missing before, was filled in.
4. **Done, 2026-10-20.** The not-yet-cited works were added to the Overleaf bibliography (now entries [6] to [15]), DiffPool and MinCutPool were read and summarised (above), and all ten now have a matching `\cite{}` in the thesis body text (motivation paragraph, specific-objectives bullets), once the researcher explicitly extended the edit authorisation beyond the references section (see [CLAUDE.md](../CLAUDE.md) rule 15). T02's acceptance criterion is met.
