```
Role: Lists the works the thesis relies on, grouped by role, and explains where each one enters the project and which task uses it.
Entry: docs/status.md
Related to: [status.md](./status.md), [phases.md](./phases.md)
```

# References and their role in the work

This document explains how the literature enters the project. For each work, it says what the thesis takes from it, which component it justifies, and which task (T01 to T19, as defined in [phases.md](./phases.md)) uses it. Read it when you need to know why a component exists, or where a design choice came from.

Works are grouped into three roles. Base works supply the method itself: the thesis either implements them or adapts their code, so they must be read in full before the related task starts. Comparison works supply baselines, metrics or design ideas that the thesis measures itself against or borrows from, without being the foundation. Auxiliary works are read and may be used, but the thesis does not depend on them. Each entry below notes whether the work is already cited in the Overleaf document, where the bibliography is maintained. The numbering [1] to [5] refers to the citation order in that document.

The citation marks below were last checked against the Overleaf bibliography on 2026-10-16, and this date must be updated whenever they are checked again. At that check, five works were cited: HieTaSumm [1], HELMNet [2], HRG/HRGNet [3], mHELMNet [4] and SLC-HELM [5]. The other works listed here still had to be added to the bibliography.

## Base works

### HieTaSumm

Cardoso, Gomes, Guimarães and Patrocínio Júnior, BRACIS 2023. Cited as [1].

This is the foundation of the whole thesis. It provides two things. The first is the time-aware sparse graph, in which each video frame is a node and only temporally close frames are connected. The second is the fixed hierarchy, built as a minimum spanning tree and cut by area, which is used only for comparison: the original HieTaSumm method is the baseline. Its reference implementation, the `HieTaSumm-lib` submodule, is the ground truth for validation: the graph built in this repository must reproduce its output, and that is the purpose of T01. The same library is used for the SumMe data in T05. The method is evaluated on VSUMM with F-score, which is the metric the thesis keeps as its primary result.

Used in: T01, T05, T06, T07, T19.

### SLC-HELM

Batisteli, Kenmochi, Bougleux, Lézoray, Guimarães and Patrocínio Júnior, IEEE GRSL 2026. Cited as [5].

This is the source of the main method. It introduces single-link clustering pooling, which learns the hierarchy during training instead of computing it beforehand. The thesis adopts this idea, and the rule it follows is simple to state: two frames belong to the same cluster when a path of low-distance edges connects them, and because edges only join temporally close frames, every cluster is a contiguous segment of the video. The paper reports about three times faster preprocessing and inference than the pre-computed approach. The repository for this work has not been shared, so the layer is implemented from the paper's description. Confirming whether the code becomes available is an open item, because it would reduce rework.

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

## Comparison works

These works are not the foundation of the method, but the thesis compares against them or borrows specific ideas from them. None of them is implemented as the main method.

### VideoSAGE

Chaves and Tripathi, 2024, arXiv:2404.10539. Not yet cited.

VideoSAGE is the closest prior work that builds a graph in the same way: frames are nodes, edges join temporally close frames, and a lightweight graph neural network is applied. It is the main reference for the rank correlation metrics, Kendall's tau and Spearman's rho, which the thesis plans to report alongside the F-score. This matters because VSUMM does not give a per-frame importance score, so how these metrics are computed on VSUMM is an open decision (T11). VideoSAGE is also the natural comparison point in the discussion of results in T18.

Used in: T10, T11, T19, and the discussion in T18.

### VOGNet

Zhang, Wu and Song, Electronics 12(23), 4757, 2023. Not yet cited. The author list here follows the project scope document and should be checked against the published article before citing.

VOGNet shows that the graph can be learned without labels, by reconstructing its structure with a variational graph autoencoder. It also scores frames with graph attention. The thesis takes two ideas from it. The first is the unsupervised objective based on graph reconstruction, which is one of the candidates for the training loss and must be decided before T13 is finished. The second is the frame-level attention used inside each hierarchy level to pick the keyframe, implemented in T17.

Used in: T13 (training objective, as a decision), T17 (frame attention).

### SumGraph

Park, Lee, Kim and Sohn, ECCV 2020. Not yet cited.

SumGraph refines the graph recursively to decide keyframes, and it works both in a supervised mode and in an unsupervised mode. It is useful mainly as related work. It shows that graph-based summarisation can be trained either way, which helps frame the thesis's choice of unsupervised training. The thesis does not implement it.

Used in: T02 (related work), T14 (methodology discussion).

### RSGN

Zhao, Li, Lu and Li, IEEE TPAMI 2021, arXiv:2105.04066. Not yet cited.

RSGN is the closest precedent for three features at once: it is hierarchical (it works at the frame level and the shot level), it uses a graph, and it is trained without labels, using a reconstruction reward. Its existence in the literature is the strongest support for the claim that a hierarchical graph model can be trained unsupervised. It informs the training objective decision in T13 and the way the methodology is justified in T14.

Used in: T02, T13 (training objective), T14.

## Auxiliary works

### DMoN and DMoN-DPR

Tsitsulin, Palowitch, Perozzi and Müller, JMLR 2023, under the title _Graph Clustering with Graph Neural Networks_. Not yet cited. DMoN-DPR is Deep Modularity Networks with Diversity-Preserving Regularization, arXiv:2501.13451, 2025.

DMoN is a graph clustering method based on a GNN. The thesis does not use it to choose clusters, because that is the job of the SLC pooling. Its regularisation idea is what matters here. DMoN-DPR adds a term that discourages redundancy between the representatives of different clusters. If the unsupervised objective includes diversity regularisation, this term can be added so that keyframes from different clusters are not near-duplicates. This is optional and depends on the decision for T13.

Used in: T13, only if the objective includes diversity regularisation.

### DiffPool and MinCutPool

Ying et al., NeurIPS 2018, and Bianchi, Grattarola and Alippi, ICML 2020. Not yet cited.

These are the two standard learned pooling methods for graphs. They are the natural alternatives to SLC pooling, so the related-work chapter must explain why the thesis chooses single-link clustering instead of a soft-assignment approach. They are not implemented. They are read in T02, and the comparison between them and SLC is part of the argument for the method.

Used in: T02 (must be read), T12 (justification of the choice).

## Gaps to close in T02

T02 closes when the four items below are done, and it is needed for PC3. First, the thesis needs at least one survey of video summarisation, to place the work in the field. Second, it needs literature on hierarchical readout in graph neural networks, to justify the adaptation of RGR. Third, the details of VOGNet, RSGN and DMoN-DPR must be checked against the primary sources, because the project notes give incomplete or differing author information for them. Fourth, the six works that are not yet cited must be added to the Overleaf bibliography, and DiffPool and MinCutPool must be read and summarised.
