```
Role: Records every architectural and methodological decision of the thesis, with context, alternatives and consequences. Identifiers match architecture.md.
Entry: docs/status.md
Related to: [status.md](./status.md), [architecture.md](./architecture.md), [phases.md](./phases.md)
```

# Decisions

This file is the record of why the project is built the way it is. Each decision has an identifier (D-01, D-02 and so on) that is used in [architecture.md](./architecture.md), in [phases.md](./phases.md) and in [status.md](./status.md), so a decision can be referenced from any document.

## Status values

**Locked** means the decision is settled. It is not revisited without a new entry that supersedes it. **Decided** means the decision is taken and in use, but may change if new evidence appears, and any change must be recorded here. **Superseded** means the decision was replaced; its entry stays in the file, with a link to the one that replaced it. **Open** means the decision has not been taken yet, and the entry records the options.

## Entry template

Every new decision follows the same structure: identifier and title, date, status, context, decision, alternatives discarded, consequences, and links to the documents it affects. Context says why the question came up. Decision says what was chosen, in one or two sentences. Alternatives say what was discarded and why. Consequences say what the choice makes easier and what it makes harder.

## Terminology

Two terms are used in every document, and each has one meaning.

**Proposed method** (also called the main method) is the thesis contribution: the time-aware graph of HieTaSumm, GATv2, a hierarchy learned by SLC pooling, a multigraph GNN, and a learned readout with frame attention that picks the keyframes. After the graph, every step is a learned component.

**HieTaSumm baseline** is the original HieTaSumm method, with no learning: the graph, the minimum spanning tree, the cut by area and a rule-based choice of keyframe. It is the "graph without learning" cell of the comparison.

## D-01 — The main method learns the hierarchy with SLC pooling; the original HieTaSumm method is the baseline

**Date:** not recorded. **Status:** Locked, confirmed with the advisor.

**Context:** HieTaSumm builds its hierarchy with a fixed rule (minimum spanning tree cut by area). The thesis needs a learned hierarchy to be the contribution.

**Decision:** The main method learns the hierarchy with single-link clustering (SLC) pooling, inspired by SLC-HELM. The original HieTaSumm method is kept as the baseline, on the same input graph. Running the same network on an MST hierarchy is not part of the final evaluation; it may be used only as an early development check of the shared steps (GATv2, multigraph, readout) before SLC pooling exists, and it is not reported as a result.

**Alternatives discarded:** Only the HieTaSumm baseline, which has no learning and so does not test the contribution. Only the SLC method, which has nothing to compare against. An MST variant of the GNN in the final evaluation, dropped by the researcher: the comparison with the HieTaSumm baseline already measures the effect of learning over the graph.

**Consequences:** Every experiment has a direct comparison between the proposed method and the original HieTaSumm method. The cost is that the SLC layer must be implemented (T13).

**Affects:** T06, T12, T13, T18.

## D-02 — Graph construction is ported from HieTaSumm with its logic kept faithful

**Date:** not recorded. **Status:** Locked.

**Context:** The time-aware graph is the base of the whole pipeline, and the thesis must show that it reproduces the reference work.

**Decision:** Port the `Frame` class of the HieTaSumm library into `src/tcc_gnn/graph/frame.py`. Keep the algorithmic logic and the variable names from the original, so that outputs can be compared directly. Validate the port against the reference (T01).

**Alternatives discarded:** Rewrite the graph from scratch. It would lose the link to the reference and make validation impossible to state.

**Consequences:** Any change to the graph code must be followed by the T01 validation again. The code has legacy naming that is kept on purpose, which is noted in the style decision (D-10).

**Affects:** T01, all later tasks that use the graph.

## D-03 — Features come from torchvision ResNet50, global average pooling, 2048 dimensions

**Date:** 2026-09-29. **Status:** Locked.

**Context:** The original code extracts features from the `predictions` layer of ResNet50, which is a softmax over 1000 classes. That vector also becomes the initial node representation of the GNN, so the choice of layer matters for the whole pipeline.

**Decision:** Extract features from the `avgpool` layer (global average pooling, 2048 dimensions, before classification), using torchvision. Implemented in `src/tcc_gnn/features/features.py`.

**Alternatives discarded:** Keep `predictions`, for strict fidelity to the original. A softmax saturates, pushing all frames toward the dominant classes, so distances between frames with different predicted classes end up roughly equal and lose fine visual detail. This is the stage that most needs that detail, the hierarchical pooling.

**Consequences:** The features are richer (texture, colour, composition) and consistent with what HELMNet and mHELMNet consume (convolutional embeddings, not classification outputs). The features differ from the original, so the T01 validation must state which extractor it used. The original softmax output is not reproduced by the production code.

**Affects:** T01, T09, T12, T13.

## D-04 — TensorFlow is replaced by torchvision in production; TensorFlow stays only for validation

**Date:** 2026-09-22 (migration), 2026-09-29 (validation-only status). **Status:** Locked.

**Context:** HieTaSumm uses TensorFlow/Keras for feature extraction. The GNN uses PyTorch Geometric, which already requires PyTorch. Keeping two deep learning engines would add dependencies without a technical gain.

**Decision:** Port feature extraction to torchvision. The TensorFlow replica in `src/tcc_gnn/features/tf_resnet_extractor.py` is kept only to check fidelity against the original. It is declared as an optional extra in `pyproject.toml` and is not used by the pipeline.

**Alternatives discarded:** Keep TensorFlow for the extraction step only. It adds a dependency with no clear benefit.

**Consequences:** One deep learning engine in production. The TensorFlow replica must stay in the repository for the validation to be repeatable, which is a maintenance cost. The validation against the original is still pending advisor review.

**Affects:** T01, T09.

## D-05 — The MST hierarchy uses higra

**Date:** not recorded. **Status:** Decided.

**Context:** The MST hierarchy, used by the HieTaSumm baseline, needs an implementation. HieTaSumm and the lab's multigraph code both use higra.

**Decision:** Implement the MST hierarchy with higra, in `src/tcc_gnn/hierarchy/`.

**Alternatives discarded:** Implement the MST by hand. It duplicates a maintained library and makes the MST hierarchy harder to trust.

**Consequences:** The MST hierarchy matches the reference behaviour. The project depends on higra, which must be installed in the environment.

**Affects:** T06, T07.

## D-06 — The GNN layer is GATv2 from PyTorch Geometric

**Date:** not recorded. **Status:** Decided. Sources: Brody, Alon and Yahav (2022), which defines GATv2 ([references.md](./references.md)), and SLC-HELM, whose graph processing block and multigraph branches use it (eqs. 1–4). Not yet validated in code.

**Context:** The base of the model needs attention over temporal neighbours, with weights that depend on each pair of frames.

**Decision:** Use `GATv2Conv` from PyTorch Geometric, applied to the time-aware graph. Edge distances may be passed as attributes.

**Alternatives discarded:** The original GAT. Its attention ranking is shared across nodes, which limits what the layer can express compared with GATv2.

**Consequences:** No custom attention code to maintain. Results depend on the library version, which must be recorded in the environment.

**Affects:** T08, T09.

## D-07 — The multigraph has two edge types, processed in separate branches and merged by an MLP

**Date:** not recorded. **Status:** Decided, from HELMNet. SLC-HELM applies the same separation in its MGP, with one GATv2 branch per relation type (eqs. 3–4).

**Context:** The hierarchy creates two kinds of relations: between frames within a level, and between a node and its parent group.

**Decision:** Process temporal and hierarchical edges in separate branches and merge the results with a multilayer perceptron, following HELMNet.

**Alternatives discarded:** A single edge type. It loses the distinction between neighbours in time and members of the same group.

**Consequences:** The model has more parameters and more code paths to test. The distinction is what makes the hierarchy usable by the model.

**Affects:** T15, T16.

## D-08 — Edge features are fixed after the initial embedding

**Date:** not recorded. **Status:** Decided, from mHELMNet, first version only. Under review: the GATv2 branches of SLC-HELM do not use edge features, so this decision may not apply to the MGP as now designed.

**Context:** Updating edge features at each layer adds computation, and the stability of training matters for a first version.

**Decision:** Keep edge features fixed after the initial embedding. The paper reports this is more stable without losing accuracy.

**Alternatives discarded:** Update edge features at every layer. More expressive in principle, but no gain reported and less stable.

**Consequences:** Simpler first version. This can be revisited later if the results suggest the edges need to adapt.

**Affects:** T15.

## D-09 — Training is unsupervised

**Date:** not recorded. **Status:** Locked.

**Context:** The thesis argues that summarisation can be done without human annotation, which is the point of HieTaSumm.

**Decision:** The main method uses no human summaries during training. Human summaries are used only for evaluation.

**Alternatives discarded:** Supervised training on the VSUMM summaries. It is kept only as an extension, as an upper bound for comparison, if the schedule allows.

**Consequences:** The method stays aligned with its claim. The training objective is still open (see D-17 when recorded), and the choice has to be made before T13.

**Affects:** T13, T14, and all training tasks.

## D-10 — Modern Python style; ported logic keeps the original names

**Date:** 2026-09-29. **Status:** Locked.

**Context:** HieTaSumm-lib and mHELMNet have inconsistent style: mixed indentation, no type hints, code wrapped in classes, dead code. The project must be maintainable and readable by agents.

**Decision:** Write new code in modern Python (PEP 8, four-space indentation, type hints, plain functions where no state is needed, no dead code). Keep the algorithmic logic and variable names of ported code faithful to the original, such as `v1`, `v2`, `delta_t` and `calc_end`.

**Alternatives discarded:** Imitate the legacy style. It would make the new code harder to test and to read.

**Consequences:** Ported code can be compared line by line with the original. New code is easier to test. The two styles coexist, which is expected and noted in the module docstrings.

**Affects:** All code.

## D-11 — Scatter operations come from PyTorch Geometric; torch_scatter is not required

**Date:** not recorded. **Status:** Locked.

**Context:** Graph layers use scatter operations. The separate `torch_scatter` package must be compiled against the exact installed PyTorch build and has no ready wheels for recent versions.

**Decision:** Rely on the scatter operations built into the PyTorch Geometric version declared in `pyproject.toml`, which has a pure PyTorch fallback.

**Alternatives discarded:** Require `torch_scatter`. It adds an installation step that often fails.

**Consequences:** Simpler installation. Performance of the fallback must be measured if it becomes a bottleneck, which is part of the functional criteria.

**Affects:** T08, T09, T13, T15.

## D-12 — Experiments are tracked with mlflow

**Date:** not recorded. **Status:** Decided.

**Context:** Experiments must be reproducible, and the lab already uses mlflow in mHELMNet.

**Decision:** Track all training and evaluation runs with mlflow, in `experiments/`.

**Alternatives discarded:** Other trackers. They would be inconsistent with the lab's practice.

**Consequences:** Runs are reproducible and comparable across the four cells. The project depends on mlflow.

**Affects:** T07, T13, T18.

## D-13 — The evaluation protocol is fixed in writing before any result is produced

**Date:** 2026-10. **Status:** Locked.

**Context:** Choosing metrics or settings after seeing the numbers can make any method look better.

**Decision:** The protocol covers how keyframes become a summary, how the F-score is computed, how the per-frame score is derived where needed, the summary length, and the hardware used for timing. It is written as a specification in task T19, before T07, and it is the same for every compared method.

**Alternatives discarded:** Decide the protocol as results come in. It invites selecting the setting that favours the method.

**Consequences:** The protocol must be written before the first result. It can be revised, but any revision is recorded here with its reason and applied to all methods again.

**Affects:** T19, T07, T18, and every evaluation task.

## D-14 — The HieTaSumm baseline and the proposed method receive the same graph

**Date:** not recorded. **Status:** Locked.

**Context:** To attribute a difference in results to the method, the input must not differ between the methods.

**Decision:** Both methods are built from the same time-aware graph, with the same features, from the same extractor.

**Alternatives discarded:** Let each method build its own graph. It would confound the effect of the method with the effect of the input.

**Consequences:** The comparison is controlled. Any change to the shared graph affects both methods and must be validated once.

**Affects:** T06, T13, T18.

## D-15 — One adapter per dataset, from the start of TCC1

**Date:** not recorded, from the scope document. **Status:** Decided.

**Context:** The datasets store ground truth in different formats. VSUMM gives user-selected keyframes, while SumMe and TVSum give importance scores.

**Decision:** Each dataset has an adapter in `src/tcc_gnn/data/` that converts its ground truth into a common format. The adapters are part of the design from TCC1, even though the full experiments on the three datasets belong to TCC2.

**Alternatives discarded:** A single format for all datasets. It hides the real differences between the ground truths.

**Consequences:** Evaluation code depends only on the common format. The difference in VSUMM ground truth raises the open question on rank metrics (Q-01). The common format was made concrete in T05: the `GroundTruth` dataclass (`video_id`, `n_frames`, `keyframes`, `scores: ScoreSeries | None`, `change_points: np.ndarray | None`) and the `ScoreSeries` dataclass (`values`, `frame_indices`), in `src/tcc_gnn/data/base.py`. For SumMe, `keyframes` maps to `user_summary`, `scores` maps to `gtscore`/`picks` with no padding, interpolation or zero-filling for frames without a score, and `change_points` maps to `change_points`.

**Affects:** T05, T10.

## D-16 — Which comparison cells belong to TCC1

**Date:** 2026-10, revised 2026-10-20. **Status:** Decided, by the researcher.

**Context:** The full comparison has four cells: with or without graph, with or without learning. Two of them need methods that sit outside the proposal: the simple baselines that use no graph and no learning, and the ablation of the network without message passing.

**Decision:** In TCC1 the proposed method is compared only with the HieTaSumm baseline. The ablation without message passing and the uniform sampling, k-means and random selection baselines are left for TCC2, which also extends the complete comparison to all three datasets. The first version of this decision put the uniform and k-means baselines and the ablation in TCC1. The researcher removed them on 2026-10-20 because they are outside the scope of the proposal and the first phase only needs preliminary results.

**Alternatives discarded:** Keep the ablation in TCC1. It would let the first phase separate the effect of the graph from the effect of learning, but it needs a definition of what is removed and a full run of the modified network, which preliminary results do not need.

**Consequences:** The only evaluation task is T19, the protocol. Tasks T20 and T21 were removed. The first phase cannot separate the effect of the graph from the effect of learning, nor show whether the proposed method beats a method with no graph and no learning, and the thesis text must say so. Before the ablation is implemented in TCC2, a decision must record what it removes: the GATv2 layer only, or also the neighbour aggregation in the multigraph branches, and whether the SLC hierarchy is kept. All compared methods produce summaries of the same length, defined by the protocol (D-13).

**Affects:** T19, T18.

## D-17 — Unsupervised training objective

**Date:** not decided. **Status:** Open (see Q-03 in architecture.md).

**Context:** Training is unsupervised, but the objective is not yet chosen. The options are graph reconstruction, as in VOGNet and RSGN, diversity regularisation, as in DMoN-DPR, or a combination.

**Decision:** Pending. When decided, record it here and mark D-09 as implemented by this objective.

**Affects:** T13.

## D-18 — After the graph, the proposed method uses no combinatorial step from HieTaSumm

**Date:** 2026-10. **Status:** Locked, stated by the researcher.

**Context:** HieTaSumm uses a graph but no learning. The contribution of the thesis is learning over the graph, so mixing rule-based steps into the proposed method would blur what is being tested.

**Decision:** In the proposed method, only the graph construction comes from HieTaSumm. The hierarchy comes from SLC pooling, and the processing of the hierarchy and the choice of keyframes are done by GNN layers. The MST, the cut by area and the rule-based keyframe choice appear only in the HieTaSumm baseline. If a learned step turns out not to be plausible, the replacement is recorded here as a new decision and is never made silently.

**Alternatives discarded:** Reuse the HieTaSumm keyframe choice on the learned hierarchy. It would make it unclear whether the results come from learning.

**Consequences:** The comparison with the HieTaSumm baseline measures learning over the graph. The readout and the attention (T17) are required for the method to produce a summary at all.

**Affects:** T13, T15, T16, T17, T18.

## D-19 — Threshold rule for SLC pooling

**Date:** 2026-10-21. **Status:** Open. The proposed rule is recorded here but not approved.

**Context:** SLC pooling merges frames along edges whose feature distance is below a threshold. The configuration leaves the threshold undefined, and T12 and T13 need one rule before they start.

**Decision:** Pending. The proposed rule is the 50th percentile of the pairwise distances at the current scale, following SLC-HELM. The paper does not say whether the percentile is taken over all pairs of nodes or only over the edges of the graph. On a sparse temporal graph the two choices give different thresholds, so the choice must be made and recorded.

**Alternatives discarded:** None yet. Options to be decided are the percentile over all pairs or over the edges, and any other fixed value.

**Consequences:** Once approved, T12 and T13 can proceed. Until then no threshold is fixed. The HieTaSumm baseline does not use this rule.

**Affects:** T12, T13.

# D-20 — Corte da baseline MST: cutNumber calculado pela referência

**Status:** Decided.

**Context:** Na T06, o cutNumber foi inicialmente fixado à mão em 20. A referência do HieTaSumm-lib calcula o valor por `Summary.bestCutNumber`, que não usa o grafo nem o ground truth: lê os frames em disco, converte cada um para pixels RGB crus e calcula o desvio padrão da similaridade de cosseno entre pares. Com `alpha = 75` (options.json), o resultado no v29 é `cutNumber = int(7.0 × 75 / 100) = 5`.

**Decision:** A baseline usa `cutNumber = 5`, reproduzindo o cálculo da referência. O valor 20 fica apenas como caso de teste de sensibilidade.

**Alternatives discarded:** cutNumber = 20 (valor manual, sem origem no original); usar o ground truth para escolher o corte (vazaria informação de avaliação, contra a proposta não supervisionada).

**Consequences:** A T07 precisa fornecer os frames em disco para o cálculo de `bestCutNumber`. O cálculo usa pixels crus, não o ResNet de D-03, e isso fica explícito na baseline. O número de clusters depende do corte, e o tamanho do resumo será fixado na T19.

**Affects:** T06, T07, T19.
