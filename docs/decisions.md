```
Role: Records every architectural and methodological decision of the thesis, with context, alternatives and consequences. Identifiers match architecture.md.
Entry: docs/status.md
Related to: [status.md](./status.md), [architecture.md](./architecture.md), [phases.md](./phases.md), [project.md](./project.md), [references.md](./references.md), [principles.md](./principles.md)
```

# Decisions

This file is the record of why the project is built the way it is. Each decision has an identifier (D-01, D-02 and so on) that is used in [architecture.md](./architecture.md), in [phases.md](./phases.md) and in [status.md](./status.md), so a decision can be referenced from any document.

## Status values

**Locked** means the decision is settled. It is not revisited without a new entry that supersedes it. **Decided** means the decision is taken and in use, but may change if new evidence appears, and any change must be recorded here. **Superseded** means the decision was replaced; its entry stays in the file, with a link to the one that replaced it.

## Entry template

Every new decision follows the same structure: identifier and title, date, status, context, decision, alternatives discarded, consequences, and links to the documents it affects. Context says why the question came up. Decision says what was chosen, in one or two sentences. Alternatives say what was discarded and why. Consequences say what the choice makes easier and what it makes harder.

## D-01 — The main method learns the hierarchy with SLC pooling; MST is the baseline

**Date:** not recorded. **Status:** Locked, confirmed with the advisor.

**Context:** HieTaSumm builds its hierarchy with a fixed rule (minimum spanning tree cut by area). The thesis needs a learned hierarchy to be the contribution.

**Decision:** The main method learns the hierarchy with single-link clustering (SLC) pooling, inspired by SLC-HELM. The fixed MST hierarchy of HieTaSumm is kept as the baseline, on the same input graph.

**Alternatives discarded:** Only the MST baseline, which has no learning and so does not test the contribution. Only the SLC method, which has nothing to compare against.

**Consequences:** Every experiment has a direct comparison between a learned and a fixed hierarchy. The cost is that the SLC layer must be implemented, which depends on D-17.

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

## D-05 — The hierarchy baseline uses higra

**Date:** not recorded. **Status:** Decided.

**Context:** The MST baseline needs a hierarchy implementation. HieTaSumm and the lab's multigraph code both use higra.

**Decision:** Implement the MST hierarchy with higra, in `src/tcc_gnn/hierarchy/`.

**Alternatives discarded:** Implement the MST by hand. It duplicates a maintained library and makes the baseline harder to trust.

**Consequences:** The baseline matches the reference behaviour. The project depends on higra, which must be installed in the environment.

**Affects:** T06, T07.

## D-06 — The GNN layer is GATv2 from PyTorch Geometric

**Date:** not recorded. **Status:** Decided, from the architecture proposal. Not yet validated in code.

**Context:** The base of the model needs attention over temporal neighbours, with weights that depend on each pair of frames.

**Decision:** Use `GATv2Conv` from PyTorch Geometric, applied to the time-aware graph. Edge distances may be passed as attributes.

**Alternatives discarded:** The original GAT. Its attention ranking is shared across nodes, which limits what the layer can express compared with GATv2.

**Consequences:** No custom attention code to maintain. Results depend on the library version, which must be recorded in the environment.

**Affects:** T08, T09.

## D-07 — The multigraph has two edge types, processed in separate branches and merged by an MLP

**Date:** not recorded. **Status:** Decided, from HELMNet.

**Context:** The hierarchy creates two kinds of relations: between frames within a level, and between a node and its parent group.

**Decision:** Process temporal and hierarchical edges in separate branches and merge the results with a multilayer perceptron, following HELMNet.

**Alternatives discarded:** A single edge type. It loses the distinction between neighbours in time and members of the same group.

**Consequences:** The model has more parameters and more code paths to test. The distinction is what makes the hierarchy usable by the model.

**Affects:** T15, T16.

## D-08 — Edge features are fixed after the initial embedding

**Date:** not recorded. **Status:** Decided, from mHELMNet, first version only.

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

**Consequences:** The method stays aligned with its claim. The training objective is still open (see D-18 when recorded), and the choice has to be made before T13.

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

**Decision:** Rely on the scatter operations built into PyTorch Geometric 2.3 or later, which have a pure PyTorch fallback.

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

**Decision:** The protocol covers how keyframes become a summary, how the F-score is computed, how the per-frame score is derived where needed, the summary length, and the hardware used for timing. It is written as a specification before task T07, and it is the same for every compared method.

**Alternatives discarded:** Decide the protocol as results come in. It invites selecting the setting that favours the method.

**Consequences:** The protocol must be written before the first result. It can be revised, but any revision is recorded here with its reason and applied to all methods again.

**Affects:** T07, T18, and every evaluation task.

## D-14 — The baseline and the main method receive the same graph

**Date:** not recorded. **Status:** Locked.

**Context:** To attribute a difference in results to the hierarchy, nothing else may differ between the methods.

**Decision:** Both hierarchies are built from the same time-aware graph, with the same features, from the same extractor.

**Alternatives discarded:** Let each method build its own graph. It would confound the effect of the hierarchy with the effect of the input.

**Consequences:** The comparison is controlled. Any change to the shared graph affects both methods and must be validated once.

**Affects:** T06, T13, T18.

## D-15 — One adapter per dataset, from the start of TCC1

**Date:** not recorded, from the scope document. **Status:** Decided.

**Context:** The datasets store ground truth in different formats. VSUMM gives user-selected keyframes, while SumMe and TVSum give importance scores.

**Decision:** Each dataset has an adapter in `src/tcc_gnn/data/` that converts its ground truth into a common format. The adapters are part of the design from TCC1, even though the full experiments on the three datasets belong to TCC2.

**Alternatives discarded:** A single format for all datasets. It hides the real differences between the ground truths.

**Consequences:** Evaluation code depends only on the common format. The difference in VSUMM ground truth raises the open question on rank metrics (Q-01).

**Affects:** T05, T10.

## D-16 — All four comparison cells belong to TCC1

**Date:** 2026-10. **Status:** Decided, by the researcher.

**Context:** The comparison has four cells: with or without graph, with or without learning. The uniform and k-means baselines are cheap to implement. The ablation of the network without message passing depends on the model working.

**Decision:** The uniform and k-means baselines and the ablation without message passing all belong to TCC1. TCC2 extends the complete comparison to all three datasets.

**Alternatives discarded:** Leave the ablation for TCC2. The TCC1 text could not then separate the effect of the graph from the effect of learning.

**Consequences:** The ablation is a task after T16, which adds work to the first phase. All compared methods produce summaries of the same length, defined by the protocol (D-13).

**Affects:** Phases, T07 and after, T16, the protocol specification.

## D-17 — The SLC pooling layer is implemented from the paper, without reference code

**Date:** 2026-09-22. **Status:** Decided. Pending confirmation of code availability (Q-04).

**Context:** SLC-HELM is the base of the main method, but its code has not been shared. Only the mHELMNet repository is available.

**Decision:** Implement the SLC pooling layer from the paper's description. Use the building blocks of mHELMNet (`Linear`, `BatchNorm`, `GraphSizeNorm`) as the style reference.

**Alternatives discarded:** Wait for the code. It would block the main method without a date.

**Consequences:** Task T13 has more risk, because every interpretation of the paper must be recorded and tested on synthetic partitions. If the code becomes available, this decision should be revisited, which would reduce rework.

**Affects:** T12, T13.

## D-18 — Unsupervised training objective

**Date:** not decided. **Status:** Open (see Q-03 in architecture.md).

**Context:** Training is unsupervised, but the objective is not yet chosen. The options are graph reconstruction, as in VOGNet and RSGN, diversity regularisation, as in DMoN-DPR, or a combination.

**Decision:** Pending. When decided, record it here and mark D-09 as implemented by this objective.

**Affects:** T13.

## Links

- [status.md](./status.md): current state of each task and the next task to do
- [architecture.md](./architecture.md): the architecture these decisions shape, with open questions and risks
- [phases.md](./phases.md): the five phases and the tasks that implement each decision
- [project.md](./project.md): objective, research question, method and scope
- [references.md](./references.md): the works behind each decision
- [principles.md](./principles.md): the ten working principles that govern how decisions are made
