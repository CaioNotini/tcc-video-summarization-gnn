```
Role: Defines the TCC stages, what each stage delivers, where its foundation comes from, and when it is considered complete.
Entry: docs/status.md
Related to: [status.md](./status.md), [project.md](./project.md), [references.md](./references.md), [architecture.md](./architecture.md), [decisions.md](./decisions.md), [principles.md](./principles.md)
```

# Stages of the work

This document explains how the thesis work is divided into stages, what each stage delivers, where its ideas come from, and when it counts as done. It is written for both people and AI agents. Read it after [status.md](./status.md), which records what is currently verified. This file defines the plan; [status.md](./status.md) records progress.

The work is a single line of tasks, numbered T01 to T18. Tasks are done one at a time, in order. Each task has an acceptance criterion, and a task is only marked as done when its evidence is saved in [docs/evidence](docs/evidence). Dates are not used to decide progress. The advisor checkpoints (PC3, PC4, PC5, PC6) are fixed external deadlines, and each one depends on a group of tasks being finished.

## Why the work is organised this way

The thesis takes the video summarisation method HieTaSumm and changes one key part of it. HieTaSumm builds a hierarchy of video frames with a fixed algorithm, a minimum spanning tree (MST) cut by area. The thesis replaces that fixed step with a hierarchy that a neural network learns during training, using single-link clustering (SLC), inspired by the SLC-HELM work from the laboratory. The fixed MST version is kept as a baseline, so the learned hierarchy can be compared against the original one on the same input.

Because of this, the stages follow the logic of the method. First the input graph must be correct. Then the fixed baseline must work end to end, so there is something to compare against. Then the neural network is built on top of that graph, and finally the learned hierarchy and its evaluation are added. A stage that depends on another one is not started until its predecessor has evidence behind it.

## Stage 1: Graph and foundations

This stage has three tasks, T01 to T03.

T01 validates the time-aware graph. Each video frame is a node, and an edge connects two frames only when they are close in time. The graph construction is ported from the code of Leonardo, the author of HieTaSumm, and the port must reproduce his output. The validation compares the graph built by this repository against the reference implementation, and the result must be saved as evidence. Right now the port exists and a small test suite passes, but the comparison with the reference has been reported and not yet saved in the repository. Until it is saved, T01 is not done. This stage's basis is the HieTaSumm paper and its reference code, which are in the `references/` submodules.

T02 is the literature review. It must cover at least ten works, and it is due for the PC3 checkpoint on 21 October. The works are the ones that justify each part of the method, and they are listed in full in `[references.md](./references.md) when that file is written. The review also has to examine two graph pooling methods, DiffPool and MinCutPool, as alternatives to the learned hierarchy.

T03 organises the documentation itself. It creates the files that let a person or an agent resume the work without the conversation history. This task is what produced the present file.

Stage 1 is complete when the graph validation is saved, the literature review has ten or more works cited in the thesis, and the documentation files exist and link to each other.

## Stage 2: Fixed hierarchy baseline

This stage has four tasks, T04 to T07, and it builds the baseline that everything else is compared against.

T04 writes the specification of the MST hierarchy. It describes the input, the output, the cut criteria and the test cases, before any code is written. This follows the principle that a non-trivial task needs its specification first.

T05 adds an adapter for the SumMe dataset. SumMe is the best dataset to start with because its ground truth and features already come with the HieTaSumm library, so it can be used to test the whole pipeline before the VSUMM dataset is available. The adapter lists the videos and turns their ground truth into a common format.

T06 implements the MST hierarchy using the higra library, the same one used by HieTaSumm, and tests it on a synthetic graph. It must also reproduce the reference hierarchy on one real video. This is the core of the baseline.

T07 runs the whole baseline from end to end on SumMe: graph, hierarchy, keyframe selection and summary. The F-score is computed and saved. This is the first complete result of the work and the reference point for the learned version.

The basis for this stage is the HieTaSumm paper, which defines the MST and the area cut, and the HieTaSumm library, which is the reference implementation. The stage is complete when the baseline produces summaries on SumMe and the F-score is saved as evidence.

## Stage 3: Graph neural network base

This stage has four tasks, T08 to T11.

T08 is a small learning project in PyTorch Geometric: a node classification model, trained and tested. It is part of the learning track, not of the thesis method, and it exists so that the training loop and the library API are familiar before the real components are written. It is kept as a task with its own evidence, but it is not counted as progress on the method.

T09 adds the first neural layer to the method: a graph attention network, GATv2, applied to the time-aware graph. Each frame's features are updated using its direct temporal neighbours. The tests check the shape and the finiteness of the outputs, and the layer is run on one real video. This follows the first processing step of the pipeline in the thesis proposal.

T10 adds the VSUMM dataset. Its ground truth is a set of keyframes chosen by five users, not a score for each frame, so the adapter must convert that into the format used for evaluation. The F-score is computed, and the output is saved. This task is blocked until the dataset is downloaded.

T11 is a decision that belongs to the advisor. It concerns whether, for VSUMM, the thesis reports only the F-score or also derives Kendall's tau and Spearman's rho from the frequency with which each frame was chosen. VideoSAGE uses these rank correlation metrics, which is why they are being considered. T11 does not block the implementation work in later stages, but it blocks the completion of T10, so it must be settled before the final evaluation.

The basis for this stage is the PyTorch Geometric implementation of GATv2, the VideoSAGE paper for the evaluation metrics, and the VOGNet paper for scoring frames. The stage is complete when GATv2 runs on a real video with tested outputs, and the VSUMM evaluation is saved, or the blocker is documented with a decision.

## Stage 4: Learned hierarchy and multigraph

This stage has five tasks, T12 to T16. It is the core of the thesis contribution.

T12 writes the specification of single-link clustering pooling. The rule is that two frames belong to the same cluster when there is a path between them along edges whose feature distance is below a threshold. Because the graph only has edges between frames that are close in time, every cluster formed this way is a continuous segment of the video. The specification defines test cases and the threshold, which is still to be chosen empirically.

T13 implements the SLC pooling layer. The layer is written from the description in the SLC-HELM paper, because the code of that work has not been shared. The layer is tested on synthetic partitions, and temporal contiguity of the clusters is verified on a real video. This is the main new component of the thesis.

T14 writes the methodology section of the first thesis report. By this point the graph, the baseline, the GNN layer and the SLC pooling have all been defined, so the section can describe the full method with evidence behind each part. Its position in the sequence is deliberate: the methodology is written from what has been built, not from what was planned.

T15 implements the multigraph modules, called MGP in the proposal. The hierarchy produces two kinds of edges: temporal edges between nodes at the same level, and hierarchical edges between a node and its parent cluster at the next level up. Each kind is processed in its own branch, and the two results are merged with a multilayer perceptron. The basis is the HELMNet and mHELMNet papers, which introduced this two-edge design. Their code is the style reference.

T16 integrates the SLC pooling with the multigraph modules and tests the combined model on one video.

The basis for this stage is the SLC-HELM paper, which defines the clustering, and the HELMNet and mHELMNet papers, which define the multigraph. The stage is complete when the SLC pooling and the multigraph modules pass their tests, and they work together on a real video. The checkpoint PC4 (18 November) depends on this stage.

## Stage 5: Scale readout and results

This stage has two tasks, T17 and T18.

T17 adds the readout that chooses the keyframes. It has two parts. The first scores each level of the hierarchy, using the region graph readout (RGR) from the HELMNet work, which weights the levels by importance. The second scores each frame inside its level using attention, following the VOGNet approach. Together they decide which frame represents each cluster. This is an initial version, meant to produce keyframes at every level so the method can be evaluated.

T18 runs the comparison between the learned hierarchy (SLC) and the fixed hierarchy (MST) on the same input, and records the F-score in a table saved under [docs/evidence](docs/evidence). These are preliminary results. They do not need to be final, but they must be real and reproducible. The checkpoint PC5 (29 November) depends on this task.

The stage is complete when the keyframes are generated and the SLC and MST results are saved as evidence.

## Summary of the stages

The table below condenses the five stages above. The state column reflects the verification of 16 October 2026 and must be updated in [status.md](./status.md) as work progresses.

| Stage                               | Tasks   | What it delivers                                                   | Basis                                         | State (16 Oct 2026)                                             | Done when                                                              |
| ----------------------------------- | ------- | ------------------------------------------------------------------ | --------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1. Graph and foundations            | T01–T03 | Validated time-aware graph, literature review, documentation files | HieTaSumm paper and Leonardo's reference code | In progress: T01 implemented but not validated; T02 in progress | Validation output saved; ten or more works cited; documentation linked |
| 2. Fixed hierarchy baseline         | T04–T07 | MST hierarchy and end-to-end summaries on SumMe with F-score       | HieTaSumm paper, higra                        | Planned                                                         | Baseline produces SumMe summaries; F-score saved                       |
| 3. Graph neural network base        | T08–T11 | GATv2 on the graph; VSUMM evaluation                               | PyTorch Geometric, VideoSAGE, VOGNet          | Planned; T10 and T11 blocked                                    | GATv2 runs on a real video; VSUMM evaluated or blocker decided         |
| 4. Learned hierarchy and multigraph | T12–T16 | SLC pooling, MGP modules, their integration, methodology section   | SLC-HELM, HELMNet, mHELMNet                   | Planned                                                         | SLC and MGP pass their tests and work together on one video            |
| 5. Readout and results              | T17–T18 | Keyframe selection; SLC versus MST results                         | HELMNet (RGR), VOGNet                         | Planned                                                         | Keyframes generated; results saved as evidence                         |

## Stages not yet covered

Two extensions are outside the guaranteed scope and are only added if the schedule allows. The first is a supervised version, trained with the human summaries from VSUMM, which serves as an upper bound for comparison. The second is unsupervised pre-training on a larger video collection, followed by evaluation on SumMe and TVSum. Both are described in the project scope document. They are not tasks in the sequence above, and they must not be added to the sequence without a decision recorded in [decisions.md](./decisions.md).

The unsupervised training objective itself, which is graph reconstruction, diversity regularisation or a combination, is still to be defined. It is part of the design of stage 4 and must be decided before T13 is finished.
