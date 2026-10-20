```
Role: Defines the TCC stages and tasks, what each one delivers, where its foundation comes from, and when it is considered complete.
Entry: docs/status.md
Related to: [status.md](./status.md), [project.md](./project.md), [references.md](./references.md), [architecture.md](./architecture.md), [decisions.md](./decisions.md), [principles.md](./principles.md)
```

# Stages of the work

This document explains how the thesis work is divided into stages and tasks, what each one delivers, where its ideas come from, and when it counts as done. It is written for both people and AI agents. This file defines the plan. [status.md](./status.md) records the progress, so no task state is written here.

The work is a single line of tasks. Tasks are done one at a time, in the order of their dependencies, and each task lists what it depends on. T01 to T18 keep the original numbering. T19 was added later, when the evaluation protocol (D-13) was turned into a task, and it sits where its dependencies place it. T20 and T21 were removed, and their numbers are not reused. Each task has an acceptance criterion, and a task is only done when its evidence is saved in [evidence/](./evidence/). Dates do not decide progress. The advisor checkpoints are fixed external deadlines, and each one depends on a group of tasks, as listed in the next section.

## Why the work is organised this way

The thesis takes the video summarisation method HieTaSumm and changes the part that comes after the graph. HieTaSumm builds a hierarchy of video frames with a fixed algorithm, a minimum spanning tree (MST) cut by area, and picks the keyframes by rule. The proposed method keeps only the time-aware graph of HieTaSumm and learns everything after it (D-18). It learns the hierarchy with single-link clustering (SLC), inspired by the SLC-HELM work from the laboratory, and it learns the choice of keyframes with a GNN. The original HieTaSumm method is kept as the HieTaSumm baseline, so the proposed method can be compared against it on the same input. The terms are defined in the Terminology section of [decisions.md](./decisions.md).

Because of this, the stages follow the logic of the method. First the input graph must be correct. Then the evaluation protocol is written and the HieTaSumm baseline must work end to end, so there is something to compare against. Then the neural network is built on top of the graph, and finally the learned hierarchy, the readout and the comparison are added. A stage that depends on another one is not started until its predecessor has evidence behind it.

## Checkpoints

The advisor checkpoints are fixed external deadlines, and their dates are in [project.md](./project.md). Each one depends on tasks, not on the calendar.

- **PC3:** T01 and T02.
- **PC4:** T16, which requires T12, T13 and T15.
- **PC5:** T18, which requires T17.
- **PC6:** T14 and the consolidated thesis text.

## Stage 1: Graph and foundations

This stage has three tasks, T01 to T03. Its basis is the HieTaSumm paper and its reference code, which are in the `references/` submodules. The stage is complete when the graph validation is saved, the literature review has ten or more works cited in the thesis, and the documentation files exist and link to each other.

### T01: Validate the time-aware graph

- **Depends on:** none.
- **Basis:** the HieTaSumm paper and the reference code of Leonardo, its author (D-02).
- **Goal:** show that the graph built by this repository reproduces the output of the reference implementation. Each video frame is a node, and an edge connects two frames only when they are close in time.
- **Acceptance criterion:** the comparison with the reference passes on at least one real video, the tests pass, and both outputs are saved in `evidence/`.
- **Code:** [`frame.py`](../src/tcc_gnn/graph/frame.py), [`validate_graph.py`](../experiments/validate_graph.py), [`test_frame_graph.py`](../tests/test_frame_graph.py).

### T02: Literature review

- **Depends on:** none.
- **Basis:** the works listed in [references.md](./references.md).
- **Goal:** cover at least ten works that justify each part of the method, and examine two graph pooling methods, DiffPool and MinCutPool, as alternatives to the learned hierarchy. It is needed for PC3.
- **Acceptance criterion:** at least ten works are cited in the thesis text, the works not yet cited are added to its bibliography, DiffPool and MinCutPool are read and summarised, and references.md matches the bibliography.

### T03: Documentation files

- **Depends on:** none.
- **Basis:** [principles.md](./principles.md).
- **Goal:** create the files that let a person or an agent resume the work without the conversation history.
- **Acceptance criterion:** an inspection reports no high-severity finding in the documents.

## Stage 2: Evaluation protocol and HieTaSumm baseline

This stage has five tasks: T04, T05, T19, T06 and T07. It builds the evaluation protocol and the HieTaSumm baseline that the proposed method is compared against. Its basis is the HieTaSumm paper, which defines the MST and the area cut, and the HieTaSumm library, which is the reference implementation. The stage is complete when the HieTaSumm baseline produces summaries on SumMe under the written protocol and its F-score is saved as evidence.

### T04: Specify the MST hierarchy

- **Depends on:** T03.
- **Basis:** the HieTaSumm paper.
- **Goal:** write the input, the output, the cut criteria and the test cases of the MST hierarchy before any code is written, following the principle that a non-trivial task needs its specification first.
- **Acceptance criterion:** the specification is written in this file, under this task, and approved by the researcher.

### T05: SumMe adapter

- **Depends on:** T03.
- **Basis:** the SumMe data that comes with the HieTaSumm library (D-15).
- **Goal:** SumMe is the first dataset used, because its ground truth and features already come with the library, so it can test the whole pipeline before VSUMM is handled. The adapter lists the videos and turns their ground truth into a common format.
- **Acceptance criterion:** the adapter lists the videos and returns the ground truth in the common format, with a test on at least one video.

### T19: Specify the evaluation protocol

- **Depends on:** T05.
- **Basis:** D-13, and the evaluation sections of the VideoSAGE and HieTaSumm papers.
- **Goal:** fix in writing, before any result is produced, how keyframes become a summary, the summary length used for every method, how the F-score is computed for each dataset, how a per-frame score is derived where it is needed (Q-01), and how running time and memory are measured (Q-09).
- **Acceptance criterion:** the specification is saved as `docs/evaluation-protocol.md`, linked from this file and from [architecture.md](./architecture.md), and approved by the researcher. Points that depend on the advisor are listed in it as open.

### T06: MST hierarchy

- **Depends on:** T04.
- **Basis:** the HieTaSumm paper, and the higra library (D-05).
- **Goal:** implement the MST hierarchy with higra, the same library HieTaSumm uses. This is the core of the HieTaSumm baseline.
- **Acceptance criterion:** the tests on a synthetic graph pass, and the hierarchy reproduces the reference hierarchy on one real video, with the output saved in `evidence/`.

### T07: HieTaSumm baseline end to end on SumMe

- **Depends on:** T05, T06 and T19.
- **Basis:** the HieTaSumm paper and library.
- **Goal:** run the original HieTaSumm method, which is the graph, the MST hierarchy and the rule-based keyframe choice, on SumMe, from video to summary. This also implements the F-score in `src/tcc_gnn/evaluation/`, under the protocol of T19, so that later tasks reuse the same evaluation code. This is the first complete result of the work and the reference point for the proposed method.
- **Acceptance criterion:** summaries are produced for the SumMe videos, the F-score is computed under the written protocol, and the results are saved in `evidence/`.

## Stage 3: Graph neural network base

This stage has four tasks, T08 to T11. Its basis is the PyTorch Geometric implementation of GATv2, the VideoSAGE paper for the evaluation metrics, and the VOGNet paper for scoring frames. The stage is complete when GATv2 runs on a real video with tested outputs, and the VSUMM evaluation is saved, or the blocker is documented with a decision.

### T08: PyTorch Geometric learning project

- **Depends on:** none.
- **Basis:** the PyTorch Geometric documentation.
- **Goal:** a small node classification model, trained and tested, so that the training loop and the library API are familiar before the real components are written. It is part of the learning track, not of the thesis method, and it is not counted as progress on the method.
- **Acceptance criterion:** the model trains and its tests pass, with the output saved in `evidence/`.

### T09: GATv2 on the time-aware graph

- **Depends on:** T07 and T08.
- **Basis:** `GATv2Conv` from PyTorch Geometric (D-06).
- **Goal:** add the first neural layer of the method. Each frame's features are updated using its direct temporal neighbours. This is step three of the pipeline in [architecture.md](./architecture.md).
- **Acceptance criterion:** tests check the shape and the finiteness of the outputs, and the layer runs on one real video, with the output saved in `evidence/`.

### T10: VSUMM adapter and F-score

- **Depends on:** T07 and T19. Its rank-metric part also depends on T11.
- **Basis:** the VSUMM dataset, and the VideoSAGE paper for the metrics.
- **Goal:** the VSUMM ground truth is a set of keyframes chosen by five users, not a score for each frame, so the adapter must convert it into the format used for evaluation (D-15). The F-score is computed.
- **Acceptance criterion:** the adapter is tested, the F-score is computed on VSUMM videos under the written protocol, and the results are saved in `evidence/`.

### T11: Decide the rank metrics on VSUMM

- **Depends on:** none. The decision belongs to the advisor (Q-01).
- **Basis:** the VideoSAGE paper, which uses Kendall's tau and Spearman's rho.
- **Goal:** decide whether, for VSUMM, the thesis reports only the F-score or also derives Kendall's tau and Spearman's rho from the frequency with which each frame was chosen by the five users. The decision does not block implementation work in later stages, but it blocks the completion of T10.
- **Acceptance criterion:** the decision is recorded in [decisions.md](./decisions.md).

## Stage 4: Learned hierarchy and multigraph

This stage has five tasks, T12 to T16. It is the core of the thesis contribution. Its basis is the SLC-HELM paper, which defines the clustering, and the HELMNet and mHELMNet papers, which define the multigraph. The stage is complete when the SLC pooling and the multigraph modules pass their tests and work together on a real video.

### T12: Specify the SLC pooling

- **Depends on:** T09.
- **Basis:** the SLC-HELM paper.
- **Goal:** write the specification of single-link clustering pooling. Two frames belong to the same cluster when there is a path between them along edges whose feature distance is below a threshold. Because the graph only has edges between frames that are close in time, every cluster formed this way is a continuous segment of the video. The specification defines the test cases and how the threshold is chosen (Q-02).
- **Acceptance criterion:** the specification is written in this file, under this task, and approved by the researcher.

### T13: SLC pooling

- **Depends on:** T12. It also depends on the answers to Q-02, Q-03 and Q-04, or on a recorded decision about each.
- **Basis:** the SLC-HELM paper, and mHELMNet for code style.
- **Goal:** implement the SLC pooling layer from the description in the paper, because the code of that work has not been shared. This is the main new component of the thesis.
- **Acceptance criterion:** the layer passes tests on synthetic partitions, temporal contiguity of the clusters is verified on a real video, and the output is saved in `evidence/`.

### T14: Methodology section

- **Depends on:** T13.
- **Basis:** the whole method as built.
- **Goal:** write the methodology section of the first thesis report. By this point the graph, the HieTaSumm baseline, the GNN layer and the SLC pooling have all been defined, so the section can describe the method with evidence behind each part. Its position in the sequence is deliberate: the methodology is written from what has been built, not from what was planned.
- **Acceptance criterion:** the section is written and reviewed by the researcher.

### T15: Multigraph modules (MGP)

- **Depends on:** T13.
- **Basis:** the HELMNet and mHELMNet papers (D-07, D-08). The mHELMNet code is the style reference.
- **Goal:** the hierarchy produces two kinds of edges: temporal edges between nodes at the same level, and hierarchical edges between a node and its parent cluster at the next level up. Each kind is processed in its own branch, and the two results are merged with a multilayer perceptron.
- **Acceptance criterion:** tests check shapes and gradient flow, and the output is saved in `evidence/`.

### T16: Integration of SLC pooling and the multigraph modules

- **Depends on:** T15.
- **Basis:** none beyond the tasks it integrates.
- **Goal:** integrate the SLC pooling with the multigraph modules and test the combined model.
- **Acceptance criterion:** the integration tests pass, and the combined model runs on one real video, with the output saved in `evidence/`.

## Stage 5: Readout and results

This stage has two tasks: T17 and T18. Its basis is the region graph readout (RGR) of HELMNet and the frame attention of VOGNet. The stage is complete when the keyframes are generated and the comparison results are saved as evidence.

### T17: Readout and keyframe selection

- **Depends on:** T16.
- **Basis:** HELMNet for the readout by scale, and VOGNet for the attention by frame (D-18).
- **Goal:** add the readout that chooses the keyframes. It has two parts. The first scores each level of the hierarchy, using the region graph readout (RGR), which weights the levels by importance. The second scores each frame inside its level using attention. Both sets of scores come from learned layers, and no combinatorial rule from HieTaSumm is used to pick keyframes. The only fixed step is taking the highest-scored frame of each group. This is an initial version, meant to produce keyframes at every level so the method can be evaluated.
- **Acceptance criterion:** the model produces keyframes for a real video, its tests pass, and the output is saved in `evidence/`.

### T18: Comparison of the methods

- **Depends on:** T17. For results on VSUMM, it also depends on T10.
- **Basis:** D-13, D-14 and D-16.
- **Goal:** run the proposed method and the compared methods on the same input and record the results in a table. The methods are the proposed method and the HieTaSumm baseline. Both use the same graph, the same features, the same evaluation code and the same summary length. The metrics are the F-score on every dataset that has an adapter at this point, Kendall's tau and Spearman's rho where the ground truth allows them, and the running time. These are preliminary results. They do not need to be final, but they must be real and reproducible.
- **Acceptance criterion:** the results table is saved in `evidence/`, and every number in it can be reproduced from the saved commands.

## Summary of the stages

The table condenses the stages. The state of each task is not recorded here. It lives only in [status.md](./status.md).

| Stage                                         | Tasks                   | What it delivers                                                                  | Basis                                         | Done when                                                                                                |
| --------------------------------------------- | ----------------------- | --------------------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1. Graph and foundations                      | T01 to T03              | Validated time-aware graph, literature review, documentation files                | HieTaSumm paper and Leonardo's reference code | Validation output saved, ten or more works cited, documentation inspected without high-severity findings |
| 2. Evaluation protocol and HieTaSumm baseline | T04, T05, T19, T06, T07 | Written protocol, MST hierarchy, HieTaSumm baseline with its F-score on SumMe     | HieTaSumm paper and library, higra            | The HieTaSumm baseline produces SumMe summaries under the protocol and its F-score is saved              |
| 3. Graph neural network base                  | T08 to T11              | GATv2 on the graph, VSUMM evaluation                                              | PyTorch Geometric, VideoSAGE, VOGNet          | GATv2 runs on a real video, VSUMM is evaluated or the blocker is decided                                 |
| 4. Learned hierarchy and multigraph           | T12 to T16              | SLC pooling, MGP modules and their integration, methodology section               | SLC-HELM, HELMNet, mHELMNet                   | SLC and MGP pass their tests and work together on one video                                              |
| 5. Readout and results                        | T17, T18                | Keyframe selection, comparison of the proposed method with the HieTaSumm baseline | HELMNet (RGR), VOGNet                         | Keyframes generated and comparison results saved as evidence                                             |

## Extensions and open decisions

Two extensions are outside the guaranteed scope and are only added if the schedule allows. The first is a supervised version, trained with the human summaries from VSUMM, which serves as an upper bound for comparison. The second is unsupervised pre-training on a larger video collection, followed by evaluation on SumMe and TVSum. Both are described in [project.md](./project.md). They are not tasks in the sequence above, and they must not be added to it without a decision recorded in [decisions.md](./decisions.md). The ablation without message passing and the simple baselines that use no graph (uniform sampling, k-means and random selection) are also not tasks in the first phase (D-16).

The unsupervised training objective, which is graph reconstruction, diversity regularisation or a combination, is still to be decided (D-17, Q-03). It is part of the design of stage 4 and must be decided before T13 is finished.
