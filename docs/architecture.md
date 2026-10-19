```
Role: Technical architecture of the thesis: pipeline, components, code structure, stack, decisions (D-xx), open questions (Q-xx), risks and invariants. Separates what exists in the code from what is only planned.
Entry: docs/status.md
Related to: [status.md](./status.md), [project.md](./project.md), [phases.md](./phases.md), [references.md](./references.md), [decisions.md](./decisions.md), [principles.md](./principles.md), [evidence/](./evidence/)
```

# Architecture

This document describes how the summarisation method is built: its pipeline, the components that implement each step, the structure of the code, the stack, the design decisions and the rules that the code must keep. It is the reference to consult before changing any code.

Every component carries one of two states. **Exists** means the code is in the repository. It does not mean the component is validated: validation is recorded in [status.md](./status.md) and only counts once its output is saved under [evidence/](./evidence/). **Planned** means the component is specified in [phases.md](./phases.md) but not yet written, or written only as an empty placeholder. When this document and the code disagree, the code is the source of truth for what exists, and this document must be corrected.

Decisions are identified as D-01, D-02 and so on, and open questions as Q-01, Q-02 and so on. The same identifiers are used in [status.md](./status.md) and [phases.md](./phases.md), so a decision or a blocker can be referenced from any document.

## Where things stand

```
src/tcc_gnn/
├── graph/        [exists]    frame.py: time-aware graph            (T01: validation not yet saved)
├── features/     [exists]    features.py, frame_extraction.py      (no tests yet)
│                             tf_resnet_extractor.py                (validation only)
├── data/         [planned]   base.py is empty                      (T05, T10)
├── hierarchy/    [planned]   MST baseline, SLC pooling             (T06, T13)
├── models/       [planned]   GATv2, MGP, readout                   (T09, T15, T17)
├── training/     [planned]   unsupervised loops                    (after T13, objective open)
└── evaluation/   [planned]   F-score, tau, rho, time, memory       (T07, T18)

experiments/      [exists]    validate_graph.py, validate_graph_torchvision.py
tests/            [exists]    test_frame_graph.py, test_smoke.py
references/       [exists]    git submodules: HieTaSumm-lib, mHELMNet (read-only)
```

## Pipeline

The method takes a video and returns a summary, through seven steps. The first step is the graph construction, which is the base of everything else. The second is the feature extraction, which gives each node its initial description. The third applies graph attention over the temporal neighbours, so that each frame's features reflect its immediate context. The fourth builds the hierarchy, which is either the fixed MST of HieTaSumm (the baseline) or the learned SLC pooling (the main method). The fifth processes the hierarchy as a multigraph with two kinds of edges, using separate branches that are merged by a perceptron. The sixth reads the hierarchy at each scale, weights the levels by importance, and picks one keyframe per group using frame-level attention. The seventh assembles the keyframes in their original order and evaluates the result.

The baseline and the main method share steps one, two, three, five, six and seven. They differ only in step four. This is the design that makes the comparison between a learned and a fixed hierarchy a controlled one.

```
 video
   │
   ▼
 [1] graph  ──────────►  time-aware frame graph        exists
   │
   ▼
 [2] features ────────►  2048-d vector per frame       exists
   │
   ▼
 [3] GATv2 ───────────►  frames updated from neighbours planned
   │
   ▼
 [4] hierarchy ───────►  SLC (main)  or  MST (baseline) planned
   │
   ▼
 [5] multigraph (MGP) ►  temporal + hierarchical edges  planned
   │
   ▼
 [6] readout + attention ► one keyframe per group       planned
   │
   ▼
 [7] summary + evaluation ► F-score, tau, rho, time     planned
```

## Graph construction (step 1, exists)

Each frame is a node. An edge joins two nodes only when their frames are within a temporal window, which keeps the graph sparse and local. The edge weight is the feature distance between the two frames, following HieTaSumm.

```
frames      1    2    3    4    5    6    7    8    9   10
            ●────●────●────●
                 ●────●────●────●
                      ●────●────●────●
                           ●────●────●────●
                                ●────●────●────●

each frame links only to frames inside its temporal window
→ the graph is a band along the diagonal, not a complete graph
```

The graph is built by `build_time_aware_graph` in `src/tcc_gnn/graph/frame.py`, a port of the `Frame` class of the HieTaSumm library, validated against the reference in T01. The window end for each frame is computed by `calc_window_end`. Variable names and algorithmic logic are kept faithful to the original so that outputs can be compared directly.

## Feature extraction (step 2, exists)

Each frame is described by a 2048-dimensional vector: the global average pooling output of a ResNet50 pre-trained on ImageNet, computed with torchvision. This replaces the 1000-dimensional softmax output of the original code, which saturates and discards information that later steps need (D-03).

```
 video file ──► frame_extraction.py (OpenCV) ──► frames ──► features.py (torchvision ResNet50, avgpool)
                                                              │
                                                              ▼
                                                 (n_frames × 2048) feature matrix

 tf_resnet_extractor.py (TensorFlow) ──► used only to check fidelity against the original; not in production
```

## Graph attention over the base graph (step 3, planned, T09)

A GATv2 layer from PyTorch Geometric (`GATv2Conv`) updates each frame's features from its direct temporal neighbours. Attention weights are learned per pair of nodes. Because the graph has only temporal edges, the attention is always local. The layer is not implemented from scratch, and the edge distance may be passed as an edge attribute. Planned location: `src/tcc_gnn/models/`.

## Hierarchy (step 4, planned)

Both implementations produce the same kind of output: a set of levels, each a partition of the frames into groups, with each node linked to its parent group. Because the output has the same shape, the two can be swapped in the same pipeline.

```
 MST baseline (T06)                          SLC pooling, main method (T13)

 graph ──► minimum spanning tree             graph ──► learned features
           │                                            │
           ▼                                            ▼
 cut by area (HieTaSumm rule, fixed)         merge frames joined by a path of
           │                                 edges below a distance threshold
           ▼                                            │
 hierarchy (levels fixed by rule)            ▼
                                             hierarchy (levels learned in training)
```

A hierarchy built by either method looks like this. Each group is a contiguous segment of the video, which holds because the edges join only temporally close frames:

```
 level 2   [────────────────── group A ──────────────────]
 level 1   [──── A1 ─────]  [────────── A2 ──────────]
 level 0    1  2  3  4  5    6  7  8  9  10  11  12
```

The SLC threshold is still open (Q-02). The MST baseline uses higra, the library the HieTaSumm code and the lab's multigraph code also use (D-05).

## Multigraph processing (step 5, planned, T15 and T16)

At each level, the graph has two kinds of edges. Temporal edges join nodes within the same level. Hierarchical edges join a node to its parent at the next level up. The two kinds are processed in separate branches, and the results are merged by a multilayer perceptron. This is the design of HELMNet and mHELMNet (D-07).

```
 level 2     [G2]
               ║  hierarchical edge (node → parent)
 level 1     [G1a] ────── temporal ────── [G1b]
               ║                           ║
 level 0     (1) ─ (2) ─ (3) ─ (4) ─ (5) ─ (6)        temporal edges inside level 0

 per level:
   temporal edges      ──► branch T ──┐
                                      ├──► MLP ──► updated node features
   hierarchical edges  ──► branch H ──┘
```

Following mHELMNet, the features of the edges are fixed after the initial embedding and are not updated in each layer (D-08). The processing is repeated each time a new level is added, so the model keeps every level, not only the last. The building blocks follow mHELMNet's style: `Linear`, `BatchNorm` and `GraphSizeNorm`, and a gated graph convolution layer as the reference for the branch design.

## Readout and keyframe selection (step 6, planned, T17)

The readout is adapted from the region graph readout (RGR) of HELMNet. It produces an importance weight for each level. Inside each level, frame-level attention, following VOGNet, scores each frame. The keyframe of a group is the frame with the highest combined score in the most important level that covers it.

```
 level 2 ──► importance β2 ──┐
 level 1 ──► importance β1 ──┼──► weights per level
 level 0 ──► importance β0 ──┘

 inside each level:  frame-level attention scores each frame in the group
 keyframe of a group = frame with the highest combined score
 summary            = keyframes, in their original temporal order
```

Planned location: `src/tcc_gnn/models/`.

## Training (planned, objective open)

Training is unsupervised: no human summary is used. The objective is still open (Q-03). The candidates are graph reconstruction, as in VOGNet and RSGN, diversity regularisation between keyframes, as in DMoN-DPR, or a combination. The decision must be recorded in `decisions.md` before T13 is finished.

```
 frame graph ──► model ──► hierarchy + keyframes
                    │
                    └──► loss = reconstruction and/or diversity   (objective open, Q-03)
                         no human labels at any point
```

Training loops go in `src/tcc_gnn/training/`, and the experiments that run them in `experiments/`, tracked with mlflow.

## Data layer (planned, T05 and T10)

Each dataset has its own adapter in `src/tcc_gnn/data/`. An adapter lists the videos and converts the ground truth into one common format, so that evaluation does not depend on the dataset. The formats differ: SumMe and TVSum give importance scores from several annotators, while VSUMM gives five sets of selected keyframes per video, without a continuous score. This difference is the reason the rank metrics do not apply directly to VSUMM (Q-01).

```
 VSUMM ──► adapter_vsumm ──┐
                           │
 SumMe ──► adapter_summe ──┼──► common format ──► evaluation/
                           │    (keyframes, scores)
 TVSum ──► adapter_tvsum ──┘
```

The base class in `src/tcc_gnn/data/base.py` is empty and will define the common interface. SumMe is the first dataset used, and its data comes from the HieTaSumm library submodule.

## Evaluation (step 7, planned, T07 and T18)

Every compared method is run through the same evaluation code and the same protocol. The protocol, covering how keyframes become a summary and how the F-score is computed, is fixed in a written specification before any result is produced (D-13, Principle 3).

```
 experiments/run_eval.py

 video ──► method (proposed, baseline, ablation or simple baseline) ──► keyframes ──► summary
                                                                                      │
 ground truth ──► adapter ──────────────────────────────────────────────────────────►  ├──► F-score
                                                                                      ├──► tau, rho  (where available)
                                                                                      ├──► running time, memory
                                                                                      ▼
                                                                                   mlflow
```

The comparison covers four cells, and each cell changes one factor at a time:

```
                       WITHOUT LEARNING              WITH LEARNING
             ┌──────────────────────────┐   ┌──────────────────────────────┐
 WITHOUT     │ uniform sampling         │   │ same network without         │
 GRAPH       │ k-means on frame features│   │ message passing (ablation)   │
             │ (floor)                  │   │                              │
             └──────────────────────────┘   └──────────────────────────────┘
             ┌──────────────────────────┐   ┌──────────────────────────────┐
 WITH        │ HieTaSumm (MST + area)   │   │ proposed GNN                 │
 GRAPH       │ (baseline)               │   │ (SLC + multigraph)           │
             └──────────────────────────┘   └──────────────────────────────┘
```

All four cells belong to TCC1 (D-16). The SLC-versus-MST comparison is a sub-question inside the bottom-right cell.

## Code structure

The package is `src/tcc_gnn/`. Directories marked as planned are empty placeholders today.

```
tcc-video-summarization-gnn/
├── README.md                   # human-facing overview
├── pyproject.toml              # dependencies and optional extras (tf-validation)
├── configs/
│   └── default.yaml
├── docs/                       # this documentation
├── src/tcc_gnn/
│   ├── graph/                  # frame.py                                  [exists]
│   ├── features/               # features.py, frame_extraction.py,          [exists]
│   │                           # tf_resnet_extractor.py (validation only)
│   ├── data/                   # base.py (empty), one adapter per dataset  [planned]
│   ├── hierarchy/              # MST baseline, SLC pooling                 [planned]
│   ├── models/                 # GATv2 base, MGP modules, readout          [planned]
│   ├── training/               # unsupervised loops                        [planned]
│   └── evaluation/             # F-score, tau, rho, time, memory           [planned]
├── tests/                      # test_frame_graph.py, test_smoke.py        [exists]
├── experiments/                # validate_graph.py, validate_graph_torchvision.py  [exists]
├── references/                 # submodules: HieTaSumm-lib, mHELMNet      [exists, read-only]
└── data/                       # raw and processed data, git-ignored       [exists, empty]
```

The dependency direction is one-way. Code in a lower layer never imports from a higher one, which keeps each piece testable on its own.

```
 experiments/
      │
      ▼
 training/ ──────────────► evaluation/ ◄────── data/ (ground truth)
      │                                              ▲
      ▼                                              │
 models/ ◄── hierarchy/ ◄── graph/ ◄── features/ ◄── data/ (videos)
```

Submodules are read-only references. They must be populated with `git submodule update --init --recursive` before the validation and the tests that depend on them can run.

## Stack

| Layer                | Tool                            | Reason                                                                               |
| -------------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| Language             | Python 3.10+                    | Required by the PyTorch Geometric version in use                                     |
| Graph and hierarchy  | networkx, higra                 | networkx for the graph; higra for watershed and MST, as HieTaSumm and mHELMNet use   |
| Features             | PyTorch, torchvision (ResNet50) | Production extractor; avgpool output (D-03)                                          |
| Frame extraction     | OpenCV                          | Reads video frames                                                                   |
| Graph neural network | PyTorch Geometric 2.3+          | Provides `GATv2Conv` and the scatter operations; `torch_scatter` not required (D-11) |
| Validation only      | TensorFlow (optional extra)     | Checks fidelity against the original feature extraction (D-04)                       |
| Data files           | h5py                            | Reads SumMe and VSUMM feature and ground truth files                                 |
| Experiment tracking  | mlflow                          | Already used by the lab (D-12)                                                       |
| Tests                | pytest                          | Unit tests per module                                                                |
| Code style           | PEP 8, type hints               | Modern style; ported logic keeps original names (D-10)                               |

Whether training runs on a local GPU or a cloud notebook is open (Q-08). Feature extraction and graph construction run on CPU.

## Decisions

The decisions that shape this architecture are recorded in [decisions.md](./decisions.md), with their context, status, alternatives and consequences. This document does not repeat them. Where a component depends on a decision, it cites the identifier (for example, D-03 for the feature extractor), and the reader follows the link to find the reasoning.

## Open questions

Open questions are blockers or pending decisions. Each one names the task it affects. The owner says who has to answer.

| ID   | Question                                                                                                                                        | Affects    | Owner      |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- |
| Q-01 | How to report Kendall's tau and Spearman's rho on VSUMM, which lacks per-frame scores: F-score only, or a derived score from the five summaries | T10, T11   | Advisor    |
| Q-02 | Value of the SLC distance threshold; choose empirically and record                                                                              | T12, T13   | Researcher |
| Q-03 | Unsupervised objective: reconstruction, diversity, or both                                                                                      | T13        | Advisor    |
| Q-04 | Whether the SLC-HELM code can be shared; otherwise implement from the paper                                                                     | T13        | Advisor    |
| Q-06 | TVSum scope: adapter in TCC1 and experiments in TCC2, or a single decision                                                                      | T05, TCC2  | Advisor    |
| Q-08 | Training hardware: local GPU or cloud notebook                                                                                                  | T13 onward | Researcher |
| Q-09 | Hardware specification used to measure running time and memory                                                                                  | T18        | Researcher |

## Risks

| Risk                                                 | Impact                                 | Mitigation                                                                                          |
| ---------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------- |
| The GNN does not beat the non-learned baselines      | High: weakens the thesis claim         | Report the result as it is; the four-cell matrix shows where the gain is missing                    |
| SLC cannot be reproduced without the original code   | High for T13                           | Implement from the paper; test on synthetic partitions; record every interpretation in decisions.md |
| VSUMM is small (50 videos), so the model may overfit | Medium                                 | Unsupervised training; SumMe and TVSum as further checks                                            |
| F-score and rank correlation disagree                | Medium: changes the reading of results | Report both and discuss divergent cases (see project.md)                                            |
| The port drifts from the reference implementation    | High for credibility                   | T01 validation saved; any change to the graph code is validated again                               |
| Task cadence is not met before PC4                   | High for the schedule                  | Linear order by dependency; the first candidate to rescope is T08, which is learning, not method    |

## Invariants

These rules hold in every change to the code. Edges join only frames within the temporal window, so the graph stays sparse and local. Clusters produced by SLC pooling are contiguous segments of the video, which holds only because the edges are temporal. Feature vectors have 2048 dimensions and come from the same extractor for every compared method. The graph construction keeps producing the output validated in T01; any change to it is followed by the validation again. Baseline and main method receive the same graph, so that only the hierarchy differs (D-14). Every compared method uses the same evaluation code and protocol, and produces summaries of the same length (D-16).

## Links

- [status.md](./status.md): current state of each task and the next task to do
- [project.md](./project.md): objective, research question, method and scope
- [phases.md](./phases.md): the five phases and the tasks, with their acceptance criteria
- [references.md](./references.md): the works behind each component
- [decisions.md](./decisions.md): recorded decisions, with context and consequences
- [principles.md](./principles.md): the ten working principles
- [evidence/](./evidence/): saved outputs that prove a task is complete
