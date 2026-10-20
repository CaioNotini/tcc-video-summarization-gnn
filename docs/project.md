```
Role: Complete description of the thesis project: problem, objective, research question, method, evaluation plan, scope, constraints and supervision. Read it for the context of the work.
Entry: docs/status.md
Related to: [status.md](./status.md), [phases.md](./phases.md), [architecture.md](./architecture.md), [references.md](./references.md), [decisions.md](./decisions.md), [principles.md](./principles.md), [evidence/](./evidence/)
```

# Project overview

This project is an undergraduate thesis (TCC) in Computer Science at PUC Minas, written by Caio Notini and supervised by Prof. Zenilton Kleber Patrocínio Júnior. It studies unsupervised video summarisation with graph neural networks. This document describes the project as a whole. It does not record progress, which lives in [status.md](./status.md), nor the detailed tasks, which live in [phases.md](./phases.md).

The README, which is still to be written, will be a short overview for human visitors. This document is the complete reference, and it is the one that agents and contributors should read. It is written to be self-contained, so that someone who has never seen the project can understand it from this file alone.

## The proposal in short

The proposed method takes the time-aware frame graph of HieTaSumm and does everything after it with learned components. A GATv2 layer updates the frame features, single-link clustering (SLC) pooling builds the hierarchy from those learned features, a multigraph GNN processes the hierarchy, and a learned readout with frame-level attention scores the frames so that the keyframes can be chosen. Nothing from the combinatorial part of HieTaSumm (the minimum spanning tree, the cut by area and the rule-based keyframe choice) is used after the graph. The only fixed steps are taking the highest-scored frame of each group and ordering the keyframes in time.

Those combinatorial steps appear in the project only as the HieTaSumm baseline, the original method used for comparison. The terms are defined once, in the Terminology section of [decisions.md](./decisions.md).

## The problem

A video summary is a short sequence of frames that represents the whole video. Producing one by hand requires a person to watch the video and choose the representative frames, which is slow and subjective. Automatic summarisation tries to do the same job with a method. Many of the methods in the literature are supervised: they learn from summaries written by people, which means they need labelled data, and the available datasets are small. This thesis works on the opposite assumption. It asks whether a method can produce useful summaries without any human annotation.

The starting point is HieTaSumm, an unsupervised method from the same research group. HieTaSumm builds a graph in which each frame is a node and only temporally close frames are connected. It then builds a hierarchy over that graph with a fixed algorithm, a minimum spanning tree cut by area, and picks one keyframe per group. HieTaSumm uses a graph, but it does not learn anything. Its hierarchy is computed by rules.

## The objective

The objective of the thesis is to build an unsupervised video summarisation method that learns over the time-aware frame graph. The method applies graph neural network layers to the graph, learns a hierarchy over it with single-link clustering (SLC) pooling, and processes that hierarchy with a multigraph that has two kinds of edges. From the resulting hierarchy, it selects one keyframe per group, and the keyframes, kept in their original order, form the summary. Once the method is built, the thesis evaluates it.

The architecture follows the lab's previous work. The graph and the comparison baseline come from HieTaSumm. The learned hierarchy comes from SLC-HELM. The multigraph design and the readout come from HELMNet and mHELMNet. The unsupervised training idea draws on VOGNet and RSGN. The full list of works and the role of each one is in [references.md](./references.md), and the technical structure is in [architecture.md](./architecture.md).

## The research question

The thesis asks one question: is applying a graph neural network to video summarisation an effective and functional method, compared with other methodologies?

The question has two parts, and each needs a precise meaning.

Effective means that the summaries score well on the evaluation datasets, using the F-score as the primary metric, and that the method's result is compared against other methods under the same evaluation protocol. A score is only meaningful relative to something else, so the evaluation always includes comparisons.

Functional means that the method runs end to end on the evaluation datasets, that its results can be reproduced with the same inputs and seeds, and that its running time is measured and reported. The SLC-HELM work claims a speedup over pre-computed hierarchies, so running time is part of the claim being tested.

The comparison is structured as two separate questions, because the word "graph" and the word "learning" mean different things here. The HieTaSumm method uses a graph but does not learn. A neural network without a graph learns but does not use the structure. The thesis therefore compares the method against four kinds of approach: graph with learning (the proposed method), graph without learning (the HieTaSumm baseline with its fixed hierarchy), learning without graph (the same network with the graph message passing removed), and neither (uniform sampling and k-means over the frame features). Only the first and the second cells differ in the presence of learning, and only the first and third differ in the presence of the graph, so each comparison isolates one factor. The comparison between the proposed method and the HieTaSumm baseline is the one that isolates the effect of learning over the graph.

## Method

The method takes a video and turns it into a sequence of frames. Each frame is described by a feature vector from a ResNet50 pre-trained on ImageNet, using the global average pooling output. The frames form a time-aware graph, with edges only between frames within a temporal window. A graph attention layer (GATv2) first updates the frame features using their direct temporal neighbours. Then single-link clustering pooling groups the frames into a hierarchy: two frames join the same cluster when a path of low-distance edges connects them, so every cluster is a continuous segment of the video. Each level of the hierarchy is kept, and the levels are processed together by a multigraph with two kinds of edges: temporal edges within a level and hierarchical edges between a node and its parent. A readout weights the levels by importance, and attention inside each level picks the keyframe. Both the level weights and the frame scores come from learned layers, and no combinatorial rule from HieTaSumm is used after the graph (D-18). The only fixed steps are taking the highest-scored frame of each group and ordering the keyframes in time. The keyframes, in their original order, are the summary.

Training is unsupervised. No human summary is used to train the model. The training objective is still to be decided: it may be graph reconstruction, diversity regularisation between keyframes, or a combination of both. Human summaries are used only to evaluate the output.

## Datasets and metrics

The main dataset is VSUMM, with 50 videos. Each video has five summaries made by different users, each one a set of selected frames. VSUMM does not give a continuous importance score per frame, which matters for the metrics. The F-score is computed against the user summaries. Kendall's tau and Spearman's rho, which rank frames by importance, can only be computed on VSUMM if a per-frame score is derived, for example from how often each frame was chosen by the five users. Whether to report those metrics on VSUMM is an open decision for the advisor.

SumMe is the second dataset. Its importance scores and features come with the HieTaSumm library, so it is the first dataset used to test the pipeline end to end, before VSUMM is ready. TVSum is the third dataset. It provides per-frame importance scores from several annotators, so the rank metrics apply directly. The data layer is designed to support all three from the start, through one adapter per dataset that normalises the ground truth. The full experiments on all three are planned for the second thesis phase.

The evaluation uses several metrics, applied wherever the ground truth allows. Each one answers a different part of the research question, so they are reported together and no single one is selected for its results.

Effectiveness in choosing the right frames is measured by the F-score, the primary metric on all datasets. It is computed against the human summaries, following the protocol used in the evaluation literature.

Effectiveness in ranking frames is measured by Kendall's tau and Spearman's rho, which compare the importance ordering produced by the method with the scores given by the annotators. They apply to SumMe and TVSum, where a per-frame score exists. On VSUMM they apply only if a per-frame score is derived from the five user summaries, which is the open decision T11.

The functional criteria are measured by the running time and the memory use of the full pipeline, reported per video and in total, on the same hardware for every compared method. These numbers are part of the claim that the method is functional, because the related work reports efficiency gains and the thesis must test them.

Every number needs a reference point. The human agreement on SumMe, where the dataset provides it, is reported alongside the methods. A random selection baseline, which on TVSum already reaches a high F-score and therefore shows what chance achieves, is added in the second phase with the other baselines that use no graph. Without these references, a score does not show whether the method is better than chance or close to human performance.

F-score and rank correlation can disagree. The VideoSAGE results show this: its F1 on SumMe is below that of A2Summ and PGL-Sum, while its Kendall's tau and Spearman's rho are higher. The thesis therefore reports both families and discusses the cases where they diverge, rather than choosing the one that favours the method.

Published numbers from other papers are used only as context, not as direct comparisons, because they use different feature extractors and some values are reproductions by their own authors. Every method compared in the thesis is run under the same protocol, which is fixed in a written specification before any result is produced, so that no metric or setting is chosen after seeing the numbers. All compared methods produce summaries of the same length, defined by the protocol, so that the comparison is fair.

## Scope

The first thesis phase (TCC1) ends on 9 December 2026. It covers the literature review with at least ten works, the time-aware graph reproduced and validated against the reference implementation, the HieTaSumm baseline (graph, MST hierarchy and rule-based keyframe choice) run end to end, the SLC pooling integrated with the multigraph modules (the minimum for the fourth checkpoint), an initial version of the readout, and preliminary results. It also covers the written evaluation protocol. With it, the first phase compares the proposed method with the HieTaSumm baseline. The ablation of the proposed network without message passing and the baselines that use no graph (uniform sampling, k-means and random selection) are left for the second phase, so the first phase cannot yet separate the effect of the graph from the effect of learning, and the text says so. It also covers the methodology section of the thesis text.

The second phase (TCC2) is planned for 2027, and its date is not yet confirmed. It covers the complete SLC pooling, the training objective, the full comparison on all three datasets, adding the ablation and the baselines that use no graph to complete the four-cell comparison, the final writing and the defence.

Two extensions are outside the guaranteed scope, and they are added only if the schedule allows and only after a decision is recorded. The first is a supervised version trained on the VSUMM human summaries, used as an upper bound for comparison. The second is unsupervised pre-training on a larger collection of videos, followed by evaluation on the other datasets, to test generalisation.

The following are explicitly out of scope: real-time or streaming summarisation, deployment as a service, audio or text modalities, and any work that introduces human annotation into the training of the main method.

## Constraints

The project is written in Python 3.10 or later. The graph and hierarchy code uses networkx and higra. The feature extraction and the graph neural network use PyTorch, torchvision and PyTorch Geometric. Experiments are tracked with mlflow, which the lab already uses. The original HieTaSumm code was written for Python 3.5 to 3.8 and uses TensorFlow; it is ported, not run as is. Whether the training runs on a local GPU or on a cloud notebook is still open. Feature extraction and graph construction run on a CPU.

The method must remain unsupervised in its main form, and the evaluation must use the same protocol for every compared method. No paid service or cloud infrastructure is introduced without explicit authorisation.

## Supervision and open decisions

The advisor, Prof. Zenilton Kleber Patrocínio Júnior, co-authored HieTaSumm and several of the lab works this thesis builds on, and he confirmed the overall architecture. Some decisions depend on him and are recorded as open items in [status.md](./status.md): how to handle Kendall's tau and Spearman's rho on VSUMM, the unsupervised training objective, and whether the SLC-HELM code can be shared, since it is not available and the layer must be implemented from the paper.

## Milestones

The thesis has four advisor checkpoints. PC3 is on 21 October 2026, and it requires the graph validation and the literature review. PC4 is on 18 November 2026, and it requires the SLC pooling integrated with the multigraph modules. PC5 is on 29 November 2026, and it requires the preliminary results. PC6 is on 9 December 2026, and it closes TCC1 with the consolidated text. The tasks that lead to each checkpoint are listed in [phases.md](./phases.md).

## How to read this project

The mandatory reading order for agents is defined in `CLAUDE.md` and is not repeated here. This file gives the context of the work.

Terms used throughout the project have specific meanings. A frame is one image of the video. A graph is the structure whose nodes are frames and whose edges join frames that are close in time. A cluster is a group of frames that the method treats as redundant with each other. A keyframe is the frame chosen to represent a cluster. A GNN is a neural network that learns from a graph structure. Single-link clustering groups nodes that are connected through low-distance edges. A multigraph has more than one kind of edge. The MST is the minimum spanning tree, the fixed hierarchy used by HieTaSumm.
