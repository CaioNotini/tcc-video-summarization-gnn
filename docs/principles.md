```
Role: Mandatory reading for agents and people. Defines the ten principles that guide how the thesis work is planned, implemented and verified.
Entry: docs/status.md
Related to: [status.md](./status.md), [phases.md](./phases.md), [architecture.md](./architecture.md), [decisions.md](./decisions.md), [references.md](./references.md), [project.md](./project.md), [evidence/](./evidence/)
```

# Working principles

This document sets out the ten principles that govern the development of this thesis. They apply to every task, whether it is done by a person or by an AI agent such as Claude Code. An agent must read this file before making any change, and must follow it even when the task seems small. The principles exist so that the work stays clean, effective and well structured, and so that anyone can resume it later without relying on the history of a conversation.

Each principle below states the rule, explains what it means for this project, and says what to do in practice. Where a principle depends on a state or a file, the state names are the ones used in `status.md`: validated, implemented but not validated, in progress, planned and blocked.

## 1. Context before execution

Before changing anything, the agent must understand the objective, the current state and the constraints. In this project, that means reading `status.md` to find the active task, then the task's entry in `phases.md`, and then the parts of `architecture.md` that the task touches. The agent should not assume that the existing code represents the intended design. The code may be an unfinished step, and `architecture.md` is the reference for what is planned.

In practice, the agent states the active task and its acceptance criterion before opening any source file. If `status.md` and the code disagree about the state of a task, the agent reports the disagreement and does not silently choose one.

## 2. Specification before implementation

A non-trivial task must be specified before it is coded. The specification covers the objective, the scope, what is explicitly out of scope, the technical constraints, the acceptance criteria and the tests that will prove the criteria are met. Ambiguities that would force an irreversible decision must be raised before the implementation starts.

For this thesis, the specification lives in `phases.md`, under the task's entry, or in a dedicated document linked from there. The MST hierarchy (T04), for example, must have its input, output, cut rule and test cases written before T06 starts. The specification for the SLC pooling (T12) must define the distance threshold approach before T13 is coded, because that threshold is an open question.

## 3. Autonomy with validation

The agent implements the smallest solution that meets the criterion, then checks the result before declaring the task complete. Checking means running the test suite or the validation script and keeping the actual output. Saying that something works is not enough. The output must be real, reproducible and saved where the status can point to it.

This principle also defines how the state of a task changes. A task is "implemented but not validated" once the code exists. It becomes "validated" only when the evidence is saved in `evidence/` and linked from `status.md`. Results recalled from memory, or from earlier conversations, are not evidence. The graph validation against the reference implementation (T01) is an example: the comparison has been reported, but until its output is saved, the task is not validated.

## 4. Small and traceable changes

The agent avoids refactoring that is unrelated to the task. It preserves existing behaviour, and it reviews its own changes before reporting them. Migrations, public interfaces and sensitive settings are not changed without first assessing the impact. In a research project, this matters most for the code that reproduces the reference work: changing the graph construction in `graph/` may silently break the agreement with HieTaSumm, so any change there must be followed by the validation in T01.

Each change should be small enough to explain in a sentence, and it should be possible to trace it back to the task that required it.

## 5. Recorded decisions

Relevant decisions are recorded in `decisions.md`, with their context, the alternatives that were considered, the decision taken, the justification and the consequences. A decision is anything that changes how the method works or how it is evaluated. Examples are the choice of the MST as the baseline, the choice of higra as the hierarchy library, and the choice of torchvision in place of TensorFlow for feature extraction.

The agent must not create abstractions for cases that might happen later. If a piece of code is needed only for the current task, it should be written for that task. Generality is added when a second real use appears, and the reason for it is recorded.

## 6. Connected documentation

The documents form a graph of knowledge, and each important fact lives in one place, with links to it from the places that need it. The rule for navigation is the three-jump rule described below. The rule for duplication is that a fact has a single source. When the same information appears in two documents, one of them must become a link to the other.

For example, the list of works behind a component belongs in `references.md`. `phases.md` links to it rather than repeating the titles. `architecture.md` links to the component and the task, and does not restate the paper's method.

## 7. Traceable decisions and their reasons

This principle is a reminder that a decision without a reason is not finished. When the agent makes or proposes a decision, it records the reason and the evidence behind it. When a decision is later changed, the old entry is kept and the new one links to it. The history of a method is part of its documentation, because a future reader needs to know why the current design is what it is.

In this thesis, the decision to learn the hierarchy with SLC pooling, instead of using only the fixed MST, is the central decision. Its reasons come from the SLC-HELM paper and from the contiguity argument, and they must be recorded in `decisions.md` before the implementation is treated as the method.

## 8. Bounded autonomy

Agents may investigate, propose and implement within the scope of a task. They may not make methodological decisions or change the scope of the thesis without a justification and the approval of the person responsible for the project. This is the boundary that protects the thesis from drift.

Concretely, the following require explicit approval before they are done: changing the datasets or the evaluation metrics, adding a supervised training objective to the main method, adding any dependency that changes how the experiments are reproduced, and moving work from the guaranteed scope into the extensions. The open decisions in `phases.md` (T11, and the choice of the unsupervised objective before T13) are examples of decisions that belong to the researcher and advisor, not to the agent.

## 9. Lean documentation

Each piece of information has one main source, and the documentation avoids repetition, contradictory rules and updates that serve no purpose. Files are kept short enough to read in one sitting. A new file is created only when an existing one cannot hold the information without becoming unclear.

The agent should prefer updating `status.md` and the relevant section of an existing file over creating a new document. Before creating a file, the agent checks whether the same content already exists under another name.

## 10. Early validation

Problems should be found as early as possible: implementation errors, methodological inconsistencies, failures to reproduce the reference work, and missing evidence. A test that runs early is worth more than a long implementation that is checked only at the end.

This is why the graph is validated before anything is built on top of it (T01), why the MST baseline runs end to end on SumMe before the learned model is written (T07), and why the first GNN layer runs on a real video before the SLC pooling depends on it (T09). When a validation fails, the agent reports the failure and its probable cause before continuing, and it does not work around the failure silently.

## Checklist before closing a task

Before a task is reported as complete, the agent confirms the following. The acceptance criterion in `phases.md` is met. The tests or the validation were run, and their real output is saved in `evidence/`. `status.md` is updated with the new state, the date and a link to the evidence. Any decision taken during the task is recorded in `decisions.md`. Any reference used is linked from `references.md`, and any change to the structure of the system is reflected in `architecture.md`. Open questions that the task raised are added to `status.md` under blockers, not left in the conversation.

## Links

- [status.md](./status.md): current state of each task and the next task to do
- [phases.md](./phases.md): the five phases and the tasks, with their acceptance criteria
- [architecture.md](./architecture.md): what exists in the code versus what is only planned
- [decisions.md](./decisions.md): recorded decisions, with context and consequences
- [references.md](./references.md): the works behind each component
- [project.md](./project.md): objective, scope, supervision and checkpoint dates
- [evidence/](./evidence/): saved outputs that prove a task is complete
