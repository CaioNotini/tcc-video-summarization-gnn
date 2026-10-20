```
Role: First file an agent reads in this repository. Sets the mandatory reading, the task flow, the approval rules and the working limits for the thesis.
Entry: none. This is the entry point of the repository for agents.
Related to: [docs/status.md](docs/status.md), [docs/principles.md](docs/principles.md), [docs/project.md](docs/project.md), [docs/phases.md](docs/phases.md), [docs/architecture.md](docs/architecture.md), [docs/decisions.md](docs/decisions.md), [docs/references.md](docs/references.md), [docs/evidence/](docs/evidence/)
```

# TCC: unsupervised video summarisation with a GNN

Undergraduate thesis at PUC Minas. The proposed method builds the time-aware frame graph of HieTaSumm and then learns everything after it: a GATv2 layer, a hierarchy learned by single-link clustering (SLC) pooling, a multigraph GNN and a learned readout that picks the keyframes. It is compared with the original HieTaSumm method, with uniform sampling and k-means, and with the same network without message passing. The full description is in [docs/project.md](docs/project.md).

## Mandatory reading before any task

1. [docs/principles.md](docs/principles.md): the ten working principles. Read it first, for every task.
2. [docs/status.md](docs/status.md): active task, blockers and real state. Confirm it before acting. Do not assume the state.
3. [docs/phases.md](docs/phases.md): read only the entry of the active task, not the whole file.

Read the other documents only when the task needs them:

- [docs/architecture.md](docs/architecture.md) before changing code in src/ or experiments/.
- [docs/decisions.md](docs/decisions.md) when a choice or an open question is involved.
- [docs/references.md](docs/references.md) before implementing a component that comes from a paper.
- [docs/project.md](docs/project.md) when the context of the work is not clear.

Do not assume that the existing code represents the intended architecture. [docs/architecture.md](docs/architecture.md) says what is intended and what already exists.

If anything in the conversation contradicts these documents, the documents prevail and the contradiction must be reported.

## Task flow

For each task, in this order:

1. **Identify:** confirm the active task, its goal, its acceptance criterion and the files involved.
2. **Specify:** define the goal, the scope, what is out of scope, the acceptance criteria and the tests. Wait for confirmation.
3. **Plan:** for non-trivial tasks, present a short approach, the affected files, the risks and the tests.
4. **Implement:** produce the smallest complete solution that meets the criteria, without anticipating future tasks.
5. **Validate:** run the tests and the validation scripts that apply. Keep the real output. Analyse failures and regressions. Save the output in [docs/evidence/](docs/evidence/) when the task needs evidence to count as validated.
6. **Document:** update [docs/status.md](docs/status.md) when the state of a task changed, with a link to the evidence. Record any decision in [docs/decisions.md](docs/decisions.md). Change [docs/phases.md](docs/phases.md) only with approval, because it defines the acceptance criteria.
7. **Summarise:** report what changed, the decisions taken, the tests run with their real results and what stays unvalidated.

## Approvals

1. Non-trivial tasks: wait for confirmation at the end of the specification and at the end of the plan. After the plan is approved, do only what it covers.
2. Trivial tasks: state the intent in one sentence and execute. Testing and the summary stay mandatory.
3. The following need explicit approval before any work:
   - a change to the research question, the method, the datasets, the metrics or the evaluation protocol
   - moving work between the guaranteed scope and the extensions
   - choosing the unsupervised training objective (D-18) or adding a supervised objective to the main method
   - bringing a combinatorial step of HieTaSumm back into the proposed method (D-19)
   - adding a dependency or a tool that changes how the experiments are reproduced
   - any change to the ported graph code beyond what the task asks (D-02)
4. Destructive changes and any action outside the approved plan also need explicit approval.
5. Contradictions and ambiguities that affect the method, the evidence or an irreversible decision: stop and ask.

## Development rules

1. Respect the active task and its acceptance criterion. Do not start a task whose dependencies in [docs/phases.md](docs/phases.md) are not met. Do not start the next task because it looks useful.
2. Do not claim that something works without evidence. A task is validated only when its real output is saved in [docs/evidence/](docs/evidence/) and linked from [docs/status.md](docs/status.md). Report the state with these words: validated, implemented but not validated, in progress, planned, blocked.
3. The proposed method uses nothing from the combinatorial part of HieTaSumm after the graph (D-19). If a learned step seems not plausible, stop and report. Do not replace it silently.
4. Ported code keeps the original logic and names (D-02, D-10). Any change to `src/tcc_gnn/graph/` is followed by the graph validation again.
5. Every compared method receives the same graph, the same features, the same evaluation code and the same summary length (D-13, D-14, D-16). No metric, threshold or setting is chosen after seeing the results.
6. The main method stays unsupervised. No human summary is used in training (D-09). The training objective is open, so do not set a default for it in code or in configuration files.
7. Use the stack in [docs/architecture.md](docs/architecture.md). TensorFlow is only for validation (D-04). Do not add `torch_scatter` (D-11). Do not add other dependencies without approval. Do not use paid services or cloud infrastructure without explicit authorisation.
8. The folders under `references/` are read-only submodules. Never edit them. Never run `git submodule update` or any command that changes them, unless asked.
9. The folder `data/` is git-ignored. Never commit videos, frames, features, checkpoints or datasets. Each dataset has its own adapter, because the ground truth formats differ (D-15).
10. Make runs reproducible. Fix the seeds, log the runs with mlflow (D-12) and record the hardware when time or memory is measured.
11. Write modern Python, with PEP 8, type hints and plain functions where no state is needed. Leave no dead code. Keep code comments to one line that says what a file or a function does.
12. Do not create abstractions, folders or files for hypothetical cases. Do not create empty folders.
13. Do not open, print or expose secrets such as `.env` files, tokens or credentials.
14. Never commit, push, reset, revert, amend, rebase, stash or checkout without an explicit request. Report the pending files and let the researcher commit.
15. When a decision or the real state changes, update the documents that own that information. Do not copy it into other files.
16. Do not resolve contradictions or ambiguities silently.

## Tests

- Every module in `src/tcc_gnn/` has tests in `tests/`. Create or update tests whenever behaviour changes.
- Cover the failure cases: invalid inputs, unexpected states and negative scenarios.
- Ported code is checked against the output of the reference implementation.
- Hierarchy code is checked on small synthetic graphs with a known answer.
- Model layers are checked for output shape, finite values, gradient flow and fixed-seed determinism.
- Evaluation code is checked on tiny cases with metrics computed by hand.
- Passing is not enough. Ask whether the tests would detect a real violation of the requirement.
- Tests written or reviewed by the same model are not independent validation. The final validation belongs to the researcher.
- Do not present a number from a toy input or from a single video as a result of the method without saying so.
- If a test fails, investigate the cause. Do not remove, skip or weaken the test to get a pass.

## Writing the documents

- Documents, headers and file names are in English. Reply to the researcher in Portuguese.
- Use simple, direct sentences.
- Use relative links instead of repeating content. Each fact has one source file.
- Every file in `docs/` starts with this header and ends with a Links section:

```
Role: <one line>
Entry: docs/status.md
Related to: <relative links>
```

- Every file in `docs/` must be reachable from [docs/status.md](docs/status.md) in at most three links.
- Identifiers are never renumbered or reused. Tasks are T-xx and live in [docs/phases.md](docs/phases.md). Decisions are D-xx and live in [docs/decisions.md](docs/decisions.md). Open questions are Q-xx and live in [docs/architecture.md](docs/architecture.md).
- Use the terms defined in the Terminology section of [docs/decisions.md](docs/decisions.md). The word "baseline" is not used alone.
- Keep apart verified facts, approved decisions, hypotheses, open questions and planned work.
