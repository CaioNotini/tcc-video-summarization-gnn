```
Role: First file an agent reads in this repository. Sets the mandatory reading, the task flow, the approval rules and the working limits for the thesis.
Entry: none. This is the entry point of the repository for agents.
Related to: [docs/status.md](docs/status.md), [docs/principles.md](docs/principles.md), [docs/project.md](docs/project.md), [docs/phases.md](docs/phases.md), [docs/architecture.md](docs/architecture.md), [docs/decisions.md](docs/decisions.md), [docs/references.md](docs/references.md), [docs/evidence/](docs/evidence/)
```

# TCC: unsupervised video summarisation with a GNN

Undergraduate thesis at PUC Minas. The proposed method builds the time-aware frame graph of HieTaSumm and then learns everything after it: a GATv2 layer, a hierarchy learned by single-link clustering (SLC) pooling, a multigraph GNN and a learned readout that picks the keyframes. It is compared with the original HieTaSumm method. The full description is in [docs/project.md](docs/project.md).

## Mandatory reading before any task

1. [docs/principles.md](docs/principles.md): the ten working principles, which this file refers to by number. Read it first, for every task.
2. [docs/status.md](docs/status.md): active task, blockers and real state. Confirm it before acting. Do not assume the state.
3. [docs/phases.md](docs/phases.md): read only the section of the active task, the heading that starts with its ID, not the whole file.

Read the other documents only when the task needs them:

- [docs/architecture.md](docs/architecture.md) before changing code in src/ or experiments/.
- [docs/decisions.md](docs/decisions.md) when a choice or an open question is involved.
- [docs/references.md](docs/references.md) before implementing a component that comes from a paper.
- [docs/project.md](docs/project.md) when the context of the work is not clear.

Do not assume that the existing code represents the intended architecture (principle 1).

If anything in the conversation contradicts these documents, the documents prevail and the contradiction must be reported.

## Task flow

For each task, in this order. The principles that each step applies are in brackets.

1. **Identify** (principle 1): confirm the active task, its goal, its acceptance criterion and the files involved.
2. **Specify** (principle 2): define the goal, the scope, what is out of scope, the acceptance criteria and the tests. Wait for confirmation.
3. **Plan** (principle 2): for non-trivial tasks, present a short approach, the affected files, the risks and the tests.
4. **Implement** (principles 4 and 5): produce the smallest complete solution that meets the criteria, without anticipating future tasks.
5. **Validate** (principles 3 and 10): run the tests and the validation scripts that apply. Keep the real output. Analyse failures and regressions. Save the output in [docs/evidence/](docs/evidence/) when the task needs evidence to count as validated.
6. **Document** (principles 5 and 6): update [docs/status.md](docs/status.md) when the state of a task changed, with a link to the evidence. Record any decision in [docs/decisions.md](docs/decisions.md). Link any reference used from [docs/references.md](docs/references.md), and reflect any change to the structure of the system in [docs/architecture.md](docs/architecture.md). Add the open questions the task raised to [docs/status.md](docs/status.md) under blockers, not left in the conversation. Do not copy information into other files. Change [docs/phases.md](docs/phases.md) only with approval, because it defines the acceptance criteria.
7. **Summarise** (principle 3): report what changed, the decisions taken, the tests run with their real results and what stays unvalidated.

## Approvals

This section applies principle 8, and it is the only place where the list of approvals is kept.

1. Non-trivial tasks: wait for confirmation at the end of the specification and at the end of the plan. After the plan is approved, do only what it covers.
2. Trivial tasks: state the intent in one sentence and execute. Testing and the summary stay mandatory.
3. The following need explicit approval before any work:
   - a change to the research question, the method, the datasets, the metrics or the evaluation protocol
   - moving work between the guaranteed scope and the extensions
   - choosing the unsupervised training objective (D-17) or adding a supervised objective to the main method
   - bringing a combinatorial step of HieTaSumm back into the proposed method (D-18)
   - adding or changing a dependency or a tool
   - any change to the ported graph code beyond what the task asks (D-02)
4. Destructive changes and any action outside the approved plan also need explicit approval.
5. Contradictions and ambiguities that affect the method, the evidence or an irreversible decision: stop and ask. Do not resolve them silently.

## Development rules

1. Respect the active task and its acceptance criterion. Do not start a task whose dependencies in [docs/phases.md](docs/phases.md) are not met. Do not start the next task because it looks useful.
2. Do not claim that something works without evidence. Principle 3 says when a task counts as validated. Report states with the state names used in [docs/status.md](docs/status.md).
3. The proposed method uses nothing from the combinatorial part of HieTaSumm after the graph (D-18). If a learned step seems not plausible, stop and report. Do not replace it silently.
4. Ported code keeps the original logic and names (D-02, D-10). A change to `src/tcc_gnn/graph/` needs the graph validation again (principle 4).
5. Keep the invariants in [docs/architecture.md](docs/architecture.md). Every compared method uses the same evaluation code and summary length (D-13, D-14). No metric, threshold or setting is chosen after seeing the results.
6. The main method stays unsupervised, and no human summary is used in training (D-09). The training objective is open, so leave `unsupervised_objective` as null until D-17 is decided.
7. Use the stack in [docs/architecture.md](docs/architecture.md). TensorFlow is only for validation (D-04). Do not add `torch_scatter` (D-11). Do not use paid services or cloud infrastructure without explicit authorisation.
8. The folders under `references/` are read-only submodules. Never edit them. Never run `git submodule update` or any command that changes them, unless asked.
9. The folder `data/` is git-ignored. Never commit videos, frames, features, checkpoints or datasets. Each dataset has its own adapter, because the ground truth formats differ (D-15).
10. Make runs reproducible. Fix the seeds, log the runs with mlflow (D-12) and record the hardware when time or memory is measured.
11. Write modern Python, with PEP 8, type hints and plain functions where no state is needed (D-10). Leave no dead code. Keep code comments to one line that says what a file or a function does.
12. Do not create empty folders, and do not create abstractions, folders or files for hypothetical cases (principles 5 and 9).
13. Do not open, print or expose secrets such as `.env` files, tokens or credentials.
14. Never commit, push, reset, revert, amend, rebase, stash or checkout without an explicit request. Report the pending files and let the researcher commit.
15. The thesis text and bibliography are maintained on Overleaf, outside this repository (see [docs/references.md](docs/references.md)). The edit link is sensitive, because this repository is public, and must never be written into a file tracked by git. Only open or edit the Overleaf project when the researcher explicitly asks for it in that conversation, and only alter the section requested, which by default is the references and bibliography section.
16. Evidence in [docs/evidence/](docs/evidence/) is written in Markdown, with the real command output or a factual description of what was done. Do not add screenshots or other images by default. Only add one when the researcher explicitly asks for it and confirms it for that task. When evidence is updated, rewrite the file so it stays accurate; do not leave references to material that was removed.

## Tests

These rules apply principles 3 and 10.

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
- Every file in `docs/` starts with this header:

```
Role: <one line>
Entry: docs/status.md
Related to: <relative links>
```

- Link a document or a code file with a relative path on its first mention in a section, and do not repeat the content of what is linked (principle 6). Do not add a Links section.
- Every file in `docs/` must be reachable from [docs/status.md](docs/status.md) in at most three links (principle 6).
- Identifiers are never renumbered or reused. Tasks are T-xx and live in [docs/phases.md](docs/phases.md). Decisions are D-xx and live in [docs/decisions.md](docs/decisions.md). Open questions are Q-xx and live in [docs/architecture.md](docs/architecture.md).
- Use the terms defined in the Terminology section of [docs/decisions.md](docs/decisions.md).
- Keep apart verified facts, approved decisions, hypotheses, open questions and planned work.
