```
Role: Operational entry point. Records the verified state of the project, the active task, the blockers and the next step. Does not repeat task definitions, decisions or architecture.
Entry: README.md
Related to: [phases.md](./phases.md), [project.md](./project.md), [references.md](./references.md), [architecture.md](./architecture.md), [decisions.md](./decisions.md), [principles.md](./principles.md), [evidence/](./evidence/)
```

# Project status

Last verified: 2026-10-16, repository commit df73d5b.

This file says where the project stands. What each task requires is in [phases.md](./phases.md). Why the project is built this way is in [decisions.md](./decisions.md). Each open question is defined in [architecture.md](./architecture.md), and its blocking effect is recorded below.

## Active task

T01, validate the time-aware graph and save the evidence.

## Task states

- **Validated:** none yet.
- **Implemented, not validated:** T01.
- **In progress:** T02.
- **Blocked:** T10 (VSUMM not downloaded), T11 (open question Q-01).
- **Planned:** T03 to T09, T12 to T18.

A task becomes validated only when its output is saved in [evidence/](./evidence/) and linked from this file. The rule is in [principles.md](./principles.md), principle 3.

## Blockers and open questions

Each question is defined in [architecture.md](./architecture.md). The IDs Q-05 and Q-07 were closed and are no longer used. The numbering keeps gaps so that existing references stay valid.

- **Q-01 (blocks T11, then T10):** how to report τ and ρ on VSUMM. Owner: advisor.
- **Q-02 (blocks T13):** value of the SLC distance threshold. To be chosen empirically.
- **Q-03 (blocks T13):** unsupervised training objective (see D-18). Owner: advisor.
- **Q-04 (affects T13):** whether the SLC-HELM code will be shared (see D-17). Owner: advisor.
- **Q-06 (affects T05 and TCC2):** scope of TVSum, adapter in TCC1 and experiments in TCC2. Owner: advisor.
- **Q-08 (from T13):** training hardware, local GPU or cloud notebook.
- **Q-09 (before T18):** hardware used to measure running time and memory.
- **Not a question:** the VSUMM dataset is not downloaded, which blocks T10.

## Verified evidence

- **2026-10-16:** `pytest` in a clean clone of the repository at commit df73d5b, with minimal dependencies (numpy, networkx, h5py, pytest). Result: 4 passed, 1 skipped. The skipped test needs the `HieTaSumm-lib` submodule. The output is not yet saved in `evidence/`, so it does not validate T01 by itself.

## Reported, not yet evidence

These results were reported in earlier work, but no saved output exists in the repository, so they do not count as validated.

- Equivalence of the time-aware graph with Leonardo's `Frame.py`: 945 matching edges and zero weight difference, reported on 2026-09-29. Needs a saved run of `experiments/validate_graph.py`.
- Correlation between edge weight and temporal distance of 0.97 on a third video, with the torchvision extractor. Needs a saved run of the sanity check.

## Next step

**T01.** The acceptance criterion is in [phases.md](./phases.md). Run the steps in this order:

1. Populate the submodules with `git submodule update --init --recursive`.
2. Run `pytest` and keep the full output.
3. Run `python experiments/validate_graph.py` with the required arguments, and keep the full output.
4. Save both outputs in `evidence/`, in one file named for T01.
5. Update this file: move T01 to validated and link the saved file.

T01 comes first because every later task depends on the graph, and PC3 on 21 October requires it.

## Links

- [phases.md](./phases.md): the five phases and the tasks, with their acceptance criteria
- [project.md](./project.md): objective, research question, method and scope
- [architecture.md](./architecture.md): what exists in the code, with open questions and risks
- [decisions.md](./decisions.md): decisions with context, status and alternatives
- [references.md](./references.md): the works behind each component
- [principles.md](./principles.md): the ten working principles
- [evidence/](./evidence/): saved outputs that prove a task is complete
