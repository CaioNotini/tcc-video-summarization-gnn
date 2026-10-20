```
Role: Operational entry point. Records the verified state of the project, the active task, the blockers and the next step. Does not repeat task definitions, decisions or architecture.
Entry: CLAUDE.md
Related to: [phases.md](./phases.md), [project.md](./project.md), [references.md](./references.md), [architecture.md](./architecture.md), [decisions.md](./decisions.md), [principles.md](./principles.md), [evidence/](./evidence/)
```

# Project status

Last verified: 2026-10-19, repository commit 2cd2f59, as reported by the read-only inspection of that date.

This file says where the project stands. What each task requires is in [phases.md](./phases.md). Why the project is built this way is in [decisions.md](./decisions.md). Each open question is defined in [architecture.md](./architecture.md), and only its blocking effect is recorded below.

## Active task

T01, validate the time-aware graph and save the evidence.

## Task states

- **Validated:** none yet.
- **Implemented, not validated:** T01.
- **In progress:** T02, and T03 (the documentation files exist; acceptance is pending).
- **Blocked:** T11 (Q-01).
- **Planned:** T04 to T10, T12 to T18. For T10, the VSUMM files are already downloaded locally; the adapter and the τ and ρ decision are pending.

A task becomes validated only when its output is saved in [evidence/](./evidence/) and linked from this file. The rule is in [principles.md](./principles.md), principle 3.

## Blockers and open questions

- **Q-01:** blocks T11 and the τ and ρ part of T10. Owner: advisor.
- **Q-02:** blocks T13.
- **Q-03:** blocks T13 (see D-18). Owner: advisor.
- **Q-04:** affects T13 (see D-17). Owner: advisor.
- **Q-06:** affects T05 and TCC2. Owner: advisor.
- **Q-08:** from T13 onward.
- **Q-09:** before T18.

Q-05 and Q-07 were closed. The numbering keeps the gaps so that existing references stay valid.

## Local state (git-ignored, reported by the inspection)

- `data/raw/vsumm/` holds the four official VSUMM archives and two extracted videos (v21 and v29).
- `data/processed/vsumm/` holds extracted frames and a `features_teste.npy` file for v21 and v29.
- The two submodules, `HieTaSumm-lib` and `mHELMNet`, are populated and in sync.

## Verified evidence

None saved in [evidence/](./evidence/) yet.

## Reported, not yet evidence

These results were reported, but no saved output exists in the repository, so they do not count as validated.

- `pytest` in an isolated copy of the repository on 2026-10-19: 5 passed, 0 skipped. The output was not saved.
- Equivalence of the time-aware graph with Leonardo's `Frame.py`: 945 matching edges and zero weight difference, reported on 2026-09-29. Needs a saved run of `experiments/validate_graph.py`.
- Correlation of 0.97 between edge weight and temporal distance on a third video, with the torchvision extractor. Needs a saved run of the sanity check.

## Next step

**T01.** The acceptance criterion is in [phases.md](./phases.md). Run the steps in this order:

1. Check the arguments of `experiments/validate_graph.py` with `--help`.
2. Run `pytest` and keep the full output.
3. Run `experiments/validate_graph.py` on the existing feature files of v21 and v29, and keep the full output.
4. Create `evidence/` and save both outputs in one file named for T01.
5. Update this file: move T01 to validated and link the saved file.

T01 comes first because every later task depends on the graph, and PC3 on 21 October requires it.
