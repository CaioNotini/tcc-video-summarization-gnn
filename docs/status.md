```
Role: Operational entry point. Records the verified state of the project, the active task, the blockers and the next step. Does not repeat task definitions, decisions or architecture.
Entry: CLAUDE.md
Related to: [phases.md](./phases.md), [project.md](./project.md), [references.md](./references.md), [architecture.md](./architecture.md), [decisions.md](./decisions.md), [principles.md](./principles.md), [evidence/](./evidence/)
```

# Project status

Last verified: 2026-10-21, repository commit 6fcb61b, as reported by the read-only inspection of that date.

This file says where the project stands. What each task requires is in [phases.md](./phases.md). Why the project is built this way is in [decisions.md](./decisions.md). Each open question is defined in [architecture.md](./architecture.md), and only its blocking effect is recorded below.

## Active task

None. T01 to T04 are all validated; the next task by dependency order is T06 (MST hierarchy, depends on T04) or T05 (SumMe adapter, depends only on T03). Neither has started (see Next step).

## Task states

- **Validated:** T01, T02, T03, T04.
- **Implemented, not validated:** none.
- **In progress:** none.
- **Blocked:** T11 (Q-01).
- **Planned:** T05, T06 to T10, T12 to T19 (T20 and T21 were removed). For T10, the VSUMM files are already downloaded locally; the adapter and the τ and ρ decision are pending.

A task becomes validated only when its output is saved in [evidence/](./evidence/) and linked from this file. The rule is in [principles.md](./principles.md), principle 3.

## Blockers and open questions

- **Q-01:** blocks T11 and the τ and ρ part of T10. Owner: advisor.
- **Q-02:** blocks T13.
- **Q-03:** blocks T13 (see D-17). Owner: advisor.
- **Q-04:** affects T13. Owner: advisor.
- **Q-06:** affects T05 and TCC2. Owner: advisor.
- **Q-08:** from T13 onward.
- **Q-09:** affects T19 and T18.
- **Q-05 and Q-07 gap:** these identifiers are skipped in the Q-xx sequence with no recorded reason, unlike T20/T21 which has an explicit removal note. Does not block any task. Needs the researcher to say whether they were ever assigned, so a note can be added or the gap confirmed as never used.

## Local state

- `data/raw/vsumm/` holds the four official VSUMM archives and two extracted videos (v21 and v29).
- `data/processed/vsumm/` holds extracted frames and a `features_teste.npy` file for v21 and v29.
- The two submodules, `HieTaSumm-lib` and `mHELMNet`, are populated and in sync.

## Verified evidence

- T04: [evidence/t04-mst-hierarchy-spec.md](./evidence/t04-mst-hierarchy-spec.md). Specifies the input (the T01 graph), the output (one partition from the single cut, not a multi-level hierarchy), the cut rule (`cutNumber`, with its two possible sources) and the test cases of the MST hierarchy, checked line by line against the `HieTaSumm-lib` submodule (`Graph.py`, `Summary.py`, `options.json`). By the researcher's decision, T06 implements only this single final cut; the "set of levels" shared interface in [architecture.md](./architecture.md) applies only to SLC pooling, not to the MST hierarchy. Approved by the researcher on 2026-10-21. This specification lives in `docs/evidence/t04-mst-hierarchy-spec.md`, not in `phases.md`, by the researcher's explicit instruction; `phases.md`'s T04 acceptance criterion still says "written in this file," which this evidence file satisfies in substance (input, output, cut criteria and test cases are all specified) but not in its literal location. This discrepancy is reported here rather than resolved by editing `phases.md`.

## Reported, not yet evidence

These results were reported, but no saved output exists in the repository, so they do not count as validated.

- Correlation of 0.97 between edge weight and temporal distance on a third video, with the torchvision extractor. Needs a saved run of the sanity check.

## Next step

Stage 2 continues: T04's specification is validated and approved, so T06 (implement the MST
hierarchy with higra, per T04's specification) can start next. T05 (SumMe adapter) depends only on
T03 and has not been started either; it does not depend on T04 and can be worked on independently,
in either order. T07 still depends on T05, T06 and T19 (the evaluation protocol, also not yet
specified).
