```
Role: Operational entry point. Records the verified state of the project, the active task, the blockers and the next step. Does not repeat task definitions, decisions or architecture.
Entry: CLAUDE.md
Related to: [phases.md](./phases.md), [project.md](./project.md), [references.md](./references.md), [architecture.md](./architecture.md), [decisions.md](./decisions.md), [principles.md](./principles.md), [evidence/](./evidence/)
```

# Project status

Last verified: 2026-10-21, repository commit 6fcb61b, as reported by the read-only inspection of that date.

This file says where the project stands. What each task requires is in [phases.md](./phases.md). Why the project is built this way is in [decisions.md](./decisions.md). Each open question is defined in [architecture.md](./architecture.md), and only its blocking effect is recorded below.

## Active task

None. T01 to T06, T11 and T19 are all validated; the next task by dependency order is T07. It has not started and is blocked by Q-10 (see Next step).

## Task states

- **Validated:** T01, T02, T03, T04, T05, T06, T11, T19
- **Implemented, not validated:** none.
- **In progress:** none.
- **Blocked:** T07 (Q-10).
- **Planned:** T08 to T10, T12 to T18 (T20 and T21 were removed). For T10, the VSUMM files are already downloaded locally; the adapter is pending. The τ and ρ question for VSUMM is decided: not reported (D-22).

A task becomes validated only when its output is saved in [evidence/](./evidence/) and linked from this file. The rule is in [principles.md](./principles.md), principle 3.

## Blockers and open questions

- **Q-02:** blocks T13.
- **Q-03:** blocks T13 (see D-17). Owner: advisor.
- **Q-04:** affects T13. Owner: advisor.
- **Q-06:** affects T05 and TCC2. Owner: advisor.
- **Q-08:** from T13 onward.
- **Q-09:** affects T19 and T18.
- **Q-10:** blocks T07 and affects T19. Baseline summaries above 15% of the duration: how many of the 25 SumMe videos exceed the budget, and how they are treated. Owner: researcher.
- **Q-11:** affects T13 and T19. SLC level used as the segmentation; must be fixed in the protocol before results. Owner: researcher.
- **Q-05 and Q-07 gap:** these identifiers are skipped in the Q-xx sequence with no recorded reason, unlike T20/T21 which has an explicit removal note. Does not block any task. Needs the researcher to say whether they were ever assigned, so a note can be added or the gap confirmed as never used.

## Local state

- `data/raw/vsumm/` holds the four official VSUMM archives and two extracted videos (v21 and v29).
- `data/processed/vsumm/` holds extracted frames and a `features_teste.npy` file for v21 and v29.
- The two submodules, `HieTaSumm-lib` and `mHELMNet`, are populated and in sync.

## Verified evidence

- T04: [evidence/t04-mst-hierarchy-spec.md](./evidence/t04-mst-hierarchy-spec.md). Specifies the input (the T01 graph), the output (one partition from the single cut, not a multi-level hierarchy), the cut rule (`cutNumber`, with its two possible sources) and the test cases of the MST hierarchy, checked line by line against the `HieTaSumm-lib` submodule (`Graph.py`, `Summary.py`, `options.json`). By the researcher's decision, T06 implements only this single final cut; the "set of levels" shared interface in [architecture.md](./architecture.md) applies only to SLC pooling, not to the MST hierarchy. Approved by the researcher on 2026-10-21. This specification lives in `docs/evidence/t04-mst-hierarchy-spec.md`, not in `phases.md`, by the researcher's explicit instruction; `phases.md`'s T04 acceptance criterion still says "written in this file," which this evidence file satisfies in substance (input, output, cut criteria and test cases are all specified) but not in its literal location. This discrepancy is reported here rather than resolved by editing `phases.md`.
- T05: [evidence/t05-summe-adapter.md](./evidence/t05-summe-adapter.md). The SumMe adapter (`src/tcc_gnn/data/summe.py`) lists the 25 SumMe videos and returns their ground truth in the common format (`GroundTruth`/`ScoreSeries`, `src/tcc_gnn/data/base.py`). A test checks `video_1` field by field against the raw `.h5` file from the `HieTaSumm-lib` submodule. `pytest` passes (8 tests, no regression).
- T06: Validated. Evidence: `docs/evidence/t06-mst-hierarchy.md`, `docs/evidence/t06-additional-verification.md`, `docs/evidence/sheets/` and `docs/evidence/images/`. Baseline with `int()` truncation and cutNumber = 5 (D-20). Identical to the HieTaSumm-lib reference on v29 (0 frames differ). The v21 case is degenerate (all edges truncate to 0) and is recorded as such. Next task: T07, which depends on T05 and T19 and needs the frames on disk for `bestCutNumber`.
- T11: Validated. Evidence: [decisions.md](./decisions.md), D-22. The decision — VSUMM reports F-score only, no rank metrics — is recorded as D-22, closing Q-01. T11's acceptance criterion ([phases.md](./phases.md)) only requires the decision to be recorded in `decisions.md`, not a file under `evidence/`.
- T19: [evaluation-protocol.md](./evaluation-protocol.md). Approved by the researcher on 2026-10-23. Fixes the summary size (15% duration budget, D-21), the F-score computation per dataset, and that VSUMM reports F-score only (D-22). Two points stay open for the advisor/researcher: the knapsack segment unit (Q-11) and the treatment of baseline summaries above the budget (Q-10).

## Reported, not yet evidence

These results were reported, but no saved output exists in the repository, so they do not count as validated.

- Correlation of 0.97 between edge weight and temporal distance on a third video, with the torchvision extractor. Needs a saved run of the sanity check.

## Next step

Stage 2 continues: T04, T05, T06 and T19 are all validated, and T11 is validated too (D-22 closes Q-01). T07 depends on T05, T06 and T19, all satisfied, but is blocked by Q-10 (how to treat HieTaSumm baseline summaries that exceed the 15% budget).
