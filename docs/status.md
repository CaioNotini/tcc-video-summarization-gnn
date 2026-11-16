```
Role: Operational entry point. Records the verified state of the project, the active task, the blockers and the next step. Does not repeat task definitions, decisions or architecture.
Entry: CLAUDE.md
Related to: [phases.md](./phases.md), [project.md](./project.md), [references.md](./references.md), [architecture.md](./architecture.md), [decisions.md](./decisions.md), [principles.md](./principles.md), [evidence/](./evidence/)
```

# Project status

Last verified: 2026-11-16, repository commit 02652a5, updated by the documentation corrections of that date.

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
- **Q-06:** affects the scope of the TVSum adapter: whether it is written in TCC1 or left for TCC2 with the TVSum experiments. Owner: advisor.
- **Q-08:** from T13 onward.
- **Q-10:** blocks T07 and affects T19. Baseline summaries above 15% of the duration: how many of the 25 SumMe videos exceed the budget, and how they are treated. Owner: researcher. The per-video measurement (2026-11-16, `delta_t = 32`) stopped at video 7 of 25 and is not evidence yet, for two reasons that need a decision: (1) `src/tcc_gnn/features/frame_extraction.py` samples every `round(fps/4)` native frames (4.17 fps at 25 fps, 4.28 at 29.97, 3.75 at 30), while the HieTaSumm-lib extractor (`HieTaSumm/__init__.py:86-142`) saves one frame every 0.25 s; `measure_q10_budget.py` converts keyshot frames to seconds with `/4`, so the percentages it prints are off by that ratio (two videos printed above 15% that are below it in real duration); (2) the frames on disk for `Car_railcrossing` are a partial extraction (272 of 725, last file truncated), which stopped the run.
- **Q-11:** affects T13 and T19. SLC level used as the segmentation; must be fixed in the protocol before results. Owner: researcher.
- **Q-05 and Q-07 gap:** closed. Confirmed (2026-11-16) that the two identifiers were assigned in earlier working versions and discarded without a recorded entry; what they covered was not recovered. The numbers are not reused, and the gap is noted in the questions table of [architecture.md](./architecture.md).

## Local state

- `data/raw/vsumm/` holds the four official VSUMM archives and two extracted videos (v21 and v29).
- `data/processed/vsumm/` holds extracted frames and a `features_teste.npy` file for v21 and v29.
- The two submodules, `HieTaSumm-lib` and `mHELMNet`, are populated and in sync (re-populated on 2026-11-16 with `git submodule update --init --recursive`, at the pinned commits).
- `data/raw/summe/SumMe/` holds the 25 SumMe videos. `data/processed/summe/` holds frames for 8 videos and cached features for 7 of them; the `Car_railcrossing` frames are a partial extraction (see Q-10).
- Python environment: `.venv/` on drive `D:` (CLAUDE.md rule 17), torch 2.5.1+cu121 on a GeForce GTX 1660 SUPER; `pytest` passes 28 tests.

## Verified evidence

- T04: [evidence/t04-mst-hierarchy-spec.md](./evidence/t04-mst-hierarchy-spec.md). Specifies the input (the T01 graph), the output (one partition from the single cut, not a multi-level hierarchy), the cut rule (`cutNumber`, with its two possible sources) and the test cases of the MST hierarchy, checked line by line against the `HieTaSumm-lib` submodule (`Graph.py`, `Summary.py`, `options.json`). By the researcher's decision, T06 implements only this single final cut; the "set of levels" shared interface in [architecture.md](./architecture.md) applies only to SLC pooling, not to the MST hierarchy. Approved by the researcher on 2026-10-21. This specification lives in `docs/evidence/t04-mst-hierarchy-spec.md`, not in `phases.md`, by the researcher's explicit instruction; `phases.md`'s T04 acceptance criterion still says "written in this file," which this evidence file satisfies in substance (input, output, cut criteria and test cases are all specified) but not in its literal location. This discrepancy is reported here rather than resolved by editing `phases.md`.
- T05: [evidence/t05-summe-adapter.md](./evidence/t05-summe-adapter.md). The SumMe adapter (`src/tcc_gnn/data/summe.py`) lists the 25 SumMe videos and returns their ground truth in the common format (`GroundTruth`/`ScoreSeries`, `src/tcc_gnn/data/base.py`). A test checks `video_1` field by field against the raw `.h5` file from the `HieTaSumm-lib` submodule. `pytest` passes (8 tests, no regression).
- T06: Validated. Evidence: `docs/evidence/t06-mst-hierarchy.md`, `docs/evidence/t06-additional-verification.md`, `docs/evidence/sheets/` and `docs/evidence/images/`. Baseline with `int()` truncation and cutNumber = 5 (D-20). Identical to the HieTaSumm-lib reference on v29 (0 frames differ). The v21 case is degenerate (all edges truncate to 0) and is recorded as such. Next task: T07, which depends on T05 and T19 and needs the frames on disk for `bestCutNumber`.
- T11: Validated. Evidence: [decisions.md](./decisions.md), D-22. The decision — VSUMM reports F-score only, no rank metrics — is recorded as D-22, closing Q-01. T11's acceptance criterion ([phases.md](./phases.md)) only requires the decision to be recorded in `decisions.md`, not a file under `evidence/`.
- T19: [evaluation-protocol.md](./evaluation-protocol.md). Approved by the researcher on 2026-10-23. Fixes the summary size (15% duration budget, D-21), the F-score computation per dataset, and that VSUMM reports F-score only (D-22). Two points stay open for the advisor/researcher: the knapsack segment unit (Q-11) and the treatment of baseline summaries above the budget (Q-10).
- bestCutNumber port (`src/tcc_gnn/hierarchy/best_cut.py`): [evidence/best-cut-port-equivalence.md](./evidence/best-cut-port-equivalence.md). Same raw value, `cutNumber` and keyshot list as the HieTaSumm-lib reference on SumMe `Jumps` and `Air_Force_One` with `delta_t = 32`; two intentional divergences (zero standard deviation, `delta_t <= 0`) recorded there. Not linked to a task in [phases.md](./phases.md); it supports the Q-10 measurement and T07.

## Reported, not yet evidence

None. The 0.97 correlation sanity check that was listed here now has saved runs: [evidence/sanity-check-features.md](./evidence/sanity-check-features.md) (v29, torchvision features, 2026-11-16: 0.9749 with `delta_t = 8`, 0.7937 with `delta_t = 32`).

## Next step

Stage 2 continues: T04, T05, T06 and T19 are all validated, and T11 is validated too (D-22 closes Q-01). T07 depends on T05, T06 and T19, all satisfied, but is blocked by Q-10 (how to treat HieTaSumm baseline summaries that exceed the 15% budget).
