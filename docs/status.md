```
Role: Operational entry point. Records the verified state of the project, the active task, the blockers and the next step. Does not repeat task definitions, decisions or architecture.
Entry: CLAUDE.md
Related to: [phases.md](./phases.md), [project.md](./project.md), [references.md](./references.md), [architecture.md](./architecture.md), [decisions.md](./decisions.md), [principles.md](./principles.md), [evidence/](./evidence/)
```

# Project status

Last verified: 2026-11-16, repository commit 02652a5, updated by the documentation corrections of that date.

This file says where the project stands. What each task requires is in [phases.md](./phases.md). Why the project is built this way is in [decisions.md](./decisions.md). Each open question is defined in [architecture.md](./architecture.md), and only its blocking effect is recorded below.

## Active task

None. T01 to T06, T11 and T19 are all validated; the next task by dependency order is T07. It has not started and is unblocked since Q-10 was closed on 2026-11-16 (D-23; see Next step).

## Task states

- **Validated:** T01, T02, T03, T04, T05, T06, T11, T19
- **Implemented, not validated:** none.
- **In progress:** none.
- **Blocked:** none. T07 was unblocked on 2026-11-16 (Q-10 closed, D-23).
- **Planned:** T08 to T10, T12 to T18 (T20 and T21 were removed). For T10, the VSUMM files are already downloaded locally; the adapter is pending. The τ and ρ question for VSUMM is decided: not reported (D-22).

A task becomes validated only when its output is saved in [evidence/](./evidence/) and linked from this file. The rule is in [principles.md](./principles.md), principle 3.

## Blockers and open questions

- **Q-02:** blocks T13.
- **Q-03:** blocks T13 (see D-17). Owner: advisor.
- **Q-04:** affects T13. Owner: advisor.
- **Q-06:** affects the scope of the TVSum adapter: whether it is written in TCC1 or left for TCC2 with the TVSum experiments. Owner: advisor.
- **Q-08:** from T13 onward.
- **Q-09:** closed (researcher, 2026-11-16). Timing and memory are measured on the local machine, GPU GeForce GTX 1660 SUPER, 6 GB.
- **Q-10:** closed (researcher, 2026-11-16, D-23). 0 of 25 SumMe videos exceed the budget with `delta_t = 32` (D-24) and frames every 0.25 s (D-25): 10.53% to 14.58%, mean 13.77%. The baseline summary is the native keyshot set, reported without adjustment. Evidence: [evidence/q10-baseline-budget.md](./evidence/q10-baseline-budget.md).
- **Pending for TCC2 (VSUMM):** the VSUMM frames on disk (v21, v29) are about one per second, while `configs/default.yaml` has `rate: 4` and D-25 requires one frame every 0.25 s. They must be extracted again, with their features, before the VSUMM experiments (T10). The T01 and T06 equivalence checks do not depend on this; their descriptive numbers and the sanity check refer to the current frames.
- **Q-11:** affects T13 and T19. SLC level used as the segmentation; must be fixed in the protocol before results. Owner: researcher.
- **Q-05 and Q-07 gap:** closed. Confirmed (2026-11-16) that the two identifiers were assigned in earlier working versions and discarded without a recorded entry; what they covered was not recovered. The numbers are not reused, and the gap is noted in the questions table of [architecture.md](./architecture.md).

## Local state

- `data/raw/vsumm/` holds the four official VSUMM archives and two extracted videos (v21 and v29).
- `data/processed/vsumm/` holds extracted frames and a `features_teste.npy` file for v21 and v29. The frames are about one per second (110 for v21, 65 for v29, both 29.97 fps), extracted before the extractor fix of 2026-11-16 and not re-extracted.
- The two submodules, `HieTaSumm-lib` and `mHELMNet`, are populated and in sync (re-populated on 2026-11-16 with `git submodule update --init --recursive`, at the pinned commits).
- `data/raw/summe/SumMe/` holds the 25 SumMe videos. `data/processed/summe/` holds frames (one every 0.25 s, 15,904 in total, none truncated) and ResNet50 avgpool features for all 25, regenerated on 2026-11-16 with the fixed `src/tcc_gnn/features/frame_extraction.py`.
- Python environment: `.venv/` on drive `D:` (CLAUDE.md rule 17), torch 2.5.1+cu121 on a GeForce GTX 1660 SUPER.
- `src/tcc_gnn/features/frame_extraction.py` was fixed on 2026-11-16 to sample one frame every `1/fps` seconds with the rule of the HieTaSumm-lib extractor (`HieTaSumm/__init__.py:86-142`), tested in `tests/test_frame_extraction.py`. Before, it kept one frame every `round(native_fps/fps)` native frames.

## Verified evidence

- T04: [evidence/t04-mst-hierarchy-spec.md](./evidence/t04-mst-hierarchy-spec.md). Specifies the input (the T01 graph), the output (one partition from the single cut, not a multi-level hierarchy), the cut rule (`cutNumber`, with its two possible sources) and the test cases of the MST hierarchy, checked line by line against the `HieTaSumm-lib` submodule (`Graph.py`, `Summary.py`, `options.json`). By the researcher's decision, T06 implements only this single final cut; the "set of levels" shared interface in [architecture.md](./architecture.md) applies only to SLC pooling, not to the MST hierarchy. Approved by the researcher on 2026-10-21. This specification lives in `docs/evidence/t04-mst-hierarchy-spec.md`, not in `phases.md`, by the researcher's explicit instruction; `phases.md`'s T04 acceptance criterion still says "written in this file," which this evidence file satisfies in substance (input, output, cut criteria and test cases are all specified) but not in its literal location. This discrepancy is reported here rather than resolved by editing `phases.md`.
- T05: [evidence/t05-summe-adapter.md](./evidence/t05-summe-adapter.md). The SumMe adapter (`src/tcc_gnn/data/summe.py`) lists the 25 SumMe videos and returns their ground truth in the common format (`GroundTruth`/`ScoreSeries`, `src/tcc_gnn/data/base.py`). A test checks `video_1` field by field against the raw `.h5` file from the `HieTaSumm-lib` submodule. `pytest` passes (8 tests, no regression).
- T06: Validated. Evidence: `docs/evidence/t06-mst-hierarchy.md`, `docs/evidence/t06-additional-verification.md`, `docs/evidence/sheets/` and `docs/evidence/images/`. Baseline with `int()` truncation and cutNumber = 5 (D-20). Identical to the HieTaSumm-lib reference on v29 (0 frames differ). The v21 case is degenerate (all edges truncate to 0) and is recorded as such. Next task: T07, which depends on T05 and T19 and needs the frames on disk for `bestCutNumber`.
- T11: Validated. Evidence: [decisions.md](./decisions.md), D-22. The decision — VSUMM reports F-score only, no rank metrics — is recorded as D-22, closing Q-01. T11's acceptance criterion ([phases.md](./phases.md)) only requires the decision to be recorded in `decisions.md`, not a file under `evidence/`.
- T19: [evaluation-protocol.md](./evaluation-protocol.md). Approved by the researcher on 2026-10-23. Fixes the summary size (15% duration budget, D-21), the F-score computation per dataset, and that VSUMM reports F-score only (D-22). Two points stay open for the advisor/researcher: the knapsack segment unit (Q-11) and the treatment of baseline summaries above the budget (Q-10).
- bestCutNumber port (`src/tcc_gnn/hierarchy/best_cut.py`): [evidence/best-cut-port-equivalence.md](./evidence/best-cut-port-equivalence.md). Same raw value, `cutNumber` and keyshot list as the HieTaSumm-lib reference on SumMe `Jumps` and `Air_Force_One` with `delta_t = 32`; two intentional divergences (zero standard deviation, `delta_t <= 0`) recorded there. Not linked to a task in [phases.md](./phases.md); it supports the Q-10 measurement and T07.

## Reported, not yet evidence

None. The 0.97 correlation sanity check that was listed here now has saved runs: [evidence/sanity-check-features.md](./evidence/sanity-check-features.md) (v29, torchvision features, 2026-11-16: Pearson 0.9749 and Spearman 1.0000 with `delta_t = 8`; Pearson 0.7937 and Spearman 0.9456 with `delta_t = 32`).

## Next step

Stage 2 continues: T04, T05, T06 and T19 are all validated, and T11 is validated too (D-22 closes Q-01). T07 depends on T05, T06 and T19, all satisfied, and Q-10 is closed (D-23), so T07 is unblocked. It has not started.
