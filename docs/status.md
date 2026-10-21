```
Role: Operational entry point. Records the verified state of the project, the active task, the blockers and the next step. Does not repeat task definitions, decisions or architecture.
Entry: CLAUDE.md
Related to: [phases.md](./phases.md), [project.md](./project.md), [references.md](./references.md), [architecture.md](./architecture.md), [decisions.md](./decisions.md), [principles.md](./principles.md), [evidence/](./evidence/)
```

# Project status

Last verified: 2026-10-20, repository commit 3d1f6ac, as reported by the read-only inspection of that date.

This file says where the project stands. What each task requires is in [phases.md](./phases.md). Why the project is built this way is in [decisions.md](./decisions.md). Each open question is defined in [architecture.md](./architecture.md), and only its blocking effect is recorded below.

## Active task

None. T01 to T03 (Stage 1) are all validated; the next task by dependency order is T04 or T05, both of which need researcher confirmation before work starts (see Next step).

## Task states

- **Validated:** T01, T02, T03.
- **Implemented, not validated:** none.
- **In progress:** none.
- **Blocked:** T11 (Q-01).
- **Planned:** T04 to T10 and T12 to T19 (T20 and T21 were removed). For T10, the VSUMM files are already downloaded locally; the adapter and the τ and ρ decision are pending.

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

- T01: [evidence/t01-graph-validation.md](./evidence/t01-graph-validation.md). `pytest` passes (5/5), and `experiments/validate_graph.py` matches Leonardo's reference exactly on v21 (742 edges) and v29 (427 edges), with zero weight difference, using `delta_t=8` from [`configs/default.yaml`](../configs/default.yaml).
- T02: [evidence/t02-bibliography-update.md](./evidence/t02-bibliography-update.md). Ten verified bibliography entries ([6] to [15]) added to the Overleaf bibliography, with two corrections found during verification (VideoSAGE's published venue, RSGN's actual publication year), DiffPool and MinCutPool read and summarised against SLC pooling, and `\cite{}` markers inserted in the thesis body (motivation paragraph and all five specific-objectives bullets) so all 15 works are actually cited, not just listed. [references.md](./references.md) updated to match. Recompiled twice in Overleaf with no error.
- T03: [evidence/t03-documentation-inspection.md](./evidence/t03-documentation-inspection.md). Inspection of every file in `docs/` and `CLAUDE.md`: headers, the three-link rule, all relative links, D/T/Q identifier continuity, and code-vs-documentation consistency. No high-severity finding; one medium finding (the Q-05/Q-07 gap above).

## Reported, not yet evidence

These results were reported, but no saved output exists in the repository, so they do not count as validated.

- Correlation of 0.97 between edge weight and temporal distance on a third video, with the torchvision extractor. Needs a saved run of the sanity check.

## Next step

Stage 1 ([phases.md](./phases.md)) is complete: T01, T02 and T03 are all validated. [project.md](./project.md)
records PC3 as 21 October 2026, requiring the graph validation (T01) and the literature review
(T02), both now validated.

Stage 2 starts next: T04 (specify the MST hierarchy) and T05 (SumMe adapter), both depending only
on T03. Neither has been started. Both are non-trivial and need a specification, with the
researcher's confirmation, before implementation (principle 2, [CLAUDE.md](../CLAUDE.md) task
flow).
