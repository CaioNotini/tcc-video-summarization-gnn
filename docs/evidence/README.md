```
Role: Index of the evidence files: which script produced each one, the last commit that changed it, its status, and the figures and data sheets it links.
Entry: ../status.md
Related to: [status.md](../status.md), [architecture.md](../architecture.md), [images/](./images/), [sheets/](./sheets/)
```

# Evidence index

One row per evidence file. "Commit" is the last commit that changed the file when this index was
last updated (2026-11-17); `git log -1 -- <file>` gives the current one. A script that no longer
exists is named with the last commit where it exists. "Superseded" means a newer evidence file
replaces it; no file is superseded today.

Since 2026-11-17 the scripts that write evidence files require the output path (`--csv-out` or
`--output-dir`), so a test run does not overwrite evidence by accident. Commands quoted in evidence
written before that date omit this argument.

| Evidence | Task | Script(s) | Commit | Status |
| -------- | ---- | --------- | ------ | ------ |
| [t01-graph-validation.md](./t01-graph-validation.md) | T01 | `experiments/validate_graph.py`, `pytest` | `aa93fc2` | current |
| [t02-bibliography-update.md](./t02-bibliography-update.md) | T02 | none (Overleaf) | `6fcb61b` | current |
| [t03-documentation-inspection.md](./t03-documentation-inspection.md) | T03 | none | `6fcb61b` | current |
| [t04-mst-hierarchy-spec.md](./t04-mst-hierarchy-spec.md) | T04 | none (specification) | `da31588` | current |
| [t05-summe-adapter.md](./t05-summe-adapter.md) | T05 | `pytest` (`tests/test_summe_adapter.py`) | `9f40163` | current |
| [t06-mst-hierarchy.md](./t06-mst-hierarchy.md) | T06 | `experiments/validate_mst_hierarchy.py`, `experiments/plot_mst_hierarchy.py` | `6e5a29d` | current |
| [t06-additional-verification.md](./t06-additional-verification.md) | T06 | `experiments/compare_mst_cases.py`, `experiments/analyze_mst_cut_edges.py`, `experiments/plot_additional_verification.py` | `02652a5` | current |
| [sanity-check-features.md](./sanity-check-features.md) | T01 support | `experiments/validate_graph_torchvision.py` | `fc4674d` | current |
| [best-cut-port-equivalence.md](./best-cut-port-equivalence.md) | Q-10, T07 support | `experiments/validate_best_cut_keyshots.py`; `experiments/validate_best_cut.py` up to `5b73d21` | `aa93fc2` | current |
| [q10-baseline-budget.md](./q10-baseline-budget.md) | Q-10 | `experiments/measure_q10_budget.py` | `fc4674d` | current |
| [t07-baseline-summe.md](./t07-baseline-summe.md) | T07 | `experiments/run_baseline_summe.py` | `aa93fc2` | current |
| [cleanup-regression.md](./cleanup-regression.md) | cleanup of 2026-11-17 | oracle script outside the repository (described in the file) | not committed yet | current |

## Figures and data sheets

| File | Produced by | Linked from |
| ---- | ----------- | ----------- |
| [images/t06-synthetic-graph.png](./images/t06-synthetic-graph.png) | `plot_mst_hierarchy.py` | t06-mst-hierarchy, t06-additional-verification |
| [images/t06-real-video-partition.png](./images/t06-real-video-partition.png) | `plot_mst_hierarchy.py` | t06-mst-hierarchy, t06-additional-verification |
| [images/t06-cut-edges-v21.png](./images/t06-cut-edges-v21.png), [v29](./images/t06-cut-edges-v29.png) | `plot_additional_verification.py` | t06-additional-verification |
| [images/t06-truncation-comparison-v21.png](./images/t06-truncation-comparison-v21.png), [v29](./images/t06-truncation-comparison-v29.png) | `plot_additional_verification.py` | t06-additional-verification |
| [images/t06-comparison-cases.png](./images/t06-comparison-cases.png) | `plot_additional_verification.py` | t06-additional-verification |
| [sheets/t06-real-video-partition.csv](./sheets/t06-real-video-partition.csv) | `validate_mst_hierarchy.py` | t06-mst-hierarchy |
| [sheets/t06-comparison-cases.csv](./sheets/t06-comparison-cases.csv) | `compare_mst_cases.py` | t06-additional-verification |
| [sheets/t06-cut-edges-v21.csv](./sheets/t06-cut-edges-v21.csv) | `analyze_mst_cut_edges.py` | t06-additional-verification |
| [sheets/q10-baseline-budget.csv](./sheets/q10-baseline-budget.csv) | `measure_q10_budget.py` | q10-baseline-budget, t07-baseline-summe, cleanup-regression |
| [sheets/t07-baseline-summe.csv](./sheets/t07-baseline-summe.csv) | `run_baseline_summe.py` | t07-baseline-summe, cleanup-regression |

The regeneration of four of these files after the cleanup, byte-identical, is recorded in
[cleanup-regression.md](./cleanup-regression.md).
