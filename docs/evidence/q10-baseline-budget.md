```
Role: Evidence for Q-10. Structural analysis of the HieTaSumm baseline summary length under the 15% budget, derived from the reference code; exact per-video measurement pending the SumMe data.
Entry: ../status.md
Related to: [status.md](../status.md), [decisions.md](../decisions.md), [evaluation-protocol.md](../evaluation-protocol.md)
```

# Q-10 evidence: baseline summaries and the 15% budget

Written on 2026-11-16, submodule `references/HieTaSumm-lib` pinned at `a554ed9`, read only.

**Scope decision (researcher, 2026-11-16):** for Q-10, "the baseline summary" is the native
keyshot output of `Summary.group_central_frames`, the default generation method in
`options.json` (`gen_summary_method = "group_central_frames"`), as used by D-20. The
measurement does not redefine the summary as whole clusters or any other unit.

## Structural result: the keyshot summary never exceeds 15% of the extracted frames

This follows from the reference code alone, before any run. Line references are to
`references/HieTaSumm-lib/HieTaSumm/Summary.py`.

1. **Total budget.** `Summary.__init__` fixes the total keyshot budget before any cluster is
   formed: `len_shot = round(n_disk * percent/100) - 1` (`Summary.py:56`), where `n_disk` is the
   number of extracted frames on disk and `percent = 15` (`options.json`). So the budget is
   `round(0.15 * n_disk) - 1 < 0.15 * n_disk` frames.
2. **Split across clusters.** `group_central_frames` divides that budget equally over the `|S|`
   connected components of the cut graph: `len_shot = int(len_shot / len(S))`
   (`Summary.py:170`). Integer division only rounds down.
3. **Per-cluster contribution.** For each component with `len_leaf_list` leaf frames and central
   index `cn = int(len_leaf_list / 2)` (`Summary.py:177`):
   - `cn == 0` (single-leaf component): contributes 0 frames (`Summary.py:178` guard);
   - `len_leaf_list < len_shot`: contributes `range(0, len_leaf_list - 1)`, i.e.
     `len_leaf_list - 1 < len_shot` frames (`Summary.py:180-182`);
   - otherwise: contributes `range(cn - int(len_shot/2), cn + int(len_shot/2))`, i.e.
     `2 * int(len_shot/2) <= len_shot` frames (`Summary.py:184-185`).
4. **Bound.** Every component contributes at most `len_shot` frames, so the summary has at most
   `|S| * int(total/|S|) <= total = round(0.15 * n_disk) - 1 < 0.15 * n_disk` frames.

Frames are extracted uniformly at `rate = 4` frames per second (`options.json`), so each
extracted frame stands for 1/4 second and the frame-count ratio equals the duration ratio, up to
one frame interval at the end of the video. Therefore, under the native keyshot rule, **no
HieTaSumm baseline summary can exceed the 15% duration budget of D-21**, for any video, any
`delta_t`, any `cutNumber` and any `bestCutNumber` outcome: the bound comes from the fixed total
budget at `Summary.py:56`, not from the clustering.

The bound also holds for the other generation methods in the library: `get_n_frames`
(`Summary.py:149`) emits one central frame per cluster and `n_fixed_keyframes` caps the count by
parameter, both far below the budget. Exceeding 15% is only possible if the baseline summary
were redefined as the full clusters of the cut, which the researcher's scope decision above
excludes for Q-10.

## What this does and does not settle

- It settles the Q-10 question "how many of the 25 SumMe videos exceed the budget": **zero, by
  construction**, under the native keyshot rule.
- It does **not** give the exact per-video summary length (how far below 15% each video lands:
  single-leaf clusters contribute nothing and small clusters contribute less than their share,
  so the realised length varies per video). That measurement needs the SumMe videos on disk
  (frames for `bestCutNumber` and for feature extraction) and is pending until the data is
  available locally; it will be added to this file when run.

## Parameter note for the pending measurement

`options.json` gives `rate = 4`, `time = 8`, and `Summary.__init__` uses
`delta_t = rate * time = 32` (`Summary.py:15`). The repository's `configs/default.yaml` records
`delta_t: 8` with `rate: 4`, and the T01/T06 validations ran with `delta_t = 8` (where the value
does not affect port equivalence). The two values give different graphs, different
`bestCutNumber` statistics and therefore different per-video lengths — not a different bound.
The pending measurement will report both until the researcher fixes one; the discrepancy is
reported here rather than resolved.
