```
Role: Evidence for Q-10. Structural bound of the HieTaSumm baseline summary length under the 15% budget, derived from the reference code, and the per-video measurement on the 25 SumMe videos with delta_t = 32.
Entry: ../status.md
Related to: [status.md](../status.md), [decisions.md](../decisions.md), [evaluation-protocol.md](../evaluation-protocol.md), [best-cut-port-equivalence.md](./best-cut-port-equivalence.md), [sheets/q10-baseline-budget.csv](./sheets/q10-baseline-budget.csv)
```

# Q-10 evidence: baseline summaries and the 15% budget

Written on 2026-11-16, submodule `references/HieTaSumm-lib` pinned at `a554ed9`, read only.
Measurement run on 2026-11-16, repository commit `c71a4ed` with uncommitted changes (listed in
[status.md](../status.md)).

**Scope decision (researcher, 2026-11-16):** for Q-10, "the baseline summary" is the native
keyshot output of `Summary.group_central_frames`, the default generation method in
`options.json` (`gen_summary_method = "group_central_frames"`), as used by D-20. The
measurement does not redefine the summary as whole clusters or any other unit.

## Result

On the 25 SumMe videos, with `delta_t = 32`, **no HieTaSumm baseline summary exceeds 15% of the
video duration**. The mean is 13.77%, the largest is 14.58% (`Cockpit_Landing` and `Notre_Dame`)
and the smallest is 10.53% (`Jumps`). This agrees with the structural bound below.

## Structural bound: the keyshot summary stays below 15%

This follows from the reference code alone. Line references are to
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
4. **Bound in frames.** Every component contributes at most `len_shot` frames, so the summary has
   at most `|S| * int(total/|S|) <= total < 0.15 * n_disk` frames.
5. **Bound in seconds.** The HieTaSumm-lib extractor (`HieTaSumm/__init__.py:86-142`) saves one
   frame at each instant `0, 0.25, ..., < int(duration)` seconds, so `n_disk = 4 * int(duration)`
   and each frame stands for 0.25 s. The summary then lasts less than
   `0.15 * n_disk * 0.25 = 0.15 * int(duration) <= 0.15 * duration` seconds.

Step 5 holds only when frames are sampled with that rule. Until 2026-11-16 the repository's
`src/tcc_gnn/features/frame_extraction.py` kept one frame every `round(fps / 4)` native frames
(4.17 frames per second at 25 fps, 4.28 at 29.97 fps, 3.75 at 30 fps), so a frame did not stand
for 0.25 s. A first, interrupted measurement with those frames converted frames to seconds with
`/ 4` and printed two videos above 15% (`Base jumping` 15.53%, `car_over_camera` 15.05%) that
were below 15% in real duration. That run is discarded. The extractor now follows the
HieTaSumm-lib rule (tested in `tests/test_frame_extraction.py`, including a byte-for-byte
comparison with the reference extractor at 25, 29.97 and 30 fps), and every SumMe frame and
feature was regenerated before the measurement below.

The bound also holds for the other generation methods in the library: `get_n_frames`
(`Summary.py:149`) emits one central frame per cluster and `n_fixed_keyframes` caps the count by
parameter. Exceeding 15% is only possible if the baseline summary were redefined as the full
clusters of the cut, which the scope decision above excludes.

## Measurement

### Setup

- Script: `experiments/measure_q10_budget.py --videos-dir data/raw/summe/SumMe/videos --delta-t 32`. The column `exceeds_15` compares with `percent` of `configs/default.yaml` (15, D-21); its name is kept from this run (note of 2026-11-17).
- Videos: the 25 `.mp4` files in `data/raw/summe/SumMe/videos/`.
- Frames: `src/tcc_gnn/features/frame_extraction.py` with `fps = 4` (one frame every 0.25 s).
  The previous frames and features under `data/processed/summe/` were deleted first, so every
  video was extracted again. The script stops if the number of frames on disk differs from the
  number of sampling instants.
- Features: ResNet50 avgpool (torchvision `IMAGENET1K_V2`), batch 16, on the GPU.
- `cutNumber`: the port `src/tcc_gnn/hierarchy/best_cut.py`, `alpha = 75`, floor 3
  (`Summary.py:57-59`). The port is checked against the reference in
  [best-cut-port-equivalence.md](./best-cut-port-equivalence.md).
- Graph, hierarchy and cut: `src/tcc_gnn/graph/frame.py` and `src/tcc_gnn/hierarchy/mst.py`.
- Keyshot length: the arithmetic of `group_central_frames` (`Summary.py:167-192`), summary
  duration `n_keyshot * 0.25 s`, video duration `native frames / fps` from OpenCV.
- `delta_t = 32` is `rate * time` of `options.json` (`Summary.py:15`). The repository's
  `configs/default.yaml` has `delta_t: 8`; the researcher asked for `delta_t = 32` only, so
  `delta_t = 8` was not run.
- Hardware: NVIDIA GeForce GTX 1660 SUPER (driver 572.42), Python 3.12.10, torch 2.5.1+cu121,
  higra 0.6.13.

### Frame check

After the run, every frame was opened and decoded with Pillow (`Image.load()`). All 25 folders
have contiguous names from `000000.jpg`, and the total is 15,904 frames:

```
TOTAL frames checked=15904 truncated_or_unreadable=0
```

### Table

Data sheet: [sheets/q10-baseline-budget.csv](./sheets/q10-baseline-budget.csv). Time is the
wall time per video in the console output: extraction, features, `bestCutNumber`, graph,
hierarchy and cut.

| Video                     | Duration (s) | Frames | bestCut raw | cutNumber | Components | Keyshot frames | Summary (s) | % of duration | Above 15% | Time (s) |
| ------------------------- | -----------: | -----: | ----------: | --------: | ---------: | -------------: | ----------: | ------------: | --------- | -------: |
| Air_Force_One             |       179.76 |    716 |           5 |         3 |          4 |            104 |        26.0 |         14.46 | no        |      176 |
| Base jumping              |       157.79 |    628 |           8 |         6 |          7 |             84 |        21.0 |         13.31 | no        |       73 |
| Bearpark_climbing         |       133.64 |    532 |           4 |         3 |          4 |             72 |        18.0 |         13.47 | no        |       99 |
| Bike Polo                 |       102.13 |    408 |           0 |         3 |          4 |             56 |        14.0 |         13.71 | no        |       69 |
| Bus_in_Rock_Tunnel        |       171.10 |    684 |           9 |         6 |          7 |             98 |        24.5 |         14.32 | no        |       43 |
| car_over_camera           |       146.21 |    584 |           0 |         3 |          4 |             80 |        20.0 |         13.68 | no        |       40 |
| Car_railcrossing          |       169.34 |    676 |           7 |         5 |          6 |             96 |        24.0 |         14.17 | no        |      110 |
| Cockpit_Landing           |       301.83 |   1204 |           0 |         3 |          4 |            176 |        44.0 |         14.58 | no        |      199 |
| Cooking                   |        85.80 |    340 |           0 |         3 |          4 |             48 |        12.0 |         13.99 | no        |       14 |
| Eiffel Tower              |       198.84 |    792 |           1 |         3 |          4 |            112 |        28.0 |         14.08 | no        |      240 |
| Excavators river crossing |       388.84 |   1552 |           8 |         6 |          7 |            224 |        56.0 |         14.40 | no        |       93 |
| Fire Domino               |        53.73 |    212 |           0 |         3 |          4 |             24 |         6.0 |         11.17 | no        |       14 |
| Jumps                     |        38.00 |    152 |           5 |         3 |          4 |             16 |         4.0 |         10.53 | no        |        8 |
| Kids_playing_in_leaves    |       106.34 |    424 |           3 |         3 |          4 |             56 |        14.0 |         13.17 | no        |       77 |
| Notre_Dame                |       192.00 |    768 |           0 |         3 |          4 |            112 |        28.0 |         14.58 | no        |      230 |
| Paintball                 |       254.25 |   1016 |           9 |         6 |          7 |            140 |        35.0 |         13.77 | no        |      161 |
| paluma_jump               |        85.89 |    340 |           5 |         3 |          4 |             48 |        12.0 |         13.97 | no        |       24 |
| playing_ball              |       103.97 |    412 |           4 |         3 |          4 |             56 |        14.0 |         13.47 | no        |       22 |
| Playing_on_water_slide    |       102.27 |    408 |           0 |         3 |          4 |             56 |        14.0 |         13.69 | no        |       22 |
| Saving dolphines          |       222.99 |    888 |           1 |         3 |          4 |            128 |        32.0 |         14.35 | no        |      117 |
| Scuba                     |        74.03 |    296 |           8 |         6 |          7 |             42 |        10.5 |         14.18 | no        |       48 |
| St Maarten Landing        |        70.04 |    280 |           3 |         3 |          4 |             40 |        10.0 |         14.28 | no        |       24 |
| Statue of Liberty         |       154.52 |    616 |           2 |         3 |          4 |             88 |        22.0 |         14.24 | no        |       41 |
| Uncut_Evening_Flight      |       322.72 |   1288 |           8 |         6 |          7 |            182 |        45.5 |         14.10 | no        |      209 |
| Valparaiso_Downhill       |       172.77 |    688 |           6 |         4 |          5 |            100 |        25.0 |         14.47 | no        |      119 |

Console summary (verbatim, mojibake from the Windows console encoding):

```
delta_t=32: 0/25 vÝdeos acima de 15%; mÚdia=13.77% mßx=14.58%
WALL_SECONDS=2324
```

Total wall time: 2,324 s (about 39 minutes) for the 25 videos.

### Observations

- In every video the summary is below the 15% budget: 0 of 25 above, so no treatment of
  over-budget summaries is needed for this baseline at `delta_t = 32`.
- The two shortest summaries are in the two shortest videos, `Jumps` (10.53%) and `Fire Domino`
  (11.17%). For `Jumps`, the budget of 22 frames split over 4 components gives 5 per component,
  and `2 * int(5/2) = 4` frames each, so 16 frames. The rounding of steps 2 and 3 loses a larger
  share of the budget when the budget is small. This is read from the arithmetic, not measured
  separately.
- The raw `bestCutNumber` is 0 to 3 on 12 videos, where the floor of 3 sets `cutNumber`
  (`Summary.py:58-59`). In total 17 videos end with `cutNumber = 3` and 4 components. The
  largest `cutNumber` is 6.
- These are lengths only. Whether the summaries are good is measured by the F-score in T07.
