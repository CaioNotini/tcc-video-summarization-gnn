```
Role: Evidence that the code cleanup of 2026-11-17 changed no result: the T07 F-scores recomputed from the saved keyshots, the pipeline on three SumMe videos against the Q-10 and T07 sheets, and the regenerated T06 artefacts.
Entry: ../status.md
Related to: [status.md](../status.md), [README.md](./README.md), [t07-baseline-summe.md](./t07-baseline-summe.md), [q10-baseline-budget.md](./q10-baseline-budget.md), [t06-mst-hierarchy.md](./t06-mst-hierarchy.md), [best-cut-port-equivalence.md](./best-cut-port-equivalence.md)
```

# Cleanup regression check

Run on 2026-11-17 on the working tree of the cleanup, on top of commit `5b73d21`; the cleanup was
then committed as `c1e41a1` to `aa93fc2`. Same hardware as the T07 evidence. The oracle script was
kept outside the repository (`.cache/scratch/oracle.py`, git-ignored); what it does is described
below. Nothing was extracted again for check 1; check 2 used the frames and features on disk.

## 1. F-scores recomputed from the saved keyshots

For each row of [sheets/t07-baseline-summe.csv](./sheets/t07-baseline-summe.csv): the `keyshots`
column, the native frame rate from the `.mp4` metadata (OpenCV, no decoding), the ground truth
from `SumMeAdapter` and the `.mat` duration. Official F: `sampled_to_native` and `f_score`
(`src/tcc_gnn/evaluation/fscore.py`). Supplementary F: `find_corresponding_frames` and `f_score`
of the submodule, through `src/tcc_gnn/reference.py`, with the 0-based keyshot ids. Criterion: the
value rounded to 4 decimals equals the CSV.

## 2. Pipeline on three videos

For `Jumps`, `Air_Force_One` and `Base jumping` (a name with a space), with `delta_t`, `alpha`,
`percent` and `rate` from `configs/default.yaml`: `best_cut_number`, `cut_number_from_best`,
`build_time_aware_graph` on the cached features, `compute_hierarchy`, `cut_graph`,
`group_central_frames`. Compared with [sheets/q10-baseline-budget.csv](./sheets/q10-baseline-budget.csv)
(`delta_t = 32` rows) and the T07 sheet: raw value, `cutNumber`, sorted keyshot list, number of
keyshots and length in percent. `run_video` of `experiments/measure_q10_budget.py` was also run and
its row compared with the Q-10 row.

## Output (verbatim)

```
== 1. F oficial e suplementar a partir dos keyshots salvos (sem reextrair)
  Air_Force_One                oficial= 32.9412 csv= 32.9412 | suplementar= 35.3208 csv= 35.3208 | IDENTICO
  Base jumping                 oficial= 25.3262 csv= 25.3262 | suplementar= 26.4550 csv=  26.455 | IDENTICO
  Bearpark_climbing            oficial= 48.0506 csv= 48.0506 | suplementar= 54.7758 csv= 54.7758 | IDENTICO
  Bike Polo                    oficial= 34.0870 csv=  34.087 | suplementar= 38.1679 csv= 38.1679 | IDENTICO
  Bus_in_Rock_Tunnel           oficial= 33.5321 csv= 33.5321 | suplementar= 34.9873 csv= 34.9873 | IDENTICO
  Car_railcrossing             oficial= 36.6197 csv= 36.6197 | suplementar= 38.9153 csv= 38.9153 | IDENTICO
  Cockpit_Landing              oficial= 37.0566 csv= 37.0566 | suplementar= 33.5994 csv= 33.5994 | IDENTICO
  Cooking                      oficial= 26.9461 csv= 26.9461 | suplementar= 30.7368 csv= 30.7368 | IDENTICO
  Eiffel Tower                 oficial= 35.7792 csv= 35.7792 | suplementar= 35.7238 csv= 35.7238 | IDENTICO
  Excavators river crossing    oficial= 22.9614 csv= 22.9614 | suplementar= 24.0227 csv= 24.0227 | IDENTICO
  Fire Domino                  oficial= 26.2887 csv= 26.2887 | suplementar= 24.4306 csv= 24.4306 | IDENTICO
  Jumps                        oficial= 37.2470 csv=  37.247 | suplementar= 54.9153 csv= 54.9153 | IDENTICO
  Kids_playing_in_leaves       oficial= 29.9065 csv= 29.9065 | suplementar= 29.1667 csv= 29.1667 | IDENTICO
  Notre_Dame                   oficial= 20.0663 csv= 20.0663 | suplementar= 26.0138 csv= 26.0138 | IDENTICO
  Paintball                    oficial= 28.2636 csv= 28.2636 | suplementar= 30.1198 csv= 30.1198 | IDENTICO
  Playing_on_water_slide       oficial= 24.0000 csv=    24.0 | suplementar= 21.5054 csv= 21.5054 | IDENTICO
  Saving dolphines             oficial= 44.6383 csv= 44.6383 | suplementar= 41.9528 csv= 41.9528 | IDENTICO
  Scuba                        oficial= 46.9428 csv= 46.9428 | suplementar= 47.3520 csv=  47.352 | IDENTICO
  St Maarten Landing           oficial= 37.0079 csv= 37.0079 | suplementar= 36.7067 csv= 36.7067 | IDENTICO
  Statue of Liberty            oficial= 30.1762 csv= 30.1762 | suplementar= 30.5499 csv= 30.5499 | IDENTICO
  Uncut_Evening_Flight         oficial= 18.5294 csv= 18.5294 | suplementar= 17.2351 csv= 17.2351 | IDENTICO
  Valparaiso_Downhill          oficial= 19.6850 csv=  19.685 | suplementar= 19.0476 csv= 19.0476 | IDENTICO
  car_over_camera              oficial= 32.0574 csv= 32.0574 | suplementar= 33.1237 csv= 33.1237 | IDENTICO
  paluma_jump                  oficial= 19.9719 csv= 19.9719 | suplementar= 17.3103 csv= 17.3103 | IDENTICO
  playing_ball                 oficial= 31.9226 csv= 31.9226 | suplementar= 32.6351 csv= 32.6351 | IDENTICO
  médias: oficial=31.20 (csv 31.20) | suplementar=32.59 (csv 32.59) | n=25
== 2. Pipeline com features em cache (bestCut -> grafo -> corte -> keyshots)
  Jumps: cut=3 n_keyshots=16 pct=10.53 | raw==q10=True cutNumber==q10==t07=True keyshots==t07=True n_keyshots==q10=True pct==q10==t07=True run_video(q10)==q10=True
  Air_Force_One: cut=3 n_keyshots=104 pct=14.46 | raw==q10=True cutNumber==q10==t07=True keyshots==t07=True n_keyshots==q10=True pct==q10==t07=True run_video(q10)==q10=True
  Base jumping: cut=6 n_keyshots=84 pct=13.31 | raw==q10=True cutNumber==q10==t07=True keyshots==t07=True n_keyshots==q10=True pct==q10==t07=True run_video(q10)==q10=True
ORACULO: OK
```

`pytest -q` on the same tree: `97 passed`.

## Other checks of the same day

- `run_video` of `experiments/run_baseline_summe.py` on `Jumps` (no mlflow run, no CSV written):
  `cut_number` 3, 16 keyshots, 10.53%, official F 37.247, supplementary F 54.9153,
  `extraction_identical` True, `features_allclose` True, `features_max_abs_diff` 0.0, keyshots
  `8 9 10 11 40 41 42 47 61 62 63 64 112 113 114 115`, the same as the T07 sheet.
- `experiments/plot_mst_hierarchy.py` (both figures), `plot_comparison_cases` of
  `experiments/plot_additional_verification.py` and `experiments/validate_mst_hierarchy.py`
  (v21, `delta_t = 8`, `cutNumber` 20) written to a scratch folder: byte-identical to
  `images/t06-synthetic-graph.png`, `images/t06-real-video-partition.png`,
  `images/t06-comparison-cases.png` and `sheets/t06-real-video-partition.csv`.
- `einsum` against `np.dot` in `best_cut.py`, weights of `delta_t = 32`, not bit-identical, so
  `einsum` was kept:

```
Jumps: n_pesos=4368 bit_a_bit_identicos=False n_diferentes=4215 max_abs=2.996e-13 std_einsum=np.float64(0.05682179449397481) std_dot=np.float64(0.05682179449397967)
Air_Force_One: n_pesos=22416 bit_a_bit_identicos=False n_diferentes=21697 max_abs=1.762e-12 std_einsum=np.float64(0.0005636518259171234) std_dot=np.float64(0.0005636518258898131)
Base jumping: n_pesos=19600 bit_a_bit_identicos=False n_diferentes=18968 max_abs=1.012e-12 std_einsum=np.float64(0.08754647400583443) std_dot=np.float64(0.08754647400591792)
```

- The `bestCutNumber` port against the reference on the current frames of the three videos is in
  [best-cut-port-equivalence.md](./best-cut-port-equivalence.md), and the T01 graph re-run in
  [t01-graph-validation.md](./t01-graph-validation.md).

## Result

The cleanup changed no number: the 25 official and 25 supplementary F-scores (means 31.20 and
32.59) and the `cutNumber`, keyshots and length of the three videos are identical to the T07 and
Q-10 sheets, and the regenerated T06 artefacts are byte-identical.

## Re-run after the audit fixes (2026-11-17)

After the fixes of the audit of the same day (on top of commit `aa93fc2`, then committed as `e4140d2` to `7190494`), the same
oracle script was run again (`python .cache/scratch/oracle.py`). Its 32 output lines are identical,
line by line (`diff` with no output), to the output above, ending in `ORACULO: OK`: official and
supplementary F-scores identical on the 25 videos (means 31.20 and 32.59), and `cutNumber`,
keyshots and length identical on `Jumps`, `Air_Force_One` and `Base jumping`.

Also on the fixed tree, written to `.cache/scratch/out/` and compared by hash with the evidence
files: the seven T06 figures (`plot_mst_hierarchy.py` and `plot_additional_verification.py` with
`--output-dir`), `sheets/t06-real-video-partition.csv` (`validate_mst_hierarchy.py`, v21,
`cutNumber` 20) and `sheets/t06-comparison-cases.csv` (`compare_mst_cases.py`, default cases plus
`--video v29 --cut-number 5 --cutnumber-fonte "bestCutNumber(alpha=75)"`) are byte-identical.
`validate_best_cut_keyshots.py --videos Jumps`, now without cache, gives `IDENTICO` for
`cutNumber` and keyshots. `pytest`: 111 passed, 1 skipped (TensorFlow not installed).
