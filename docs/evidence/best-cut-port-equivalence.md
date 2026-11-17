```
Role: Evidence that the vectorised port of Summary.bestCutNumber (src/tcc_gnn/hierarchy/best_cut.py) reproduces the HieTaSumm-lib reference on real SumMe videos, for cutNumber and for the keyshot list, and records its three intentional divergences.
Entry: ../status.md
Related to: [status.md](../status.md), [q10-baseline-budget.md](./q10-baseline-budget.md), [t06-mst-hierarchy.md](./t06-mst-hierarchy.md)
```

# bestCutNumber port: equivalence with the reference

Run on 2026-11-16, repository commit `c71a4ed` with uncommitted changes (listed in
[status.md](../status.md)); submodule `references/HieTaSumm-lib` at `a554ed9`. Hardware: NVIDIA
GeForce GTX 1660 SUPER (driver 572.42), Python 3.12.10, torch 2.5.1+cu121, higra 0.6.13, numpy
2.5.4. Videos: SumMe `Jumps` (159 frames) and `Air_Force_One` (749 frames), frames already on disk
under `data/processed/summe/`, `delta_t = 32` (`rate * time` of `options.json`, `Summary.py:15`),
`alpha = 75`, `percent = 15`.

These frames came from `frame_extraction.py` before its fix of the same day (one frame every
`round(fps/4)` native frames) and were later replaced by frames sampled every 0.25 s (152 frames
for `Jumps`, 716 for `Air_Force_One`, see [q10-baseline-budget.md](./q10-baseline-budget.md)).
Port and reference read the same frames in each comparison, so the equivalence does not depend
on how the frames were sampled. The comparison was repeated on the new frames on 2026-11-17, with
a third video (see "Re-run on the current frames").

## What is compared

- The raw `bestCutNumber` value and the derived `cutNumber` (`int(raw * alpha/100)`, floor 3,
  `Summary.py:57-59`), port against `Summary.bestCutNumber` called on the real submodule. In the
  first run this was `experiments/validate_best_cut.py`, which exists up to commit `5b73d21`; on
  2026-11-17 it was merged into `experiments/validate_best_cut_keyshots.py`, which prints the same
  comparison with the time of each `bestCutNumber`.
- `experiments/validate_best_cut_keyshots.py`: the full keyshot list. The port path is
  `best_cut_number` → `cut_number_from_best` → `build_time_aware_graph` → `compute_hierarchy` →
  `cut_graph`. The reference path is `Summary.bestCutNumber` → `Frame.calc_end` /
  `Frame.calculate_similarity` → `Graph.compute_hierarchy` → `Graph.cut_graph`. Both cuts go
  through the same reproduction of `Summary.group_central_frames` (`Summary.py:167-192`), and the
  reference side is also run through the real `Summary.group_central_frames` method, with its
  `cp` calls disabled, to check that the reproduction matches the original method.
  The criterion is identical sorted lists of keyshot frame indices.

## Output of the first run (verbatim, 2026-11-16)

`validate_best_cut.py --frames-dirs .../Jumps/frames .../Air_Force_One/frames --delta-t 32`:

```
Jumps delta_t=32: ref raw=5.0 cut=3 [2.1s] | port raw=5.0 cut=3 [0.9s] | IDENTICO
Air_Force_One delta_t=32: ref raw=5.0 cut=3 [108.1s] | port raw=5.0 cut=3 [72.1s] | IDENTICO
```

`validate_best_cut_keyshots.py` (default videos `Jumps` and `Air_Force_One`, `--delta-t 32`):

```
Device: cuda
Jumps delta_t=32 [ref_bestCut=computed]:
  cutNumber: port raw=5.0 cut=3 | ref raw=5.0 cut=3 | IDENTICO
  keyshots: port=16 ref=16 | IDENTICO
  reproducao vs metodo real: OK (repro=16 actual=16)
  tempos: port=1.0s ref=1.5s
Air_Force_One delta_t=32 [ref_bestCut=computed]:
  cutNumber: port raw=5.0 cut=3 | ref raw=5.0 cut=3 | IDENTICO
  keyshots: port=104 ref=104 | IDENTICO
  reproducao vs metodo real: OK (repro=104 actual=104)
  tempos: port=69.0s ref=82.8s
```

Both scripts exited with code 0. The `Jumps` features were not cached and were extracted in this
run with the extractor then in `experiments/measure_q10_budget.py` (now
`src/tcc_gnn/features/features.py`); `Air_Force_One` used its cached
features. The keyshot times include graph construction and hierarchy on both sides, so they do
not isolate the `bestCutNumber` step; the first output does.

## Re-run on the current frames (verbatim, 2026-11-17)

Frames sampled every 0.25 s (152 for `Jumps`, 716 for `Air_Force_One`, 628 for `Base jumping`),
cached features, same hardware, `delta_t`, `alpha` and `percent` read from `configs/default.yaml`.
The reference `bestCutNumber` was computed again (`--no-cache` for the first two videos; the third
had no cache entry), because the cache entries of 2026-11-16 came from the old frames.

`python experiments/validate_best_cut_keyshots.py --videos Jumps Air_Force_One --no-cache`:

```
Device: cuda
Jumps delta_t=32 [ref_bestCut=computed]:
  cutNumber: port raw=5.0 cut=3 [0.8s] | ref raw=5.0 cut=3 [1.4s] | IDENTICO
  keyshots: port=16 ref=16 | IDENTICO
  reproducao vs metodo real: OK (repro=16 actual=16)
  tempos: port=0.9s ref=1.4s
Air_Force_One delta_t=32 [ref_bestCut=computed]:
  cutNumber: port raw=5.0 cut=3 [58.6s] | ref raw=5.0 cut=3 [66.2s] | IDENTICO
  keyshots: port=104 ref=104 | IDENTICO
  reproducao vs metodo real: OK (repro=104 actual=104)
  tempos: port=58.8s ref=66.5s
```

`python experiments/validate_best_cut_keyshots.py --videos Jumps Air_Force_One "Base jumping"`,
lines of `Base jumping` only (the other two videos of that run read the old cache, so their lines
are replaced by the run above):

```
Base jumping delta_t=32 [ref_bestCut=computed]:
  cutNumber: port raw=8.0 cut=6 [10.8s] | ref raw=8.0 cut=6 [10.1s] | IDENTICO
  keyshots: port=84 ref=84 | IDENTICO
  reproducao vs metodo real: OK (repro=84 actual=84)
  tempos: port=11.0s ref=10.4s
```

Both runs exited with code 0. The cache key was `video|delta_t` and did not depend on the frames,
so a cache entry was only valid for the frames it was computed on. By the researcher's decision of
2026-11-17 the cache was removed: the script now always computes the reference `bestCutNumber`,
and the `--no-cache` option and the `[ref_bestCut=...]` tag in the output no longer exist.

## Result

On both videos the port gives the same raw `bestCutNumber` (5.0), the same `cutNumber` (3) and
the same keyshot list (16 frames on `Jumps`, 104 on `Air_Force_One`) as the reference, and the
reproduction of `group_central_frames` gives the same list as the original method. On the current
frames the same holds for the three videos of the re-run (`Base jumping`: raw 8.0, `cutNumber` 6,
84 keyshots). Three videos and one `delta_t` are covered; the unit tests in `tests/test_best_cut.py` cover synthetic cases with
`delta_t` of 3, 8 and 32.

## Intentional divergences

The port differs from the reference only on inputs that the HieTaSumm pipeline never produces.
The three cases raise `ValueError` and are covered by `tests/test_best_cut.py`.

1. **Standard deviation of the weights equal to zero.** The original loop
   `while cut <= 1: cut = cut * 10` (`Summary.py:123-124`) never ends when `cut = 0`, because
   `0 * 10 = 0`. This happens, for example, with a single frame, where the only pair is `(0, 0)`
   and the weight list is `[1.0]`. The port raises `ValueError` instead of hanging
   (`test_best_cut_number_desvio_zero_levanta_erro`). Reason: a process that never returns gives
   no result to compare, and an explicit error is the only observable behaviour that can be tested.
2. **`delta_t <= 0`.** In the original, `calc_end` (`Summary.py:256-260`) returns the full length
   when `delta_t < 0`, so every pair `(v1, v2 >= v1)` enters the statistic; with `delta_t = 0` it
   returns `v1`, the pair list is empty and, by reading the code (not run), `np.std([])` gives
   `nan` and the function returns `nan`. The port raises `ValueError`
   (`test_best_cut_number_delta_t_invalido`). Reason: the pipeline always uses
   `delta_t = rate * time = 32 > 0` (`Summary.py:15`), the window-free branch is not part of the
   time-aware graph, and reproducing it would mean a dense computation that the port was written to
   avoid.
3. **A frame whose half-resolution pixels are all zero (all black).** Its norm is zero, so the
   cosine of every pair that includes it is `0/0`. In the original, `spatial.distance.cosine`
   returns `nan`, `nan < 20` is false, the weight stays `nan`, `np.std` gives `nan`, the
   `while cut <= 1` loop is skipped, `bestCutNumber` returns `nan`, and the pipeline fails at
   `int(nan * alpha/100)` (`Summary.py:57`). Before 2026-11-17 the port dropped those pairs with
   the same `~isnan` filter that marks the pairs outside the window, and returned a number. It now
   raises `ValueError` (`test_frame_preto_levanta_erro_como_o_original_falha`, which also checks
   that the reference fails). Decided by the researcher on 2026-11-17. No result is affected:
   none of the 16,079 frames on disk on that date (25 SumMe videos and VSUMM v21, v29) is all
   black at half resolution.
