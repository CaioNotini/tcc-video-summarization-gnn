```
Role: Evidence that the vectorised port of Summary.bestCutNumber (src/tcc_gnn/hierarchy/best_cut.py) reproduces the HieTaSumm-lib reference on real SumMe videos, for cutNumber and for the keyshot list, and records its two intentional divergences.
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

## What is compared

- `experiments/validate_best_cut.py`: the raw `bestCutNumber` value and the derived `cutNumber`
  (`int(raw * alpha/100)`, floor 3, `Summary.py:57-59`), port against `Summary.bestCutNumber`
  called on the real submodule.
- `experiments/validate_best_cut_keyshots.py`: the full keyshot list. The port path is
  `best_cut_number` → `cut_number_from_best` → `build_time_aware_graph` → `compute_hierarchy` →
  `cut_graph`. The reference path is `Summary.bestCutNumber` → `Frame.calc_end` /
  `Frame.calculate_similarity` → `Graph.compute_hierarchy` → `Graph.cut_graph`. Both cuts go
  through the same reproduction of `Summary.group_central_frames` (`Summary.py:167-192`), and the
  reference side is also run through the real `Summary.group_central_frames` method, with its
  `cp` calls disabled, to check that the reproduction matches the original method.
  The criterion is identical sorted lists of keyshot frame indices.

## Output (verbatim)

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
run with the extractor of `experiments/measure_q10_budget.py`; `Air_Force_One` used its cached
features. The keyshot times include graph construction and hierarchy on both sides, so they do
not isolate the `bestCutNumber` step; the first output does.

## Result

On both videos the port gives the same raw `bestCutNumber` (5.0), the same `cutNumber` (3) and
the same keyshot list (16 frames on `Jumps`, 104 on `Air_Force_One`) as the reference, and the
reproduction of `group_central_frames` gives the same list as the original method. Two videos and
one `delta_t` are covered; the unit tests in `tests/test_best_cut.py` cover synthetic cases with
`delta_t` of 3, 8 and 32.

## Intentional divergences

The port differs from the reference only on inputs that the HieTaSumm pipeline never produces.
Both cases raise `ValueError` and are covered by `tests/test_best_cut.py`.

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
