```
Role: Evidence for T07. HieTaSumm baseline run end to end on the 25 SumMe videos, with the official F-score (D-26), a supplementary F-score under the original HieTaSumm-lib rule, running time and memory per stage, and a reproducibility check of frames and features.
Entry: ../status.md
Related to: [status.md](../status.md), [phases.md](../phases.md), [evaluation-protocol.md](../evaluation-protocol.md), [decisions.md](../decisions.md), [q10-baseline-budget.md](./q10-baseline-budget.md), [best-cut-port-equivalence.md](./best-cut-port-equivalence.md), [sheets/t07-baseline-summe.csv](./sheets/t07-baseline-summe.csv)
```

# T07 evidence: HieTaSumm baseline on SumMe

Run on 2026-11-16, repository commit `573e556` with uncommitted changes (the T07 code listed in
[status.md](../status.md)). Submodule `references/HieTaSumm-lib` at `a554ed9`.

## Result

| Metric (25 SumMe videos) | Value |
| ------------------------ | ----: |
| **F-score, official rule (D-26), maximum over annotators** | **31.20** |
| F-score, original HieTaSumm-lib rule (supplementary, not official) | 32.59 |
| Mean summary length, % of video duration | 13.77 |
| Videos above the 15% budget | 0 of 25 |

The official number is the one defined by the protocol
([evaluation-protocol.md](../evaluation-protocol.md)): every sampled keyshot frame stands for the
native frames of its 0.25 s interval (D-26), and the F-score is the maximum over annotators
(`Evaluation.py:132-155`). The supplementary number uses the original
`Evaluation.find_corresponding_frames` and is reported only to compare with published numbers.
It receives the 0-based leaf id of each keyshot, which is what the original passes (the file name `str(leaf).zfill(6)` written by `Summary.group_central_frames`), although the original extractor numbers its jpgs from `000001`; this off-by-one is a feature of the original, kept on purpose (researcher, 2026-11-17).

**Published reference (to be filled in by the researcher):**

| Source (paper, table, page) | Dataset and setting | Published F-score | Our F-score, original rule | Our F-score, official rule |
| --------------------------- | ------------------- | ----------------: | -------------------------: | -------------------------: |
| _to be filled in_ | SumMe | _to be filled in_ | 32.59 | 31.20 |

The published value was not found in the repository, and the technical report link found on
2026-11-16 (imscience.icei.pucminas.br) returned HTTP 404. The researcher will check the paper.

## Setup

- Command: `python experiments/run_baseline_summe.py` (all 25 videos, `--delta-t 32` by default).
- Pipeline per video, in this order:
  1. frame extraction, one frame every 0.25 s (`src/tcc_gnn/features/frame_extraction.py`, D-25);
  2. ResNet50 avgpool features on the GPU (`experiments/measure_q10_budget.py`,
     `extract_features_batched`);
  3. `bestCutNumber` and `cutNumber` (`src/tcc_gnn/hierarchy/best_cut.py`, `alpha = 75`, floor 3);
  4. time-aware graph with `delta_t = 32` (`src/tcc_gnn/graph/frame.py`, D-24);
  5. MST hierarchy and cut (`src/tcc_gnn/hierarchy/mst.py`);
  6. keyshots (`src/tcc_gnn/hierarchy/keyshots.py`, port of `Summary.group_central_frames`);
  7. official evaluation: `sampled_to_native` and `f_score` (`src/tcc_gnn/evaluation/fscore.py`);
  8. supplementary evaluation: `Evaluation.find_corresponding_frames` and `Evaluation.f_score`
     from the submodule, with the `.mat` duration, as in `Evaluation.evaluate`.
- Ground truth: `user_summary` and `n_frames` from `eccv16_dataset_summe_google_pool5.h5`
  through `SumMeAdapter`. Each `.mp4` is linked to its `video_N` by the `video_name` field of the
  `.h5` (`SumMeAdapter.video_names()`); the names match the `.mp4` file names exactly, including
  the 8 names with spaces. The `.mat` files use `_` instead of spaces.
- Frame rate from the `.mp4` (OpenCV), summary vector length from the `.h5` `n_frames`, video
  duration from OpenCV (researcher's decision, 2026-11-16).
- Hardware and software: Intel Core i5-8400 @ 2.80 GHz, 15.9 GB RAM, NVIDIA GeForce GTX 1660 SUPER
  6 GB (driver 572.42), Windows 10 Pro 64-bit 10.0.19045, Python 3.12.10, torch 2.5.1+cu121
  (CUDA 12.1), mlflow 3.18.0, psutil 7.2.2.

## Experiment tracking

The run is in mlflow, experiment `T07-hietasumm-baseline-summe` (id 2), run
`c02e2eb0b1c74fea9399fb5c199a1126`, status FINISHED. It holds the parameters (`delta_t`, `alpha`,
`percent`, `rate`, extractor, frame sampling, mapping rule, commit, number of videos), the hardware
tags, the per-video F-scores, length and `cutNumber` as metric histories, the means, the wall time
and the CSV as an artifact. The two-video development run is in experiment `T07-dev`.

**Change of tracking backend.** `configs/default.yaml` had `tracking_uri: "./mlruns"`, a file
store. mlflow 3.18.0 refuses it and stops with "The filesystem tracking backend (e.g., './mlruns')
is in maintenance mode". By the researcher's decision (2026-11-16) the tracking URI is now
`sqlite:///mlruns/mlflow.db` and the artifacts are in `./mlruns/` (`artifact_root`), both inside
the git-ignored `mlruns/` folder.

## Checks

- **Against the Q-10 measurement.** For all 25 videos, `bestCutNumber` raw value, `cutNumber`,
  number of sampled frames, number of keyshots and summary length in % are equal to
  [sheets/q10-baseline-budget.csv](./sheets/q10-baseline-budget.csv) (0 differences). Mean length
  13.77%, as in Q-10.
- **Official mapping.** The % of native frames marked by the D-26 rule differs from the % of
  duration by at most 0.04 points.
- **Reference values of the T07 note in [phases.md](../phases.md).** `Jumps` and `Air_Force_One`
  give raw 5.0 and `cutNumber = 3`, as the reference does
  ([best-cut-port-equivalence.md](./best-cut-port-equivalence.md)).
- **Values.** Every F-score is in [0, 100], with no NaN.

### Reproducibility of frames and features

For each video the frames were extracted again into a temporary folder under `.cache/` and
compared byte for byte with `data/processed/summe/<video>/frames`. The features were computed
again on the GPU without overwriting the cache and compared with the `.npy` file
(`np.allclose`, default tolerances). The rest of the pipeline used the files on disk.

| Check | Result |
| ----- | -----: |
| Frames identical byte for byte | 25 of 25 videos |
| Features `np.allclose` to the cached `.npy` | 25 of 25 videos |
| Largest absolute difference in the features | 0.0 |

## Per-video results

Data sheet: [sheets/t07-baseline-summe.csv](./sheets/t07-baseline-summe.csv) (also the keyshot
list of each video, the raw `bestCutNumber`, frame counts, both durations, times and memory per
stage).

| Video | fps | cutNumber | Keyshots | % duration | F official (D-26) | F original rule | % native, original rule |
| ----- | --: | --------: | -------: | ---------: | ----------------: | --------------: | ----------------------: |
| Air_Force_One | 25 | 3 | 104 | 14.46 | 32.94 | 35.32 | 15.58 |
| Base jumping | 29.97 | 6 | 84 | 13.31 | 25.33 | 26.45 | 13.74 |
| Bearpark_climbing | 25 | 3 | 72 | 13.47 | 48.05 | 54.78 | 15.71 |
| Bike Polo | 30 | 3 | 56 | 13.71 | 34.09 | 38.17 | 16.32 |
| Bus_in_Rock_Tunnel | 30 | 6 | 98 | 14.32 | 33.53 | 34.99 | 15.59 |
| Car_railcrossing | 29.97 | 5 | 96 | 14.17 | 36.62 | 38.92 | 15.27 |
| Cockpit_Landing | 29.97 | 3 | 176 | 14.58 | 37.06 | 33.60 | 12.99 |
| Cooking | 15 | 3 | 48 | 13.99 | 26.95 | 30.74 | 22.55 |
| Eiffel Tower | 25 | 3 | 112 | 14.08 | 35.78 | 35.72 | 15.59 |
| Excavators river crossing | 25 | 6 | 224 | 14.40 | 22.96 | 24.02 | 18.26 |
| Fire Domino | 30 | 3 | 24 | 11.17 | 26.29 | 24.43 | 17.06 |
| Jumps | 25 | 3 | 16 | 10.53 | 37.25 | 54.92 | 15.58 |
| Kids_playing_in_leaves | 29.97 | 3 | 56 | 13.17 | 29.91 | 29.17 | 14.90 |
| Notre_Dame | 24 | 3 | 112 | 14.58 | 20.07 | 26.01 | 16.78 |
| Paintball | 23.976 | 6 | 140 | 13.77 | 28.26 | 30.12 | 16.65 |
| Playing_on_water_slide | 29.97 | 3 | 56 | 13.69 | 24.00 | 21.51 | 15.50 |
| Saving dolphines | 29.97 | 3 | 128 | 14.35 | 44.64 | 41.95 | 13.09 |
| Scuba | 30 | 6 | 42 | 14.18 | 46.94 | 47.35 | 20.26 |
| St Maarten Landing | 25 | 3 | 40 | 14.28 | 37.01 | 36.71 | 18.56 |
| Statue of Liberty | 25 | 3 | 88 | 14.24 | 30.18 | 30.55 | 16.18 |
| Uncut_Evening_Flight | 29.97 | 6 | 182 | 14.10 | 18.53 | 17.24 | 16.54 |
| Valparaiso_Downhill | 29.97 | 4 | 100 | 14.47 | 19.68 | 19.05 | 15.45 |
| car_over_camera | 29.97 | 3 | 80 | 13.68 | 32.06 | 33.12 | 17.69 |
| paluma_jump | 29.97 | 3 | 48 | 13.97 | 19.97 | 17.31 | 15.54 |
| playing_ball | 30 | 3 | 56 | 13.47 | 31.92 | 32.64 | 16.03 |
| **Mean** | | | | 13.77 | 31.20 | 32.59 | 16.30 |

Observations, read from the table:

- The official F-score ranges from 18.53 (`Uncut_Evening_Flight`) to 48.05 (`Bearpark_climbing`).
- The original rule gives a higher F-score on 15 videos and a lower one on 10. Under it, the
  evaluated summary covers 12.99% to 22.55% of the native frames (mean 16.30%), above 15% on 21
  videos. The rule marks 25 native frames from the start of each second that contains a keyshot,
  so a keyshot run counts for whole seconds, and on 15 fps (`Cooking`) 25 frames are more than
  one second. This is why the two numbers differ and why the protocol uses D-26.
- The human summaries of SumMe have a mean length of 13.14% of the frames (10.76% to 14.95% per
  video), close to the baseline's 13.77%.

## Running time and memory

Wall time per stage in seconds, from `time.perf_counter`. CPU peak is the largest resident set
size (RSS) of the whole process during the video's stages, sampled every 20 ms with psutil; it
includes the ResNet50 model and memory not yet released from earlier videos, so it is an upper
bound for the video. GPU peak is `torch.cuda.max_memory_allocated` during feature extraction.
Per-stage values of both are in the CSV.

| Video | Extraction | Features | bestCut | Graph | Hierarchy + cut | Keyshots | Eval official | Eval original | Total (s) | CPU peak (MB) | GPU peak (MB) |
| ----- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Air_Force_One | 28.2 | 18.6 | 61.7 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 108.8 | 5552 | 260 |
| Base jumping | 7.7 | 7.0 | 11.6 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 26.6 | 1919 | 260 |
| Bearpark_climbing | 12.2 | 10.0 | 22.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 44.5 | 2919 | 260 |
| Bike Polo | 9.6 | 7.3 | 16.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 33.1 | 2865 | 260 |
| Bus_in_Rock_Tunnel | 4.1 | 6.2 | 7.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 17.5 | 1478 | 260 |
| Car_railcrossing | 17.8 | 11.5 | 27.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 56.7 | 3056 | 260 |
| Cockpit_Landing | 27.9 | 20.0 | 48.8 | 0.3 | 0.1 | 0.0 | 0.0 | 0.0 | 97.2 | 3491 | 260 |
| Cooking | 1.0 | 2.7 | 1.2 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 4.9 | 1111 | 260 |
| Eiffel Tower | 32.7 | 21.0 | 69.0 | 0.2 | 0.1 | 0.0 | 0.0 | 0.0 | 123.0 | 5775 | 260 |
| Excavators river crossing | 8.9 | 11.9 | 15.6 | 0.3 | 0.1 | 0.0 | 0.0 | 0.0 | 36.9 | 1608 | 260 |
| Fire Domino | 1.1 | 2.3 | 1.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.0 | 1212 | 260 |
| Jumps | 0.4 | 1.6 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.1 | 1161 | 260 |
| Kids_playing_in_leaves | 10.4 | 8.0 | 17.2 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 35.7 | 2917 | 260 |
| Notre_Dame | 30.1 | 21.3 | 67.1 | 0.2 | 0.1 | 0.0 | 0.0 | 0.0 | 118.6 | 6202 | 260 |
| Paintball | 20.4 | 16.8 | 40.5 | 0.2 | 0.1 | 0.0 | 0.0 | 0.0 | 78.0 | 3394 | 260 |
| Playing_on_water_slide | 2.0 | 3.5 | 3.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 8.8 | 1294 | 260 |
| Saving dolphines | 17.0 | 12.6 | 26.3 | 0.2 | 0.1 | 0.0 | 0.0 | 0.0 | 56.2 | 2726 | 260 |
| Scuba | 6.6 | 5.7 | 12.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 24.4 | 2805 | 260 |
| St Maarten Landing | 2.4 | 3.5 | 5.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 11.1 | 1701 | 260 |
| Statue of Liberty | 3.4 | 6.0 | 8.3 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 17.9 | 1631 | 260 |
| Uncut_Evening_Flight | 30.7 | 21.4 | 51.6 | 0.3 | 0.1 | 0.0 | 0.0 | 0.0 | 104.1 | 3344 | 260 |
| Valparaiso_Downhill | 16.8 | 12.1 | 27.6 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 56.7 | 3131 | 260 |
| car_over_camera | 3.6 | 5.5 | 8.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 17.3 | 1558 | 260 |
| paluma_jump | 2.4 | 3.9 | 4.9 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 11.3 | 1503 | 260 |
| playing_ball | 1.8 | 3.7 | 3.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 8.7 | 1282 | 260 |
| **Total** | 299.3 | 243.8 | 557.5 | 3.5 | 1.1 | 0.2 | 0.0 | 0.5 | 1106.0 | 6202 | 260 |

The stages add up to 1,106 s; the whole run took 1,245 s (about 21 minutes), the rest being the
byte comparison, loading the cached features and the mlflow writes. `bestCutNumber` is half of
the time: it reads every frame at half resolution in raw pixels (D-14, D-20). The baseline after
the features (graph, hierarchy, cut and keyshots) takes under 0.5 s per video.

## Data divergences

- **Native frame counts, OpenCV against the `.h5`.** They differ on 3 videos:
  `Bus_in_Rock_Tunnel` 5133 against 5131, `Cooking` 1287 against 1286, and `playing_ball` 3119
  against 3120. The summary vector has the `.h5` length, so evaluation uses the ground truth's
  frame count. Frame sampling and duration use OpenCV.
- **Duration, `.mat` against OpenCV.** The `.mat` `video_duration`, used only by the
  supplementary rule, is longer than the OpenCV duration (frames / fps) on these videos, in
  seconds: Paintball +1.42, Jumps +1.31, Statue of Liberty +1.12, Base jumping +1.04, Eiffel
  Tower +0.99, Fire Domino +0.99, Bike Polo +0.88, Cooking +0.78, Cockpit_Landing +0.47,
  Bus_in_Rock_Tunnel +0.10. On the other 15 videos the difference is at most 0.04 s. Both
  durations are in the CSV.

## Console output (verbatim)

```
Device: cuda | 25 vídeos | tracking sqlite:///D:/Estudos/TCC/tcc-video-summarization-gnn/mlruns/mlflow.db | artefatos file:///D:/Estudos/TCC/tcc-video-summarization-gnn/mlruns
[1/25] Air_Force_One: cut=3 keyshots=104 pct=14.46% F_oficial=32.9412 F_regra_original=35.3208 frames_iguais=True features_allclose=True [109s]
[2/25] Base jumping: cut=6 keyshots=84 pct=13.31% F_oficial=25.3262 F_regra_original=26.455 frames_iguais=True features_allclose=True [27s]
[3/25] Bearpark_climbing: cut=3 keyshots=72 pct=13.47% F_oficial=48.0506 F_regra_original=54.7758 frames_iguais=True features_allclose=True [45s]
[4/25] Bike Polo: cut=3 keyshots=56 pct=13.71% F_oficial=34.087 F_regra_original=38.1679 frames_iguais=True features_allclose=True [33s]
[5/25] Bus_in_Rock_Tunnel: cut=6 keyshots=98 pct=14.32% F_oficial=33.5321 F_regra_original=34.9873 frames_iguais=True features_allclose=True [18s]
[6/25] Car_railcrossing: cut=5 keyshots=96 pct=14.17% F_oficial=36.6197 F_regra_original=38.9153 frames_iguais=True features_allclose=True [57s]
[7/25] Cockpit_Landing: cut=3 keyshots=176 pct=14.58% F_oficial=37.0566 F_regra_original=33.5994 frames_iguais=True features_allclose=True [97s]
[8/25] Cooking: cut=3 keyshots=48 pct=13.99% F_oficial=26.9461 F_regra_original=30.7368 frames_iguais=True features_allclose=True [5s]
[9/25] Eiffel Tower: cut=3 keyshots=112 pct=14.08% F_oficial=35.7792 F_regra_original=35.7238 frames_iguais=True features_allclose=True [123s]
[10/25] Excavators river crossing: cut=6 keyshots=224 pct=14.4% F_oficial=22.9614 F_regra_original=24.0227 frames_iguais=True features_allclose=True [37s]
[11/25] Fire Domino: cut=3 keyshots=24 pct=11.17% F_oficial=26.2887 F_regra_original=24.4306 frames_iguais=True features_allclose=True [5s]
[12/25] Jumps: cut=3 keyshots=16 pct=10.53% F_oficial=37.247 F_regra_original=54.9153 frames_iguais=True features_allclose=True [3s]
[13/25] Kids_playing_in_leaves: cut=3 keyshots=56 pct=13.17% F_oficial=29.9065 F_regra_original=29.1667 frames_iguais=True features_allclose=True [36s]
[14/25] Notre_Dame: cut=3 keyshots=112 pct=14.58% F_oficial=20.0663 F_regra_original=26.0138 frames_iguais=True features_allclose=True [119s]
[15/25] Paintball: cut=6 keyshots=140 pct=13.77% F_oficial=28.2636 F_regra_original=30.1198 frames_iguais=True features_allclose=True [78s]
[16/25] Playing_on_water_slide: cut=3 keyshots=56 pct=13.69% F_oficial=24.0 F_regra_original=21.5054 frames_iguais=True features_allclose=True [9s]
[17/25] Saving dolphines: cut=3 keyshots=128 pct=14.35% F_oficial=44.6383 F_regra_original=41.9528 frames_iguais=True features_allclose=True [56s]
[18/25] Scuba: cut=6 keyshots=42 pct=14.18% F_oficial=46.9428 F_regra_original=47.352 frames_iguais=True features_allclose=True [24s]
[19/25] St Maarten Landing: cut=3 keyshots=40 pct=14.28% F_oficial=37.0079 F_regra_original=36.7067 frames_iguais=True features_allclose=True [11s]
[20/25] Statue of Liberty: cut=3 keyshots=88 pct=14.24% F_oficial=30.1762 F_regra_original=30.5499 frames_iguais=True features_allclose=True [18s]
[21/25] Uncut_Evening_Flight: cut=6 keyshots=182 pct=14.1% F_oficial=18.5294 F_regra_original=17.2351 frames_iguais=True features_allclose=True [104s]
[22/25] Valparaiso_Downhill: cut=4 keyshots=100 pct=14.47% F_oficial=19.685 F_regra_original=19.0476 frames_iguais=True features_allclose=True [57s]
[23/25] car_over_camera: cut=3 keyshots=80 pct=13.68% F_oficial=32.0574 F_regra_original=33.1237 frames_iguais=True features_allclose=True [17s]
[24/25] paluma_jump: cut=3 keyshots=48 pct=13.97% F_oficial=19.9719 F_regra_original=17.3103 frames_iguais=True features_allclose=True [11s]
[25/25] playing_ball: cut=3 keyshots=56 pct=13.47% F_oficial=31.9226 F_regra_original=32.6351 frames_iguais=True features_allclose=True [9s]
CSV salvo em D:\Estudos\TCC\tcc-video-summarization-gnn\docs\evidence\sheets\t07-baseline-summe.csv
médias: F_oficial=31.20 F_regra_original=32.59 pct=13.77% | frames idênticos 25/25 | features allclose 25/25 | tempo total 1245s
```

## Not covered

- Rank metrics (Kendall, Spearman): the protocol does not report them for the baseline, which has
  no per-frame score.
- `delta_t = 8`, VSUMM and TVSum.
- The published comparison, pending the reference above.
