```
Role: Fixes, before any result is produced, how every method is turned into a summary and scored.
Entry: docs/status.md
Related to: [status.md](./status.md), [phases.md](./phases.md), [decisions.md](./decisions.md), [architecture.md](./architecture.md)
```

# Evaluation protocol (T19)

**Status: Approved (2026-10-23).** This is the specification required by [phases.md](./phases.md) T19, approved by the researcher. Points marked **OPEN** below are not decided and must not be resolved by an agent.

## Scope and principle

The protocol is fixed in writing before any method produces a result, and it is the same for every compared method. No metric, threshold or setting in this file may be chosen or adjusted after seeing a result. Source: [decisions.md](./decisions.md) D-13 (Locked), which requires exactly this and names T19 as the task that writes it.

## Input

- **Graph:** the time-aware graph is shared by every compared method; it is built once per video and reused. Source: D-14 (Locked).
- **Features:** node features come from the production feature extractor (torchvision ResNet50, `avgpool`, 2048 dimensions), the same for every method. Source: D-03 (Locked).
- **Datasets:** SumMe in TCC1; VSUMM and TVSum in TCC2. In TCC1, the proposed method is compared only with the HieTaSumm baseline on SumMe. Per-dataset adapters exist from TCC1, even though the VSUMM and TVSum experiments belong to TCC2. Source: D-16 (Decided, revised 2026-10-21), D-15 (Decided).

## Summary size

Each summary has a duration budget of **15% of the original video duration, in seconds**. A summary may be shorter than the budget but never exceeds it. Source: D-21 (Decided, 2026-10-21).

## Output of each method

- **Proposed method:** segments are selected by a 0/1 knapsack under the 15% budget, applied after the learned per-segment scores. This is a standard selection rule, not a combinatorial step inherited from HieTaSumm. Source: D-21 and D-18 (Locked).
- **HieTaSumm baseline:** segments are selected by the original cut rule. `bestCutNumber` is computed per video from the raw frame pixels on disk (`rgbSim`/`spatialSim`), and the final cut is `cut_number = int(bestCutNumber() * (alpha / 100))` with `alpha = 75` from `options.json`. There is no fixed cut value; it varies by video. Source: D-20 (Decided); `references/HieTaSumm-lib/HieTaSumm/Summary.py:57` (cut_number), `Summary.py:104-125` (bestCutNumber), `references/HieTaSumm-lib/HieTaSumm/options.json:5` (alpha = 75).
- **Exception to the shared graph:** the baseline's cut statistic is the one explicit exception to D-14. It works on raw pixels of the frames on disk, not on the shared graph or its features. The baseline's resulting summary is still subject to the same 15% budget as every method. Source: D-14 and D-20.

## Knapsack segment unit for the proposed method — OPEN

Not decided. Two options are on the table, and neither is chosen here:

1. Clusters of one level of the SLC hierarchy.
2. Shots from KTS (kernel temporal segmentation).

Source: D-21, consequences ("the unit of segments used by the knapsack ... is open").

## Baseline summaries and the 15% budget — closed (Q-10, D-23)

Decided (researcher, 2026-11-16): the HieTaSumm baseline summary is the set of native keyshots of `Summary.group_central_frames`. Measured on the 25 SumMe videos with `delta_t = 32` and frames every 0.25 s, none exceeds the budget (10.53% to 14.58%, mean 13.77%), and summaries below the budget are reported without adjustment. Source: D-23, D-24, D-25; [evidence/q10-baseline-budget.md](./evidence/q10-baseline-budget.md).

## From sampled frames to native frames

Every method selects sampled frames, one every 0.25 s (D-25). For evaluation, the sampled frame `k` stands for exactly the native frames `c` with `0.25 * k <= c / fps < 0.25 * (k + 1)`, `fps` being the native frame rate of the video. The evaluated summary is the union of these native frames over the selected sampled frames, and the F-score below is computed on it. This rule is the same for every compared method. Source: D-26 (Decided, 2026-11-16).

The original rule of `Evaluation.find_corresponding_frames` (`Evaluation.py:62-93`) is not the official rule. It may be used only as a supplementary check of the HieTaSumm baseline against published numbers, reported separately and labelled as such. Source: D-26.

## F-score

Definition and aggregation follow `references/HieTaSumm-lib/HieTaSumm/Evaluation.py:132-155` (`f_score` method): for each annotator, `precision = overlap / sum(predicted_summary)`, `recall = overlap / sum(user_summary)`, `f_score = 2 * precision * recall * 100 / (precision + recall)`.

- **SumMe:** maximum F-score over annotators (`Evaluation.py:152-153`, `eval_method='max'`, called at `Evaluation.py:31`).
- **TVSum:** mean F-score over annotators (`Evaluation.py:154-155`).
- **Empty summary:** the original code only returns 0 when `precision + recall == 0` (`Evaluation.py:147-148`), which does not actually trigger when the predicted summary is empty (`sum(predicted_summary) == 0` makes `precision` a division by zero, not a clean zero). This protocol declares, as a protocol-level decision not present in the original code, that an empty predicted summary scores F-score = 0.

Source for aggregation choice: D-21 ("maximum over annotators on SumMe, and mean over annotators on TVSum").

## Rank metrics (Kendall's tau and Spearman's rho)

Computed on the positions of `picks` in SumMe (the SumMe ground-truth field for annotator click positions; D-15), and only for methods that produce a per-frame score. The HieTaSumm baseline does not receive these metrics, because its rule-based cut produces a binary segment/no-segment choice, not a per-frame score (D-18, D-20) — unless a derived score is explicitly defined for it, which is not done here. Source: D-21, consequences ("the report includes ... for the proposed method, the rank metrics").

## Mandatory reporting

Every results table reports the mean summary length as a percentage of the original video duration, for every method in that table. Source: D-21 ("Each table reports the mean summary length as a percentage of the video duration").

## VSUMM

Metric: F-score, computed against the five user-selected keyframe sets, with the mean over annotators, since VSUMM provides no continuous importance score (D-15; common format in `src/tcc_gnn/data/base.py`). Summary length follows the 15% duration budget (D-21).

Rank metrics (Kendall's tau and Spearman's rho) are not reported on VSUMM. The derivation of a per-frame score from the user keyframes is therefore not needed for this dataset. Rank metrics are reported only on SumMe and TVSum, where a per-frame importance score exists.

## Hardware

Decided (researcher, 2026-11-16): every timing and memory measurement runs on the researcher's local desktop, whose GPU is a GeForce GTX 1660 SUPER with 6 GB. This closes Q-09 and settles the local-versus-cloud part of Q-08; only the training details of Q-08 stay open. The remaining fields below are environment facts, to be recorded at the first measured run; none of them is chosen from results, and no pending field is to be filled by an agent.

| Field                 | Value                                                                       |
| --------------------- | --------------------------------------------------------------------------- |
| CPU model (exact)     | Intel Core i5-8400 @ 2.80 GHz (`Win32_Processor`)                           |
| GPU model and variant | NVIDIA GeForce GTX 1660 SUPER                                               |
| GPU memory (VRAM)     | 6 GB (6144 MiB, `nvidia-smi`)                                               |
| Operating system      | Windows 10 Pro 64-bit, version 10.0.19045 (`Win32_OperatingSystem`)         |
| GPU driver version    | 572.42 (`nvidia-smi`)                                                       |
| CUDA version          | 12.1 used by PyTorch (`torch.version.cuda`); the driver supports up to 12.8 |
| PyTorch version       | 2.5.1+cu121                                                                 |

The GPU, driver, CUDA and PyTorch values were read on 2026-11-16 from `nvidia-smi` and from the project environment (`.venv/`); the CPU and operating system values were read the same day from Windows (CIM classes named in the table). The machine has 15.9 GB of RAM.

Source: Q-08 (local desktop GPU decided; training details still open, owner Researcher) and Q-09 (closed: GeForce GTX 1660 SUPER, 6 GB), both in [architecture.md](./architecture.md), questions table; D-13 (the protocol must cover "the hardware used for timing"); `docs/project.md:61` (time and memory are reported "on the same hardware for every compared method").

## Pending measurements

Running time and memory per pipeline stage, per video and in total, for every compared method. These measurements are pending execution on the hardware above and are not performed by this document. Source: `docs/project.md:61`, Q-09.
