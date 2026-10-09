# v2 pipeline runs: 2026-10-08 to 2026-10-09

**Notebook:** `experiments/tn3k_v2_pipeline_colab.ipynb` (Colab L4, bf16). Commits used: `09453c3` (first v2 run), `6f202b8` (all later runs: near-duplicate filter and full-precision prediction).
**Split:** `splits/pooled_seed42.csv` (train 2,095 / validation 436 / calibration 524 / test 438).
**Common settings:** 16 mRMR features, 5 epochs, batch 8 × accumulation 2, LR 2e-4, LoRA rank 16 on Q/V, seed 42, unless the table says otherwise. LTT risks as in the main notebook (R1 ≤ 5%, R2 ≤ 20%, δ = 0.05).

**Files:** one folder per run with `config.json` (incl. the selected features), `metrics.json`, `train_log.csv` and `probs_{validation,calibration,test}.csv` (one row per nodule). `summary.csv` collects the metrics; `xgboost_v1_vs_v2_pooled.csv` is Part B; `notebooks/` holds the executed notebooks. Adapters and checkpoints are not committed (on Drive).

## Feature extraction (Part A)

3,493 images, 0 errors, 102 features. On single-component masks the shared features match v1 exactly. The official-source check (trainval vs official test) is not smaller with relative shape features (AUC 0.709 vs 0.654 for pixel shape features), so the two TN3K sources differ beyond pixel scale. The pooled split mixes both sources in every split, so the conformal guarantees are not affected.

## XGBoost benchmark (Part B)

Test AUC; same split and mRMR order, untuned XGBoost.

| features | 8 | 16 | 32 | all |
|---|---|---|---|---|
| v1 | 0.695 | 0.723 | 0.738 | 0.793 (67) |
| v1, near-duplicates removed (\|r\| > 0.95) | 0.691 | 0.717 | 0.743 | 0.797 (44) |
| v2 | 0.723 | 0.780 | 0.803 | 0.800 (102) |
| v2, near-duplicates removed | 0.723 | **0.805** | 0.796 | 0.800 (54) |

No setting certifies the 5% auto-benign rule; the lowest 15% of calibration holds 2–11 cancers out of 78.

## LLM runs (Part C)

| run | model | features | filter | format | val | cal | test | minutes |
|---|---|---|---|---|---|---|---|---|
| A | Qwen2.5-1.5B | v2 | – | decimal | 0.753 | 0.757 | 0.750 | 26 |
| B | Qwen2.5-1.5B | v1 | 0.95 | decimal | 0.709 | 0.705 | 0.699 | 27 |
| C | Qwen2.5-7B | v2 | 0.95 | decimal | 0.708 | 0.705 | 0.713 | 90 |
| D | Qwen2.5-1.5B | v2 | 0.95 | decimal | 0.750 | 0.752 | 0.754 | 26 |
| E | Qwen2.5-1.5B | v2 | 0.95 | zscore | 0.729 | 0.756 | 0.759 | 37 |
| F | Qwen2.5-1.5B | v2 | 0.95 | percentile | 0.709 | 0.713 | 0.734 | 37 |

Paired bootstrap (2,000 resamples of the same nodules), 95% interval of the AUC difference:

| comparison | what changes | calibration | test |
|---|---|---|---|
| A − B | features (and filter) | [+0.015, +0.090] | [+0.013, +0.089] |
| C − A | model size (and filter) | [−0.083, −0.022] | [−0.069, −0.004] |
| E − D | z-score wording | [−0.016, +0.022] | [−0.017, +0.029] |
| F − D | percentile values | [−0.075, −0.005] | [−0.057, +0.022] |

Findings:

- **v2 features help the LLM**: about +0.05 AUC over v1 on calibration and test (B vs D differ only in the feature set: 0.699 → 0.754 on test).
- **Qwen2.5-7B is worse than 1.5B** with the same 16 features (C vs D). Its training did not settle: validation sensitivity at 0.5 swings 0.00 → 0.01 → 0.93 → 0.04 → 0.62, gradient norms reach 200, and the best epoch was epoch 2. LR 2e-4 is likely too high for 7B.
- **The near-duplicate filter does not change the LLM** (A vs D, within ±0.005), although it lifts XGBoost with 16 v2 features from 0.780 to 0.805.
- **Number format**: stating the reference (z-score) changes nothing; percentiles, which compress the tails, lose 0.04 on validation and calibration. The LLM uses the magnitude of the values.
- **LTT certifies nothing in any run.** Best tail: run E, lowest 15% of calibration with 1 cancer out of 78; at threshold 0.20, 90 nodules with 3 cancers (1 error would need ≥ 93 nodules, 3 errors ≥ 153).
- **Exploratory miss-rate rule** (cancers sent to auto-benign ≤ 5%): the test miss rate ranges from 2.6% to 8.6% across runs (run D: 8.6% at t = 0.16), so a single run can exceed the target.

Notes:

- Run A predicted under the Trainer's bf16 autocast, so its probabilities are rounded (195 distinct values on 524 calibration nodules). Runs B–F predict in full precision (524 of 524).
- Differences below about 0.02 AUC between single runs are within seed-to-seed variation (the two earlier 1.5B runs on the official split gave 0.672 and 0.667 on test).
- XGBoost with the same 16 v2 features (0.805) stays about 0.05 above the best LLM run.
