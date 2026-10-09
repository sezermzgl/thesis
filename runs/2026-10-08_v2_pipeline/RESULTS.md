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
| H | Qwen2.5-1.5B | v2 | 0.95 | rich | 0.732 | 0.711 | 0.744 | 83 |
| I | Qwen2.5-1.5B | v2 + 6 TI-RADS features | 0.95 | decimal | 0.795 | 0.812 | **0.804** | 36 |
| G | Llama-2-7B, LR 1e-4, LoRA all layers (Ra et al. settings), batch 4 × 4, MAX_LENGTH 512 | v2 | 0.95 | decimal | 0.705 | 0.709 | 0.731 | 182 |

Paired bootstrap (2,000 resamples of the same nodules), 95% interval of the AUC difference:

| comparison | what changes | calibration | test |
|---|---|---|---|
| A − B | features (and filter) | [+0.015, +0.090] | [+0.013, +0.089] |
| C − A | model size (and filter) | [−0.083, −0.022] | [−0.069, −0.004] |
| E − D | z-score wording | [−0.016, +0.022] | [−0.017, +0.029] |
| F − D | percentile values | [−0.075, −0.005] | [−0.057, +0.022] |
| I − D | + TI-RADS-motivated features | [+0.028, +0.093] | [+0.015, +0.084] |
| H − D | rich (plain-language meaning added) | [−0.074, −0.011] | [−0.041, +0.023] |
| G − D | Llama-2-7B with Ra et al. settings vs Qwen2.5-1.5B | [−0.070, −0.016] | [−0.052, +0.007] |
| G − C | Llama-2-7B vs Qwen2.5-7B | [−0.014, +0.022] | [−0.002, +0.036] |

Findings:

- **v2 features help the LLM**: about +0.05 AUC over v1 on calibration and test (B vs D differ only in the feature set: 0.699 → 0.754 on test).
- **Qwen2.5-7B is worse than 1.5B** with the same 16 features (C vs D). Its training did not settle: validation sensitivity at 0.5 swings 0.00 → 0.01 → 0.93 → 0.04 → 0.62, gradient norms reach 200, and the best epoch was epoch 2. LR 2e-4 is likely too high for 7B.
- **Llama-2-7B with the settings of Ra et al. (G) is not better than Qwen2.5-1.5B** (test 0.731 vs 0.754; worse on validation and calibration) and about equal to Qwen2.5-7B. Its training also started badly (epoch 2: all validation nodules called benign, validation loss 1.21) and recovered to validation AUC 0.70–0.71; 182 minutes on one L4.
- **The near-duplicate filter does not change the LLM** (A vs D, within ±0.005), although it lifts XGBoost with 16 v2 features from 0.780 to 0.805.
- **Number format**: stating the reference (z-score) changes nothing; percentiles, which compress the tails, lose 0.04 on validation and calibration. The LLM uses the magnitude of the values.
- **The rich format (H) does not help**: plain-language bands and meanings on top of the z-score give test 0.744 vs 0.754 and are worse on calibration (interval excludes zero). Validation AUC rose faster (0.719 after epoch 2) and then plateaued at 0.73. Prompts up to 780 tokens; 83 minutes.
- **TI-RADS-motivated features (run I, `conformal_triage/clinical.py`) help both models most.** XGBoost with mRMR 16 + the six features: 0.828/0.848/0.831 (vs 0.799/0.823/0.805; all 54 radiomics features give 0.796, the six alone 0.775). The LLM rises to 0.795/0.812/0.804 (+0.05 vs D, intervals exclude zero on all splits); the gap to XGBoost shrinks from ~0.05 to 0.027 and training is stable. Train AUCs of the features: taller-than-wide 0.658, solidity 0.367, anechoic fraction 0.409 (as TI-RADS expects); echogenicity ratio 0.552 and margin sharpness 0.563 (opposite to expectation), punctate foci 0.487 (no signal). Per-split AUCs are consistent. The echogenicity ratio correlates −0.78 with the anechoic fraction: it measures fluid, not tissue echogenicity.
- **LTT certifies nothing in any run under the planned procedure.** Best tail: run E, lowest 15% of calibration with 1 cancer out of 78; at threshold 0.20, 90 nodules with 3 cancers (1 error would need ≥ 93 nodules, 3 errors ≥ 153).
- **Exploratory miss-rate rule** (cancers sent to auto-benign ≤ 5%): the test miss rate ranges from 2.6% to 8.6% across runs (run D: 8.6% at t = 0.16), so a single run can exceed the target (run G: t = 0.17, test miss rate 0.7% with 7.8% automated).

## LTT with a validation-ordered sequence (exploratory)

The planned fixed sequence starts at the strictest threshold (u = 0.95 / t = 0.05), where few nodules fall, and stops at the first failure. Ordering the candidates by their validation p-values instead (validation is independent of calibration, so the guarantee holds) certifies, for run I, auto-malignant at u = 0.85: calibration 50 nodules, 4 benign, p = 0.018; test 43 nodules (9.8% of test), 5 benign (11.6% ≤ 20%). Nothing for runs D and E; nothing on the auto-benign side. The rule was chosen after seeing run I, so it must be fixed in advance and confirmed on another seed or split.

## Split conformal prediction (computed afterwards from the saved probabilities)

Score 1 − P(label); thresholds from calibration, applied to test. Single-label sets are automatic decisions, two-label sets are referred. Full table (marginal α 0.10 and 0.05, Mondrian): `split_cp_test.csv`. The notebook computes the same table in cell C12 for new runs.

Mondrian, α = 0.10 for benign and 0.05 for malignant nodules (at most ~5% of cancers get the single label "benign"):

| run | cancers covered | auto-benign | cancers in auto-benign | auto-malignant | benign in auto-malignant | referred |
|---|---|---|---|---|---|---|
| A | 0.901 | 22.8% | 15.0% | 19.6% | 30.2% | 57.5% |
| B | 0.914 | 17.1% | 17.3% | 17.8% | 34.6% | 65.1% |
| C | 0.941 | 18.7% | 11.0% | 16.4% | 41.7% | 64.8% |
| D | 0.908 | 21.7% | 14.7% | 17.4% | 30.3% | 61.0% |
| E | 0.934 | 22.8% | 10.0% | 16.7% | 27.4% | 60.5% |
| F | 0.928 | 18.5% | 13.6% | 15.8% | 39.1% | 65.8% |
| G | 0.954 | 18.0% | 8.9% | 14.6% | 35.9% | 67.4% |
| H | 0.987 | 8.7% | 5.3% | 12.3% | 27.8% | 79.0% |
| I | 0.941 | 18.3% | 11.3% | 22.8% | 22.0% | 58.9% |

- Marginal coverage holds (α = 0.10: 0.90–0.92; α = 0.05: 0.94–0.95), but it is carried by the benign majority: at α = 0.10 only 70–86% of cancers are covered.
- Mondrian gives a guarantee on missed cancers (like the miss-rate rule) for about a fifth of the nodules, but the auto-benign sets still hold 10–17% cancers, so it does not meet R1 (≤ 5% cancers among auto-benign calls). This is why LTT certifies nothing while split CP "works".
- Cancer coverage is 0.90–0.94 in runs A–F, below the 0.95 target (G: 0.954, H: 0.987). All runs share one calibration and one test draw, so this is a single realization, not six independent failures; other random splits are needed to separate chance from a systematic gap.

Notes:

- The executed Llama notebook had a cell that printed the Hugging Face token; that output was removed before committing.
- Run A predicted under the Trainer's bf16 autocast, so its probabilities are rounded (195 distinct values on 524 calibration nodules). Runs B–F predict in full precision (524 of 524).
- Differences below about 0.02 AUC between single runs are within seed-to-seed variation (the two earlier 1.5B runs on the official split gave 0.672 and 0.667 on test).
- XGBoost with the same 16 v2 features (0.805) stays about 0.05 above the best LLM run.
