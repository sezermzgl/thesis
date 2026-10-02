# Conformal Triage for Thyroid Nodules: Full Report

*Status as of 2026-10-02. Run records: [`runs/2026-10-02_part_d_m10`](../runs/2026-10-02_part_d_m10/RESULTS.md) (run 1) and [`runs/2026-10-02_part_d_jitter`](../runs/2026-10-02_part_d_jitter/RESULTS.md) (run 2).*

## 1. Summary

The conformal triage layer from the design note ("Conformal Triage for Thyroid Nodules", 30 Sep 2026) is implemented as notebook Part D plus the `conformal_triage` package, and was run twice on TN3K.

- With the current model (Qwen2.5-1.5B + 4-bit QLoRA, test AUC 0.672), **Learn-then-Test certifies no automatic decision** at α₁ = 5% (cancers among auto-benign) and α₂ = 20% (benign among auto-malignant), δ = 0.05 each.
- The limit is the base model, not the method. The model is **consistently wrong** on about 20% of its most confident benign calls, and no consistency-based signal can flag such errors.
- **Perturbation-based votes do not beat the softmax probability** as an uncertainty signal for this model.
- **Calibration and official test are not exchangeable** (classifier two-sample AUC 0.63, p = 0.002), so test-split rates describe behaviour under shift rather than testing the guarantee.

## 2. Setup

### Base pipeline (unchanged)

TN3K ground-truth masks → PyRadiomics (shape2D, first-order, GLCM, GLSZM; 67 usable features) → train-only median imputation and z-scoring → mRMR selects 8 features → each nodule becomes one text prompt ("The 'Major Axis Length' feature is measured at 0.335. …") → Qwen2.5-1.5B with a linear classification head, LoRA rank 16 on Q/V, 4-bit QLoRA, class-weighted cross-entropy, 5 epochs.

Selected features: MajorAxisLength, glszm HighGrayLevelZoneEmphasis, Elongation, glcm ClusterShade, firstorder Range, glszm LargeAreaHighGrayLevelEmphasis, MaximumDiameter, firstorder Maximum.

| split | n | malignant share | role in Part D |
|---|---|---|---|
| train | 2015 | 0.338 | fine-tuning only |
| validation | 432 | 0.338 | tuning (noise level, feasibility); checkpoint selection in Part B |
| calibration | 432 | 0.338 | LTT thresholds and CP quantiles only |
| official test | 614 | 0.384 | reporting the locked rules only |

Base model: test Acc 0.656, F1 0.577, AUC 0.672, sensitivity 0.610. Validation AUC is 0.704, calibration 0.707.

### Part D method

1. **Queries.** Each nodule is sent to the unchanged model several times. Query 0 (the base query) is always the exact training prompt.
2. **Votes.** v = number of queries with P(malignant) ≥ 0.5.
3. **Three-way rule.** v ≤ t → auto-benign; v ≥ u → auto-malignant; otherwise refer.
4. **LTT calibration.** For each candidate t (safest first), test H₀: "share of cancers among auto-benign > α₁" with the binomial p-value P(Bin(n, α₁) ≤ k). Use fixed-sequence testing, stop at the first failure, and lock the most permissive certified value. Do the same for u with α₂. Guarantee: both risks hold jointly with probability ≥ 1 − 2δ = 0.90 over the calibration draw.
5. **Comparisons on the same data.**
   - Classic split CP: marginal and Mondrian, α = 0.10, with score = votes against the label. A singleton set counts as an automatic decision.
   - A softmax baseline: the same LTT applied to the base-query probability.
6. **Exchangeability check.** A logistic-regression classifier tries to tell calibration from test using the 8 standardized features (5-fold CV AUC, 500 permutations). Validation vs calibration is the control.

The statistical code is covered by 21 unit tests. Among them, the worked example from the design note is reproduced exactly (p-values 0.002 … 0.024, t = 4, u = 6, q̂ = 4).

## 3. Run 1: mixed perturbations broke the model

Queries 1–9 each shuffled the feature order, dropped one feature with probability 0.5, and used one of three sentence templates.

| query type (validation) | flip rate vs base | share predicted malignant | AUC |
|---|---|---|---|
| base | 0 | 0.477 | 0.704 |
| shuffle only | 0.290 | 0.348 | 0.618 |
| shuffle + drop | 0.291 | 0.391 | 0.594 |
| shuffle + paraphrased template (T1 / T2) | 0.484 / 0.474 | **0.000 / 0.000** | 0.577 / 0.604 |

- Paraphrased templates were never classified as malignant in any split. This held for all 2,536 validation, 2,553 calibration and 3,674 test queries.
- No nodule got more than 6/10 malignant votes, so the auto-malignant arm was structurally impossible.
- Shuffle-only flip rate was 0.10 when MajorAxisLength stayed first and 0.24–0.39 otherwise: the model relies on feature position.

Conclusion: these perturbations measured distance from the training format, not uncertainty about the nodule.

## 4. Run 2: format-preserving perturbations

The training template and feature order are kept. Per nodule, 36 queries:

- the base query;
- 9 jitter queries at each noise SD in {0.05, 0.1, 0.2}, with Gaussian noise added to every standardized value;
- 8 drop-one queries, each removing one feature and changing nothing else.

There are two vote signals:

- **jitter votes:** base + 9 jitter queries, m = 10. The noise SD is chosen on validation by a pre-specified rule: the highest AUC of the vote fraction.
- **drop votes:** base + 8 drop-one queries, m = 9.

The cached adapter was reused; base probabilities match run 1 within 0.002.

### 4.1 The model stays in distribution

| kind | noise SD | flip rate vs base (val / cal / test) | AUC (val / cal / test) |
|---|---|---|---|
| base | – | 0 | 0.704 / 0.707 / 0.672 |
| jitter | 0.05 | 0.056 / 0.057 / 0.052 | 0.700 / 0.703 / 0.673 |
| jitter | 0.10 | 0.074 / 0.074 / 0.069 | 0.699 / 0.705 / 0.670 |
| jitter | 0.20 | 0.090 / 0.106 / 0.088 | 0.694 / 0.692 / 0.669 |
| drop | – | 0.105 / 0.117 / 0.097 | 0.686 / 0.696 / 0.658 |

The selected noise SD is 0.2 (validation vote-fraction AUC 0.684, vs 0.678 for 0.05 and 0.669 for 0.1).

### 4.2 Votes do not beat softmax

Difference in AUC for separating correct from incorrect base predictions, vote agreement minus softmax confidence, with a 2000-sample bootstrap 95% CI:

| signal | validation | calibration | test |
|---|---|---|---|
| jitter (SD 0.2) | −0.042 [−0.089, +0.005] | −0.050 [−0.100, −0.006] | −0.044 [−0.087, −0.002] |
| drop | +0.018 [−0.018, +0.052] | +0.022 [−0.017, +0.062] | −0.005 [−0.036, +0.026] |

### 4.3 The model is confidently wrong

Jitter votes, calibration split:

| v | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 133 | 20 | 14 | 7 | 8 | 7 | 6 | 10 | 13 | 15 | 53 |
| malignant | 32 | 5 | 7 | 2 | 3 | 7 | 2 | 3 | 9 | 13 | 63 |

The votes are bimodal. 19.4% of 0/10 nodules are malignant and 45.7% of 10/10 nodules are benign.

### 4.4 LTT: nothing certified

| signal | best auto-benign group | best auto-malignant group |
|---|---|---|
| jitter votes | v = 0: 165 nodules, 19.4% malignant (target ≤ 5%) | v = 10: 116 nodules, 45.7% benign (target ≤ 20%) |
| drop votes | v = 0: 175 nodules, 20.0% malignant | v = 9: 58 nodules, 31.0% benign |
| softmax | not certified at any candidate | not certified at any candidate |

All p-values are ≈ 1. As a check on run 1, LTT on the softmax baseline was also re-run with candidate grids restricted to thresholds with enough validation nodules, and with α₁ relaxed up to 25%. Nothing was certified.

### 4.5 Split CP (test split)

| method | auto-benign (cancer rate) | auto-malignant (benign rate) | referred | coverage (benign / malignant) |
|---|---|---|---|---|
| marginal, jitter votes | 0 | 0 | 614 | 1.000 (1.000 / 1.000) |
| Mondrian, drop votes | 0 | 107 (0.449) | 507 | 0.922 (0.873 / 1.000) |
| marginal, softmax | 213 (0.230) | 109 (0.376) | 292 | 0.853 (0.892 / 0.792) |
| Mondrian, softmax | 209 (0.234) | 98 (0.378) | 307 | 0.860 (0.902 / 0.792) |

- Vote-based CP needs both labels in almost every set to reach 90% coverage, because about 22% of malignant nodules get 0/10 votes.
- Softmax CP meets coverage on calibration (0.903) but not on test (0.853).
- On test, softmax CP's automatic benign calls contain 23% cancers: a coverage guarantee does not bound the clinical error rate.

### 4.6 Feature dependence (clean drop-one, validation)

| dropped feature | flip rate | mean abs. change in P(malignant) |
|---|---|---|
| MajorAxisLength | 0.280 | 0.115 |
| LargeAreaHighGrayLevelEmphasis | 0.137 | 0.047 |
| Elongation | 0.106 | 0.061 |
| firstorder Maximum | 0.100 | 0.049 |
| HighGrayLevelZoneEmphasis | 0.062 | 0.025 |
| MaximumDiameter | 0.062 | 0.039 |
| ClusterShade | 0.056 | 0.039 |
| firstorder Range | 0.039 | 0.017 |

The ranking is the same on test.

## 5. Calibration vs official test: not exchangeable

| comparison (run 2, D2) | two-sample AUC | permutation p |
|---|---|---|
| validation vs calibration (control) | 0.506 | 0.37 |
| calibration vs test | 0.632 | 0.002 |
| benign only | 0.644 | 0.002 |
| malignant only | 0.633 | 0.002 |

Test nodules are larger. Mean standardized values, calibration → test: MajorAxisLength −0.04 → +0.34, MaximumDiameter −0.03 → +0.31, LargeAreaHighGrayLevelEmphasis −0.01 → +0.44. The model's most relied-on features (Section 4.6) are the ones that shift most. Because the shift holds within each class, it is not explained by the different malignant share.

## 6. Number of features (classical models)

AUC with the model trained on train, as validation / official test:

| features | LR | RF | GBM |
|---|---|---|---|
| mRMR 8 | 0.689 / 0.654 | 0.691 / 0.673 | 0.681 / 0.674 |
| shape2D only (9) | 0.707 / 0.654 | 0.743 / 0.661 | 0.736 / 0.646 |
| all 67 | 0.740 / 0.699 | 0.755 / 0.663 | 0.786 / 0.654 |

More features help within the trainval distribution but barely on the official test split, consistent with Section 5. This has not yet been tested with the LLM.

## 7. Limitations

- **Weak base model.** AUC is 0.70 on validation and 0.67 on test, using a 1.5B model with 4-bit QLoRA, 5 epochs and a free T4. Ra et al. used LLaMA2-7B for 500 epochs.
- **Numbers as text.** The model reads standardized values as tokens. Run 1 showed it depends on feature position more than on content.
- **Only 8 features** (Section 6).
- **Noise level not grounded in real measurement variability.** A principled alternative is to perturb the segmentation masks and re-extract the features, which needs the images.
- **Small calibration set.** It has 146 malignant nodules. At α = δ = 0.05, a threshold needs at least 59 nodules on its side even with zero errors (0.95⁵⁹ < 0.05).
- **Exchangeability violated** between calibration and official test (Section 5).
- **Training instability.** Validation sensitivity collapsed to 0.014 in epoch 2 before recovering. The committed adapter was trained before a gradient-accumulation fix, so loss and gradients were scaled by GRAD_ACCUM = 2. The effect should be small with Adam, and retraining with `USE_CACHE = False` uses the fixed loss.
- **Single dataset.** The scoping review recommends at least four; TN5000 is the planned second.

## 8. Open questions

1. **Handling the calibration/test shift.**
   - (a) Split the official test set into calibration and evaluation halves (about 307 each): exchangeable by construction, but fewer calibration nodules.
   - (b) Weighted conformal prediction (Tibshirani et al., 2019) with likelihood-ratio weights from the two-sample classifier: assumes covariate shift only, and is not standard with LTT.
   - (c) Report the official test as an evaluation under shift.

   Proposed: (a) for the guarantee, (c) as an additional analysis.
2. **A stronger model.**
   - More features with the current 1.5B model: runs on a free T4, but needs retraining and a longer `MAX_LENGTH`.
   - A 7B model (Qwen2.5-7B, or LLaMA2-7B as in Ra et al.): fits a T4 with 4-bit QLoRA, but takes several hours per run, so Kaggle, Colab Pro or a university GPU would be needed.

   Section 6 suggests features are the more promising lever.

## 9. Reproducibility

- Code: `conformal_triage/` (tests: `python -m unittest discover tests`), notebook Part D (D1–D9).
- The adapter used in both runs is at `tn3k_gt_radiomics/classification/lora_adapter_seed42_qwen15b/` and is loaded automatically in `repo` mode, so B5 skips training.
- Per-query probabilities for both runs are in `runs/`, so every table above can be recomputed without a GPU.
- Seed 42 throughout. Each nodule's perturbation noise is seeded by its `sample_id`.
