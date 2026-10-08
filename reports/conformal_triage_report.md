# Conformal Triage for Thyroid Nodules: Full Report

*Status as of 2026-10-02; Section 10 added 2026-10-07, Section 11 added 2026-10-08. Run records: [`runs/2026-10-02_part_d_m10`](../runs/2026-10-02_part_d_m10/RESULTS.md) (run 1) and [`runs/2026-10-02_part_d_jitter`](../runs/2026-10-02_part_d_jitter/RESULTS.md) (run 2).*

## 1. Summary

The conformal triage layer from the design note ("Conformal Triage for Thyroid Nodules", 30 Sep 2026) is implemented as notebook Part D plus the `conformal_triage` package, and was run twice on TN3K.

- With the current model (Qwen2.5-1.5B + 4-bit QLoRA, test AUC 0.672), **Learn-then-Test certifies no automatic decision** at α₁ = 5% (cancers among auto-benign) and α₂ = 20% (benign among auto-malignant), δ = 0.05 each.
- The limit is the base model, not the method. The model is **consistently wrong** on about 20% of its most confident benign calls, and no consistency-based signal can flag such errors.
- **Perturbation-based votes do not beat the softmax probability** as an uncertainty signal for this model.
- **Calibration and official test are not exchangeable** (classifier two-sample AUC 0.63, p = 0.002), so test-split rates describe behaviour under shift rather than testing the guarantee.
- **More features did not improve performance on the official test set** (classical models). Going from 8 to all 67 radiomics features raised validation AUC from about 0.69 to 0.74–0.79, but test AUC stayed at 0.65–0.70.

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

Test nodules are larger **in pixels**. Mean standardized values, calibration → test: MajorAxisLength −0.04 → +0.34, MaximumDiameter −0.03 → +0.31, LargeAreaHighGrayLevelEmphasis −0.01 → +0.44. The model's most relied-on features (Section 4.6) are the ones that shift most. Because the shift holds within each class, it is not explained by the different malignant share.

**Correction (2026-10-08, from the image QC table; see Section 11.4).** The official-test *images* are larger (median 436×375 vs 390×336 pixels), and image width alone separates official test from trainval with AUC 0.77. Relative to the image, test nodules are slightly *smaller* (median nodule area 7.0% vs 8.6% of the image). Shape features are in pixels because no physical pixel spacing is available, so the size shift is largely an image-size artefact rather than a difference between nodules.

## 6. Number of features (classical models)

AUC with the model trained on train, as validation / official test:

| features | LR | RF | GBM |
|---|---|---|---|
| mRMR 8 | 0.689 / 0.654 | 0.691 / 0.673 | 0.681 / 0.674 |
| shape2D only (9) | 0.707 / 0.654 | 0.743 / 0.661 | 0.736 / 0.646 |
| all 67 | 0.740 / 0.699 | 0.755 / 0.663 | 0.786 / 0.654 |

More features did not improve performance on the official test set: validation AUC rises from about 0.69 to 0.74–0.79, but test AUC stays at 0.65–0.70 (0.65–0.67 with 8 features). The gain appears only within the trainval distribution, consistent with Section 5. This has not yet been tested with the LLM. On an exchangeable split the gain does carry over to the test set (Section 11.3).

## 7. Limitations

- **Weak base model, which drives the main negative results.** AUC is 0.70 on validation and 0.67 on test, using a 1.5B model with 4-bit QLoRA, 5 epochs and a free T4. Ra et al. used LLaMA2-7B for 500 epochs. The errors that block LTT come from the base model itself, not from the perturbations. The base prompt alone is wrong on 35% of calibration nodules, and 56% of those errors are repeated in all 10 queries, so no vote threshold can flag them (Section 10.1). The current limitations of the triage layer are therefore limitations of the model it sits on. A 7B backbone gives the same AUC as the 1.5B model (Section 11.1), so the limit appears to be the input features rather than model size.
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

   On validation, features look like the more promising lever (Section 6), but that gain has not yet carried over to the official test set.

## 9. Reproducibility

- Code: `conformal_triage/` (tests: `python -m unittest discover tests`), notebook Part D (D1–D9).
- The adapter used in both runs is at `tn3k_gt_radiomics/classification/lora_adapter_seed42_qwen15b/` and is loaded automatically in `repo` mode, so B5 skips training.
- Per-query probabilities for both runs are in `runs/`, so every table above can be recomputed without a GPU.
- Seed 42 throughout. Each nodule's perturbation noise is seeded by its `sample_id`.

## 10. Addendum (2026-10-07): clarifications and an exploratory analysis

This section was added after the report was first shared. Sections 10.1–10.2 restate existing results more explicitly. Section 10.3 is an **exploratory analysis that was not planned in advance**. All numbers are computed from the run 2 per-query probabilities in `runs/2026-10-02_part_d_jitter/`; no new model queries were made.

### 10.1 The errors come from the base model; perturbations neither create nor reveal them

v = 0 means that all 10 queries, including the unperturbed base prompt, answered benign. So the 32 cancers in the calibration v = 0 group were already misclassified by the base prompt alone. The perturbations did not cause these errors. They also did not expose them: the model gives the same wrong answer under noise and under feature removal.

| split | base accuracy | base errors | unanimous nodules (v = 0 or 10) | wrong among unanimous | share of base errors that are unanimous |
|---|---|---|---|---|---|
| validation | 0.648 | 152 | 301 (69.7%) | 95 (31.6%) | 62.5% |
| calibration | 0.648 | 152 | 281 (65.0%) | 85 (30.2%) | 55.9% |
| test | 0.658 | 210 | 450 (73.3%) | 145 (32.2%) | 69.0% |

The design assumed the model would be unstable on the nodules it gets wrong, so that their votes would split and they would be referred. Instead, 56–69% of the base model's errors are unanimous. Consistency here measures how stable the model's answer is, not whether it is correct.

### 10.2 Separating correct from incorrect predictions: absolute values

Section 4.2 reports differences. The underlying AUCs for separating correct from incorrect base predictions are:

| confidence measure | validation | calibration | test |
|---|---|---|---|
| softmax confidence (base prompt) | 0.602 | 0.629 | 0.577 |
| jitter vote agreement (SD 0.2) | 0.560 | 0.579 | 0.533 |
| drop vote agreement | 0.620 | 0.652 | 0.571 |

All three are close to 0.5–0.65, so none separates the model's errors well.

### 10.3 Exploratory: a different guarantee (miss rate) with the same model

R1 bounds the share of cancers among auto-benign nodules. That is a demanding target with a 34% base rate: the auto-benign group must be at least 95% clean. A standard alternative is to bound the **miss rate**: the share of all cancers that are sent to auto-benign (false-negative-rate control, as in conformal risk control). This was tested with the same binomial LTT test, now with n = 146 calibration cancers. Thresholds were tested in increasing order, stopping at the first failure, with δ = 0.05.

| score | target miss rate | threshold | calibration: auto-benign | test: auto-benign | test: cancers missed |
|---|---|---|---|---|---|
| softmax (base prompt) | ≤ 5% | P(malignant) ≤ 0.215 | 19 (4.4%) | 36 (5.9%) | 10 / 236 = 4.2% |
| softmax (base prompt) | ≤ 10% | P(malignant) ≤ 0.283 | 65 (15.0%) | 156 (25.4%) | 36 / 236 = 15.3% |
| jitter votes | ≤ 5% or ≤ 10% | none certified | – | – | – |

- With the same small model, a miss-rate guarantee can be certified, but only for a small share of nodules (about 5% at the 5% level).
- At the 10% level the guarantee is violated on the official test set (15.3% missed). Under the calibration → test shift (Section 5), many more test nodules fall below the threshold. This is a concrete example of why exchangeability matters.
- Vote counts certify nothing even under this guarantee. They take only 11 values, and the cleanest group (v = 0) already holds 32 of the 146 calibration cancers (22%). Thresholds cannot be set more finely than that.
- Caveats: the analysis is exploratory and post hoc. The thresholds in the table were searched over the calibration scores themselves; a valid version fixes the candidates in advance. With a grid fixed in advance (0.01 to 0.50 in steps of 0.01, as in `experiments/model_sweep.py`), the 5% level certifies P(malignant) ≤ 0.22: 4.4% automation on calibration, 7.2% on test, and a **test miss rate of 5.1%**, at the edge of the target under the shift. The 10% level gives t = 0.28, with 14.1% automation on calibration and 24.4% on test, and a 15.3% test miss rate. It uses a single run and a single dataset.

The two guarantees answer different clinical questions. R1 is a per-patient statement ("if the system says benign, it is right with high probability"). The miss rate is a programme-level statement ("the system refers at least 95% of all cancers").

### 10.4 Relation to the literature

The method combines two established ideas, which places the results in context:

- **Abstention with a guaranteed error rate.** This is selective classification with guaranteed risk (Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks", 2017), generalised to several risks by Learn then Test (Angelopoulos et al., "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control", 2021). The miss-rate variant corresponds to conformal risk control (Angelopoulos et al., "Conformal Risk Control", 2022).
- **Uncertainty from repeated or perturbed queries.** This follows the consistency- and sampling-based uncertainty literature for LLMs: self-consistency (Wang et al.), semantic entropy (Kuhn et al., 2023), and conformal prediction from sampling frequencies (Su et al., "API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access", 2024). The scoping review (Ashby et al., 2026, Finding 3) and Noorani et al. (2025) belong to the same line.

A recurring point in this literature is that the guarantee is always valid, but its usefulness depends on how well the score ranks errors. This matches the results here. The contribution of this work is to apply the approach to a radiomics → LLM pipeline and to show that consistency signals that help in general NLP do not beat the softmax probability here, because the model's errors are stable.

## 11. Addendum (2026-10-08): backbone sweep, exchangeable split, image QC

This section was added after the report was first shared. 11.1–11.2 use new fine-tuning runs (`experiments/model_sweep_colab.ipynb`, Colab L4, bf16, the gradient-accumulation fix). 11.3–11.5 are offline analyses of the existing radiomics features and the image QC table. XGBoost uses fixed hyperparameters (300 trees, depth 3, learning rate 0.05, class weighting); they were not tuned.

### 11.1 Larger backbone (official splits, 8 features)

| model | validation AUC | calibration AUC | test AUC | LTT | training time (L4) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B, earlier adapter (T4, fp16) | 0.704 | 0.707 | 0.672 | nothing certified | ~30 min (T4) |
| Qwen2.5-1.5B, retrained (control) | 0.697 | 0.714 | 0.667 | nothing certified | 16 min |
| Qwen2.5-7B | 0.697 | 0.686 | 0.669 | nothing certified | 57 min |

- The retrained 1.5B control agrees with the earlier adapter (Spearman 0.97 between their test probabilities), so the earlier results are robust to the precision change and the loss fix.
- The 7B model gives the same AUC. Its best checkpoint (by validation AUC) came from epoch 3, where validation loss was highest. That checkpoint over-predicts malignancy (test sensitivity 0.77, accuracy 0.58), and its softmax confidence no longer separates correct from incorrect predictions (AUC 0.45–0.50).
- Early-epoch instability recurs: in the control run, validation sensitivity was 0 after epoch 1.
- A 7B run with 16 features is in progress.

### 11.2 Classical benchmark (official splits)

| model | features | AUC (val / cal / test) | test sensitivity |
|---|---|---|---|
| LLM, Qwen2.5-1.5B | 8 | 0.697 / 0.714 / 0.667 | 0.62 |
| XGBoost | 8 | 0.702 / 0.731 / 0.696 | 0.56 |
| XGBoost | 16 | 0.729 / 0.764 / 0.709 | 0.57 |
| logistic regression (balanced) | 8 / 16 | test 0.652 / 0.671 | 0.58 / 0.54 |
| RF, SVM (as in Part C, no class weights) | 8 | test 0.663 / 0.648 | 0.33 / 0.31 |

XGBoost matches or exceeds the LLM. The LLM stays the main model of this work and XGBoost is reported as a benchmark. No classical model gets an LTT certificate either. The low RF/SVM sensitivity comes from training without class weights.

### 11.3 Exchangeable split (pooled and re-split)

All 3,493 images (official trainval and test) were pooled and re-split at random, stratified by label and official source: train 2,095, validation 436, calibration 524, test 438. Each split holds about 17.5% official-test images and 34.7% malignant nodules. The split is in `splits/pooled_seed42.csv` (made by `experiments/make_pooled_splits.py`) and is the notebook default. Standardization and mRMR are refitted on the new train split, and two of the eight selected features change.

- Exchangeability is restored. Two-sample AUC is 0.476 for validation vs calibration and 0.434 for calibration vs test (p = 1.0); with the official splits it was 0.63.
- More features now help on the test set too. XGBoost AUC (val / cal / test): 8 features 0.733 / 0.756 / 0.694; 16 features 0.756 / 0.775 / 0.715; all 67 features 0.788 / 0.822 / **0.792**. The lack of a test gain in Section 6 was due to the shift.
- The LLM still has to be retrained on this split. The existing adapters belong to the official splits.

### 11.4 Image QC findings

- Image sizes vary, and official-test images are larger (Section 5 correction). Shape features in pixels partly measure image size: MajorAxisLength correlates with image width at r = 0.42. Size still carries real signal (smaller nodules are more often malignant), and relative size carries about the same signal (AUC 0.67 vs 0.66 for pixel size), so size should be normalised rather than dropped.
- Image width alone predicts malignancy weakly (AUC 0.57), which suggests an acquisition-related confounder.
- Multi-component masks make up 7.8% of trainval and 11.7% of test; masks touching the image border make up 11.2% and 5.4%. Both groups are less often malignant (22% vs 36%).
- Texture features use `binWidth = 25`, which leaves a median of about 9 grey levels inside a nodule (IQR 7–10).

### 11.5 Where LTT stands

On the exchangeable split, the best classical model (XGBoost, 67 features) ranks calibration nodules by malignancy score. Its 78 lowest-scored nodules contain 3 cancers (3.8%), already below the 5% target. LTT still cannot certify this group, because with 78 nodules the evidence is too weak (zero cancers would be needed). To certify the same 3.8% rate, the group would need roughly 1,000 nodules (a rough estimate). The two levers are therefore:

- cleaner features, which make this group purer;
- a larger calibration set, which strengthens the evidence. ThyroidXL can help here, used as a separate dataset.

A binormal simulation with the current calibration size points the same way: a 5% certificate becomes likely only for a much stronger model. Its exact threshold depends on the shape of the score distribution, so it is not a fixed requirement.

### 11.6 Planned re-extraction (PyRadiomics)

| setting | original extraction | planned |
|---|---|---|
| grey levels for texture | `binWidth = 25` (about 9 levels in a nodule) | fixed bin count, 32–64 |
| shape features | pixels, sizes vary across images | normalised to image size (or image size recorded as a covariate) |
| feature families | shape2D, first-order, GLCM, GLSZM (67 features) | add GLRLM, GLDM, NGTDM; consider wavelet/LoG-filtered images |
| multi-component masks | used as they are | rule to be fixed in advance (e.g. keep the largest component) |
| border-touching masks | used as they are | flag kept, effect checked |

The new features will first be checked quickly with XGBoost on the exchangeable split, then used for the LLM.
