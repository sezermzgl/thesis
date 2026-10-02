# Part D run: 2026-10-02 (m = 10, seed 42, Qwen2.5-1.5B QLoRA)

**Code:** `sezermzgl/thesis` @ branch `conformal-triage`, commit `6ff6e85`
**Setup:** Colab T4, `DATA_SOURCE = "repo"` (CACHE_MODE). The model was retrained in this run (no cached adapter).
**Settings:** N_QUERIES = 10, DROP_PROB = 0.5, VOTE_THRESHOLD = 0.5, ALPHA_BENIGN = 0.05, ALPHA_MALIGNANT = 0.20, DELTA = 0.05, CP_ALPHA = 0.10
**Files in this folder:** `perturbed_probs_{validation,calibration,test}_seed42_qwen15b_m10.csv` (one row per nodule × query: perturbation used, prompt, P(malignant)), `triage_test_*.csv`, `cp_sets_test_*.csv`, `ltt_rules_*.json`. The adapter used in this run is committed at `tn3k_gt_radiomics/classification/lora_adapter_seed42_qwen15b/`.

**Perturbation scheme in this run (superseded):** query 0 = training prompt; queries 1–9 each shuffle the feature order, drop one feature with probability 0.5, and pick one of three sentence templates uniformly.

## Data

| split | n | benign | malignant | malignant share |
|---|---|---|---|---|
| train | 2015 | 1333 | 682 | 0.338 |
| validation | 432 | 286 | 146 | 0.338 |
| calibration | 432 | 286 | 146 | 0.338 |
| test | 614 | 378 | 236 | 0.384 |

mRMR selection (train only): MajorAxisLength, glszm HighGrayLevelZoneEmphasis, Elongation, glcm ClusterShade, firstorder Range, glszm LargeAreaHighGrayLevelEmphasis, MaximumDiameter, firstorder Maximum.

## B5: training (630 steps, 29:54; best checkpoint chosen by validation AUC)

| epoch | train loss | val loss | acc | F1 | AUC | sens |
|---|---|---|---|---|---|---|
| 1 | 1.426 | 0.627 | 0.616 | 0.556 | 0.689 | 0.712 |
| 2 | 1.291 | 0.688 | 0.664 | 0.027 | 0.686 | 0.014 |
| 3 | 1.316 | 0.772 | 0.574 | 0.562 | 0.697 | 0.808 |
| 4 | 1.174 | 0.636 | 0.653 | 0.559 | 0.702 | 0.651 |
| 5 | 1.139 | 0.635 | 0.648 | 0.568 | **0.704** | 0.685 |

Epoch 2 collapses to "almost always benign" (sens 0.014), so training is unstable from epoch to epoch.

## B6: test (n = 614)

Acc 0.656 · F1 0.577 · AUC 0.672 · Sens 0.610 (README: 0.656 / 0.574 / 0.675 / 0.602)

## D2–D3: sanity checks

- Query 0 equals the training prompt on 100% of nodules in all three splits.
- max |query-0 prob − B6 prob| on test = 0.0015, so inference matches B6.

## D4: feasibility (validation)

| metric | value |
|---|---|
| auc_base_softmax | 0.704 |
| auc_mean_prob | 0.662 |
| auc_vote_fraction | 0.652 |
| base_accuracy | 0.648 |
| auc_correct_by_vote_agreement | 0.628 |
| auc_correct_by_softmax_conf | 0.602 |
| share_unanimous | 0.326 |
| share_flipped_vs_base | 0.415 |

Vote distribution (validation):

| v | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 115 | 56 | 52 | 34 | 22 | 6 | 1 | 0 | 0 | 0 | 0 |
| malignant | 26 | 29 | 31 | 29 | 18 | 9 | 4 | 0 | 0 | 0 | 0 |

Flip rate when a feature is dropped (validation): 0.394–0.443 for every feature (MaximumDiameter 0.443, ClusterShade 0.443, LAHGLE 0.422, Maximum 0.421, Range 0.420, HGLZE 0.404, MajorAxisLength 0.397, Elongation 0.394).

Dry-run LTT on validation: t = None, u = None.

## D5: LTT (calibration)

Vote distribution (calibration):

| v | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 119 | 62 | 44 | 34 | 15 | 10 | 2 | 0 | 0 | 0 | 0 |
| malignant | 28 | 22 | 42 | 36 | 14 | 1 | 3 | 0 | 0 | 0 | 0 |

Votes, R1 (auto-benign, α = 0.05):

| t | n | malignant | rate | p |
|---|---|---|---|---|
| 0 | 147 | 28 | 0.190 | 1.0 |
| 1 | 231 | 50 | 0.216 | 1.0 |
| 2 | 317 | 92 | 0.290 | 1.0 |
| 3 | 387 | 128 | 0.331 | 1.0 |
| 4 | 416 | 142 | 0.341 | 1.0 |

Votes, R2 (auto-malignant, α = 0.20): u = 10…7 have n = 0; u = 6 has n = 5, 2 benign, p = 0.94.

Softmax (query 0), R1: p ≤ 0.20 → 7 nodules, 0 malignant (p = 0.70); p ≤ 0.25 → 43 nodules, 6 malignant (0.140); p ≤ 0.45 → 195 nodules, 40 malignant (0.205). None certified.
Softmax, R2: p ≥ 0.80 → 9 nodules, 3 benign (0.333); p ≥ 0.75 → 41 nodules, 14 benign (0.341). None certified.

**Locked rules: votes t = u = None; softmax t = u = None, so every nodule is referred.**

## D6: split-CP thresholds (calibration)

| signal | kind | q̂ benign | q̂ malignant |
|---|---|---|---|
| votes | marginal | 9 | 9 |
| votes | Mondrian | 3 | 10 |
| softmax | marginal | 0.6654 | 0.6654 |
| softmax | Mondrian | 0.6706 | 0.6697 |

## D7: test (n = 614)

Triage outcomes:

| method | auto-benign | cancers in auto-benign | rate | auto-malignant | benign in auto-malignant | rate | referred | automation |
|---|---|---|---|---|---|---|---|---|
| LTT votes | 0 | 0 | – | 0 | 0 | – | 614 | 0.000 |
| LTT softmax | 0 | 0 | – | 0 | 0 | – | 614 | 0.000 |
| CP marginal votes | 235 | 58 | 0.247 | 0 | 0 | – | 379 | 0.383 |
| CP Mondrian votes | 0 | 0 | – | 60 | 29 | 0.483 | 554 | 0.098 |
| CP marginal softmax | 213 | 49 | 0.230 | 109 | 41 | 0.376 | 292 | 0.524 |
| CP Mondrian softmax | 209 | 49 | 0.234 | 97 | 37 | 0.381 | 308 | 0.498 |

Prediction sets (target coverage 0.90):

| method | coverage | benign | malignant | singleton | both labels | empty |
|---|---|---|---|---|---|---|
| CP marginal votes | 0.906 | 1.000 | 0.754 | 0.383 | 0.617 | 0 |
| CP Mondrian votes | 0.953 | 0.923 | 1.000 | 0.098 | 0.902 | 0 |
| CP marginal softmax | 0.853 | 0.892 | 0.792 | 0.524 | 0.476 | 0 |
| CP Mondrian softmax | 0.860 | 0.902 | 0.792 | 0.498 | 0.502 | 0 |

## Interpretation (preliminary)

1. **The perturbations push the model out of distribution.** 41.5% of perturbed queries flip the training-prompt answer, no nodule gets more than 6/10 malignant votes, and the flip rate is about the same whichever feature is dropped. The vote count seems to measure distance from the training prompt format more than uncertainty about the nodule. *Needs to be confirmed per perturbation type; this requires the per-query CSVs.*
2. **LTT certifies nothing, for votes or softmax.** Even the most confident auto-benign subgroup is 14–19% malignant against a 5% target, and auto-malignant subgroups are 33–41% benign against a 20% target. The limit is the base model's separation, not the method.
3. **Classic CP behaves as the design note predicted.** Marginal coverage is met (0.906) while the malignant class is covered at only 0.754, and CP singletons called "benign" contain 23–25% cancers.
4. **Possible calibration/test shift.** Softmax Mondrian targets 0.90 malignant coverage on calibration but gets 0.792 on test (n_mal = 236). That is roughly 3 SD below target and should be checked with an exchangeability test.
5. The auc_correct_by_vote_agreement vs softmax-confidence gap (0.628 vs 0.602, n = 432) is small and not yet a finding.

## Follow-up analysis (offline, from the per-query CSVs)

### Perturbations broken down by type (validation; calibration and test look the same)

Every query 1–9 shuffles the order, so shuffling cannot be separated from the other two perturbations.

| query type | n | flip rate vs query 0 | share predicted malignant | mean Δp | AUC |
|---|---|---|---|---|---|
| query 0 (training prompt) | 432 | 0 | 0.477 | 0 | 0.704 |
| shuffle only | 679 | 0.290 | 0.348 | −0.037 | 0.618 |
| shuffle + drop | 673 | 0.291 | 0.391 | −0.019 | 0.594 |
| shuffle + template T1 | 653 | 0.484 | **0.000** | −0.303 | 0.577 |
| shuffle + template T2 | 623 | 0.474 | **0.000** | −0.315 | 0.604 |
| shuffle + drop + T1 | 614 | 0.495 | **0.000** | −0.310 | 0.607 |
| shuffle + drop + T2 | 646 | 0.475 | **0.000** | −0.313 | 0.651 |

- Paraphrased templates (T1, T2) are **never** classified as malignant in any split (0 of 2,536 validation, 2,553 calibration and 3,674 test queries): P(malignant) drops by about 0.30. About two-thirds of the perturbed queries used T1/T2, which is why no nodule exceeded 6/10 malignant votes.
- Shuffling alone flips 29% of answers and lowers the AUC from 0.70 to 0.62. Shuffle-only flip rate is 0.10 when MajorAxisLength stays in first position and 0.24–0.39 otherwise, so the model relies on feature position.
- Dropping a feature adds nothing measurable on top of shuffling (0.291 vs 0.290).
- Conclusion: these perturbations measure distance from the training format, not uncertainty about the nodule. The next scheme keeps the training format (value jitter + clean drop-one).

### Exchangeability: calibration vs test

Classifier two-sample test on the 8 standardized features (5-fold CV, 500 permutations):

| comparison | AUC | permutation p |
|---|---|---|
| validation vs calibration (control) | 0.51 (RF) | – |
| calibration vs test (LR / RF) | 0.623 / 0.592 | < 0.002 |
| benign only | 0.648 / 0.603 | < 0.002 |
| malignant only | 0.619 / 0.588 | < 0.002 |

KS test on query-0 P(malignant), calibration vs test: p = 0.0045 overall, 0.0074 benign, 0.11 malignant. Test nodules are larger: mean standardized MajorAxisLength +0.34 vs −0.04, MaximumDiameter +0.31 vs −0.03, LargeAreaHighGrayLevelEmphasis +0.44 vs −0.01. The shift holds within each class, so it is not just the different malignant share. **Calibration and official test are not exchangeable**, and test-split rates cannot confirm the LTT/CP guarantees.

### Number of features (classical models, train → validation / official test AUC)

| features | LR | RF | GBM |
|---|---|---|---|
| mRMR 8 | 0.689 / 0.654 | 0.691 / 0.673 | 0.681 / 0.674 |
| shape2D only (9) | 0.707 / 0.654 | 0.743 / 0.661 | 0.736 / 0.646 |
| all 67 | 0.740 / 0.699 | 0.755 / 0.663 | 0.786 / 0.654 |

More features help within the trainval distribution (validation +0.05 to +0.10 AUC) but barely on the official test split, consistent with the shift above.
