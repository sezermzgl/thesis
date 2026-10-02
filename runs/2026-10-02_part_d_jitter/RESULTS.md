# Part D run: 2026-10-02, format-preserving perturbations (jitter + drop)

**Code:** `sezermzgl/thesis` @ branch `conformal-triage`, commit `6a27d01`
**Setup:** Colab T4, `DATA_SOURCE = "repo"`. Committed adapter loaded (no training).
**Settings:** N_JITTER = 9 at noise SD ∈ {0.05, 0.1, 0.2}, one drop-one query per feature, VOTE_THRESHOLD = 0.5, ALPHA_BENIGN = 0.05, ALPHA_MALIGNANT = 0.20, DELTA = 0.05, CP_ALPHA = 0.10
**Queries:** 36 per nodule (1 base + 27 jitter + 8 drop), 53,208 in total
**Files:** `perturbed_probs_{validation,calibration,test}_*.csv` (one row per nodule × query), `feasibility_validation_*.csv`, `ltt_rules_*.json`, `triage_test_*.csv`, `cp_sets_{test,calibration}_*.csv`, `exchangeability.csv`

## Sanity checks

- Base-prompt probabilities match the first run's query 0 within 0.002 on every split, so the cached adapter loaded correctly.
- Base AUC: validation 0.704, calibration 0.707, test 0.672 (same as the first run).

## Exchangeability (D2): rejected again

| comparison | AUC | permutation p |
|---|---|---|
| validation vs calibration (control) | 0.506 | 0.37 |
| calibration vs test | 0.632 | 0.002 |
| benign only | 0.644 | 0.002 |
| malignant only | 0.633 | 0.002 |

Symptom in split CP with softmax scores: coverage 0.903 on calibration vs 0.853 on test; malignant coverage 0.904 → 0.792 (Mondrian 0.911 → 0.792).

## Perturbations no longer break the model

| kind | noise SD | flip rate vs base (val / cal / test) | AUC (val / cal / test) |
|---|---|---|---|
| base | – | 0 | 0.704 / 0.707 / 0.672 |
| jitter | 0.05 | 0.056 / 0.057 / 0.052 | 0.700 / 0.703 / 0.673 |
| jitter | 0.10 | 0.074 / 0.074 / 0.069 | 0.699 / 0.705 / 0.670 |
| jitter | 0.20 | 0.090 / 0.106 / 0.088 | 0.694 / 0.692 / 0.669 |
| drop | – | 0.105 / 0.117 / 0.097 | 0.686 / 0.696 / 0.658 |

Discrimination is preserved, there is no class collapse (share predicted malignant stays at 0.39–0.48), and votes now cover the whole range 0–10. First run for comparison: paraphrased templates gave 0% malignant and shuffling alone flipped 29%.

Noise level by the pre-specified rule (highest validation `auc_vote_fraction`): **SD = 0.2** (0.684, vs 0.678 for 0.05 and 0.669 for 0.1).

## Main result: the vote signal does not beat softmax, and the model is confidently wrong

Does vote agreement separate correct from incorrect base predictions better than softmax confidence? Difference in AUC(correct ~ confidence), with a 2000-sample bootstrap 95% CI:

| signal | validation | calibration | test |
|---|---|---|---|
| jitter (SD 0.2) − softmax | −0.042 [−0.089, +0.005] | −0.050 [−0.100, −0.006] | −0.044 [−0.087, −0.002] |
| drop − softmax | +0.018 [−0.018, +0.052] | +0.022 [−0.017, +0.062] | −0.005 [−0.036, +0.026] |

Jitter votes are slightly **worse** than softmax confidence; drop votes are indistinguishable from it. For this model, sampling-based signals bring no gain over the logit (the design note's research question, review Finding 3).

Jitter vote distribution (calibration, m = 10):

| v | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 133 | 20 | 14 | 7 | 8 | 7 | 6 | 10 | 13 | 15 | 53 |
| malignant | 32 | 5 | 7 | 2 | 3 | 7 | 2 | 3 | 9 | 13 | 63 |

Votes are bimodal: most nodules get unanimous answers. But **19% of nodules with 0/10 malignant votes are malignant (32/165), and 46% of nodules with 10/10 are benign (53/116)**. The model's errors are stable under perturbation, so no consistency-based signal can flag them. The design note warned about this ("consistency is not correctness").

## LTT (D6): nothing certified

| signal | best auto-benign group (calibration) | best auto-malignant group (calibration) |
|---|---|---|
| jitter votes | v = 0: 165 nodules, 19.4% malignant (target ≤ 5%) | v = 10: 116 nodules, 45.7% benign (target ≤ 20%) |
| drop votes | v = 0: 175 nodules, 20.0% malignant | v = 9: 58 nodules, 31.0% benign |
| softmax | (no certification, as in the first run) | |

All p-values ≈ 1; no candidate is even close. Unlike the first run this is not a structural artefact, since the candidates are populated. The limit is the model's separation: the cleanest subgroup it can pick out is still about 20% malignant (base rate 34%).

## Split CP (D7–D8), test

| method | auto-benign (cancer rate) | auto-malignant (benign rate) | referred | coverage (benign / malignant) |
|---|---|---|---|---|
| CP marginal, jitter votes | 0 | 0 | 614 | 1.000 (1.000 / 1.000) |
| CP Mondrian, drop votes | 0 | 107 (0.449) | 507 | 0.922 (0.873 / 1.000) |
| CP marginal, softmax | 213 (0.230) | 109 (0.376) | 292 | 0.853 (0.892 / 0.792) |
| CP Mondrian, softmax | 209 (0.234) | 98 (0.378) | 307 | 0.860 (0.902 / 0.792) |

With vote scores, about 20% of malignant nodules get 0/10 votes. Reaching 90% coverage of the true label therefore forces both labels into almost every set, and almost nothing is decided automatically.

## Feature dependence (clean drop-one, validation)

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

Same ranking on test. The model relies most on nodule size (MajorAxisLength) and LargeAreaHighGrayLevelEmphasis, which are also the features with the largest calibration → test shift (+0.37 and +0.45 SD in the first run's analysis).

## Takeaways

1. The technical problem from the first run is solved: format-preserving perturbations keep the model in distribution.
2. With this model (AUC ≈ 0.70), LTT cannot certify any automatic decision at α₁ = 5% / α₂ = 20%, for votes or softmax. The model is confidently wrong on about 20% of its most consistent "benign" calls.
3. Perturbation-consistency signals do not beat softmax here; jitter is slightly worse.
4. Calibration and official test are not exchangeable, so test-split rates describe behaviour under shift.
5. The model's most relied-on feature (size) is the one that shifts most between splits.
