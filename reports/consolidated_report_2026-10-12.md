# Radiomics-as-text LLM classification and conformal triage on TN3K: consolidated report

*Status 2026-10-12. Supersedes the summaries in `reports/conformal_triage_report.md` (which stays as the record of the October 2–8 work). Run records: [`runs/2026-10-08_v2_pipeline/RESULTS.md`](../runs/2026-10-08_v2_pipeline/RESULTS.md); conformal checks: [`runs/2026-10-08_v2_pipeline/conformal_checks.txt`](../runs/2026-10-08_v2_pipeline/conformal_checks.txt) (`experiments/conformal_checks.py`); LLM + XGBoost fusion: [`hybrid_xgb_llm/`](../runs/2026-10-08_v2_pipeline/hybrid_xgb_llm) (`experiments/hybrid_xgb_llm.py`).*

## 1. Summary

The thesis asks two questions on TN3K thyroid ultrasound: (1) can a fine-tuned LLM classify nodules from radiomics features written as text (the approach of Ra et al., 2025), and (2) which statistical guarantees can a conformal layer on top of it give for automatic triage decisions.

1. **Best LLM setting**: Qwen2.5-1.5B, 16 mRMR radiomics features plus 6 TI-RADS-motivated features, plain decimal values. Test AUC **0.785 ± 0.018** (mean ± SD over three training seeds). A single run (seed 42) gave 0.804, which was the upper end.
2. **The TI-RADS-motivated features are the largest gain**: +0.059 test AUC on average over three matched seeds (smallest +0.050; every matched pair's 95% interval excludes zero). They also help XGBoost (+0.026).
3. **The LLM stays below a tabular model with the same information**: XGBoost on the same 22 features reaches test AUC 0.835. Fusing the two does not help: a stacker fitted on validation gives the LLM a weight of −0.04, and the probabilities correlate 0.87. The LLM adds no information beyond XGBoost.
4. **Larger backbones, other prompt formats, the settings of Ra et al. and peritumoral radiomics do not help**: Qwen2.5-7B (two learning rates), Llama-2-7B, z-score/percentile/plain-language ("rich") formats are equal or worse (larger backbones: single seed and no learning-rate search for 7B, so "did not help in the settings tried"), and PyRadiomics features of a ring around the nodule add nothing to XGBoost (calibration +0.002, interval includes zero).
5. **The planned LTT triage certifies nothing**, for the LLM *and* for XGBoost. Two reasons: the 5% auto-benign target needs a large, very clean tail (e.g. at most 1 cancer in 93 nodules), which neither model has; and the planned fixed sequence starts at the strictest threshold, where the groups are too small, and stops there.
6. **Guarantees that are attainable**: Mondrian split CP and an LTT rule on the *miss rate* (share of all cancers sent to auto-benign ≤ 5%) both hold. Re-splitting calibration and test 4,000 times confirms the Mondrian coverage of cancers averages 0.950–0.951, as the theory says. But the auto-benign group they produce still contains about 10% cancers for the LLM (6% for XGBoost), against a 35% base rate. **A valid guarantee is not the same as a safe decision.**

## 2. Pipeline

```
TN3K image + ground-truth mask
  → PyRadiomics v2 (102 features) + 6 TI-RADS-motivated features (conformal_triage/clinical.py)
  → train-only median imputation and z-scoring
  → near-duplicate filter |r| > 0.95 (train only), mRMR selects 16 radiomics features; TI-RADS features always kept
  → one prompt per nodule: "The 'Elongation' feature is measured at -1.140." (value = train z-score)
  → Qwen2.5-1.5B with a linear classification head on the last token, 4-bit QLoRA (LoRA rank 16 on Q/V),
    class-weighted cross-entropy, 5 epochs, LR 2e-4, effective batch 16, best epoch by validation AUC
  → P(malignant) per nodule
  → conformal layer, calibrated on the calibration split only, reported on test
```

The model is not generative: it does not answer in words; a new linear layer maps the last hidden state to two logits. Ra et al. use the same construction ("LLaMA2-based classifier" with binary cross-entropy).

## 3. Data and splits

- **TN3K**: 3,493 ultrasound images (from 2,421 patients), each with a nodule mask and a benign/malignant label from cytology/pathology (FNA). Malignant share 34.6%, far above a screening population.
- **Pooled split** (`splits/pooled_seed42.csv`): all images pooled and re-split, stratified by label and by official source: train 2,095 / validation 436 / calibration 524 / test 438, 34.6–34.7% malignant in each.
  - Why: on the official split, calibration (from official trainval) and the official test are not exchangeable (two-sample classifier AUC 0.63, p = 0.002; test nodules larger). Conformal guarantees need exchangeable calibration and test data. After pooling, the same check gives AUC 0.43.
  - Caveat: the split is at image level and patient IDs are not available, so images of the same patient can fall into different splits. This inflates absolute AUCs for all models alike. It does not break exchangeability between calibration and test (a random split of a fixed pool is exchangeable), but the guarantees then refer to new images from this pool, not to new patients.
- **Roles**: train for fine-tuning; validation for epoch selection (and the fusion stacker); calibration only for conformal thresholds; test only for reporting.

## 4. Features

**v1 → v2 (PyRadiomics re-extraction).** v1 used binWidth 25 (about 9 grey levels inside a nodule), pixel-unit shape features on images of different sizes, and 298 multi-component masks as they were. v2: binCount 64, size features divided by the image scale (`…Relative`), seven feature families (102 features), largest mask component only. v2 lifts the LLM by about +0.05 test AUC (run B 0.699 → run D 0.754, interval excludes zero).

**TI-RADS-motivated features** (`conformal_triage/clinical.py`, fixed settings chosen before looking at labels, computed on the raw image):

| feature | TI-RADS category | definition | train AUC (higher value → malignant) |
|---|---|---|---|
| TallerThanWideRatio | shape | vertical / horizontal extent of the mask | 0.658 (as expected) |
| EchogenicityRatio | echogenicity | mean brightness inside / in a ring around the nodule | 0.552 (opposite to expected) |
| PunctateFociDensity | echogenic foci | small bright spots per 1,000 nodule pixels | 0.487 (no signal) |
| Solidity | margin | area / convex-hull area | 0.367 (lower = irregular = malignant, as expected) |
| MarginSharpness | margin | boundary gradient relative to the surrounding spread | 0.563 |
| AnechoicFraction | composition | share of nodule pixels much darker than the surroundings | 0.409 (solid = malignant, as expected) |

The echogenicity ratio correlates −0.78 with the anechoic fraction: it mostly measures fluid, not tissue echogenicity. A seventh feature computed on the solid part only (run J) removed the reversed direction but added no signal and was slightly worse, so it is left out.

**Background: what ACR TI-RADS is** (Tessler et al., 2017). Radiologists score a thyroid nodule on five ultrasound categories, add the points, and the total gives a risk level (TR1 benign … TR5 highly suspicious) that decides whether a biopsy (FNA) is recommended, depending on the nodule's size.

| TI-RADS category | what the radiologist looks at | options and points | our feature |
|---|---|---|---|
| composition | cystic or solid | cystic or spongiform 0 · mixed cystic-solid 1 · solid 2 | anechoic fraction |
| echogenicity | darker or brighter than the surrounding thyroid | anechoic 0 · hyper- or isoechoic 1 · hypoechoic 2 · very hypoechoic 3 | echogenicity ratio |
| shape | taller than wide on a transverse view | wider-than-tall 0 · taller-than-wide 3 | taller-than-wide ratio |
| margin | smooth, lobulated or irregular | smooth 0 · ill-defined 0 · lobulated or irregular 2 · extrathyroidal extension 3 | solidity, margin sharpness |
| echogenic foci | bright spots, e.g. microcalcifications | none or large comet-tail 0 · macrocalcifications 1 · peripheral (rim) 2 · punctate 3 (types add up) | punctate foci density |

Total points → TR1 (0), TR2 (2), TR3 (3), TR4 (4–6), TR5 (≥ 7). FNA is recommended for TR3 from 2.5 cm, TR4 from 1.5 cm and TR5 from 1 cm. Our features are automatic, continuous approximations of these signs; they do not reproduce the radiologist's points.

**Note on margin sharpness.** ACR TI-RADS gives an ill-defined margin 0 points; the margin points come from a lobulated or irregular outline, which solidity measures. Margin sharpness is therefore an extra measurement inspired by the margin category, not a direct copy of a scored sign.

**Feature engineering: what TI-RADS adds to PyRadiomics.** PyRadiomics already measures shape, brightness and texture, but by design it (a) is rotation-invariant, so it ignores how the nodule is oriented, and (b) uses only the pixels inside the mask, so it never compares the nodule with the tissue around it. Two TI-RADS signs rest exactly on these.

| type | what TI-RADS looks at | our feature | why PyRadiomics misses it |
|---|---|---|---|
| new view | orientation | taller-than-wide ratio = vertical / horizontal extent of the mask | MajorAxisLength, Elongation use the nodule's own axes: a lying and a standing ellipse get the same values |
| new view | contrast with the surroundings | echogenicity ratio = mean inside / mean of a ring of surrounding tissue | first-order sees only absolute brightness inside the mask, which depends on scanner gain |
| new view | boundary (inside vs outside) | margin sharpness = gradient across the boundary / spread of the surroundings | needs pixels on both sides of the boundary; PyRadiomics sees only the inside |
| partial overlap | outline irregularity | solidity = area / convex-hull area | related to Sphericity, PerimeterSurfaceRatio; solidity targets lobulation directly |
| partial overlap | fluid content | anechoic fraction = share of pixels much darker than the ring | related to first-order percentiles, but relative to the surroundings |
| partial overlap | punctate bright spots | punctate foci density = small, separate bright spots per 1,000 pixels | first-order max / 90th percentile see bright pixels, not whether they form small separate spots |

So the engineering contribution is two views PyRadiomics structurally lacks (orientation and context), plus three clinically defined summaries of information PyRadiomics only spreads over many features. Evidence that this is new information: with matched seeds, adding these features raised the LLM's AUC by +0.059 on average (minimum +0.050).

## 5. Classification results

All numbers are AUC on validation / calibration / test. Intervals are 95% paired bootstrap intervals of the AUC difference (2,000 resamples of the same nodules). Full table of runs in the appendix.

### 5.1 Training noise is large

Three seeds of the same setting (runs I, M, N) give test AUC 0.804, 0.768, 0.783. Seed 1 vs seed 42 differs significantly on every split (test [−0.058, −0.014]), and a repeat of run H with the same seed differed by 0.046 on test. **Single-run differences below about 0.03–0.05 are not interpretable.** The main comparisons below are therefore repeated over seeds.

### 5.2 TI-RADS-motivated features (three matched seeds)

| | val | cal | test |
|---|---|---|---|
| with TI-RADS features (runs I, M, N), mean ± SD | 0.785 ± 0.012 | 0.794 ± 0.017 | **0.785 ± 0.018** |
| without (runs D, O, P), mean ± SD | 0.727 ± 0.022 | 0.721 ± 0.029 | **0.726 ± 0.024** |
| paired difference per seed, mean (smallest) | +0.058 (+0.045) | +0.074 (+0.059) | **+0.059 (+0.050)** |
| average of the three seeds' probabilities: difference [95% CI] | +0.054 [+0.018, +0.092] | +0.067 [+0.033, +0.100] | +0.055 [+0.019, +0.094] |

XGBoost (mRMR 16): 0.805 → 0.831 on test with the six features; the six alone give 0.775.

### 5.3 Prompt format (single runs, Qwen2.5-1.5B)

| format | example | test AUC | vs decimal |
|---|---|---|---|
| decimal (D) | "… is measured at -1.140." | 0.754 | – |
| z-score wording (E) | "… has a z-score of -1.14 relative to the training nodules." | 0.759 | no difference |
| percentile (F) | "… is at percentile 13 …" | 0.734 | worse on cal (tails compressed) |
| rich (H) | z-score + percentile + band + plain-language meaning | 0.744 (repeat: 0.698) | worse on cal |
| rich + TI-RADS (K) vs decimal + TI-RADS (I) | | 0.801 vs 0.804 | equal on val/test, worse on cal; 4× slower |

The LLM uses the magnitude of the numbers; adding words around them does not help.

### 5.4 Model size and the settings of Ra et al.

| run | model | features | test AUC |
|---|---|---|---|
| D | Qwen2.5-1.5B | v2 | 0.754 |
| C | Qwen2.5-7B, LR 2e-4 | v2 | 0.713 |
| G | Llama-2-7B, LR 1e-4, LoRA on all layers (Ra et al.) | v2 | 0.731 |
| I | Qwen2.5-1.5B | v2 + TI-RADS | 0.804 (3-seed mean 0.785) |
| L | Qwen2.5-7B, LR 5e-5 | v2 + TI-RADS | 0.693 |

The 7B models trained unstably (validation sensitivity swinging between 0 and 0.9, gradient norms up to 200); a lower LR calmed the swings but not the result. The TI-RADS features, which lift 1.5B by about 0.06, give Qwen2.5-7B nothing (L vs C, interval includes zero). Within this setup (QLoRA on Q/V, about 2,000 training prompts, 5 epochs) a larger backbone does not help.

**Caveat.** These are single-seed runs, and no learning-rate search was done for 7B. Run C used the learning rate chosen for 1.5B; run L used a four times lower one, and its validation AUC was still rising at epoch 5 (0.691 → 0.736 → 0.742 → 0.751 → 0.753), so it may have stopped before converging. Llama-2 (run G) had no TI-RADS features. The supported claim is therefore limited: *in the settings tried, a larger backbone did not help*; it is not shown that a larger backbone cannot help.

**Why Qwen2.5-1.5B stays the main model.**
1. In the one like-for-like comparison (same 16 features, learning rate, epochs and effective batch: run C vs run D), 7B is lower: 0.713 vs 0.754, paired interval [−0.069, −0.004].
2. All three large-model runs, under three different settings, land below the matching 1.5B run; none comes near 0.804.
3. The 1.5B result is stable and replicated over three seeds (0.785 ± 0.018); the 7B trainings were unstable (gradient norms up to 200).
4. Information ceiling: XGBoost on the same 22 numbers reaches 0.835 and the LLM adds nothing on top of it (stacking weight −0.04). A larger LLM could at best approach this ceiling; it cannot raise it.
5. Cost: three to seven times the training time.


### 5.5 LLM vs XGBoost, and fusion

XGBoost (300 trees, depth 3, untuned) on exactly the 22 features of the best LLM setting, same preprocessing:

| model | val | cal | test |
|---|---|---|---|
| XGBoost | 0.827 | 0.848 | **0.835** |
| LLM, average of 3 seeds | 0.792 | 0.803 | 0.793 |
| mean of the two probabilities | 0.820 | 0.838 | 0.824 |
| stacking (logistic regression on both logits, fitted on validation) | 0.827 | 0.848 | 0.834 |

- LLM − XGBoost on test: −0.042 [−0.067, −0.017].
- Correlation of the two models' probabilities: 0.87–0.88 on every split (two LLM seeds correlate 0.91).
- The stacker's weights are 0.99 for XGBoost and −0.04 for the LLM: it learns to ignore the LLM.

**Conclusion: the fine-tuned LLM reproduces a weaker version of the tabular model's ranking and carries no additional information.** This does not show that LLMs are useless for this task in general; it shows that feeding them the same numbers a tabular model sees does not add anything.

### 5.6 Peritumoral radiomics (2026-10-13)

PyRadiomics features (all families except shape) in a ring around the nodule: width half the equivalent radius, at least 5 px; 93 features, 49 after the near-duplicate filter (`conformal_triage/peritumoral.py`, notebook A4b/B2). Rule fixed before running: pass them to the LLM only if adding mRMR 8 of them to XGBoost improves calibration AUC with an interval above zero.

| XGBoost features | val | cal | test |
|---|---|---|---|
| nodule mRMR 16 + TI-RADS | 0.828 | 0.848 | 0.831 |
| + ring mRMR 8 | 0.832 | 0.851 | 0.834 |
| + all 49 ring features | 0.831 | 0.853 | 0.832 |
| ring mRMR 16 only | 0.699 | 0.671 | 0.693 |

Calibration +0.002 [−0.009, +0.014]: no gain, so no LLM run. The ring carries signal (0.69 alone) but it is redundant with the nodule and TI-RADS features; mRMR picks mostly brightness statistics, close to what the echogenicity ratio already compares.

## 6. The conformal layer: what each method guarantees

All methods use only the calibration split to choose thresholds and assume calibration and test nodules are exchangeable.

| method | risk controlled | type of guarantee | in this work |
|---|---|---|---|
| **LTT, planned** (Angelopoulos et al.) | R1: share of cancers **among** auto-benign ≤ 5%; R2: share of benign among auto-malignant ≤ 20% | with probability ≥ 1 − δ over the calibration draw, the true risk of the chosen threshold is below α (δ = 0.05 per risk, 0.90 jointly) | binomial test per candidate threshold, fixed sequence from the strictest threshold, stop at the first failure |
| **miss-rate LTT** | share of **all** cancers sent to auto-benign ≤ 5% | same high-probability form | same test, denominator = all calibration cancers |
| **split CP, marginal** | true label in the set for ≥ 1 − α of nodules | in expectation over calibration and test | one quantile for both labels |
| **split CP, Mondrian** | true label in the set for ≥ 1 − α_y of the nodules of each class (0.90 benign, 0.95 malignant) | in expectation, per class | one quantile per class; a single-label set is an automatic decision |

The key distinction: Mondrian and the miss-rate rule bound **how many cancers are missed** (a rate over cancers); R1 bounds **how many of the auto-benign calls are cancers** (a rate over decisions). With 35% prevalence, a rule can miss only 5% of cancers while its auto-benign group still holds 10% cancers. R1 is the clinically meaningful one for a patient told "benign", and it is far harder.

**Why the binomial test is valid.** Conditional on the number n of calibration nodules below a threshold, these nodules are an i.i.d. sample from the nodules below that threshold, so the number of cancers among them is Binomial(n, R(t)). p = P(Bin(n, α) ≤ k) is then a valid p-value for H0: R(t) > α. The fixed sequence controls the family-wise error rate without a multiplicity correction because the order is fixed before seeing the calibration data.

## 7. Conformal results

### 7.1 LTT certifies nothing, for any model

Smallest group that can be certified at δ = 0.05:

| errors in the group | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| α = 5% (auto-benign) | 59 | 93 | 124 | 153 | 208 |
| α = 20% (auto-malignant) | 14 | 22 | 30 | 37 | 50 |

What the models offer (calibration, 524 nodules, 182 cancers):

| | cleanest 10% | 15% | 20% | first auto-malignant candidates (u = 0.95, 0.90, 0.85) |
|---|---|---|---|---|
| XGBoost | 2/52 cancers | 2/78 | 3/104 | 3/7 (p = 0.97), 3/32, 4/54 (p = 0.010) |
| LLM (3-seed average) | 4/52 | 7/78 | 10/104 | 1/8 (p = 0.50), 2/30 (p = 0.044), 4/43 |

- **Auto-benign (5%)**: XGBoost's tail is close (2 cancers in 78) but still needs 124 nodules for 2 errors; the LLM's tail is two to three times dirtier. Neither certifies at any threshold.
- **Auto-malignant (20%)**: XGBoost at u = 0.85 would pass on its own (4/54, p = 0.010), but the fixed sequence starts at u = 0.95 (7 nodules, 3 benign), fails there and stops. **This is a design weakness of the planned sequence, not only a power problem**: the strictest candidates contain too few nodules to ever pass.
- Earlier simulation: certifying the 5% auto-benign rule with about 500 calibration nodules needs a model with AUC around 0.9 or higher.

### 7.2 Validation-ordered sequence (exploratory)

Ordering the candidates by their p-values on the validation split (independent of calibration, so the guarantee still holds) avoids starting at an empty threshold. It certifies auto-malignant at u = 0.85 for run I (test: 43 nodules, 11.6% benign) and at u = 0.90 for the 3-seed LLM average (test: 28 nodules, none benign); nothing for XGBoost (its validation order puts u = 0.95 first) and nothing on the auto-benign side. Because this rule was found after looking at the results, it has to be fixed in advance and confirmed on a new split before it can be claimed.

### 7.2b A design fixed in advance (2026-10-13)

Written and committed before it was run (`experiments/ltt_design_resplits.py`, output `ltt_design_resplits.txt`): candidates at the 5%, 10%, …, 30% quantiles of the validation scores, ordered by validation p-values, fixed-sequence test on calibration with the planned α and δ. Models fixed; calibration and test re-split 2,000 times.

| model | auto-benign 5%, either design | auto-malignant 20%, planned | auto-malignant 20%, quantile design | decided on test | benign among them (test / all 962) |
|---|---|---|---|---|---|
| XGBoost | 0% | 0% | 38% | 10.7% | 14.9% / 11.2% |
| LLM, 3-seed average | 0% | 1% | **99%** | 7.9% | 7.4% / 6.4% |
| LLM run I | 0% | 13% | 22% | 11.5% | 18.4% / 13.6% |
| LLM run D | 0% | 0% | 8% | 6.8% | 22.6% / 13.8% |

Auto-benign at 5% is never certified: the limit there is the model. Auto-malignant at 20% becomes attainable for the seed-averaged LLM (99% of re-splits; on our split u = 0.833, 9.8% of test nodules, 14% benign). The ordering helps only where validation and calibration agree (XGBoost's top-5% candidate is dirty on calibration). The benign share over all 962 nodules never exceeds 20%. Caveat: the design followed the exploratory run-I result and the models were selected on these data; confirmation needs new data.

### 7.3 Miss-rate rule (LTT on the share of cancers missed)

| model | threshold t | test automation | test miss rate | cancers among auto-benign (test) |
|---|---|---|---|---|
| XGBoost | 0.14 | 23.7% | 4.6% | 7/104 (6.7%) |
| LLM (3-seed average) | 0.14 | 12.1% | 4.6% | 7/53 (13.2%) |

Single LLM runs: test miss rate 0.7–8.6% across runs (a 95% guarantee allows occasional excess; the test set has only 152 cancers, so one cancer is 0.7 points).

### 7.4 Mondrian split CP (α 0.10 benign / 0.05 malignant)

| model | cancers covered (our test split) | over 4,000 re-splits of cal + test | auto-benign | cancers among auto-benign (re-splits, 5–95%) |
|---|---|---|---|---|
| XGBoost | 0.941 | 0.951 ± 0.024 | 25.7% | 6.4% (2.3–10.5%) |
| LLM, 3-seed average | 0.934 | 0.950 ± 0.024 | 16.7% | 9.8% (4.7–14.6%) |
| LLM run I (seed 42) | 0.941 | 0.951 ± 0.024 | 16.9% | 9.6% (4.1–14.0%) |
| LLM run D (no TI-RADS) | 0.908 | 0.951 ± 0.023 | 17.6% | 9.3% (3.6–14.9%) |

On our single test split, cancer coverage fell below 0.95 in 12 of 16 runs. This looked like a systematic failure, but all runs share the same calibration/test draw. Re-splitting the 962 calibration + test nodules (model fixed) gives an average coverage of 0.950–0.951 for every model, below 0.95 in about 45% of draws: **the guarantee holds; our particular split was unlucky by about one standard deviation.**

### 7.5 Valid guarantee vs safe decision

The attainable guarantees (Mondrian, miss rate) send 17–26% of nodules to auto-benign with about 6–10% cancers among them, down from the 35% base rate but far above the 5% the design asked for. In ACR TI-RADS terms, a 5–20% malignancy risk corresponds to TR4, where FNA is still recommended above a size threshold. These rules are therefore valid but would not justify skipping a biopsy.

## 8. Theory check

What was verified for this report:

- **LTT implementation** (`conformal_triage/ltt.py`): binomial p-value, fixed sequence, joint guarantee 1 − 2δ by the union bound. Correct as implemented; unit tests cover it (40 tests pass).
- **Split CP** (`split_cp.py`): quantile index ⌈(n + 1)(1 − α)⌉, per-class quantiles for Mondrian. Correct. The empirical coverage over re-splits (0.950–0.951) matches the theory.
- **Miss-rate rule**: is a valid LTT procedure (binomial count over all calibration cancers, fixed increasing order). The notebook called it "exploratory" because it was added after the planned rule failed, not because it is invalid. The notebook text claiming calibration and test are not exchangeable referred to the old official split and has been corrected.
- **Validation-ordered LTT**: valid because the order depends only on validation data. It was chosen post hoc, so it is a hypothesis for a new split, not a result.
- **No leakage into calibration/test**: imputation, scaling, near-duplicate filter and mRMR are fitted on train; epoch selection and the stacker use validation only.

Caveats that remain:

- **Bootstrap intervals** treat the trained model as fixed and resample nodules only; they do not include training noise (handled separately with seeds) or patient clustering (patient IDs unavailable), so they are somewhat too narrow.
- **Several design choices were informed by calibration/test results**: excluding the seventh TI-RADS feature, keeping decimal over rich, and the validation-ordered LTT. The main conclusions (TI-RADS gain, LLM < XGBoost, no LTT certification) do not depend on these choices.
- **Many comparisons, no multiplicity correction**: individual borderline intervals (e.g. J − I on test) should not be over-read; the headline findings hold across seeds and splits.
- **Prevalence**: 35% cancers is a referral (FNA) population. Precision-type risks such as R1 depend on prevalence, so they would be easier in a lower-prevalence population and do not transfer directly.

## 9. Relation to Ra et al. (2025)

Our pipeline replicates theirs (PyRadiomics → mRMR → text → LLM classifier with LoRA r16). Differences in their evaluation that explain why their LLM appears to beat the tabular baselines:

- their baseline AUCs equal their accuracies to within 0.004 for all seven classical models (XGBoost AUC 0.605), which suggests AUC computed from hard labels;
- their own ablation shows a frozen LLM with only the classification head trained (AUC 0.710) is as good as LoRA fine-tuning (0.700);
- hiding the feature names changes AUC by 0.005 (internal) and 0.013 (external), without a test;
- labels are BI-RADS categories, not pathology; no confidence intervals, a single seed, 500+ epochs.

With a properly evaluated XGBoost and seed repeats, we find the LLM about 0.04–0.05 AUC below the tabular model and contributing nothing to it.

## 10. Limitations

- One dataset, one pooled split, image-level splitting without patient IDs.
- 2D single images; radiomics from expert masks (no segmentation error).
- The TI-RADS features are heuristic approximations; echogenic foci and echogenicity are not captured well.
- LLM hyperparameters were not tuned systematically (5 epochs, one LR per model size).
- Conformal results use one calibration set of 524 nodules; the re-split analysis shows how much a single draw can vary.

## 11. Open options (not run)

In order of expected value:

1. **Confirm the quantile-grid LTT design on new data** (Section 7.2b): retraining on new random splits or a second dataset.
2. **Repeat the conformal analysis over several random splits** (needs re-training per split, or the re-split approach above for a fixed model).
3. **External dataset** (e.g. ThyroidXL) to test whether the TI-RADS gain and the LLM–XGBoost gap transfer.
4. Inputs a tabular model cannot use (report text, clinical history) would be the setting where an LLM could add information.

## Appendix: run index (pooled split, Qwen2.5-1.5B, 16 mRMR features, decimal, seed 42 unless stated)

| run | change | val | cal | test |
|---|---|---|---|---|
| A | v2 features, no filter | 0.753 | 0.757 | 0.750 |
| B | v1 features | 0.709 | 0.705 | 0.699 |
| C | Qwen2.5-7B | 0.708 | 0.705 | 0.713 |
| D | v2 + filter (reference) | 0.750 | 0.752 | 0.754 |
| E | z-score wording | 0.729 | 0.756 | 0.759 |
| F | percentile values | 0.709 | 0.713 | 0.734 |
| G | Llama-2-7B, Ra et al. settings | 0.705 | 0.709 | 0.731 |
| H | rich format | 0.732 | 0.711 | 0.744 |
| I | + 6 TI-RADS features | 0.795 | 0.812 | 0.804 |
| J | + 7 TI-RADS features | 0.786 | 0.802 | 0.781 |
| K | + 6 TI-RADS, rich | 0.803 | 0.785 | 0.801 |
| L | + 6 TI-RADS, Qwen2.5-7B, LR 5e-5 | 0.751 | 0.712 | 0.693 |
| M | I, seed 1 | 0.771 | 0.777 | 0.768 |
| N | I, seed 2 | 0.787 | 0.794 | 0.783 |
| O | D, seed 1 | 0.706 | 0.695 | 0.712 |
| P | D, seed 2 | 0.723 | 0.715 | 0.713 |
