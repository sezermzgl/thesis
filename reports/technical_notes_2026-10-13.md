# Technical notes: every step and parameter

*How each part of the work was done, as implemented in `conformal_triage/`, `experiments/` and `experiments/tn3k_v2_pipeline_colab.ipynb`; numbers from `runs/`. Chronology: `reports/thesis_journey_2026-10-13.md`. A Turkish version is kept with the study material.*

## 1. Data: TN3K

| item | value |
|---|---|
| images | 3,493 (2,421 patients) |
| official split | trainval 2,879 (1,905 benign, 974 malignant) + test 614 (378 benign, 236 malignant), patient-disjoint |
| malignant share | 34.6% (trainval 33.8%, official test 38.4%) |
| labels | cytology/pathology (FNA), not radiologist scores |
| masks | pixel-level nodule masks, binarised at 127 |
| image size | varies; median 390×336 (trainval) and 436×375 (official test) pixels |
| patient IDs | not released |

In the first experiments (2–8 October), trainval was split at image level into train 2,015 / validation 432 / calibration 432; the test set was the official test (614).

## 2. Pooled split

`experiments/make_pooled_splits.py`, seed 42. All 3,493 images are pooled; each gets a stratum (official source × label); scikit-learn `train_test_split` is applied three times, stratified by stratum: 60% train; of the rest, 12.5/40 validation; of the rest, 15/27.5 calibration and the remainder test.

| split | images | malignant | from official test |
|---|---|---|---|
| train | 2,095 | 34.6% | 17.6% |
| validation | 436 | 34.6% | 17.4% |
| calibration | 524 | 34.7% | 17.7% |
| test | 438 | 34.7% | 17.6% |

Price: the test set is no longer an external sample, and images of one patient can fall into different splits (TN3K has 1,072 more images than patients; in the worst case about 40% of test images could have a same-patient image in train).

## 3. Exchangeability test (classifier two-sample test)

`conformal_triage/evaluation.py`, `two_sample_test`: stack the two sets with a set label (0/1); a logistic regression predicts the set label from the features; 5-fold stratified cross-validated probabilities give an AUC; the p-value comes from 500 label permutations, p = (1 + #{permuted AUC ≥ observed}) / 501.

| comparison | AUC | p |
|---|---|---|
| validation – calibration (official, control) | 0.506 | 0.37 |
| calibration – official test | 0.632 | 0.002 |
| validation – calibration (pooled) | 0.476 | – |
| calibration – test (pooled) | 0.434 | 1.0 |

## 4. Feature extraction

**v1 (2–8 October).** PyRadiomics in 2D on the original image only; image converted to grey and rescaled to 0–255 by its own min–max; pixel spacing 1 (no physical spacing known); texture with a fixed bin width of 25; families shape2D, first-order, GLCM, GLSZM; 67 usable features.

**Four problems found by the audit.**
1. Bin width 25 left a median of about 9 grey levels inside a nodule (IQR 7–10).
2. Shape features in pixels on images of different sizes; MajorAxisLength correlates 0.42 with image width.
3. 298 multi-component masks (7.8% of trainval, 11.7% of test).
4. 356 masks touching the image border (11.2% and 5.4%).

Side findings: those masks are less often malignant (22% vs 36%); image width alone predicts malignancy weakly (AUC 0.57).

**v2 (from 8 October).**

| setting | v1 | v2 |
|---|---|---|
| grey levels | bin width 25 | **bin count 64** |
| size features | pixels | lengths / √(WH), areas / (WH), perimeter-to-surface ratio × √(WH) |
| families (PyRadiomics) | 4 | 7 (+ GLRLM, GLDM, NGTDM): shape 9, first-order 18, GLCM 24, GLRLM 16, GLSZM 16, GLDM 14, NGTDM 5 |
| multi-component masks | as they are | largest component only |
| border-touching masks | as they are | kept and flagged |
| minimum mask | – | 20 pixels |
| features | 67 | 102 |

On single-component masks the shared features reproduce v1 exactly. With relative shape features the two official sources remain separable (0.709 vs 0.654 with pixel shape), so they differ beyond pixel scale; the pooled split spreads this difference evenly.

## 5. Preprocessing and selection (fitted on train only)

1. Infinite values set to missing; constant features removed.
2. Median imputation; z-score with the train mean and SD.
3. **Near-duplicate filter** (`selection.py`, |r| > 0.95): visit features from most to least correlated with the label; keep a feature only if its |r| with every kept feature is ≤ 0.95. Of v2's 102 features, 48 are dropped.
4. **mRMR** (mrmr-selection `mrmr_classif`): greedy; at each step the feature maximising relevance (F statistic with the label) divided by redundancy (mean |correlation| with the features already chosen). The first 16 are used, in mRMR order. The filter was needed because the *mean* correlation lets a copy of one chosen feature through (PixelSurfaceRelative and MeshSurfaceRelative are the same number in 2D and were both chosen).
5. TI-RADS-motivated features bypass the filter and mRMR and are placed first in the prompt.

## 6. The features in the prompt

**16 radiomics features of the best setting (mRMR order).**

| # | feature (family) | what it measures |
|---|---|---|
| 1 | MajorAxisLengthRelative (shape) | major axis of the ellipse with the nodule's second moments, relative to the image: length |
| 2 | SmallAreaLowGrayLevelEmphasis (GLSZM) | weight of small, dark zones (connected pixels of equal grey level) |
| 3 | Elongation (shape) | √(minor / major eigenvalue); 1 = circle, lower = elongated |
| 4 | PixelSurfaceRelative (shape) | mask area / image area |
| 5 | Range (first-order) | brightest minus darkest pixel |
| 6 | PerimeterSurfaceRatioRelative (shape) | perimeter / area, scaled; high for irregular or small outlines |
| 7 | DependenceVariance (GLDM) | variance of the number of similar neighbours per pixel |
| 8 | MeanAbsoluteDeviation (first-order) | mean absolute deviation of brightness |
| 9 | Idm (GLCM) | inverse difference moment: local homogeneity |
| 10 | Sphericity (shape) | perimeter of the equal-area circle / actual perimeter; roundness |
| 11 | DependenceNonUniformityNormalized (GLDM) | how evenly dependence sizes are spread, normalised |
| 12 | DependenceNonUniformity (GLDM) | same, not normalised (grows with size) |
| 13 | MaximumProbability (GLCM) | share of the most frequent neighbouring grey-level pair: dominance of one texture |
| 14 | SizeZoneNonUniformity (GLSZM) | variability of zone sizes, not normalised |
| 15 | ZoneEntropy (GLSZM) | randomness of zone sizes and grey levels: texture complexity |
| 16 | SizeZoneNonUniformityNormalized (GLSZM) | normalised version of 14 |

**8 features of the first runs (v1, official split):** MajorAxisLength, HighGrayLevelZoneEmphasis, Elongation, ClusterShade, Range, LargeAreaHighGrayLevelEmphasis, MaximumDiameter, Maximum. Dropping MajorAxisLength alone changed 28% of the answers.

**Six TI-RADS-motivated features** (`clinical.py`; raw image, no per-image rescaling; ratios or densities; settings fixed before looking at labels). Ring: equivalent radius r = √(area/π); width max(4, round(0.25 r)); the dilated mask minus the nodule, excluding pixels darker than 5 (outside the ultrasound sector).

| feature | definition | TI-RADS sign | train AUC |
|---|---|---|---|
| TallerThanWideRatio | vertical / horizontal extent of the mask (image as acquired) | shape | 0.658 |
| EchogenicityRatio | mean inside / mean of the ring | echogenicity | 0.552 (reversed) |
| PunctateFociDensity | bright spots per 1,000 nodule pixels | echogenic foci | 0.487 |
| Solidity | area / convex-hull area | margin | 0.367 |
| MarginSharpness | mean boundary gradient / √(0.5 (var inside + ring SD²)) | margin | 0.563 |
| AnechoicFraction | share of nodule pixels below 25% of the ring mean | composition | 0.409 |

Bright spots: inside the nodule eroded by 2 pixels; white top-hat with a radius-3 disk; threshold median + 6 × MAD (scaled by 1.4826, at least 1) and brighter than the ring mean; connected regions of 2–40 pixels. Margin gradient: Gaussian smoothing σ = 1, Sobel magnitude / 8. The seventh feature (run J) computes the echogenicity ratio on the non-fluid pixels only.

**Peritumoral ring** (`peritumoral.py`): width max(5, round(0.5 r)) via a Euclidean distance transform; nodule and outside-sector pixels excluded; PyRadiomics with v2 settings, families first-order, GLCM, GLSZM, GLRLM, GLDM, NGTDM (93 features, 49 after the filter); computed on a crop around the ring (identical to the full image).

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

## 7. Prompt

"Classify this thyroid nodule as benign or malignant based on the following standardized radiomic features." followed by one sentence per feature, "The '{name}' feature is measured at {z:.3f}.", TI-RADS features first, then mRMR order. Readable names (`original_shape2D_Elongation` → Elongation); a name shared by families gets the family added. The longest prompt is checked before training and the run stops if it would be truncated (≤ 384 tokens with 16 features, ≤ 512 with TI-RADS).

Other formats: z-score wording; train-split percentile; rich (z-score, percentile, band: < 10 "much lower than typical", < 30 "lower", < 70 "close to typical", < 90 "higher", else "much higher", plus a phrase on what the feature measures, never on which direction is malignant); semantic only (not run).

## 8. Language model and training

| setting | value |
|---|---|
| backbone | Qwen2.5-1.5B (also Qwen2.5-7B, Llama-2-7B) |
| head | `AutoModelForSequenceClassification`, 2 outputs on the last non-padding token, randomly initialised |
| weights | 4-bit NF4 with double quantisation, compute in bf16 (QLoRA) |
| LoRA | rank 16, α = 32, dropout 0.05, on q_proj and v_proj (Llama run: all seven linear projections) |
| loss | cross-entropy with class weights n / (2 n_c): benign 0.765, malignant 1.445 |
| optimiser | AdamW, LR 2 × 10⁻⁴ (Llama 10⁻⁴; run L 5 × 10⁻⁵), cosine schedule, 5% warm-up |
| batch | 8 × 2 accumulation = 16 (7B: 4 × 4) |
| epochs | 5 (131 steps per epoch, 655 in total) |
| selection | best epoch by validation AUC, restored at the end |
| precision | bf16 training on L4 (fp16 on T4), gradient checkpointing; prediction in fp32 |
| seed | 42; repeats with 1 and 2 |

Two fixes: the loss under gradient accumulation was scaled incorrectly in the first runs (fixed; the retrained model agrees, Spearman 0.97); predictions ran under the Trainer's bf16 autocast, rounding probabilities to steps of 1/64 in logit space (195 distinct values on 524 nodules; fixed by predicting in fp32 on the unwrapped model).

## 9. XGBoost

300 trees, depth 3, learning rate 0.05, subsample 0.8, colsample_bytree 0.8, scale_pos_weight = n_benign / n_malignant; not tuned; the same preprocessing as the LLM.

## 10. Evaluation

AUC from P(malignant); accuracy and sensitivity at 0.5. Paired bootstrap: 2,000 resamples of the same nodules, AUC difference per resample (single-class resamples skipped), 2.5–97.5% percentiles. It holds the trained model fixed (no training noise, covered by seeds) and resamples images (no patient clustering). Main comparison over matched seeds 42, 1, 2: mean ± SD, per-seed differences, and the interval of the seed-averaged probabilities.

## 11. Perturbation votes (first phase)

Run 1: query 0 = training prompt; queries 1–9 each shuffle the feature order, drop one feature with probability 0.5 and use one of three templates. Run 2: 36 queries per nodule: the base query; 9 jitter queries at each SD ∈ {0.05, 0.1, 0.2} (Gaussian noise on every standardised value; SD chosen on validation by the highest vote-fraction AUC, a rule fixed in advance → 0.2); 8 drop-one queries. Vote v = number of queries with P ≥ 0.5; jitter votes m = 10, drop votes m = 9. Noise seeded per nodule (crc32 of its ID).

## 12. Conformal methods

**Definitions.** For auto-benign calls (P ≤ t): miss rate = FN / all cancers = 1 − sensitivity (false negative rate); R1 = FN / all auto-benign calls = 1 − NPV (false omission rate). For auto-malignant calls (P ≥ u): R2 = FP / all auto-malignant calls = 1 − PPV (false discovery rate).

**LTT** (`ltt.py`). Candidates t ∈ {0.05, …, 0.45}, u ∈ {0.95, …, 0.55} (votes: t ∈ {0..4}, u ∈ {10..6}); for each, n calibration nodules on the decided side, k errors, p = P(Bin(n, α) ≤ k) (p = 1 if n = 0); fixed sequence from the safest candidate, stop at the first p > δ; the most permissive certified threshold is used. Valid because, given n, the decided nodules are an independent sample of that region, so k ~ Bin(n, R(t)); the order is fixed without calibration data, so the family-wise error is ≤ δ. Guarantee: with probability ≥ 1 − δ over the calibration draw, the true risk of the chosen threshold is ≤ α; both sides jointly ≥ 1 − 2δ = 0.90.

| smallest certifiable group (δ = 0.05) | 0 errors | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| α = 5% | 59 | 93 | 124 | 153 | 181 | 208 |
| α = 20% | 14 | 22 | 30 | 37 | 44 | 50 |

**Miss-rate LTT.** Candidates t ∈ {0.01, …, 0.50} increasing; n = all calibration cancers (182 on the pooled split), k = cancers with P ≤ t; p = P(Bin(182, 0.05) ≤ k); stop at the first failure.

**Split CP** (`split_cp.py`). Non-conformity s(x, y) = 1 − P(y | x); threshold = the k-th smallest calibration score of the true labels with k = ⌈(n + 1)(1 − α)⌉ (∞ if k > n); set {y : s(x, y) ≤ q̂}; a single-label set is an automatic decision. Marginal: one threshold, α = 0.10 or 0.05. Mondrian: one per class, α_benign = 0.10, α_malignant = 0.05. Guarantee in expectation over the calibration and test draws: P(y ∈ C(x) | Y = y) ≥ 1 − α_y.

**Pre-registered LTT design** (`ltt_design_resplits.py`, committed in `4f42659` before running): candidates at the q-quantile (auto-benign) or (1 − q)-quantile (auto-malignant) of the validation scores, q ∈ {5%, …, 30%}; ordered by validation p-value (ties: smaller q first); fixed-sequence test on calibration with the planned α and δ. Evaluated with the models fixed over 2,000 stratified re-splits of the 962 calibration + test nodules.

## 13. Fusion and conformal checks

**Fusion** (`hybrid_xgb_llm.py`): XGBoost on exactly run I's 22 features; the LLM score is the mean of three seeds; combined by mean probability, mean rank, or a logistic regression on the two logits (clipped to [10⁻⁶, 1 − 10⁻⁶]) fitted on validation only. Reported: AUCs, Pearson/Spearman correlation, paired bootstrap against XGBoost, Mondrian table, stacking weights.

**Conformal checks** (`conformal_checks.py`): the smallest-group table; planned and validation-ordered LTT and the miss-rate rule for XGBoost and the seed-averaged LLM; Mondrian over 4,000 re-splits of calibration + test with the model fixed (524 calibration nodules, 182 of them cancers), recalibrated each time.

## 14. All runs

| run | model | features | format | other | val | cal | test |
|---|---|---|---|---|---|---|---|
| official | Qwen2.5-1.5B | v1, 8 | decimal | T4, first model | 0.704 | 0.707 | 0.672 |
| official | Qwen2.5-1.5B | v1, 8 | decimal | L4, retrained | 0.697 | 0.714 | 0.667 |
| official | Qwen2.5-7B | v1, 8 | decimal | | 0.697 | 0.686 | 0.669 |
| official | Qwen2.5-7B | v1, 16 | decimal | | 0.702 | 0.715 | 0.666 |
| A | Qwen2.5-1.5B | v2, 16, no filter | decimal | | 0.753 | 0.757 | 0.750 |
| B | Qwen2.5-1.5B | v1, 16 | decimal | | 0.709 | 0.705 | 0.699 |
| C | Qwen2.5-7B | v2, 16 | decimal | batch 4×4 | 0.708 | 0.705 | 0.713 |
| D | Qwen2.5-1.5B | v2, 16 | decimal | reference | 0.750 | 0.752 | 0.754 |
| E | Qwen2.5-1.5B | v2, 16 | z-score | | 0.729 | 0.756 | 0.759 |
| F | Qwen2.5-1.5B | v2, 16 | percentile | | 0.709 | 0.713 | 0.734 |
| G | Llama-2-7B | v2, 16 | decimal | LR 1e-4, LoRA all | 0.705 | 0.709 | 0.731 |
| H | Qwen2.5-1.5B | v2, 16 | rich | | 0.732 | 0.711 | 0.744 |
| I | Qwen2.5-1.5B | v2, 16 + 6 TI-RADS | decimal | best setting | 0.795 | 0.812 | 0.804 |
| J | Qwen2.5-1.5B | v2, 16 + 7 TI-RADS | decimal | | 0.786 | 0.802 | 0.781 |
| K | Qwen2.5-1.5B | v2, 16 + 6 TI-RADS | rich | | 0.803 | 0.785 | 0.801 |
| L | Qwen2.5-7B | v2, 16 + 6 TI-RADS | decimal | LR 5e-5 | 0.751 | 0.712 | 0.693 |
| M | Qwen2.5-1.5B | as I | decimal | seed 1 | 0.771 | 0.777 | 0.768 |
| N | Qwen2.5-1.5B | as I | decimal | seed 2 | 0.787 | 0.794 | 0.783 |
| O | Qwen2.5-1.5B | as D | decimal | seed 1 | 0.706 | 0.695 | 0.712 |
| P | Qwen2.5-1.5B | as D | decimal | seed 2 | 0.723 | 0.715 | 0.713 |

Large-model runs (C, G, L) are single-seed, and no learning-rate search was done for 7B (run L's validation AUC was still rising at epoch 5); the like-for-like comparison is C vs D (same features, LR, epochs and effective batch). Why Qwen2.5-1.5B stays the main model: (1) in the like-for-like comparison C vs D, 7B is lower (0.713 vs 0.754, paired interval [−0.069, −0.004]); (2) all three large-model runs fall below the matching 1.5B run; (3) the 1.5B result is replicated over three seeds (0.785 ± 0.018), the 7B trainings were unstable; (4) XGBoost on the same 22 numbers reaches 0.835 and the LLM adds nothing to it, so the features, not the backbone size, set the limit; (5) three to seven times the training time.

Training times: 1.5B 26–37 min, rich 83–132 min, Qwen 7B 90–123 min, Llama 182 min. Hardware: Colab T4, then Colab Pro L4.

## 15. Code and records

`conformal_triage/` (LTT, split CP, prompts, meaning dictionary, near-duplicate filter, TI-RADS and ring features, evaluation; 43 unit tests in `tests/`); `experiments/tn3k_v2_pipeline_colab.ipynb` (Part A extraction, Part B XGBoost, Part C LLM and conformal layer); `experiments/make_pooled_splits.py`, `hybrid_xgb_llm.py`, `conformal_checks.py`, `ltt_design_resplits.py`; `runs/` (config, metrics, training log, per-nodule probabilities and executed notebook per run); `manuscript/` (thesis, LaTeX).
