# From design to results: how the thesis work unfolded (30 September – 13 October 2026)

*A chronological account of the work, written for the advisor meeting: what was done at each step, why, what it showed, and how it shaped the next step. Detailed tables: `reports/conformal_triage_report.md` (2–8 October), `runs/2026-10-08_v2_pipeline/RESULTS.md` and `reports/consolidated_report_2026-10-12.md` (from 8 October).*

## 0. Starting point

**Goal.** A triage layer for thyroid nodules that makes an automatic decision only where its error rate is guaranteed, and refers every other nodule to a clinician:

- auto-benign: at most 5% cancers among the nodules called benign (R1, α₁ = 5%);
- auto-malignant: at most 20% benign nodules among those called malignant (R2, α₂ = 20%);
- each guarantee with probability ≥ 95% over the calibration data (δ = 0.05), via Learn then Test (LTT).

**Classifier.** The pipeline of Ra et al. (2025), moved from mammography to thyroid ultrasound: PyRadiomics features of the nodule → mRMR selection → one English sentence per feature ("The 'Elongation' feature is measured at −1.140.") → a language model (Qwen2.5-1.5B) fine-tuned with 4-bit QLoRA and a classification head.

**Uncertainty signal (original design, 30 September).** Ask the model the same question several times with small perturbations of the prompt, count the "malignant" votes, and expect the wrong answers to be unstable, so that their votes split and they get referred.

**Data.** TN3K: 3,493 ultrasound images (2,421 patients) with nodule masks and biopsy-based labels, 35% malignant.

## 1. First triage runs on the official split (2 October)

**What was done.** The triage layer was implemented as a tested Python package (`conformal_triage/`, 21 unit tests at the time, reproducing the design note's worked example exactly) and run on the official TN3K split with eight v1 features. Base model: test AUC 0.672.

**Run 1, mixed perturbations** (shuffled feature order, a dropped feature, paraphrased templates). The perturbations broke the model rather than probing it: paraphrased templates were never classified malignant, shuffling alone changed 29% of answers, and no nodule received more than 6/10 malignant votes. The votes measured distance from the training format, not uncertainty.

**Run 2, format-preserving perturbations** (Gaussian jitter on the standardised values; drop-one). The model stayed in distribution (5–12% of answers changed), but:

- vote agreement separated correct from incorrect predictions no better than the softmax probability (0.53–0.65 AUC for all signals);
- the model was **consistently wrong**: 19% of nodules with 0/10 malignant votes were cancers, and 56–69% of its errors were unanimous across all ten queries.

| run 2 signal (official split) | flip rate vs base prompt (cal) | AUC for telling correct from wrong predictions (cal / test) |
|---|---|---|
| softmax confidence | – | 0.629 / 0.577 |
| jitter votes (SD 0.2) | 0.106 | 0.579 / 0.533 |
| drop-one votes | 0.117 | 0.652 / 0.571 |

| LTT on calibration (official split) | cleanest auto-benign group | cleanest auto-malignant group |
|---|---|---|
| jitter votes | v = 0: 165 nodules, 19.4% cancers (target ≤ 5%) | v = 10: 116 nodules, 45.7% benign (target ≤ 20%) |
| drop votes | v = 0: 175 nodules, 20.0% cancers | v = 9: 58 nodules, 31.0% benign |
| softmax | not certified | not certified |

**LTT certified nothing**, with votes or with softmax. Split conformal prediction met its coverage on calibration but its automatic benign calls on test contained 23% cancers.

**A data problem surfaced.** A classifier could tell calibration nodules from official-test nodules (two-sample AUC 0.63, p = 0.002). Conformal guarantees require calibration and test data to be exchangeable; on this split they are not.

**Conclusions that set the direction.**
1. The limit is the base model, not the triage method: errors are stable, so no consistency signal can flag them.
2. The softmax probability is as good a score as any vote; the perturbation machinery was dropped.
3. The split has to be fixed before any guarantee can be tested.

## 2. Error analysis and a weaker guarantee (7 October)

**The question.** Why did LTT certify nothing? The run-2 outputs were looked at again, without any new training.

**Finding 1: the votes do not find the errors.** The cancers the model calls benign are already called benign by the plain prompt, and the perturbed prompts give the same answer. The model is confidently wrong, so counting votes cannot flag these nodules.

**Finding 2: there is an easier guarantee.** Both guarantees are about the same mistake: a cancer called benign (a false negative, FN). They divide it by different things.

| guarantee | the question it answers | formula |
|---|---|---|
| R1 (planned) | "If I call a nodule benign, how likely is it to be a cancer?" (each patient) | FN / all nodules called benign |
| miss rate (new) | "What share of all cancers do I miss?" (the whole programme) | FN / all cancers |

*Example:* 1,000 nodules, 350 of them cancers. 200 are called benign, and 20 of those are cancers. Miss rate = 20 / 350 = 5.7%; R1 = 20 / 200 = 10%. The same 20 mistakes, but R1 is twice as large, because the benign group is much smaller than the set of all cancers. With 35% cancers in the data, holding R1 at 5% is much harder.

**Result with the miss rate** (same LTT test, official split):

| target | nodules sent to auto-benign (test) | cancers missed (test) | verdict |
|---|---|---|---|
| miss ≤ 5% of cancers | 7.2% | 5.1% | about on target, but few nodules decided |
| miss ≤ 10% of cancers | 24.4% | 15.3% | **broken**: calibration and test were not alike |

**Takeaway.** The miss rate can be certified; R1 could not. They are different promises, and this distinction runs through the rest of the work. The broken 10% row was also the first concrete sign that the official split was not exchangeable.

## 3. Backbone, benchmark, split and data audit (8 October)

**Larger backbone (official split).** Qwen2.5-7B gave the same test AUC as 1.5B (0.669 vs 0.667); a retrained 1.5B matched the first model (Spearman 0.97). Size did not look like the bottleneck.

**Classical benchmark.** XGBoost on the same features matched or beat the LLM (0.696 vs 0.667 with 8 features; 0.709 with 16). From here on XGBoost is the reference.

| official split, test AUC | 8 features | 16 features |
|---|---|---|
| Qwen2.5-1.5B (retrained) | 0.667 | – |
| Qwen2.5-7B | 0.669 | 0.666 |
| XGBoost | 0.696 | 0.709 |

**Pooled, exchangeable split.** All 3,493 images were pooled and re-split, stratified by label and source: train 2,095 / validation 436 / calibration 524 / test 438. The two-sample test dropped to AUC 0.43. With this split, more features started to help on test as well (XGBoost with all 67 features: 0.792), which showed that the earlier "more features do not help" was an artefact of the shift.

**Image quality audit.** Four problems in the original feature extraction:
- texture computed with about 9 grey levels inside a nodule (bin width 25);
- shape features in pixels on images of different sizes (official-test images are larger, which explained much of the "shift");
- 298 masks with several components;
- 356 masks touching the image border.

| two-sample test (can a classifier tell the sets apart?) | AUC |
|---|---|
| validation vs calibration (control) | 0.51 |
| calibration vs official test | **0.63** (p = 0.002) |
| calibration vs test, pooled split | **0.43** |

**One limitation to keep in mind.** TN3K does not say which images come from the same patient, so the split is made per image: two images of one patient can end up one in training and one in test. The model may then partly recognise the patient rather than judge the nodule, which makes every model's AUC look somewhat higher than it would on new patients. It affects all models in the same way, so the comparisons between them still hold. It is reported as a limitation.

## 4. Revised features and the first pooled-split LLM runs (8–9 October)

**Why new features.** The audit above showed that the v1 features themselves were the weak point: texture was computed on only about 9 grey levels inside a nodule, sizes were in pixels although image sizes differ, and many masks had several pieces. Training more models on these numbers would not help, so the features were extracted again first.

**v2 extraction.** 64 grey levels, size features relative to the image, the seven PyRadiomics feature families (102 features: shape 9, first-order 18, GLCM 24, GLRLM 16, GLSZM 16, GLDM 14, NGTDM 5; shape measures geometry, first-order the brightness histogram, the five texture families how grey levels are arranged: neighbouring pairs, runs along a line, connected zones, neighbour similarity, difference from the neighbourhood mean), largest mask component only. On single-component masks it reproduces v1 exactly.

**Two fixes found along the way.**
- mRMR kept selecting near-identical features (pixel surface and mesh surface are the same number in 2D), so a near-duplicate filter (|r| > 0.95, train only) was added before mRMR.
- Predictions were computed under bfloat16 autocast, which rounded the probabilities (195 distinct values on 524 nodules); prediction was moved to full precision.

**The 16 features the LLM sees** (selected by mRMR on train, after the near-duplicate filter; values are train z-scores):

| group | features | what they describe |
|---|---|---|
| size and shape | MajorAxisLengthRelative, PixelSurfaceRelative, Elongation, PerimeterSurfaceRatioRelative, Sphericity | how long and large the nodule is relative to the image, how elongated, how round or irregular its outline |
| brightness | Range, MeanAbsoluteDeviation | spread of grey values inside the nodule |
| texture (GLCM) | Idm, MaximumProbability | local homogeneity; dominance of one grey-level pattern |
| texture (GLSZM) | SmallAreaLowGrayLevelEmphasis, SizeZoneNonUniformity, SizeZoneNonUniformityNormalized, ZoneEntropy | small dark patches; how varied and random the zones of equal grey level are |
| texture (GLDM) | DependenceVariance, DependenceNonUniformity, DependenceNonUniformityNormalized | how uniformly pixels resemble their neighbours |

**Results (runs A–F, pooled split, 16 features).**

| run | change | val | cal | test |
|---|---|---|---|---|
| B | v1 features | 0.709 | 0.705 | 0.699 |
| D | v2 features (reference) | 0.750 | 0.752 | **0.754** |
| E | z-score wording | 0.729 | 0.756 | 0.759 |
| F | percentile values | 0.709 | 0.713 | 0.734 |
| C | Qwen2.5-7B | 0.708 | 0.705 | 0.713 |
| – | XGBoost, same 16 features | 0.799 | 0.823 | **0.805** |

- v1 → v2: +0.055 on test, paired interval [+0.013, +0.089].
- Stating that values are z-scores changed nothing; percentiles hurt, so the model uses the magnitude of the numbers.
- Qwen2.5-7B trained unstably (validation sensitivity 0.00, 0.01, 0.93, 0.04, 0.62 across epochs).
- XGBoost stays about 0.05 above the LLM.

## 5. Ra et al.'s settings, plain-language prompts, split CP (9 October)

| run | change | val | cal | test | minutes |
|---|---|---|---|---|---|
| D | reference (Qwen2.5-1.5B, decimal) | 0.750 | 0.752 | 0.754 | 26 |
| G | Llama-2-7B, Ra et al.'s settings | 0.705 | 0.709 | 0.731 | 182 |
| H | rich prompt (plain-language meaning added) | 0.732 | 0.711 | 0.744 | 83 |

- **Llama-2-7B with Ra et al.'s settings** (LR 1e-4, LoRA on all layers): below Qwen2.5-1.5B, seven times slower.
- **"Rich" prompts** (z-score + percentile + a meaning such as "lower than typical, meaning a less round outline"): no help.
- **Split conformal prediction** was added to every run. Mondrian CP (95% coverage of cancers) works as promised on average, but its auto-benign group still holds 10–17% cancers. Marginal coverage is carried by the benign majority.
- **An exploratory finding on LTT**: the planned fixed sequence starts at the strictest threshold, where only a handful of nodules fall, and stops at the first failure. Ordering candidates by their validation p-values (which keeps the guarantee) certified auto-malignant for one run. Flagged as exploratory, to be confirmed.

## 6. TI-RADS-motivated features (9–10 October)

**Idea.** PyRadiomics describes the nodule on its own. Several ACR TI-RADS signs depend on what it leaves out: the orientation of the nodule (taller-than-wide) and its contrast with the surrounding tissue.

**Background: what ACR TI-RADS is** (Tessler et al., 2017). Radiologists score a thyroid nodule on five ultrasound categories, add the points, and the total gives a risk level (TR1 benign … TR5 highly suspicious) that decides whether a biopsy (FNA) is recommended.

| TI-RADS category | what the radiologist looks at | our feature |
|---|---|---|
| composition | cystic or solid | anechoic fraction |
| echogenicity | darker or brighter than the surrounding thyroid | echogenicity ratio |
| shape | taller than wide | taller-than-wide ratio |
| margin | smooth, lobulated or irregular | solidity, margin sharpness |
| echogenic foci | bright spots, e.g. microcalcifications | punctate foci density |

Our features are automatic measurements of these signs; they do not reproduce the radiologist's score.

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

**What was done.** Six features were written from the image and mask, with settings fixed before looking at labels: taller-than-wide ratio, echogenicity ratio, punctate foci density, solidity, margin sharpness, anechoic fraction. Three behaved as TI-RADS predicts; the echogenicity ratio pointed the wrong way because it mostly measures fluid (correlation −0.78 with the anechoic fraction).

| our feature | TI-RADS sign | train AUC alone | direction |
|---|---|---|---|
| taller-than-wide ratio | shape | 0.658 | as expected |
| solidity | margin | 0.367 | as expected (lower = irregular = malignant) |
| anechoic fraction | composition | 0.409 | as expected (cystic = benign) |
| echogenicity ratio | echogenicity | 0.552 | reversed (measures fluid) |
| margin sharpness | margin | 0.563 | unclear |
| punctate foci density | echogenic foci | 0.487 | no signal |

**Results.**

| model | features | val | cal | test |
|---|---|---|---|---|
| XGBoost | radiomics 16 | 0.799 | 0.823 | 0.805 |
| XGBoost | radiomics 16 + 6 TI-RADS | 0.828 | 0.848 | **0.831** |
| XGBoost | 6 TI-RADS only | 0.786 | 0.787 | 0.775 |
| LLM (run D) | radiomics 16 | 0.750 | 0.752 | 0.754 |
| LLM (run I) | radiomics 16 + 6 TI-RADS | 0.795 | 0.812 | **0.804** |
| LLM (run J) | + 7th feature (solid-part echogenicity) | 0.786 | 0.802 | 0.781 |
| LLM (run K) | + 6 TI-RADS, rich format | 0.803 | 0.785 | 0.801 |

- The largest gain of any change: LLM +0.050 on test, paired interval [+0.015, +0.084].
- The seventh feature fixed the reversed direction but added nothing; the rich format equalled decimal at four times the training time (132 vs 36 minutes).

## 7. A second try with the larger model (11 October)

The 7B instability pointed to a too-high learning rate, so Qwen2.5-7B was retrained with LR 5e-5 and the TI-RADS features (run L).

| model | features | test AUC |
|---|---|---|
| Qwen2.5-1.5B (run I) | radiomics + TI-RADS | **0.804** |
| Qwen2.5-7B, LR 2e-4 (run C) | radiomics | 0.713 |
| Llama-2-7B, LR 1e-4 (run G) | radiomics | 0.731 |
| Qwen2.5-7B, LR 5e-5 (run L) | radiomics + TI-RADS | 0.693 |

Training was calmer, but the result far below 1.5B (interval [−0.148, −0.076]), and the TI-RADS features gave 7B nothing. Three large-model attempts point the same way: in this setup a larger backbone does not help.

**Limits of this comparison.** These are single-seed runs, and no learning-rate search was done for 7B. Run C used the learning rate chosen for 1.5B; run L used a four times lower one, and its validation AUC was still rising at epoch 5 (0.691 → 0.736 → 0.742 → 0.751 → 0.753), so it may have stopped before converging. Llama-2 (run G) had no TI-RADS features. The supported claim is therefore limited: *in the settings tried, a larger backbone did not help*; it is not shown that a larger backbone cannot help.

**Why Qwen2.5-1.5B stays the main model.**
1. In the one like-for-like comparison (same 16 features, learning rate, epochs and effective batch: run C vs run D), 7B is lower: 0.713 vs 0.754, paired interval [−0.069, −0.004].
2. All three large-model runs, under three different settings, land below the matching 1.5B run; none comes near 0.804.
3. The 1.5B result is stable and replicated over three seeds (0.785 ± 0.018); the 7B trainings were unstable (gradient norms up to 200).
4. Information ceiling: XGBoost on the same 22 numbers reaches 0.835 and the LLM adds nothing on top of it (stacking weight −0.04). A larger LLM could at best approach this ceiling; it cannot raise it.
5. Cost: three to seven times the training time.

## 8. Is the result real? Seeds, fusion, conformal checks (12 October)

**Seed repeats.** The best setting was retrained with seeds 1 and 2, and so was the setting without TI-RADS features.

| test AUC | seed 42 | seed 1 | seed 2 | mean ± SD |
|---|---|---|---|---|
| with TI-RADS features (runs I, M, N) | 0.804 | 0.768 | 0.783 | **0.785 ± 0.018** |
| without (runs D, O, P) | 0.754 | 0.712 | 0.713 | 0.726 ± 0.024 |
| paired gain | +0.050 | +0.056 | +0.070 | **+0.059** |
| 95% interval of the gain | [+0.015, +0.084] | [+0.013, +0.098] | [+0.026, +0.112] | |

Seed 42 was the lucky upper end on both sides. The training seed alone moves AUC by 0.03–0.04, as much as many single-run comparisons.

**Fusion with XGBoost.** Does the LLM know something XGBoost does not?

| model (same 22 features) | val | cal | test |
|---|---|---|---|
| XGBoost | 0.827 | 0.848 | **0.835** |
| LLM, average of 3 seeds | 0.792 | 0.803 | 0.793 |
| mean of the two probabilities | 0.820 | 0.838 | 0.824 |
| stacking (fitted on validation) | 0.827 | 0.848 | 0.834 |

The stacker's weights: XGBoost 0.99, LLM −0.04. The two models' scores correlate 0.87. **The LLM adds no information beyond XGBoost.**

**Conformal checks.**

**How to read the LTT tables.** In the auto-benign group an *error* is a nodule called benign that is in fact a cancer. The guarantee is about the **true rate**: the share of cancers among all future nodules the rule would call benign, which is never known. The calibration group only gives the **observed rate**, a sample of it. LTT asks: *if the true rate were 5%, how likely would it be to see as few cancers as we did?* If that chance is below δ = 5%, the true rate is certified to be below 5%.

**Example 1: no cancers in the group.** With a true rate of 5%, each nodule is benign with probability 0.95.

| group | chance that all are benign if the true rate were 5% | conclusion |
|---|---|---|
| 30 nodules, 0 cancers | 0.95³⁰ ≈ 21% | common; proves nothing |
| 59 nodules, 0 cancers | 0.95⁵⁹ ≈ 4.8% | below 5%: **certified** |

**Example 2: cancers in the group.** Each cancer raises the evidence needed:

| cancers in the group | smallest group that can be certified | observed rate then |
|---|---|---|
| 0 | 59 | 0% |
| 1 | 93 | 1.1% |
| 2 | 124 | 1.6% |
| 3 | 153 | 2.0% |

So the observed rate has to be about 0–2%, not just below 5%; the gap is the uncertainty of a small sample. On the auto-malignant side the target is 20%, so 14 clean nodules already suffice, which is why that side can pass.

**The cleanest tail.** Each model gives every one of the 524 calibration nodules a probability of cancer. Sorted from the lowest, the first 10% (52 nodules) are the ones the model is most sure are benign; a threshold there would make them auto-benign. Slices of 10–20% are shown because a 5% slice (26 nodules) cannot be certified even with no cancers (59 needed), and wider slices take in more cancers. The LTT itself does not use these slices; they illustrate why it fails.

**Applied.** XGBoost's cleanest 52 nodules hold 2 cancers: observed rate 3.8%, below 5%. But if the true rate were 5%, 2 or fewer cancers in 52 nodules would happen with probability ≈ 51%, so this is no evidence. Two cancers would need 124 nodules; widening the group to 104 brings in a third cancer, which needs 153. The LLM's tail (4 in 52, ≈ 8%) is above 5% even as observed.

**The three tables below.** The first is pure arithmetic from the binomial test: how large a group must be for a given number of errors; it is the same for any model. The second shows our models' real calibration numbers. The third is Mondrian split CP, a different guarantee: it promises that 95% of cancers are flagged (miss rate ≤ 5%). Its last column, cancers among the auto-benign, is R1, which it does not promise. The 20% target belongs only to the auto-malignant side (R2: benign nodules among those called malignant).

| smallest group LTT can certify (δ = 0.05) | 0 errors | 1 | 2 | 3 |
|---|---|---|---|---|
| auto-benign, α = 5% | 59 | 93 | 124 | 153 |
| auto-malignant, α = 20% | 14 | 22 | 30 | 37 |

| calibration tail (524 nodules) | lowest 10% | 15% | 20% | planned LTT |
|---|---|---|---|---|
| XGBoost | 2 cancers / 52 | 2 / 78 | 3 / 104 | nothing certified |
| LLM, 3-seed average | 4 / 52 | 7 / 78 | 10 / 104 | nothing certified |

| Mondrian split CP (95% of cancers covered) | coverage on our split | over 4,000 re-splits | auto-benign | cancers among them |
|---|---|---|---|---|
| XGBoost | 0.941 | 0.951 ± 0.024 | 25.7% | 6.4% |
| LLM, 3-seed average | 0.934 | 0.950 ± 0.024 | 16.7% | 9.8% |

- Even XGBoost, the strongest model, certifies nothing at 5%: 2 cancers among its cleanest 78 would need 124 nodules.
- Mondrian coverage of cancers was below 0.95 in 12 of 16 runs on our split, but averages 0.95 over re-splits: **the guarantee holds; our split was unlucky**. The 6.4% and 9.8% in the last column are R1, which Mondrian does not promise, and they are above 5%.

## 9. A design that can pass, and one more feature idea (13 October)

**Pre-registered LTT design.** Written and committed before it was run: candidate thresholds at quantiles of the validation scores (5%–30% of nodules per side), ordered by validation p-values, fixed-sequence test on calibration. Evaluated over 2,000 re-splits:

| share of re-splits certified | auto-benign 5% | auto-malignant 20%, planned design | auto-malignant 20%, new design | decided / benign among them |
|---|---|---|---|---|
| XGBoost | 0% | 0% | 38% | 10.7% / 14.9% |
| LLM, 3-seed average | 0% | 1% | **99%** | 7.9% / 7.4% |
| LLM run I | 0% | 13% | 22% | 11.5% / 18.4% |

Auto-malignant becomes attainable; auto-benign is never certified, for any model or design.

**Why the planned design failed.** LTT tests candidate thresholds in a fixed order and stops at the first failure; this keeps the guarantee. The planned order began at the safest auto-malignant threshold, u = 0.95, which almost no nodule exceeds. With fewer than 14 nodules nothing can be certified, so the test stopped at the first step and never reached u ≈ 0.80, where it would have passed: a staircase with a missing first step.

**The two changes.**

1. Candidates are defined by the share of nodules (the top 5%, 10%, …, 30% on validation) instead of by probability values, so no candidate is empty (5% of calibration ≈ 26 nodules).
2. The order is set on the validation split, most promising candidate first, and then tested on calibration.

**Worked example (LLM, 3-seed average, auto-malignant side).** On validation each candidate gets a p-value: *if the true error rate were 20%, how likely is it to see this few benign nodules?* Smaller means more promising. None passes on validation; it only sets the order. Calibration then tests in that order and stops at the first failure.

| candidate (threshold) | validation: nodules / benign / p | order | calibration: nodules / benign / p | result |
|---|---|---|---|---|
| top 5% (0.891) | 22 / 5 / 0.73 | 1 | 34 / 2 / 0.023 | passes |
| top 10% (0.833) | 44 / 12 / 0.91 | 2 | 50 / 4 / 0.019 | passes |
| top 20% (0.710) | 88 / 28 / 0.997 | 3 | 105 / 24 / 0.81 | fails: stop |
| top 15%, 25%, 30% | p 0.999–1.00 | 4–6 | – | not reached |

The most permissive certified candidate is used: threshold 0.833, the top 10%.

**Why the guarantee still holds.** Fixed-sequence testing only requires the order to be fixed without calibration data; validation is a separate, independent split. Every detail (candidates, order, α, δ) was committed to Git before the design was run.

**What "auto-malignant" means.** The triage puts each nodule in one of three boxes: *auto-benign* (very low probability; no biopsy), *referred* (in between; a clinician decides, most nodules land here) and *auto-malignant* (very high probability; the system calls it malignant on its own and flags it as high risk). The auto-malignant guarantee: at most 20% of these calls are in fact benign. On our split the certified threshold is 0.833: 9.8% of the 438 test nodules (about 43) become auto-malignant, and 14% of them (about 6) are benign, within the 20%. Over 2,000 re-splits about 8% are decided, with 7.4% benign among them.

**Why auto-benign stays at 0%.** There the problem is not the order but the tail itself (see the conformal checks above). **Why the LLM passes more often than XGBoost (99% vs 38%) despite a lower AUC:** AUC scores the whole ranking; this test looks only at the top 5–30%, where the 3-seed LLM average is cleaner. **Keep in mind:** the design was written after the main results; it is pre-registered but tested on the same data, so it is exploratory until confirmed on new data (e.g. ThyroidXL).

**Peritumoral radiomics.** PyRadiomics features in a ring around the nodule (93 features), with a rule fixed in advance: pass them to the LLM only if they improve XGBoost on calibration.

| XGBoost features | val | cal | test |
|---|---|---|---|
| nodule + TI-RADS | 0.828 | 0.848 | 0.831 |
| + 8 ring features | 0.832 | 0.851 | 0.834 |
| ring only | 0.699 | 0.671 | 0.693 |

Calibration +0.002 [−0.009, +0.014]: no gain, so no LLM run. The ring carries signal on its own but repeats what the nodule and TI-RADS features already contain.

## 10. Results at a glance

**Classification (test AUC, pooled split unless stated).**

| step | LLM (Qwen2.5-1.5B) | XGBoost |
|---|---|---|
| official split, v1, 8 features | 0.672 | 0.696 |
| pooled split, v1, 16 features | 0.699 | 0.723 |
| pooled split, v2, 16 features | 0.754 | 0.805 |
| + 6 TI-RADS-motivated features | **0.785 ± 0.018** (3 seeds) | **0.835** |
| + ring features | not run (no XGBoost gain) | 0.834 |
| larger backbones (7B) | 0.693–0.731 | – |
| LLM and XGBoost fused | 0.834 (= XGBoost) | |

**Conformal guarantees (best model setting, pooled split).**

| guarantee | attainable? | what it delivers |
|---|---|---|
| R1: ≤ 5% cancers among auto-benign (LTT) | **no**, for any model or design | – |
| R2: ≤ 20% benign among auto-malignant (LTT, new design) | **yes**, 99% of re-splits (LLM) | about 8% of nodules decided, 6–7% of them benign |
| ≤ 5% of all cancers missed (miss-rate LTT) | **yes** | 12% (LLM) or 24% (XGBoost) of test nodules auto-benign, 4.6% of cancers missed |
| 95% of cancers covered (Mondrian split CP) | **yes**, 0.95 on average | 17% (LLM) or 26% (XGBoost) auto-benign, with about 10% / 6% cancers among them |

## 11. Where the work stands

| research question | answer |
|---|---|
| RQ1: can an LLM classify nodules from radiomics prompts, and how does it compare with a classical model? | Yes, moderately: test AUC 0.785 ± 0.018 with the best features. About 0.05 below XGBoost on the same features, and it adds no information to XGBoost. |
| RQ2: which pipeline choices matter? | Feature content matters (v2 +0.05, TI-RADS-motivated +0.06). Prompt wording, prompt format, model size and peritumoral features do not. |
| RQ3: can LTT certify automatic decisions, and do perturbation votes help? | Votes do not help. Auto-benign at 5% is out of reach of every model tried, including XGBoost. Auto-malignant at 20% becomes attainable with a pre-registered design. Guarantees on missed cancers hold but leave about 10% cancers among automatic benign calls. |

**Main message.** *Which guarantee is attainable, when, and what it is worth.* The LTT layer behaved honestly: it refused guarantees the evidence did not support. With 524 calibration nodules and a 35% cancer rate, a 5% guarantee on each benign call needs a model with AUC around 0.9; a guarantee on missed cancers is attainable now but leaves a group at TI-RADS TR4-like risk, where biopsy is still recommended. A valid guarantee is not the same as a safe decision.

## 12. Methodological lessons

- **Check exchangeability before trusting a guarantee.** The official split broke it; pooling fixed it.
- **Audit the features.** The v2 re-extraction was worth as much as any modelling change.
- **Repeat over seeds.** Single runs moved by 0.03–0.05; the main comparison is reported over matched seeds.
- **Compare against a properly evaluated tabular model, and test for added information.** Fusion answers "does the LLM add anything?" more directly than an AUC difference.
- **Fix decision rules before running.** The LTT design and the peritumoral experiment were both decided by rules written in advance.
- **New information, not new wording or size, moved the score.**

## 13. Limitations

One dataset; image-level split without patient IDs; single 2D images with expert masks; heuristic TI-RADS-motivated features; untuned LLM hyperparameters; one calibration set of 524 nodules; several design choices informed by calibration or test results (stated where they occur); bootstrap intervals that do not include training noise or patient clustering; a referral-population cancer rate (35%) that does not transfer to screening.

## 14. Options for the next phase (for discussion)

1. Confirm the pre-registered LTT design on new data: retrain on new random splits, or apply it to a second dataset (ThyroidXL).
2. Use the image itself (a CNN or a vision-language model): the only realistic route towards a model good enough for the auto-benign guarantee, but a change of scope.
3. Write up: the experimental questions about the classifier are answered.
