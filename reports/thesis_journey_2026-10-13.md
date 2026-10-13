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

**LTT certified nothing**, with votes or with softmax. Split conformal prediction met its coverage on calibration but its automatic benign calls on test contained 23% cancers.

**A data problem surfaced.** A classifier could tell calibration nodules from official-test nodules (two-sample AUC 0.63, p = 0.002). Conformal guarantees require calibration and test data to be exchangeable; on this split they are not.

**Conclusions that set the direction.**
1. The limit is the base model, not the triage method: errors are stable, so no consistency signal can flag them.
2. The softmax probability is as good a score as any vote; the perturbation machinery was dropped.
3. The split has to be fixed before any guarantee can be tested.

## 2. Error analysis and a weaker guarantee (7 October)

Without new training, the run-2 outputs were re-analysed:

- the errors that block LTT come from the base prompt itself; the perturbations neither create nor reveal them;
- **a different guarantee was tried: the miss rate**, the share of *all* cancers sent to auto-benign (≤ 5%), tested with the same binomial LTT test. It can be certified, but only for about 5% of the nodules, and at 10% it was violated on the official test (15% of cancers missed), a concrete effect of the non-exchangeable split.

This introduced the distinction that runs through the rest of the work: a guarantee on **missed cancers** (programme level) versus a guarantee on **each benign call** (patient level, R1).

## 3. Backbone, benchmark, split and data audit (8 October)

**Larger backbone (official split).** Qwen2.5-7B gave the same test AUC as 1.5B (0.669 vs 0.667); a retrained 1.5B matched the first model (Spearman 0.97). Size did not look like the bottleneck.

**Classical benchmark.** XGBoost on the same features matched or beat the LLM (0.696 vs 0.667 with 8 features; 0.709 with 16). From here on XGBoost is the reference.

**Pooled, exchangeable split.** All 3,493 images were pooled and re-split, stratified by label and source: train 2,095 / validation 436 / calibration 524 / test 438. The two-sample test dropped to AUC 0.43. With this split, more features started to help on test as well (XGBoost with all 67 features: 0.792), which showed that the earlier "more features do not help" was an artefact of the shift.

**Image quality audit.** Four problems in the original feature extraction:
- texture computed with about 9 grey levels inside a nodule (bin width 25);
- shape features in pixels on images of different sizes (official-test images are larger, which explained much of the "shift");
- 298 masks with several components;
- 356 masks touching the image border.

**Caveat recorded.** Patient IDs are not released, so the pooled split is at image level; images of one patient can fall into different splits. This inflates absolute AUCs for all models alike and is reported as a limitation.

## 4. Revised features and the first pooled-split LLM runs (8–9 October)

**v2 extraction.** 64 grey levels, size features relative to the image, seven feature families (102 features), largest mask component only. On single-component masks it reproduces v1 exactly.

**Two fixes found along the way.**
- mRMR kept selecting near-identical features (pixel surface and mesh surface are the same number in 2D), so a near-duplicate filter (|r| > 0.95, train only) was added before mRMR.
- Predictions were computed under bfloat16 autocast, which rounded the probabilities (195 distinct values on 524 nodules); prediction was moved to full precision.

**Results (runs A–F, test AUC).**
- v1 → v2 features: 0.699 → 0.754, a significant gain (+0.05).
- Prompt format: stating that values are z-scores changed nothing (0.759); replacing them by percentiles hurt (0.734), so the model uses the magnitude of the numbers.
- Qwen2.5-7B: 0.713, unstable training (sensitivity swinging between 0 and 0.93 across epochs).
- XGBoost with the same 16 v2 features: 0.805, still about 0.05 above the LLM.

## 5. Ra et al.'s settings, plain-language prompts, split CP (9 October)

- **Llama-2-7B with Ra et al.'s settings** (run G: LR 1e-4, LoRA on all layers): 0.731, below Qwen2.5-1.5B, 182 minutes per run.
- **"Rich" prompts** (z-score + percentile + a plain-language meaning such as "lower than typical, meaning a less round outline"; run H): 0.744, no help.
- **Split conformal prediction** was added to every run. Mondrian CP (95% coverage of cancers) works as promised on average, but its auto-benign group still holds 10–17% cancers. Marginal coverage is carried by the benign majority.
- **An exploratory finding on LTT**: the planned fixed sequence starts at the strictest threshold, where only a handful of nodules fall, and stops at the first failure. Ordering candidates by their validation p-values (which keeps the guarantee) certified auto-malignant for one run. Flagged as exploratory, to be confirmed.

## 6. TI-RADS-motivated features (9–10 October)

**Idea.** PyRadiomics describes the nodule on its own. Several ACR TI-RADS signs depend on what it leaves out: the orientation of the nodule (taller-than-wide) and its contrast with the surrounding tissue.

**What was done.** Six features were written from the image and mask, with settings fixed before looking at labels: taller-than-wide ratio, echogenicity ratio, punctate foci density, solidity, margin sharpness, anechoic fraction. Three behaved as TI-RADS predicts; the echogenicity ratio pointed the wrong way because it mostly measures fluid (correlation −0.78 with the anechoic fraction).

**Results.**
- XGBoost: 0.805 → 0.831.
- LLM (run I): 0.754 → 0.804, the largest gain of any change.
- A seventh feature computed on the solid part only (run J) fixed the reversed direction but added nothing.
- The rich format with these features (run K) equalled decimal, at four times the training time.

## 7. A fair test of the larger model (11 October)

The 7B instability pointed to a too-high learning rate, so Qwen2.5-7B was retrained with LR 5e-5 and the TI-RADS features (run L). Training was calmer, but test AUC was 0.693, far below 1.5B, and the TI-RADS features gave 7B nothing. Three large-model attempts now agree: in this setup a larger backbone does not help.

## 8. Is the result real? Seeds, fusion, conformal checks (12 October)

**Seed repeats.** The best setting was retrained with seeds 1 and 2, and so was the setting without TI-RADS features.
- With TI-RADS: 0.804, 0.768, 0.783 → **0.785 ± 0.018**. Seed 42 was the lucky upper end.
- Without: 0.754, 0.712, 0.713 → 0.726 ± 0.024.
- **Paired gain +0.059** on test, positive for every seed (smallest +0.050), every matched interval above zero.
- Lesson: the training seed alone moves AUC by 0.03–0.04, as much as many single-run comparisons.

**Fusion with XGBoost.** Does the LLM know something XGBoost does not? XGBoost on exactly the same 22 features: 0.835. Combining both models: a stacker fitted on validation gave the LLM a weight of −0.04 and XGBoost 0.99; the two models' scores correlate 0.87. **The LLM adds no information beyond XGBoost.**

**Conformal checks.**
- LTT on XGBoost, the strongest model: **still nothing certified**. Its cleanest 15% holds 2 cancers in 78; certification at 5% would need 124 nodules with 2 errors.
- Mondrian coverage of cancers had fallen below 0.95 in 12 of 16 runs on our split. Re-splitting calibration and test 4,000 times with the models fixed gave an average coverage of 0.950–0.951 for every model: **the guarantee holds; our split was unlucky**.
- Expected share of cancers among Mondrian's automatic benign calls: about 10% for the LLM, 6% for XGBoost.

## 9. A design that can pass, and one more feature idea (13 October)

**Pre-registered LTT design.** Written and committed before it was run: candidate thresholds at quantiles of the validation scores (5%–30% of nodules per side), ordered by validation p-values, fixed-sequence test on calibration. Evaluated over 2,000 re-splits:
- auto-malignant at 20%: certified in **99%** of re-splits for the seed-averaged LLM (about 8% of nodules decided, 6–7% of them benign), against 1% with the planned design;
- auto-benign at 5%: **never certified**, for any model or design.

**Peritumoral radiomics.** PyRadiomics features in a ring around the nodule (93 features), with a rule fixed in advance: pass them to the LLM only if they improve XGBoost on calibration. Calibration +0.002 [−0.009, +0.014]: no gain, so no LLM run. The ring carries signal on its own (0.69) but repeats what the nodule and TI-RADS features already contain.

## 10. Where the work stands

| research question | answer |
|---|---|
| RQ1: can an LLM classify nodules from radiomics prompts, and how does it compare with a classical model? | Yes, moderately: test AUC 0.785 ± 0.018 with the best features. About 0.05 below XGBoost on the same features, and it adds no information to XGBoost. |
| RQ2: which pipeline choices matter? | Feature content matters (v2 +0.05, TI-RADS-motivated +0.06). Prompt wording, prompt format, model size and peritumoral features do not. |
| RQ3: can LTT certify automatic decisions, and do perturbation votes help? | Votes do not help. Auto-benign at 5% is out of reach of every model tried, including XGBoost. Auto-malignant at 20% becomes attainable with a pre-registered design. Guarantees on missed cancers hold but leave about 10% cancers among automatic benign calls. |

**Main message.** *Which guarantee is attainable, when, and what it is worth.* The LTT layer behaved honestly: it refused guarantees the evidence did not support. With 524 calibration nodules and a 35% cancer rate, a 5% guarantee on each benign call needs a model with AUC around 0.9; a guarantee on missed cancers is attainable now but leaves a group at TI-RADS TR4-like risk, where biopsy is still recommended. A valid guarantee is not the same as a safe decision.

## 11. Methodological lessons

- **Check exchangeability before trusting a guarantee.** The official split broke it; pooling fixed it.
- **Audit the features.** The v2 re-extraction was worth as much as any modelling change.
- **Repeat over seeds.** Single runs moved by 0.03–0.05; the main comparison is reported over matched seeds.
- **Compare against a properly evaluated tabular model, and test for added information.** Fusion answers "does the LLM add anything?" more directly than an AUC difference.
- **Fix decision rules before running.** The LTT design and the peritumoral experiment were both decided by rules written in advance.
- **New information, not new wording or size, moved the score.**

## 12. Limitations

One dataset; image-level split without patient IDs; single 2D images with expert masks; heuristic TI-RADS-motivated features; untuned LLM hyperparameters; one calibration set of 524 nodules; several design choices informed by calibration or test results (stated where they occur); bootstrap intervals that do not include training noise or patient clustering; a referral-population cancer rate (35%) that does not transfer to screening.

## 13. Options for the next phase (for discussion)

1. Confirm the pre-registered LTT design on new data: retrain on new random splits, or apply it to a second dataset (ThyroidXL).
2. Use the image itself (a CNN or a vision-language model): the only realistic route towards a model good enough for the auto-benign guarantee, but a change of scope.
3. Write up: the experimental questions about the classifier are answered.
