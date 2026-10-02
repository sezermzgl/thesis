# TN3K Radiomics → LLM Classification (Ra et al. pipeline)

Reproducible implementation of the pipeline from **Ra et al. (2025)**,
*"Enhancing radiomics features via a large language model for classifying benign and malignant
tumors in mammography"*, applied to the **TN3K thyroid ultrasound** dataset.

Ground-truth masks → PyRadiomics features → mRMR (8 features) → feature-name + value prompts →
LLM backbone + linear classification head + LoRA fine-tuning → benign/malignant classification,
compared against RF / SVM / LR baselines.

## Repository layout

```
.
├── TN3K_pipeline_end_to_end.ipynb   # the whole pipeline (Parts A, B, C)
├── README.md
├── requirements.txt
├── .gitignore
├── conformal_triage/                # Part D: perturbed prompts, votes, LTT, split CP, reports
├── tests/                           # unit tests for conformal_triage (python -m unittest discover tests)
├── runs/                            # Part D run records (results + per-query probabilities)
├── tn3k_gt_radiomics/classification/lora_adapter_seed42_qwen15b/   # fine-tuned adapter (B5 skips training)
└── features/                        # cached radiomics tables (committed so extraction is skipped)
    ├── radiomics_train.csv
    ├── radiomics_validation.csv
    ├── radiomics_calibration.csv
    └── radiomics_test.csv
```

## How to run

### Option A — Colab, manual upload (what most people will do)

This is the simplest path and needs **no Google Drive and no images** — only the four
`radiomics_*.csv` files.

1. Open `TN3K_pipeline_end_to_end.ipynb` in Google Colab (Upload notebook, or
   File → Open notebook → GitHub).
2. In the Colab **Files** panel (left sidebar), create a folder named `features` under
   `/content/`, i.e. `/content/features/`.
3. Upload the four CSVs into that folder:
   `radiomics_train.csv`, `radiomics_validation.csv`, `radiomics_calibration.csv`,
   `radiomics_test.csv`.
4. In the **Configuration** cell set `DATA_SOURCE = "local"`.
5. Run the install cell, then **Runtime → Restart session**, then **Run all**.

The notebook detects the CSVs in `/content/features/`, enters **CACHE_MODE**, and skips the
image-dependent steps (manifest, QC, split, PyRadiomics extraction). It runs straight through
mRMR → prompts → LLM fine-tuning → evaluation. You should see a line like
`Found cached radiomics CSVs in: /content/features` and `CACHE_MODE = True`.

> Colab clears `/content/` when the session ends. After a **Restart session**, check that
> `/content/features/` still holds the four CSVs (run `import os; print(os.listdir('/content/features'))`);
> if empty, re-upload them before **Run all**.

### Option B — Colab via git clone

1. In a Colab cell: `!git clone https://github.com/sezermzgl/thesis.git then `%cd <repo>`.
2. Open the notebook, set `DATA_SOURCE = "repo"`. Part D imports `conformal_triage/` from the
   cloned repo; with Option A, upload that folder to `/content/` as well.
3. Run install → **Restart session** → **Run all**.

Because `features/` is committed at the repo root, CACHE_MODE triggers automatically.

### Option C — full pipeline from images



Provide the TN3K images/masks (Google Drive or a local folder), set `DATA_SOURCE` to `"drive"`
or `"local"`, and run all. Extraction runs and produces the same `features/` CSVs.

> After changing package versions in the install cell, always **Restart session** before running
> the rest — pip updates the disk but the kernel keeps the old versions until a restart.

## Pipeline (matches the paper stage by stage)

- **Part A – Data preparation:** manifest + QC, disjoint train/validation/calibration/test splits
  (with leakage checks), PyRadiomics extraction, train-only z-score standardization, mRMR
  selection of 8 features, and construction of the feature-name + value prompts.
- **Part B – LLM classification:** LLM backbone + linear head, adapted with LoRA (rank 16, Q/V
  attention), class-weighted cross-entropy, evaluated on the internal test set.
- **Part C – Baselines:** RF / SVM / LR on the same 8 standardized features, reported alongside
  the LLM.
- **Part D – Conformal triage:** each validation/calibration/test nodule is queried several
  times in the training prompt format: the base prompt, Gaussian noise added to the standardized
  values (jitter, several noise levels; one is picked on validation), and one prompt per removed
  feature (drop). The vote count `v` drives a three-way rule: `v <= t` auto-benign, `v >= u`
  auto-malignant, otherwise refer. `t` and `u` are chosen on the calibration split with
  Learn-then-Test so that the share of cancers among auto-benign nodules and the share of benign
  nodules among auto-malignant nodules are bounded with high probability. Part D also runs a
  calibration-vs-test exchangeability check, classic split CP (marginal and Mondrian), a
  softmax-only baseline, and zips its outputs for download (D9).

## Current results (internal test set, n = 614)

| Model | Acc | F1 | AUC | Sens |
|---|---|---|---|---|
| RF | 0.651 | 0.415 | 0.663 | 0.322 |
| SVM | 0.664 | 0.411 | 0.648 | 0.305 |
| LR | 0.629 | 0.546 | 0.652 | 0.581 |
| **Ours (LLM+LoRA)** | 0.656 | **0.574** | **0.675** | **0.602** |

Consistent with the paper's trend: the LLM method is the most balanced, with the best F1 and AUC
and a much higher sensitivity than RF/SVM (which reach comparable accuracy only by defaulting to
the majority benign class). Selected features match the paper's set (major axis length,
elongation, maximum diameter, high/large gray-level zone emphasis, cluster shade, range, maximum).

## Limitations

- The paper used **LLaMA2-7B, 500 epochs, on a 48 GB GPU**. This runs a **~1.5B model
  (Qwen2.5-1.5B) with 4-bit QLoRA and few epochs** to fit a free Colab T4, so absolute scores are
  **not directly comparable**. To scale toward the paper: set `MODEL_NAME` to a 7B model, raise
  `NUM_EPOCHS`, and use a larger GPU.
- Part D assumes calibration and test nodules are exchangeable. Calibration comes from the official
  trainval split and test from the official test split, and in the first run a two-sample test
  separated them clearly (AUC 0.62, p < 0.01; test nodules are larger). Test-split rates
  therefore describe behaviour under shift rather than testing the guarantee. See
  `runs/2026-10-02_part_d_m10/RESULTS.md`.
- The committed adapter (trained 2026-10-02, test AUC 0.672) is loaded automatically in
  `repo` mode, so B5 skips training. It was trained before the gradient-accumulation fix in B5.
  Set `USE_CACHE = False` to retrain with the fixed loss.
- Part D adds 36 forward passes per nodule on three splits (about 53k, roughly 25 minutes on a T4).
  The per-query probabilities are cached in `tn3k_gt_radiomics/conformal/perturbed_probs_*.csv`,
  so the analysis cells can be re-run without the GPU. In Colab, `/content` is wiped when the
  session ends, so download the zip from D9.

## Reproducibility

`SEED = 42` throughout (numpy, torch, splits, mRMR, baselines). The imputer, scaler and mRMR
selection are fit on the **train split only**. Small run-to-run variation in the LLM scores is
expected from stochastic fine-tuning.

## Environment

See `requirements.txt`. Note: on recent Colab runtimes, `bitsandbytes>=0.46.1` is required for
4-bit; the install cell handles this.
