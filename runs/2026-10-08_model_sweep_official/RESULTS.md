# Backbone sweep on the official splits: 2026-10-08

**Notebook:** `experiments/model_sweep_colab.ipynb` (Colab L4, bf16), v1 features, official splits (calibration from trainval, test = official test, which is not exchangeable with calibration; see report Section 5). Settings: 5 epochs, LR 2e-4, LoRA rank 16 on Q/V, seed 42.

**Files:** one folder per run (`config.json`, `metrics.json`, `train_log.csv`, `probs_{validation,calibration,test}.csv`), `summary.csv`, and the executed notebooks in `notebooks/`. Adapters are on Drive.

| run | features | val AUC | cal AUC | test AUC | minutes |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 8 | 0.697 | 0.714 | 0.667 | 16 |
| Qwen2.5-7B | 8 | 0.697 | 0.686 | 0.669 | 57 |
| Qwen2.5-7B | 16 | 0.702 | 0.715 | 0.666 | 85 |

The retrained 1.5B model matches the original main-notebook model (test 0.672; Spearman 0.97 between their scores). The 7B model is not better, and LTT certifies nothing in any run. These runs predicted under the Trainer's bf16 autocast, so their probabilities are rounded to steps of 1/64 in logit space (fixed in commit `6f202b8`).
