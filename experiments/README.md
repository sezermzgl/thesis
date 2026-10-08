# Backbone sweep

`model_sweep.py` re-runs the notebook pipeline (Part A prompts, Part B QLoRA fine-tuning) with a different LLM backbone. It then reports classification metrics, the softmax LTT result and the exploratory miss-rate guarantee (report Section 10.3) on the same splits. Each call fine-tunes one model, so long runs can be spread over several sessions.

## Candidate backbones

All of these have a sequence-classification head in `transformers` and fit a 16 GB T4 with 4-bit QLoRA. Run times are rough extrapolations from the measured 30 minutes for Qwen2.5-1.5B (5 epochs); they have not been measured.

| model id | size | access / license | suggested flags | rough T4 time |
|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 1.5B | open, Apache-2.0 | (defaults) | ~30 min, measured |
| `Qwen/Qwen3-1.7B-Base` | 1.7B | open, Apache-2.0 | (defaults) | ~35 min |
| `Qwen/Qwen2.5-3B` | 3B | open, Qwen Research License | (defaults) | ~1 h |
| `meta-llama/Llama-3.2-3B` | 3B | gated, Llama 3.2 license | (defaults) | ~1 h |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | open, MIT | `--batch 4 --grad-accum 4` | ~1.5 h |
| `Qwen/Qwen3-4B-Base` | 4B | open, Apache-2.0 | `--batch 4 --grad-accum 4` | ~1.5 h |
| `Qwen/Qwen2.5-7B` | 7B | open, Apache-2.0 | `--batch 4 --grad-accum 4` | ~3 h |
| `meta-llama/Llama-2-7b-hf` | 7B | gated, Llama 2 license; the model used by Ra et al. | `--batch 4 --grad-accum 4` | ~3 h |
| `mistralai/Mistral-7B-v0.3` | 7B | Apache-2.0 (may ask to accept terms) | `--batch 4 --grad-accum 4` | ~3 h |
| `meta-llama/Llama-3.1-8B` | 8B | gated, Llama 3.1 license | `--batch 4 --grad-accum 4` | ~3.5 h |

Gated models need the license to be accepted on the model page with your own Hugging Face account, plus a login in the session (`from huggingface_hub import login; login()`).

**Run `Qwen/Qwen2.5-1.5B` first.** It is the control. The notebook's committed adapter was trained before the gradient-accumulation fix, so comparisons should use models trained by this same script.

## Colab

**Step-by-step notebook (recommended):** [`model_sweep_colab.ipynb`](model_sweep_colab.ipynb). Every step is a visible cell with its own output. Open it directly in Colab:
https://colab.research.google.com/github/sezermzgl/thesis/blob/main/experiments/model_sweep_colab.ipynb

**Script version** (the same steps in one command):

```
!git clone https://github.com/sezermzgl/thesis.git
%cd /content/thesis
!pip install -q -U bitsandbytes peft accelerate mrmr-selection
from google.colab import drive; drive.mount("/content/drive")
!python experiments/model_sweep.py --model Qwen/Qwen2.5-1.5B --out /content/drive/MyDrive/thesis_model_sweep
!python experiments/model_sweep.py --model Qwen/Qwen2.5-3B --out /content/drive/MyDrive/thesis_model_sweep
```

Writing to Google Drive keeps the results when the Colab session ends.

**Colab Pro.** Choose the GPU under Runtime → Change runtime type. L4 is the cheaper default for 7–8B models; A100 is faster but uses compute units roughly 2–3× faster. Precision is picked automatically: bf16 on A100/L4 (more stable for large models), fp16 on T4. `--precision fp16` forces fp16, for example to match earlier T4 runs. When the runs finish, use Runtime → Disconnect and delete runtime, because an idle GPU still uses compute units. Each run creates `<out>/<model>_f<features>_e<epochs>_s<seed>/`:

| file | contents |
|---|---|
| `config.json` | arguments and the selected features |
| `train_log.csv` | training log |
| `probs_{validation,calibration,test}.csv` | per-nodule P(malignant) |
| `metrics.json` | all metrics |
| `adapter/` | the LoRA adapter |

Every run also appends one row to `<out>/summary.csv`.

`--n-features 16 --max-length 512` tries more features instead of a larger model. The script stops before training if any prompt would be truncated. `--dry-run` checks data preparation and prompts without a GPU.

## Metrics in `summary.csv`

- `gpu`, `precision`, `minutes`: where and how the run was trained. Compare models trained with the same precision where possible.
- `{split}_acc/_f1/_auc/_sens`: classification at the 0.5 threshold.
- `{split}_auc_correct_softmax`: how well softmax confidence separates correct from incorrect predictions (report Section 10.2).
- `ltt_t`, `ltt_u`: thresholds certified by LTT on calibration (R1 ≤ 5%, R2 ≤ 20%, δ = 0.05, notebook D6 grids); empty means nothing was certified. `ltt_test_auto_*` gives the resulting test counts.
- `miss5_*`, `miss10_*`: miss-rate guarantee at 5% and 10% with a grid fixed in advance. `_t` is the certified threshold; `_cal_automation` and `_test_automation` are the shares sent to auto-benign; `_test_miss_rate` is the realised share of test cancers sent to auto-benign. Calibration and test are not exchangeable (report Section 5), so test rates can exceed the target.
