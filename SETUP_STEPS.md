# Repo setup — step by step

## 1. Create the repo structure locally

Put these at the repo root:

```
TN3K_pipeline_end_to_end.ipynb
README.md
requirements.txt
.gitignore
features/
    radiomics_train.csv
    radiomics_validation.csv
    radiomics_calibration.csv
    radiomics_test.csv
```

The four `radiomics_*.csv` are the ones you already have in Colab at `/content/features/`.
Download them from the Colab Files panel (or the cell below) and place them in `features/`.

## 2. (Optional) grab the extra deterministic files from Colab

If you also want the run to be fully deterministic without re-doing mRMR/prompts/training,
copy these from `/content/tn3k_gt_radiomics/` in Colab into the repo, preserving folders:

```
features/selected_standardized_{train,validation,calibration,test}.csv
prompts/llm_prompts_{train,validation,calibration,test}.csv
artifacts/radiomics_preprocessing.joblib
artifacts/selected_features_mrmr.json
classification/comparison_table_seed42_qwen15b.csv
classification/lora_adapter_seed42_qwen15b/      # only if you want to skip LLM training too
```

A quick way to collect everything into one downloadable zip, run in Colab:

```python
import shutil, os
root = "/content/tn3k_gt_radiomics"
os.makedirs("/content/repo_export/features", exist_ok=True)
# always: the 4 radiomics CSVs
for s in ["train","validation","calibration","test"]:
    shutil.copy(f"/content/features/radiomics_{s}.csv", f"/content/repo_export/features/")
# optional deterministic extras (ignore errors if a folder is absent)
for sub in ["features","prompts","artifacts","classification"]:
    src = os.path.join(root, sub)
    if os.path.isdir(src):
        shutil.copytree(src, f"/content/repo_export/{sub}", dirs_exist_ok=True)
# drop the big checkpoint dir if present
shutil.rmtree("/content/repo_export/classification/checkpoint", ignore_errors=True)
shutil.make_archive("/content/repo_export", "zip", "/content/repo_export")
print("Download /content/repo_export.zip from the Files panel")
```

## 3. Initialize git and push

```bash
cd your-repo-folder
git init
git add TN3K_pipeline_end_to_end.ipynb README.md requirements.txt .gitignore features/
# optional extras if you copied them:
# git add prompts/ artifacts/ classification/comparison_table_seed42_qwen15b.csv
git commit -m "TN3K radiomics->LLM pipeline (Ra et al.): end-to-end notebook + cached radiomics CSVs + results"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

## 4. Verify a clean clone runs

On a fresh Colab:
```bash
!git clone https://github.com/<you>/<repo>.git
%cd <repo>
```
Open the notebook, set `DATA_SOURCE = "repo"`, run the install cell, **Restart session**,
then **Run all**. The Configuration/CACHE_MODE cell should print
`Found cached radiomics CSVs in: .../features` and `CACHE_MODE = True`.

## What NOT to commit

- TN3K images/masks (licensing + size) — kept in Drive/local only.
- `classification/checkpoint/` (large training intermediates).
- Hugging Face caches / base model weights.

These are already covered by `.gitignore`.
