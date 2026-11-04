"""Ablation on the language model: the same chain as ablation_xgboost.py, from finished Colab runs.

Cells (Qwen2.5-1.5B, 16 mRMR features, |r| > 0.95 filter, decimal prompt, 5 epochs, seeds 42, 1, 2):

    official v1            official TN3K split, original features
    pooled v1              pooled split, original features
    pooled v2              pooled split, re-extracted features
    pooled v2 + TI-RADS    pooled split, re-extracted features + six TI-RADS-motivated features

Each change is read in two ways: the difference within each seed, and a paired bootstrap on the
seed-averaged probabilities (same test nodules, so only for the pooled cells). The official and
pooled test sets differ, so the split change is shown through the validation -> test gap instead.

    python experiments/ablation_llm.py > runs/2026-10-08_v2_pipeline/ablation_llm.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from hybrid_xgb_llm import paired_bootstrap  # noqa: E402

RUNS = REPO / "runs" / "2026-10-08_v2_pipeline"
SEEDS = [42, 1, 2]
CELLS = {
    "official v1": "Qwen2.5-1.5B_f16_e5_s{seed}_corr95",
    "pooled v1": "Qwen2.5-1.5B_f16_e5_s{seed}_pooled_seed42_corr95",
    "pooled v2": "Qwen2.5-1.5B_f16_e5_s{seed}_pooled_seed42_v2_corr95",
    "pooled v2 + TI-RADS": "Qwen2.5-1.5B_f16_e5_s{seed}_pooled_seed42_v2_corr95_clin6",
}
# run I (seed 42) was trained before the folder name carried the number of TI-RADS features
RENAMED = {("pooled v2 + TI-RADS", 42): "Qwen2.5-1.5B_f16_e5_s42_pooled_seed42_v2_corr95_clin"}
SETS = ["validation", "calibration", "test"]


def run_dir(cell, seed):
    return RUNS / RENAMED.get((cell, seed), CELLS[cell].format(seed=seed))


def probs(cell, seed, split):
    return pd.read_csv(run_dir(cell, seed) / f"probs_{split}.csv").sort_values("sample_id", ignore_index=True)


def main():
    rows = []
    for cell in CELLS:
        for seed in SEEDS:
            m = json.load(open(run_dir(cell, seed) / "metrics.json"))
            rows.append({"cell": cell, "seed": seed, **{s: m[f"{s}_auc"] for s in SETS}})
    df = pd.DataFrame(rows)

    print("AUC per cell, mean ± SD over seeds 42, 1, 2\n")
    g = df.groupby("cell", sort=False)
    print(pd.DataFrame({s: g[s].mean().map("{:.3f}".format) + " ± " + g[s].std().map("{:.3f}".format)
                        for s in SETS}).to_string(), "\n")

    print("One change at a time on the pooled split (same test nodules)")
    print("  within seed: test AUC difference per seed; bootstrap: seed-averaged probabilities, paired, 95%")
    for name, a, b in [("v1 -> v2", "pooled v1", "pooled v2"),
                       ("TI-RADS added to v2", "pooled v2", "pooled v2 + TI-RADS"),
                       ("both", "pooled v1", "pooled v2 + TI-RADS")]:
        per_seed = df[df.cell == b]["test"].to_numpy() - df[df.cell == a]["test"].to_numpy()
        pa = [probs(a, s, "test") for s in SEEDS]
        pb = [probs(b, s, "test") for s in SEEDS]
        assert all((x["sample_id"] == pa[0]["sample_id"]).all() for x in pa + pb), "different test nodules"
        y = pa[0]["label"].to_numpy()
        mean_a = np.mean([x["prob_malignant"] for x in pa], axis=0)
        mean_b = np.mean([x["prob_malignant"] for x in pb], axis=0)
        d, lo, hi = paired_bootstrap(y, mean_b, mean_a)
        print(f"  {name}: per seed {', '.join(f'{x:+.3f}' for x in per_seed)} (mean {per_seed.mean():+.3f}); "
              f"seed-averaged {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nSplit change (v1 features): the test sets differ, so compare the validation -> test gap")
    for cell in ["official v1", "pooled v1"]:
        c = df[df.cell == cell]
        gap = c["test"] - c["validation"]
        print(f"  {cell}: validation {c['validation'].mean():.3f}, test {c['test'].mean():.3f}, "
              f"gap {gap.mean():+.3f} (per seed {', '.join(f'{x:+.3f}' for x in gap)})")

    print("\nSeed-averaged probabilities, test AUC")
    for cell in CELLS:
        p = [probs(cell, s, "test") for s in SEEDS]
        print(f"  {cell}: {roc_auc_score(p[0]['label'], np.mean([x['prob_malignant'] for x in p], axis=0)):.3f}")


if __name__ == "__main__":
    main()
