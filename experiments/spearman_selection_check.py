"""Robustness check: does feature selection depend on Pearson vs Spearman correlation?

Both places that use a correlation between features are switched together:

    near-duplicate filter  |r| > 0.95 between features (train only)
    mRMR redundancy        mean |r| between a candidate and the features already chosen

mRMR relevance (F-statistic) is the same in both. Everything else follows Part C of the v2
notebook: train-only median imputation and z-scoring, TI-RADS features never compete with the
radiomics features, 16 radiomics features selected. The Pearson selection must reproduce the
features of run I; the script stops if it does not.

Each selection, plus the six TI-RADS features, is scored with the untuned XGBoost of Part B.
Rule fixed before running: an LLM run with the Spearman features is worth doing only if the
calibration AUC gain over Pearson has a 95% paired bootstrap interval above zero.

    python experiments/spearman_selection_check.py --features radiomics_v2_clinical.csv \
        > runs/2026-10-08_v2_pipeline/spearman_selection_check.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
import conformal_triage as ct  # noqa: E402
from hybrid_xgb_llm import EVAL, paired_bootstrap, xgboost_probs  # noqa: E402

RUN_I = REPO / "runs" / "2026-10-08_v2_pipeline" / "Qwen2.5-1.5B_f16_e5_s42_pooled_seed42_v2_corr95_clin"
META_COLUMNS = ["id", "sample_id", "image_name", "official_split", "analysis_split",
                "label", "label_name", "mask_source", "source_split",
                "image_width", "image_height", "n_components_raw", "kept_component_fraction", "touches_border", "mask_pixels"]
CLINICAL_PREFIX = "original_clinical_"
MAX_CORR, K = 0.95, 16




def select(train: pd.DataFrame, y: pd.Series, method: str) -> tuple[list[str], int]:
    candidates, dropped = ct.drop_correlated(train, y, MAX_CORR, method=method)
    X = train[candidates]
    if method == "pearson":
        chosen = mrmr_classif(X=X, y=y, K=K, show_progress=False)
    else:   # mRMR redundancy as Spearman r: Pearson r on ranks (mrmr takes the absolute value)
        ranks = X.rank()

        def redundancy(target_column, features, X):
            return ranks.loc[:, features].corrwith(ranks.loc[:, target_column])

        chosen = mrmr_classif(X=X, y=y, K=K, redundancy=redundancy, show_progress=False)
    return list(chosen), len(dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="radiomics_v2_clinical.csv (v2 features + TI-RADS features)")
    ap.add_argument("--split", default=str(REPO / "splits" / "pooled_seed42.csv"))
    args = ap.parse_args()

    table = pd.read_csv(args.features)
    assignment = pd.read_csv(args.split).set_index("sample_id")["split"]
    t = table.assign(split=table["sample_id"].map(assignment))
    assert t["split"].notna().all(), "nodules missing from the split file"
    train = t[t["split"] == "train"].reset_index(drop=True)

    reference = json.load(open(RUN_I / "config.json"))["selected_features"]
    clinical = [f for f in reference if f.startswith(CLINICAL_PREFIX)]
    columns = [c for c in table.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(table[c])]
    raw = train[columns].replace([np.inf, -np.inf], np.nan)
    valid = [c for c in columns if raw[c].notna().sum() > 0 and raw[c].nunique(dropna=True) > 1]
    radiomic = [c for c in valid if not c.startswith(CLINICAL_PREFIX) and not c.startswith(ct.peritumoral.PREFIX)]
    scaled = pd.DataFrame(StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(raw[valid])),
                          columns=valid)[radiomic]
    y = train["label"].astype(int)

    selections, n_dropped = {}, {}
    for method in ["pearson", "spearman"]:
        selections[method], n_dropped[method] = select(scaled, y, method)
    expected = [f for f in reference if not f.startswith(CLINICAL_PREFIX)]
    assert selections["pearson"] == expected, "Pearson selection does not reproduce run I"

    print(f"{len(radiomic)} usable radiomics features; near-duplicates dropped: "
          f"Pearson {n_dropped['pearson']}, Spearman {n_dropped['spearman']}")
    shared = [f for f in selections["pearson"] if f in selections["spearman"]]
    print(f"mRMR 16: {len(shared)} features shared\n")
    rows = []
    for i in range(K):
        p, s = selections["pearson"][i], selections["spearman"][i]
        rows.append({"rank": i + 1, "Pearson": p.replace("original_", ""), "Spearman": s.replace("original_", "")})
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    probs = {m: xgboost_probs(table, clinical + selections[m], Path(args.split)) for m in selections}
    auc = pd.DataFrame([{"selection": m, **{s: round(roc_auc_score(probs[m][s]["label"], probs[m][s]["prob_malignant"]), 3)
                                             for s in EVAL}} for m in probs])
    print("XGBoost, 16 radiomics + 6 TI-RADS features (AUC)\n", auc.to_string(index=False), "\n")

    for s in ["calibration", "test"]:
        a, b = probs["spearman"][s], probs["pearson"][s]
        assert (a["sample_id"] == b["sample_id"]).all()
        diff, lo, hi = paired_bootstrap(a["label"].to_numpy(), a["prob_malignant"].to_numpy(), b["prob_malignant"].to_numpy())
        print(f"{s}: Spearman - Pearson AUC {diff:+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}]")
        if s == "calibration":
            decision = "run the LLM with the Spearman features" if lo > 0 else "no LLM run (rule fixed in advance)"
    print(f"\nDecision: {decision}")


if __name__ == "__main__":
    main()
