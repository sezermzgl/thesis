"""Does the LLM add information beyond XGBoost? Late fusion of the two models' probabilities.

XGBoost is trained on exactly the features of an LLM run (its config.json `selected_features`), with
the preprocessing of the v2 notebook (median imputation and z-scoring fitted on train) and the
untuned settings of Part B. The two models' per-nodule probabilities are then combined:

    mean       average of the two probabilities
    rank mean  average of the two ranks (insensitive to how each model is calibrated)
    stacking   logistic regression on the two logits, fitted on validation only

Calibration and test are never used for fitting, so split conformal prediction on top stays valid
(thresholds from calibration, applied to test). Paired bootstrap intervals compare each model with
XGBoost on the same nodules.

    python experiments/hybrid_xgb_llm.py --features radiomics_v2_clinical.csv \
        --llm-runs runs/2026-10-08_v2_pipeline/<run> [<run> ...] --out runs/2026-10-08_v2_pipeline/hybrid

With --xgb-probs <folder> instead of --features, the saved XGBoost probabilities are reused.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from conformal_triage import split_cp as cp  # noqa: E402
from conformal_triage.evaluation import set_report  # noqa: E402

SPLITS = ["train", "validation", "calibration", "test"]
EVAL = SPLITS[1:]
SEED = 42


def xgboost_probs(table: pd.DataFrame, features: list[str], split_csv: Path) -> dict[str, pd.DataFrame]:
    assignment = pd.read_csv(split_csv).set_index("sample_id")["split"]
    t = table.assign(split=table["sample_id"].map(assignment))
    assert t["split"].notna().all(), "nodules missing from the split file"
    parts = {s: t[t["split"] == s].reset_index(drop=True) for s in SPLITS}
    clean = lambda df: df[features].replace([np.inf, -np.inf], np.nan)
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    X = {"train": sc.fit_transform(imp.fit_transform(clean(parts["train"])))}
    for s in EVAL:
        X[s] = sc.transform(imp.transform(clean(parts[s])))
    y = parts["train"]["label"].to_numpy()
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=(y == 0).sum() / (y == 1).sum(), random_state=SEED, verbosity=0)
    model.fit(X["train"], y)
    return {s: pd.DataFrame({"sample_id": parts[s]["sample_id"], "label": parts[s]["label"].astype(int),
                             "prob_malignant": model.predict_proba(X[s])[:, 1]}) for s in EVAL}


def paired_bootstrap(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return roc_auc_score(y, a) - roc_auc_score(y, b), *np.percentile(diffs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="radiomics_v2_clinical.csv (v2 features + TI-RADS features)")
    src.add_argument("--xgb-probs", help="folder with xgboost_probs_{split}.csv from an earlier run (skips training)")
    ap.add_argument("--llm-runs", nargs="+", required=True, help="LLM run folders with the same selected features")
    ap.add_argument("--split", default=str(REPO / "splits" / "pooled_seed42.csv"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = [Path(r) for r in args.llm_runs]
    feature_sets = [json.load(open(r / "config.json"))["selected_features"] for r in runs]
    assert all(f == feature_sets[0] for f in feature_sets), "LLM runs use different features"
    features = feature_sets[0]

    if args.features:
        xgb = xgboost_probs(pd.read_csv(args.features), features, Path(args.split))
    else:
        xgb = {s: pd.read_csv(Path(args.xgb_probs) / f"xgboost_probs_{s}.csv") for s in EVAL}
    for s in EVAL:
        xgb[s].to_csv(out / f"xgboost_probs_{s}.csv", index=False)

    # Align all models on the same nodules; the LLM is the average over its runs (seeds).
    data = {}
    for s in EVAL:
        d = xgb[s].rename(columns={"prob_malignant": "xgb"})
        for i, r in enumerate(runs):
            llm = pd.read_csv(r / f"probs_{s}.csv").rename(columns={"prob_malignant": f"llm{i}"})
            d = d.merge(llm, on=["sample_id", "label"], how="inner")
        assert len(d) == len(xgb[s]), f"{s}: LLM and XGBoost nodules differ"
        d["llm"] = d[[f"llm{i}" for i in range(len(runs))]].mean(axis=1)
        data[s] = d

    eps = 1e-6
    lg = lambda p: logit(np.clip(p, eps, 1 - eps))
    stack = LogisticRegression().fit(np.column_stack([lg(data["validation"]["xgb"]), lg(data["validation"]["llm"])]),
                                     data["validation"]["label"])
    for s in EVAL:
        d = data[s]
        d["mean"] = (d["xgb"] + d["llm"]) / 2
        d["rank mean"] = (rankdata(d["xgb"]) + rankdata(d["llm"])) / (2 * len(d))
        d["stacking"] = stack.predict_proba(np.column_stack([lg(d["xgb"]), lg(d["llm"])]))[:, 1]

    models = ["xgb", "llm", "mean", "rank mean", "stacking"]
    auc = pd.DataFrame({s: {m: roc_auc_score(data[s]["label"], data[s][m]) for m in models} for s in EVAL})
    single = {f"llm{i} ({r.name})": {s: roc_auc_score(data[s]["label"], data[s][f"llm{i}"]) for s in EVAL}
              for i, r in enumerate(runs)}
    auc = pd.concat([pd.DataFrame(single).T, auc])
    corr = pd.DataFrame({s: {"pearson": np.corrcoef(data[s]["xgb"], data[s]["llm"])[0, 1],
                             "spearman": spearmanr(data[s]["xgb"], data[s]["llm"]).statistic} for s in EVAL})

    boot = []
    for m in ["llm", "mean", "rank mean", "stacking"]:
        for s in ["calibration", "test"] + ([] if m == "stacking" else ["validation"]):
            d = data[s]
            diff, lo, hi = paired_bootstrap(d["label"].to_numpy(), d[m].to_numpy(), d["xgb"].to_numpy())
            boot.append({"model": m, "split": s, "AUC minus XGBoost": diff, "low": lo, "high": hi})
    boot = pd.DataFrame(boot)

    cp_rows = []
    for m in models:
        c, t = data["calibration"], data["test"]
        q = cp.calibrate_mondrian(cp.softmax_scores(c[m].to_numpy()), c["label"].to_numpy(), (0.10, 0.05))
        r = set_report(cp.prediction_sets(cp.softmax_scores(t[m].to_numpy()), q), t["label"].to_numpy())
        cp_rows.append({"model": m, **r.to_dict()})
    cp_table = pd.DataFrame(cp_rows)

    pd.set_option("display.width", 200)
    print(f"{len(features)} features, LLM = average of {len(runs)} run(s)\n")
    print("AUC\n", auc.round(3).to_string(), "\n")
    print("Correlation of XGBoost and LLM probabilities\n", corr.round(3).to_string(), "\n")
    print("Stacking weights (logit XGBoost, logit LLM):", stack.coef_.round(3), "intercept", stack.intercept_.round(3), "\n")
    print("Paired bootstrap vs XGBoost (95% interval)\n", boot.round(3).to_string(index=False), "\n")
    print("Mondrian split CP on test (alpha 0.10 benign / 0.05 malignant)\n",
          cp_table[["model", "coverage_malignant", "benign_only_rate", "cancer_share_in_benign_only",
                    "malignant_only_rate", "benign_share_in_malignant_only", "both_labels_rate"]].round(3).to_string(index=False))

    auc.to_csv(out / "auc.csv")
    corr.to_csv(out / "correlation.csv")
    boot.to_csv(out / "bootstrap_vs_xgboost.csv", index=False)
    cp_table.to_csv(out / "split_cp_test.csv", index=False)
    json.dump({"features": features, "llm_runs": [r.name for r in runs],
               "stacking_coef": stack.coef_.tolist(), "stacking_intercept": stack.intercept_.tolist()},
              open(out / "config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
