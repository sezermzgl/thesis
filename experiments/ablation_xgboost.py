"""Ablation: what did each change contribute? One factor at a time, with XGBoost.

Factors (all combinations):
    split      official TN3K split  vs  pooled split (all images pooled and re-split)
    features   v1 (original extraction)  vs  v2 (re-extraction)
    TI-RADS    without  vs  with the six TI-RADS-motivated features

Every cell uses the same pipeline as the LLM runs: train-only median imputation and z-scoring,
near-duplicate filter (|r| > 0.95), mRMR selects 16 radiomics features, TI-RADS features added
when used, untuned XGBoost of Part B. XGBoost is used because it is cheap and reproducible;
the LLM follows it (the TI-RADS gain is +0.026 for XGBoost and +0.059 for the LLM).

The official split has one fixed test set. The pooled split is repeated with 20 random seeds
(same stratification as splits/pooled_seed42.csv) to give a mean and SD. The official split's
validation set comes from the same distribution as its training set while its test set does not,
so on the official split the gap between validation and test AUC measures the cost of the shift.

    python experiments/ablation_xgboost.py --clinical clinical_features.csv \
        [--v2 radiomics_v2_all.csv] > runs/2026-10-08_v2_pipeline/ablation_xgboost.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
import conformal_triage as ct  # noqa: E402
from hybrid_xgb_llm import paired_bootstrap  # noqa: E402
from make_pooled_splits import pooled_split  # noqa: E402
from model_sweep import META_COLUMNS, SPLITS  # noqa: E402

CLINICAL_PREFIX = "original_clinical_"
CLINICAL = ["TallerThanWideRatio", "EchogenicityRatio", "PunctateFociDensity", "Solidity",
            "MarginSharpness", "AnechoicFraction"]
POOLED_SEEDS = [42] + list(range(1, 20))
K, MAX_CORR, SEED = 16, 0.95, 42


def official_assignment() -> pd.Series:
    parts = [pd.read_csv(REPO / "features" / f"radiomics_{s}.csv", usecols=["id", "official_split"]).assign(split=s)
             for s in SPLITS]
    a = pd.concat(parts, ignore_index=True)
    return a.set_index(a["official_split"].astype(str) + "/" + a["id"].astype(str))["split"]


def load_v1() -> pd.DataFrame:
    t = pd.concat([pd.read_csv(REPO / "features" / f"radiomics_{s}.csv") for s in SPLITS], ignore_index=True)
    t["sample_id"] = t["official_split"].astype(str) + "/" + t["id"].astype(str)
    return t


def fit_cell(table: pd.DataFrame, assignment: pd.Series, clinical: list[str]) -> dict:
    t = table.assign(split=table["sample_id"].map(assignment))
    assert t["split"].notna().all()
    parts = {s: t[t["split"] == s].reset_index(drop=True) for s in SPLITS}
    columns = [c for c in table.columns if c not in META_COLUMNS + ["sample_id", "split"]
               and pd.api.types.is_numeric_dtype(table[c])]
    raw = {s: parts[s][columns].replace([np.inf, -np.inf], np.nan) for s in SPLITS}
    valid = [c for c in columns if raw["train"][c].notna().sum() > 0 and raw["train"][c].nunique(dropna=True) > 1]
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    X = {"train": pd.DataFrame(sc.fit_transform(imp.fit_transform(raw["train"][valid])), columns=valid)}
    for s in SPLITS[1:]:
        X[s] = pd.DataFrame(sc.transform(imp.transform(raw[s][valid])), columns=valid)
    y = {s: parts[s]["label"].astype(int).to_numpy() for s in SPLITS}

    radiomic = [c for c in valid if not c.startswith(CLINICAL_PREFIX)]
    candidates, _ = ct.drop_correlated(X["train"][radiomic], y["train"], MAX_CORR)
    chosen = list(mrmr_classif(X=X["train"][candidates], y=pd.Series(y["train"]), K=K, show_progress=False))
    features = [c for c in clinical if c in valid] + chosen
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=(y["train"] == 0).sum() / (y["train"] == 1).sum(),
                          random_state=SEED, verbosity=0).fit(X["train"][features], y["train"])
    return {s: pd.DataFrame({"sample_id": parts[s]["sample_id"], "label": y[s],
                             "p": model.predict_proba(X[s][features])[:, 1]}).sort_values("sample_id", ignore_index=True)
            for s in ["validation", "test"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True, help="clinical_features.csv (sample_id + six TI-RADS features)")
    ap.add_argument("--v2", help="radiomics_v2_all.csv or radiomics_v2_clinical.csv; without it only v1 cells run")
    args = ap.parse_args()

    clin = pd.read_csv(args.clinical)
    clin_cols = [CLINICAL_PREFIX + c for c in CLINICAL]
    clin = clin[["sample_id"] + clin_cols]
    tables = {"v1": load_v1()}
    if args.v2:
        v2 = pd.read_csv(args.v2)
        if "sample_id" not in v2:
            v2["sample_id"] = v2["official_split"].astype(str) + "/" + v2["id"].astype(int).astype(str)
        tables["v2"] = v2.drop(columns=[c for c in v2.columns if c.startswith(CLINICAL_PREFIX)])

    def with_clinical(t):
        m = t.merge(clin, on="sample_id", how="left")
        assert m[clin_cols].notna().all(axis=None), "TI-RADS features missing for some images"
        return m

    splits = {"official": [official_assignment()]}
    splits["pooled"] = [pooled_split(s).set_index("sample_id")["split"] for s in POOLED_SEEDS]

    rows, keep = [], {}
    for fname, table in tables.items():
        for tirads in (False, True):
            t = with_clinical(table) if tirads else table
            for sname, assignments in splits.items():
                for i, a in enumerate(assignments):
                    out = fit_cell(t, a, clin_cols if tirads else [])
                    rows.append({"split": sname, "features": fname, "TI-RADS": "yes" if tirads else "no",
                                 "rep": i, **{f"AUC {s}": roc_auc_score(out[s]["label"], out[s]["p"]) for s in out}})
                    if i == 0:
                        keep[(sname, fname, tirads)] = out["test"]
    df = pd.DataFrame(rows)

    print("Test AUC per cell (pooled: mean ± SD over 20 random splits; official: its one test set)\n")
    summary = []
    for (sname, fname, tirads), g in df.groupby(["split", "features", "TI-RADS"], sort=False):
        row = {"split": sname, "features": fname, "TI-RADS": tirads}
        if sname == "official":
            row["validation AUC"] = f"{g['AUC validation'].iloc[0]:.3f}"
            row["test AUC"] = f"{g['AUC test'].iloc[0]:.3f}"
        else:
            row["validation AUC"] = f"{g['AUC validation'].mean():.3f} ± {g['AUC validation'].std():.3f}"
            row["test AUC"] = f"{g['AUC test'].mean():.3f} ± {g['AUC test'].std():.3f}"
        summary.append(row)
    print(pd.DataFrame(summary).to_string(index=False), "\n")

    print("One change at a time (same split, same test nodules; pooled: split seed 42; paired bootstrap 95%)")
    changes = []
    for sname in splits:
        for fname in tables:
            changes.append((f"TI-RADS added ({sname}, {fname})", (sname, fname, False), (sname, fname, True)))
        if "v2" in tables:
            for tirads in (False, True):
                changes.append((f"v1 -> v2 ({sname}, TI-RADS {'yes' if tirads else 'no'})",
                                (sname, "v1", tirads), (sname, "v2", tirads)))
    for name, a, b in changes:
        m = keep[a].merge(keep[b], on=["sample_id", "label"], suffixes=("_a", "_b"))
        assert len(m) == len(keep[a]) == len(keep[b]), "the two cells have different test nodules"
        d, lo, hi = paired_bootstrap(m["label"].to_numpy(), m["p_b"].to_numpy(), m["p_a"].to_numpy())
        print(f"  {name}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nSame changes on the pooled split, averaged over the 20 random splits (difference in test AUC within each split):")
    pooled = df[df.split == "pooled"].pivot_table(index="rep", columns=["features", "TI-RADS"], values="AUC test")
    pairs = [("TI-RADS added, v1", ("v1", "no"), ("v1", "yes"))]
    if "v2" in tables:
        pairs += [("TI-RADS added, v2", ("v2", "no"), ("v2", "yes")), ("v1 -> v2, no TI-RADS", ("v1", "no"), ("v2", "no")),
                  ("v1 -> v2, with TI-RADS", ("v1", "yes"), ("v2", "yes")), ("both (v1 -> v2 + TI-RADS)", ("v1", "no"), ("v2", "yes"))]
    for name, a, b in pairs:
        d = pooled[b] - pooled[a]
        print(f"  {name}: {d.mean():+.3f} ± {d.std():.3f} (positive in {(d > 0).mean():.0%} of splits)")

    print("\nCost of the shift on the official split (validation is in-distribution, test is not):")
    for fname in tables:
        for tirads in ("no", "yes"):
            g = df[(df.split == "official") & (df.features == fname) & (df["TI-RADS"] == tirads)].iloc[0]
            print(f"  {fname}, TI-RADS {tirads}: validation {g['AUC validation']:.3f}, test {g['AUC test']:.3f}, "
                  f"gap {g['AUC test'] - g['AUC validation']:+.3f}")


if __name__ == "__main__":
    main()
