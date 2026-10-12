"""Checks of the conformal layer on the saved probabilities (no GPU, no retraining).

1. LTT feasibility: the smallest group that can be certified with k errors at level alpha.
2. LTT (planned fixed sequence, validation-ordered sequence) and the miss-rate rule, applied to
   XGBoost and to the LLM averaged over seeds, on the same calibration and test nodules.
3. Mondrian split CP over random re-splits of calibration + test: the guarantee is an average over
   calibration draws, so the realised coverage on our single test split is compared with its
   distribution. The model is fixed; only the calibration/test assignment changes (stratified by
   label, same sizes as `splits/pooled_seed42.csv`).

    python experiments/conformal_checks.py > runs/2026-10-08_v2_pipeline/conformal_checks.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import conformal_triage as ct  # noqa: E402

RUNS = REPO / "runs" / "2026-10-08_v2_pipeline"
LLM_SEEDS = ["Qwen2.5-1.5B_f16_e5_s42_pooled_seed42_v2_corr95_clin",
             "Qwen2.5-1.5B_f16_e5_s1_pooled_seed42_v2_corr95_clin6",
             "Qwen2.5-1.5B_f16_e5_s2_pooled_seed42_v2_corr95_clin6"]
NO_TIRADS = "Qwen2.5-1.5B_f16_e5_s42_pooled_seed42_v2_corr95"
XGB_DIR = RUNS / "hybrid_xgb_llm"
SPLITS = ["validation", "calibration", "test"]
ALPHA_BENIGN, ALPHA_MALIGNANT, DELTA = 0.05, 0.20, 0.05
T_GRID = [round(x, 2) for x in np.arange(0.05, 0.50, 0.05)]     # as notebook cell C10
U_GRID = [round(x, 2) for x in np.arange(0.95, 0.50, -0.05)]
MISS_GRID = [round(x, 2) for x in np.arange(0.01, 0.51, 0.01)]  # as notebook cell C11


def load_models() -> dict[str, dict[str, pd.DataFrame]]:
    xgb = {s: pd.read_csv(XGB_DIR / f"xgboost_probs_{s}.csv").set_index("sample_id") for s in SPLITS}
    models = {"XGBoost (22 features)": xgb}
    llm = {}
    for s in SPLITS:
        ids = xgb[s].index
        probs = [pd.read_csv(RUNS / r / f"probs_{s}.csv").set_index("sample_id").loc[ids, "prob_malignant"]
                 for r in LLM_SEEDS]
        llm[s] = pd.DataFrame({"label": xgb[s]["label"], "prob_malignant": np.mean(probs, axis=0)}, index=ids)
    models["LLM (mean of 3 seeds)"] = llm
    for name, run in [("LLM seed 42 (run I)", LLM_SEEDS[0]), ("LLM without TI-RADS (run D)", NO_TIRADS)]:
        models[name] = {s: pd.read_csv(RUNS / run / f"probs_{s}.csv").set_index("sample_id") for s in SPLITS}
    return models


def feasibility():
    print("1. Smallest group certifiable with k errors (binomial test, delta = 0.05)")
    for alpha in (ALPHA_BENIGN, ALPHA_MALIGNANT):
        row = []
        for k in range(6):
            n = k + 1
            while binom.cdf(k, n, alpha) > DELTA:
                n += 1
            row.append(f"k={k}: n>={n}")
        print(f"   alpha = {alpha:.2f}: " + ", ".join(row))
    print()


def ltt_checks(name, m):
    pv, yv = m["validation"]["prob_malignant"].to_numpy(), m["validation"]["label"].to_numpy()
    pc, yc = m["calibration"]["prob_malignant"].to_numpy(), m["calibration"]["label"].to_numpy()
    pt, yt = m["test"]["prob_malignant"].to_numpy(), m["test"]["label"].to_numpy()
    print(f"== {name}")
    rule = ct.calibrate_triage(pc, yc, T_GRID, U_GRID, ALPHA_BENIGN, ALPHA_MALIGNANT, DELTA)
    print(f"   planned LTT (safest first): t = {rule.t}, u = {rule.u}")
    for label, table in (("auto-benign", rule.benign_table), ("auto-malignant", rule.malignant_table)):
        first = table.iloc[:4][["threshold", "n", "errors", "p_value"]].round(3)
        print(f"   {label}, first candidates: " + "; ".join(
            f"{r.threshold}: {int(r.errors)}/{int(r.n)}, p={r.p_value}" for r in first.itertuples()))
    for side, grid, alpha in (("benign", T_GRID, ALPHA_BENIGN), ("malignant", U_GRID, ALPHA_MALIGNANT)):
        order = list(ct.ltt.fixed_sequence_test(pv, yv, grid, side, alpha, DELTA)
                     .sort_values("p_value", kind="stable")["threshold"])
        table = ct.ltt.fixed_sequence_test(pc, yc, order, side, alpha, DELTA)
        certified = table.loc[table["certified"], "threshold"].tolist()
        if certified:
            mask = pt <= max(certified) if side == "benign" else pt >= min(certified)
            wrong = (yt[mask] == (1 if side == "benign" else 0)).mean()
            print(f"   validation-ordered LTT, {side}: certified {certified}; test {mask.sum()} nodules "
                  f"({mask.mean():.1%}), error share {wrong:.3f}")
        else:
            r = table.iloc[0]
            print(f"   validation-ordered LTT, {side}: nothing (first candidate {r.threshold}: "
                  f"{int(r.errors)}/{int(r.n)}, p={r.p_value:.3f})")
    n_cancer, t = int((yc == 1).sum()), None
    for cand in MISS_GRID:
        if binom.cdf(int(((pc <= cand) & (yc == 1)).sum()), n_cancer, ALPHA_BENIGN) > DELTA:
            break
        t = cand
    if t is not None:
        auto = pt <= t
        print(f"   miss-rate rule (<= 5% of cancers auto-benign): t = {t}; test automation {auto.mean():.1%}, "
              f"miss rate {(auto & (yt == 1)).sum() / (yt == 1).sum():.3f}, "
              f"cancers among auto-benign {(auto & (yt == 1)).sum()}/{auto.sum()}")
    o = np.argsort(pc)
    print("   cleanest calibration tail: " + ", ".join(
        f"lowest {f:.0%} {int(yc[o[:int(f * len(pc))]].sum())}/{int(f * len(pc))} cancers" for f in (0.10, 0.15, 0.20)))
    print()


def mondrian_resplits(name, m, n_rep=4000, seed=0):
    p = np.r_[m["calibration"]["prob_malignant"].to_numpy(), m["test"]["prob_malignant"].to_numpy()]
    y = np.r_[m["calibration"]["label"].to_numpy(), m["test"]["label"].to_numpy()]
    n_cal, n_cal_cancer = len(m["calibration"]), int(m["calibration"]["label"].sum())
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rep):
        a1, a0 = rng.permutation(i1), rng.permutation(i0)
        cal = np.r_[a1[:n_cal_cancer], a0[:n_cal - n_cal_cancer]]
        test = np.r_[a1[n_cal_cancer:], a0[n_cal - n_cal_cancer:]]
        q = ct.calibrate_mondrian(ct.softmax_scores(p[cal]), y[cal], (0.10, 0.05))
        rows.append(ct.set_report(ct.prediction_sets(ct.softmax_scores(p[test]), q), y[test]))
    r = pd.DataFrame(rows)
    q = ct.calibrate_mondrian(ct.softmax_scores(m["calibration"]["prob_malignant"].to_numpy()),
                              m["calibration"]["label"].to_numpy(), (0.10, 0.05))
    actual = ct.set_report(ct.prediction_sets(ct.softmax_scores(m["test"]["prob_malignant"].to_numpy()), q),
                           m["test"]["label"].to_numpy())
    cov = r["coverage_malignant"]
    print(f"   {name}: cancer coverage mean {cov.mean():.3f} (SD {cov.std():.3f}, below 0.95 in {np.mean(cov < 0.95):.0%}); "
          f"our split {actual['coverage_malignant']:.3f}. Auto-benign {r['benign_only_rate'].mean():.1%}, "
          f"cancers among them {r['cancer_share_in_benign_only'].mean():.1%} "
          f"(5-95%: {r['cancer_share_in_benign_only'].quantile(0.05):.1%}-{r['cancer_share_in_benign_only'].quantile(0.95):.1%})")


def main():
    models = load_models()
    feasibility()
    print("2. LTT and the miss-rate rule (thresholds from calibration, applied to test)\n")
    for name in ["XGBoost (22 features)", "LLM (mean of 3 seeds)"]:
        ltt_checks(name, models[name])
    print("3. Mondrian split CP (alpha 0.10 benign / 0.05 malignant) over 4,000 random re-splits of calibration + test")
    for name, m in models.items():
        mondrian_resplits(name, m)


if __name__ == "__main__":
    main()
