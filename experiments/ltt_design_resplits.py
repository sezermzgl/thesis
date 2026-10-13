"""A Learn-then-Test design that can pass, evaluated over random re-splits of calibration and test.

Pre-registered design (fixed in this file before it was run; see the git history):

    Candidates. Thresholds at quantiles of the VALIDATION scores, so every candidate holds a
    known share of nodules: auto-benign t_q = q-quantile, auto-malignant u_q = (1 - q)-quantile,
    q in {0.05, 0.10, ..., 0.30}. At most 30% of nodules per side, so t < u always.
    Order. By the binomial p-value of each candidate on the validation split (same alpha),
    smallest first; ties broken by q (smaller q first). Validation is independent of
    calibration, so the order does not use calibration data and fixed-sequence testing keeps
    its guarantee.
    Test. Fixed-sequence binomial test on calibration, stop at the first p > delta. The rule uses
    the most permissive certified candidate. alpha_benign = 0.05, alpha_malignant = 0.20,
    delta = 0.05 per side, as planned.

Compared with the planned design (fixed probability grid t = 0.05..0.45, u = 0.95..0.55, safest
first) on the same re-splits.

Evaluation. The model is fixed; the 962 calibration + test nodules are randomly re-split with the
sizes and class proportions of splits/pooled_seed42.csv. For each re-split and design: whether a
side is certified, the share of test nodules decided automatically, and the error share among
them on test. Because a test group can be small, the error share on test is noisy; the share
computed on all 962 nodules below the chosen threshold is reported as a proxy for the true risk.
Validation is the same in every re-split.

    python experiments/ltt_design_resplits.py > runs/2026-10-08_v2_pipeline/ltt_design_resplits.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNS = REPO / "runs" / "2026-10-08_v2_pipeline"
LLM_SEEDS = ["Qwen2.5-1.5B_f16_e5_s42_pooled_seed42_v2_corr95_clin",
             "Qwen2.5-1.5B_f16_e5_s1_pooled_seed42_v2_corr95_clin6",
             "Qwen2.5-1.5B_f16_e5_s2_pooled_seed42_v2_corr95_clin6"]
SPLITS = ["validation", "calibration", "test"]
ALPHA = {"benign": 0.05, "malignant": 0.20}
DELTA = 0.05
QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PLANNED = {"benign": [round(x, 2) for x in np.arange(0.05, 0.50, 0.05)],
           "malignant": [round(x, 2) for x in np.arange(0.95, 0.50, -0.05)]}
N_REP = 2000


def load_models():
    xgb = {s: pd.read_csv(RUNS / "hybrid_xgb_llm" / f"xgboost_probs_{s}.csv").set_index("sample_id") for s in SPLITS}
    models = {"XGBoost (22 features)": xgb}
    llm = {}
    for s in SPLITS:
        ids = xgb[s].index
        p = [pd.read_csv(RUNS / r / f"probs_{s}.csv").set_index("sample_id").loc[ids, "prob_malignant"] for r in LLM_SEEDS]
        llm[s] = pd.DataFrame({"label": xgb[s]["label"], "prob_malignant": np.mean(p, axis=0)}, index=ids)
    models["LLM (mean of 3 seeds)"] = llm
    models["LLM run I (seed 42)"] = {s: pd.read_csv(RUNS / LLM_SEEDS[0] / f"probs_{s}.csv").set_index("sample_id") for s in SPLITS}
    models["LLM run D (no TI-RADS)"] = {s: pd.read_csv(RUNS / "Qwen2.5-1.5B_f16_e5_s42_pooled_seed42_v2_corr95" / f"probs_{s}.csv").set_index("sample_id") for s in SPLITS}
    return {k: {s: (v[s]["prob_malignant"].to_numpy(), v[s]["label"].to_numpy()) for s in SPLITS} for k, v in models.items()}


def decided(p, c, side):
    return p <= c if side == "benign" else p >= c


def pvalue(p, y, c, side):
    m = decided(p, c, side)
    n = int(m.sum())
    k = int((y[m] == (1 if side == "benign" else 0)).sum())
    return 1.0 if n == 0 else float(binom.cdf(k, n, ALPHA[side]))


def quantile_design(pv, yv):
    """Candidates and their order, from validation only."""
    out = {}
    for side in ("benign", "malignant"):
        cands = [(q, float(np.quantile(pv, q if side == "benign" else 1 - q))) for q in QUANTILES]
        ranked = sorted(cands, key=lambda qc: (pvalue(pv, yv, qc[1], side), qc[0]))
        out[side] = [c for _, c in ranked]
    return out


def certify(pc, yc, order, side):
    """Fixed-sequence test on calibration; returns the most permissive certified threshold or None."""
    certified = []
    for c in order:
        if pvalue(pc, yc, c, side) > DELTA:
            break
        certified.append(c)
    if not certified:
        return None
    return max(certified) if side == "benign" else min(certified)


def evaluate(name, data, rng):
    pv, yv = data["validation"]
    pool_p = np.r_[data["calibration"][0], data["test"][0]]
    pool_y = np.r_[data["calibration"][1], data["test"][1]]
    n_cal, n_cal_pos = len(data["calibration"][1]), int(data["calibration"][1].sum())
    i1, i0 = np.where(pool_y == 1)[0], np.where(pool_y == 0)[0]
    designs = {"planned (probability grid, safest first)": PLANNED, "quantile grid, validation-ordered": quantile_design(pv, yv)}

    def one(cal, test, design, side):
        c = certify(pool_p[cal], pool_y[cal], design[side], side)
        if c is None:
            return {"certified": 0}
        mt, ma = decided(pool_p[test], c, side), decided(pool_p, c, side)
        err = 1 if side == "benign" else 0
        return {"certified": 1, "threshold": c, "test_auto": mt.mean(),
                "test_error": (pool_y[test][mt] == err).mean() if mt.any() else np.nan,
                "pool_error": (pool_y[ma] == err).mean()}

    actual_cal = np.arange(n_cal)
    actual_test = np.arange(n_cal, len(pool_y))
    rows = []
    for r in range(N_REP + 1):
        if r == 0:
            cal, test = actual_cal, actual_test
        else:
            a1, a0 = rng.permutation(i1), rng.permutation(i0)
            cal = np.r_[a1[:n_cal_pos], a0[:n_cal - n_cal_pos]]
            test = np.r_[a1[n_cal_pos:], a0[n_cal - n_cal_pos:]]
        for dname, design in designs.items():
            for side in ("benign", "malignant"):
                rows.append({"rep": r, "design": dname, "side": side, **one(cal, test, design, side)})
    df = pd.DataFrame(rows)

    print(f"== {name}")
    for side in ("benign", "malignant"):
        a = ALPHA[side]
        print(f"   auto-{side} (alpha = {a:.0%}); validation-ordered quantile candidates: "
              + ", ".join(f"{c:.3f}" for c in designs["quantile grid, validation-ordered"][side]))
        for dname in designs:
            d = df[(df.design == dname) & (df.side == side)]
            act, rs = d[d.rep == 0].iloc[0], d[d.rep > 0]
            cert = rs[rs.certified == 1]
            line = f"     {dname}: certified in {rs.certified.mean():.0%} of re-splits"
            if len(cert):
                line += (f"; when certified: test automation {cert.test_auto.mean():.1%} "
                         f"(5-95%: {cert.test_auto.quantile(0.05):.1%}-{cert.test_auto.quantile(0.95):.1%}), "
                         f"test error {cert.test_error.mean():.1%}, error on all 962 {cert.pool_error.mean():.1%}, "
                         f"test error > alpha in {(cert.test_error > a).mean():.0%}, all-962 error > alpha in {(cert.pool_error > a).mean():.0%}")
            line += (f". Our split: " + ("not certified" if act.certified == 0 else
                     f"threshold {act.threshold:.3f}, test automation {act.test_auto:.1%}, test error {act.test_error:.1%}"))
            print(line)
    print()


def main():
    rng = np.random.default_rng(0)
    for name, data in load_models().items():
        evaluate(name, data, rng)


if __name__ == "__main__":
    main()
