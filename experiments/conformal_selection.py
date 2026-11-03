"""Conformal selection (Jin & Candès, "Selection by prediction with conformal p-values") for the triage.

R1 (cancers among the auto-benign) and R2 (benign nodules among the auto-malignant) are false
discovery rates of a selection: auto-benign selects nodules declared benign, auto-malignant selects
nodules declared malignant. Conformal selection controls them directly:

    auto-benign     H_j: nodule j is a cancer.  p_j = (1 + #{calibration cancers with score < s_j}) / (n + 1)
    auto-malignant  H_j: nodule j is benign.    p_j = (1 + #{calibration benign with score > s_j}) / (n + 1)

where s is the model's probability of cancer and n the number of calibration nodules (all labels).
This is the clipped score V(x, y) = M * y - mu(x) of the paper with the "+1" p-value (their Eq. 6),
valid when scores have no ties. Benjamini-Hochberg at level q on the test p-values selects the
nodules; then FDR = E[R1] <= 0.05 and E[R2] <= 0.20, in finite samples, when calibration and test
nodules are exchangeable (their Theorem 2.6 covers sampling without replacement from a finite set,
as in the re-splits below) and the model was trained without them.

The guarantee is on the expectation over calibration and test draws, not with probability 1 - delta
as in LTT; the script reports both the mean and how often a single split exceeds the target.
The same 2,000 re-splits of the 962 calibration + test nodules as ltt_design_resplits.py.

    python experiments/conformal_selection.py > runs/2026-10-08_v2_pipeline/conformal_selection.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from ltt_design_resplits import N_REP, load_models  # noqa: E402

Q = {"benign": 0.05, "malignant": 0.20}


def conformal_pvalues(p_cal, y_cal, p_test, side):
    """Conformal p-values for H_j: the test nodule is NOT of the class this side selects."""
    n = len(y_cal)
    if side == "benign":    # nulls are cancers; small p when the test score is below most calibration cancers
        null = np.sort(p_cal[y_cal == 1])
        below = np.searchsorted(null, p_test, side="left")
    else:                   # nulls are benign; small p when the test score is above most calibration benign
        null = np.sort(p_cal[y_cal == 0])
        below = len(null) - np.searchsorted(null, p_test, side="right")
    return (1 + below) / (n + 1)


def benjamini_hochberg(p, q):
    m = len(p)
    order = np.argsort(p)
    passed = np.nonzero(p[order] <= q * np.arange(1, m + 1) / m)[0]
    selected = np.zeros(m, dtype=bool)
    if len(passed):
        selected[order[: passed[-1] + 1]] = True
    return selected


def run(p_cal, y_cal, p_test, y_test, side):
    sel = benjamini_hochberg(conformal_pvalues(p_cal, y_cal, p_test, side), Q[side])
    error_label = 1 if side == "benign" else 0
    fdp = (y_test[sel] == error_label).mean() if sel.any() else 0.0
    return {"selected": sel.mean(), "fdp": fdp, "any": bool(sel.any())}


def evaluate(name, data, rng):
    pool_p = np.r_[data["calibration"][0], data["test"][0]]
    pool_y = np.r_[data["calibration"][1], data["test"][1]]
    n_cal, n_cal_pos = len(data["calibration"][1]), int(data["calibration"][1].sum())
    i1, i0 = np.where(pool_y == 1)[0], np.where(pool_y == 0)[0]
    rows = []
    for r in range(N_REP + 1):
        if r == 0:
            cal, test = np.arange(n_cal), np.arange(n_cal, len(pool_y))
        else:
            a1, a0 = rng.permutation(i1), rng.permutation(i0)
            cal = np.r_[a1[:n_cal_pos], a0[:n_cal - n_cal_pos]]
            test = np.r_[a1[n_cal_pos:], a0[n_cal - n_cal_pos:]]
        for side in ("benign", "malignant"):
            rows.append({"rep": r, "side": side, **run(pool_p[cal], pool_y[cal], pool_p[test], pool_y[test], side)})
    df = pd.DataFrame(rows)

    print(f"== {name}")
    for side in ("benign", "malignant"):
        d = df[df.side == side]
        act, rs = d[d.rep == 0].iloc[0], d[d.rep > 0]
        what = "cancers among auto-benign (R1)" if side == "benign" else "benign among auto-malignant (R2)"
        print(f"   auto-{side}, target E[{what}] <= {Q[side]:.0%}")
        print(f"     our split: {act.selected:.1%} of test nodules selected, {what} {act.fdp:.1%}")
        print(f"     {N_REP} re-splits: something selected in {rs['any'].mean():.0%}; selected {rs.selected.mean():.1%} "
              f"(5-95%: {rs.selected.quantile(0.05):.1%}-{rs.selected.quantile(0.95):.1%}); "
              f"mean {what} {rs.fdp.mean():.1%} (the guaranteed quantity); "
              f"single split above {Q[side]:.0%} in {(rs.fdp > Q[side]).mean():.0%}")
    print()


def main():
    rng = np.random.default_rng(0)
    for name, data in load_models().items():
        evaluate(name, data, rng)


if __name__ == "__main__":
    main()
