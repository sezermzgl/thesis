"""Learn-then-Test (LTT) calibration of the three-way triage rule.

Rule, for a score s (vote count, or P(malignant) for the softmax baseline):
    s <= t         -> auto-benign
    s >= u         -> auto-malignant
    t < s < u      -> refer to an expert

Risks, each tested at level delta with fixed-sequence testing:
    R1: share of truly malignant nodules among auto-benign   <= alpha_benign
    R2: share of truly benign nodules among auto-malignant   <= alpha_malignant

With both risks tested at delta, the two guarantees hold together with probability >= 1 - 2*delta.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binom

BENIGN, MALIGNANT, REFER = 0, 1, -1


def binomial_pvalue(n: int, k: int, alpha: float) -> float:
    """p-value for H0: true risk > alpha, given k errors among n decisions.

    p = P(Binomial(n, alpha) <= k). With n = 0 there is no evidence, so p = 1.
    """
    if n == 0:
        return 1.0
    return float(binom.cdf(k, n, alpha))


def fixed_sequence_test(
    scores: np.ndarray,
    y: np.ndarray,
    candidates: list,
    side: str,
    alpha: float,
    delta: float,
) -> pd.DataFrame:
    """Test candidate thresholds in the given order (safest first) and stop at the first failure.

    side="benign":    decisions are s <= t, an error is y == 1.
    side="malignant": decisions are s >= u, an error is y == 0.
    """
    if side == "benign":
        decided, error_label = (lambda c: scores <= c), 1
    elif side == "malignant":
        decided, error_label = (lambda c: scores >= c), 0
    else:
        raise ValueError("side must be 'benign' or 'malignant'.")

    rows, still_certifying = [], True
    for c in candidates:
        mask = decided(c)
        n, k = int(mask.sum()), int((y[mask] == error_label).sum())
        p = binomial_pvalue(n, k, alpha)
        still_certifying = still_certifying and p <= delta
        rows.append({"threshold": c, "n": n, "errors": k,
                     "observed_rate": k / n if n else np.nan,
                     "p_value": p, "certified": still_certifying})
    return pd.DataFrame(rows)


@dataclass
class TriageRule:
    t: float | None          # None -> no auto-benign decisions certified
    u: float | None          # None -> no auto-malignant decisions certified
    benign_table: pd.DataFrame = field(repr=False)
    malignant_table: pd.DataFrame = field(repr=False)

    def decide(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)
        out = np.full(scores.shape, REFER, dtype=int)
        if self.t is not None:
            out[scores <= self.t] = BENIGN
        if self.u is not None:
            out[scores >= self.u] = MALIGNANT
        return out


def _last_certified(table: pd.DataFrame):
    certified = table.loc[table["certified"], "threshold"]
    return None if certified.empty else certified.iloc[-1]


def calibrate_triage(
    scores: np.ndarray,
    y: np.ndarray,
    t_candidates: list,
    u_candidates: list,
    alpha_benign: float,
    alpha_malignant: float,
    delta: float,
) -> TriageRule:
    """Pick the most permissive certified t and u.

    t_candidates must be increasing and u_candidates decreasing (safest first), and every t must
    lie strictly below every u so that the two automatic regions never overlap.
    """
    scores, y = np.asarray(scores), np.asarray(y)
    if list(t_candidates) != sorted(t_candidates):
        raise ValueError("t_candidates must be increasing (safest first).")
    if list(u_candidates) != sorted(u_candidates, reverse=True):
        raise ValueError("u_candidates must be decreasing (safest first).")
    if max(t_candidates) >= min(u_candidates):
        raise ValueError("Every t candidate must be below every u candidate.")

    benign_table = fixed_sequence_test(scores, y, t_candidates, "benign", alpha_benign, delta)
    malignant_table = fixed_sequence_test(scores, y, u_candidates, "malignant", alpha_malignant, delta)
    return TriageRule(_last_certified(benign_table), _last_certified(malignant_table),
                      benign_table, malignant_table)


def vote_candidates(n_queries: int) -> tuple[list[int], list[int]]:
    """Default grids for vote scores: t = 0..floor((m-1)/2), u = m..floor(m/2)+1.

    For m = 10 this is t = 0..4 and u = 10..6, so v = 5 is always referred.
    """
    t = list(range(0, (n_queries - 1) // 2 + 1))
    u = list(range(n_queries, n_queries // 2, -1))
    return t, u
