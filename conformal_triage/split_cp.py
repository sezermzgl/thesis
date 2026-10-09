"""Classic split conformal prediction, marginal and class-conditional (Mondrian).

Scores are an (n, 2) matrix: column 0 is the nonconformity of "benign", column 1 of "malignant".
A label enters the prediction set when its score is <= the calibrated threshold q_hat.
"""

import math

import numpy as np


def vote_scores(votes: np.ndarray, n_queries: int) -> np.ndarray:
    """s(x, benign) = v, s(x, malignant) = m - v (votes against each label)."""
    votes = np.asarray(votes, dtype=np.float64)
    return np.column_stack([votes, n_queries - votes])


def softmax_scores(prob_malignant: np.ndarray) -> np.ndarray:
    """s(x, y) = 1 - p(y | x)."""
    p = np.asarray(prob_malignant, dtype=np.float64)
    return np.column_stack([p, 1.0 - p])


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """k-th smallest calibration score with k = ceil((n + 1)(1 - alpha)); inf if k > n."""
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else float(s[k - 1])


def calibrate_marginal(scores: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """One threshold shared by both labels, returned as a length-2 array."""
    true_scores = scores[np.arange(len(y)), y]
    q = conformal_quantile(true_scores, alpha)
    return np.array([q, q])


def calibrate_mondrian(scores: np.ndarray, y: np.ndarray, alpha) -> np.ndarray:
    """One threshold per label, each from that label's calibration nodules only.

    `alpha` is one level for both labels or a pair (alpha_benign, alpha_malignant); e.g.
    (0.10, 0.05) keeps "malignant" in the set for at least 95% of cancers, so at most 5% of
    cancers end up with the single label "benign".
    """
    alphas = (alpha, alpha) if np.isscalar(alpha) else tuple(alpha)
    return np.array([conformal_quantile(scores[y == c, c], alphas[c]) for c in (0, 1)])


def prediction_sets(scores: np.ndarray, q_hat: np.ndarray) -> np.ndarray:
    """Boolean (n, 2) matrix: True where the label is in the set."""
    return scores <= np.asarray(q_hat)[None, :]
