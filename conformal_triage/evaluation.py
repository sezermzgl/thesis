"""Reports: feasibility of the vote signal, triage outcomes, and prediction-set quality."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .ltt import BENIGN, MALIGNANT, REFER


def _safe_auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def feasibility_report(probs: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> pd.Series:
    """Does vote disagreement carry information beyond the single training-prompt softmax?

    probs: (n, m) P(malignant) per query; column 0 is the unperturbed training prompt.
    """
    probs, y = np.asarray(probs), np.asarray(y)
    m = probs.shape[1]
    base = probs[:, 0]
    vote_frac = (probs >= threshold).sum(axis=1) / m
    base_correct = ((base >= threshold).astype(int) == y).astype(int)

    # Confidence measures: 1 = fully confident, 0 = undecided.
    vote_agreement = np.abs(vote_frac - 0.5) * 2
    softmax_confidence = np.abs(base - 0.5) * 2

    return pd.Series({
        "n": len(y),
        "queries": m,
        "auc_base_softmax": _safe_auc(y, base),
        "auc_mean_prob": _safe_auc(y, probs.mean(axis=1)),
        "auc_vote_fraction": _safe_auc(y, vote_frac),
        "base_accuracy": base_correct.mean(),
        # > 0.5 means higher confidence goes with a correct base prediction.
        "auc_correct_by_vote_agreement": _safe_auc(base_correct, vote_agreement),
        "auc_correct_by_softmax_conf": _safe_auc(base_correct, softmax_confidence),
        "share_unanimous": float(np.isin(vote_frac, [0.0, 1.0]).mean()),
        "share_flipped_vs_base": float(((probs >= threshold) != (base >= threshold)[:, None])[:, 1:].mean()),
    })


def _with_base(long_table: pd.DataFrame, threshold: float, key_column: str) -> pd.DataFrame:
    base = long_table[long_table["kind"] == "base"].set_index(key_column)["prob_malignant"]
    out = long_table[long_table["kind"] != "base"].copy()
    out["base_prob"] = base.loc[out[key_column]].to_numpy()
    out["flipped"] = (out["prob_malignant"] >= threshold) != (out["base_prob"] >= threshold)
    return out


def perturbation_report(long_table: pd.DataFrame, threshold: float = 0.5,
                        key_column: str = "sample_id") -> pd.DataFrame:
    """Per perturbation kind (and noise level): how far the answers move from the base prompt.

    A useful perturbation keeps the discrimination (auc) close to the base prompt's and does
    not push every answer to one class (share_pred_malignant).
    """
    base = long_table[long_table["kind"] == "base"]
    rows = {("base", 0.0): {
        "n": len(base), "flip_rate": 0.0,
        "share_pred_malignant": (base["prob_malignant"] >= threshold).mean(),
        "mean_shift": 0.0, "auc": _safe_auc(base["label"], base["prob_malignant"])}}
    for (kind, sd), g in _with_base(long_table, threshold, key_column).groupby(["kind", "noise_sd"]):
        rows[(kind, sd)] = {
            "n": len(g), "flip_rate": g["flipped"].mean(),
            "share_pred_malignant": (g["prob_malignant"] >= threshold).mean(),
            "mean_shift": (g["prob_malignant"] - g["base_prob"]).mean(),
            "auc": _safe_auc(g["label"], g["prob_malignant"])}
    out = pd.DataFrame(rows).T
    out.index.names = ["kind", "noise_sd"]
    return out


def feature_dependence(long_table: pd.DataFrame, threshold: float = 0.5,
                       key_column: str = "sample_id") -> pd.DataFrame:
    """How often removing each feature (and nothing else) flips the base-prompt answer."""
    drops = _with_base(long_table, threshold, key_column)
    drops = drops[drops["kind"] == "drop"]
    drops["abs_shift"] = (drops["prob_malignant"] - drops["base_prob"]).abs()
    return (drops.groupby("dropped_feature")
            .agg(n=("flipped", "size"), flip_rate=("flipped", "mean"),
                 mean_abs_shift=("abs_shift", "mean"))
            .sort_values("flip_rate", ascending=False))


def two_sample_test(X_a: np.ndarray, X_b: np.ndarray, n_permutations: int = 500,
                    seed: int = 0) -> pd.Series:
    """Classifier two-sample test: can a model tell sample a from sample b?

    Cross-validated AUC of a logistic regression separating the two samples, with a
    permutation p-value. AUC near 0.5 is consistent with exchangeability; a small p-value
    means the two samples come from different distributions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = np.vstack([X_a, X_b])
    t = np.r_[np.zeros(len(X_a), int), np.ones(len(X_b), int)]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = cross_val_predict(LogisticRegression(max_iter=1000), X, t, cv=cv,
                               method="predict_proba")[:, 1]
    auc = roc_auc_score(t, scores)
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(rng.permutation(t), scores) for _ in range(n_permutations)])
    return pd.Series({"n_a": len(X_a), "n_b": len(X_b), "auc": auc,
                      "p_value": (1 + (null >= auc).sum()) / (1 + n_permutations)})


def triage_report(decisions: np.ndarray, y: np.ndarray) -> pd.Series:
    """Realized outcome of a three-way rule on labelled data."""
    decisions, y = np.asarray(decisions), np.asarray(y)
    ab, am, ref = decisions == BENIGN, decisions == MALIGNANT, decisions == REFER
    n_mal = int((y == 1).sum())
    return pd.Series({
        "n": len(y),
        "auto_benign": int(ab.sum()),
        "auto_benign_malignant": int((ab & (y == 1)).sum()),
        "auto_benign_malignant_rate": (y[ab] == 1).mean() if ab.any() else np.nan,   # R1
        "auto_malignant": int(am.sum()),
        "auto_malignant_benign": int((am & (y == 0)).sum()),
        "auto_malignant_benign_rate": (y[am] == 0).mean() if am.any() else np.nan,   # R2
        "referred": int(ref.sum()),
        "automation_rate": float((ab | am).mean()),
        # Share of all cancers that the system sent home without review.
        "cancers_auto_benign_share": int((ab & (y == 1)).sum()) / n_mal if n_mal else np.nan,
    })


def sets_to_decisions(sets: np.ndarray) -> np.ndarray:
    """Singleton -> automatic decision; both labels or empty -> refer."""
    out = np.full(len(sets), REFER, dtype=int)
    out[sets[:, 0] & ~sets[:, 1]] = BENIGN
    out[sets[:, 1] & ~sets[:, 0]] = MALIGNANT
    return out


def set_report(sets: np.ndarray, y: np.ndarray) -> pd.Series:
    """Coverage (marginal and per class) and set-size breakdown."""
    sets, y = np.asarray(sets), np.asarray(y)
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(axis=1)
    return pd.Series({
        "n": len(y),
        "coverage": covered.mean(),
        "coverage_benign": covered[y == 0].mean() if (y == 0).any() else np.nan,
        "coverage_malignant": covered[y == 1].mean() if (y == 1).any() else np.nan,
        "singleton_rate": (size == 1).mean(),
        "both_labels_rate": (size == 2).mean(),
        "empty_rate": (size == 0).mean(),
        "non_singleton_rate": (size != 1).mean(),
    })
