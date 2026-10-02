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


def feature_dependence(long_table: pd.DataFrame, threshold: float = 0.5,
                       key_column: str = "sample_id") -> pd.DataFrame:
    """How often dropping each feature flips the answer relative to the training prompt."""
    base = (long_table[long_table["query"] == 0]
            .set_index(key_column)["prob_malignant"] >= threshold)
    dropped = long_table[long_table["dropped_feature"].notna()].copy()
    dropped["flipped"] = (dropped["prob_malignant"] >= threshold).to_numpy() != \
        base.loc[dropped[key_column]].to_numpy()
    return (dropped.groupby("dropped_feature")["flipped"]
            .agg(n_queries="size", flip_rate="mean")
            .sort_values("flip_rate", ascending=False))


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
