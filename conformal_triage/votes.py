"""From per-query model outputs to vote counts."""

import numpy as np
import pandas as pd


def predict_malignant_proba(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    """P(malignant) for each text, one forward pass per text. Needs torch (Colab GPU)."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(texts[start:start + batch_size], truncation=True,
                            max_length=max_length, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def signal_matrix(
    long_table: pd.DataFrame,
    kind: str,
    noise_sd: float | None = None,
    key_column: str = "sample_id",
):
    """(n, q) matrix of P(malignant): column 0 is the base prompt, then the `kind` queries.

    kind="jitter" needs `noise_sd` to pick one noise level; kind="drop" uses every drop query.
    Rows follow the first-appearance order of `key_column` in the long table.
    """
    sel = long_table["kind"] == kind
    if kind == "jitter":
        sel &= np.isclose(long_table["noise_sd"], noise_sd)
    rows = long_table[(long_table["kind"] == "base") | sel]

    keys = rows[key_column].drop_duplicates().tolist()
    wide = rows.pivot(index=key_column, columns="query", values="prob_malignant").loc[keys]
    if wide.isna().any().any():
        raise ValueError("Some (nodule, query) pairs have no probability.")
    if wide.shape[1] < 2:
        raise ValueError(f"No '{kind}' queries found (noise_sd={noise_sd}).")
    labels = rows.drop_duplicates(key_column).set_index(key_column).loc[keys, "label"]
    return wide.to_numpy(dtype=np.float64), labels.to_numpy(dtype=int), keys


def count_votes(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """v = number of queries answering malignant, per nodule."""
    return (np.asarray(probs) >= threshold).sum(axis=1)


def vote_table(votes: np.ndarray, y: np.ndarray, n_queries: int) -> pd.DataFrame:
    """Counts of benign / malignant nodules at each vote value (as in the design note)."""
    values = np.arange(n_queries + 1)
    return pd.DataFrame(
        {
            "truly_benign": [int(((votes == v) & (y == 0)).sum()) for v in values],
            "truly_malignant": [int(((votes == v) & (y == 1)).sum()) for v in values],
        },
        index=pd.Index(values, name="votes"),
    ).T
