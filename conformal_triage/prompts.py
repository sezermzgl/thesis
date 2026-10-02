"""Perturbed feature prompts for the conformal triage layer.

Each nodule is shown to the fine-tuned model several times. Query 0 is always the exact prompt
used in training (Part A, A11); the remaining queries shuffle the feature order, optionally drop
one feature, and pick a paraphrased sentence template. Every variant is recorded so that the
votes can later be traced back to the perturbation that produced them.
"""

import re
import zlib

import numpy as np
import pandas as pd

# Template 0 reproduces build_prompt() from the notebook word for word.
TEMPLATES = [
    {
        "intro": ("Classify this thyroid nodule as benign or malignant based on "
                  "the following standardized radiomic features."),
        "sentence": "The '{name}' feature is measured at {value:.3f}.",
    },
    {
        "intro": ("Decide whether this thyroid nodule is benign or malignant using "
                  "these standardized radiomic measurements."),
        "sentence": "'{name}': {value:.3f}.",
    },
    {
        "intro": ("Based on the standardized radiomic features below, classify the "
                  "thyroid nodule as benign or malignant."),
        "sentence": "The standardized value of '{name}' is {value:.3f}.",
    },
]


def readable_feature_name(raw_name: str) -> str:
    name = raw_name.split("_")[-1]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip()


def render_prompt(
    feature_names: list[str],
    values: np.ndarray,
    order: list[int] | None = None,
    dropped: int | None = None,
    template: int = 0,
) -> str:
    """Render one prompt. `order` and `dropped` index into `feature_names`."""
    spec = TEMPLATES[template]
    order = list(range(len(feature_names))) if order is None else list(order)
    sentences = [
        spec["sentence"].format(name=readable_feature_name(feature_names[i]), value=values[i])
        for i in order
        if i != dropped
    ]
    return " ".join([spec["intro"], *sentences])


def sample_seed(base_seed: int, sample_key: str) -> int:
    """Per-nodule seed, so variants do not depend on row order or on the other nodules."""
    return (base_seed * 1_000_003 + zlib.crc32(str(sample_key).encode())) % (2**32)


def make_variants(
    feature_names: list[str],
    values: np.ndarray,
    n_queries: int,
    rng: np.random.Generator,
    drop_prob: float = 0.5,
) -> list[dict]:
    """Query 0 is the training prompt; queries 1..n_queries-1 are random perturbations."""
    n_features = len(feature_names)
    variants = [{"query": 0, "template": 0, "dropped": None,
                 "order": list(range(n_features))}]

    for q in range(1, n_queries):
        order = rng.permutation(n_features).tolist()
        dropped = int(rng.integers(n_features)) if rng.random() < drop_prob else None
        template = int(rng.integers(len(TEMPLATES)))
        variants.append({"query": q, "template": template, "dropped": dropped, "order": order})

    for v in variants:
        v["prompt"] = render_prompt(feature_names, values, v["order"], v["dropped"], v["template"])
    return variants


def build_perturbed_table(
    table: pd.DataFrame,
    feature_names: list[str],
    n_queries: int,
    seed: int,
    key_column: str = "sample_id",
    drop_prob: float = 0.5,
) -> pd.DataFrame:
    """Long table: one row per (nodule, query) with the prompt and the perturbation used."""
    rows = []
    values = table[feature_names].to_numpy(dtype=np.float64)

    for i, (_, row) in enumerate(table.iterrows()):
        key = str(row[key_column])
        rng = np.random.default_rng(sample_seed(seed, key))
        for v in make_variants(feature_names, values[i], n_queries, rng, drop_prob):
            rows.append({
                key_column: key,
                "label": int(row["label"]),
                "query": v["query"],
                "template": v["template"],
                "dropped_feature": None if v["dropped"] is None else feature_names[v["dropped"]],
                "order": ",".join(map(str, v["order"])),
                "prompt": v["prompt"],
            })

    return pd.DataFrame(rows)
