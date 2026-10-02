"""Perturbed feature prompts for the conformal triage layer.

Every prompt keeps the training format (same template, same feature order), because the
fine-tuned model was trained on that single format and moves far off-distribution otherwise:
in the first run, paraphrased templates were never classified as malignant and shuffling the
feature order alone flipped 29% of the answers.

Per nodule the long table holds three kinds of queries:
    base    the exact training prompt (one per nodule)
    jitter  Gaussian noise N(0, sd^2) added to every standardized value; one block of
            `n_jitter` queries per noise level. Base + one jitter block = one vote signal.
    drop    one feature removed, values unchanged; one query per feature. Used for the
            feature-dependence analysis and as a second vote signal (base + all drops).
"""

import re
import zlib

import numpy as np
import pandas as pd

INTRO = ("Classify this thyroid nodule as benign or malignant based on "
         "the following standardized radiomic features.")
SENTENCE = "The '{name}' feature is measured at {value:.3f}."


def readable_feature_name(raw_name: str) -> str:
    name = raw_name.split("_")[-1]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip()


def render_prompt(feature_names: list[str], values: np.ndarray, dropped: int | None = None) -> str:
    """Training-format prompt (identical to build_prompt in A11), optionally without one feature."""
    sentences = [
        SENTENCE.format(name=readable_feature_name(name), value=value)
        for i, (name, value) in enumerate(zip(feature_names, values))
        if i != dropped
    ]
    return " ".join([INTRO, *sentences])


def sample_seed(base_seed: int, sample_key: str) -> int:
    """Per-nodule seed, so variants do not depend on row order or on the other nodules."""
    return (base_seed * 1_000_003 + zlib.crc32(str(sample_key).encode())) % (2**32)


def make_variants(
    feature_names: list[str],
    values: np.ndarray,
    rng: np.random.Generator,
    n_jitter: int,
    jitter_sds: list[float],
    include_drops: bool = True,
) -> list[dict]:
    values = np.asarray(values, dtype=np.float64)
    variants = [{"kind": "base", "noise_sd": 0.0, "dropped": None, "values": values}]
    for sd in jitter_sds:
        for _ in range(n_jitter):
            variants.append({"kind": "jitter", "noise_sd": float(sd), "dropped": None,
                             "values": values + rng.normal(0.0, sd, size=values.shape)})
    if include_drops:
        for j in range(len(feature_names)):
            variants.append({"kind": "drop", "noise_sd": 0.0, "dropped": j, "values": values})

    for q, v in enumerate(variants):
        v["query"] = q
        v["prompt"] = render_prompt(feature_names, v["values"], v["dropped"])
    return variants


def build_perturbed_table(
    table: pd.DataFrame,
    feature_names: list[str],
    n_jitter: int,
    jitter_sds: list[float],
    seed: int,
    key_column: str = "sample_id",
    include_drops: bool = True,
) -> pd.DataFrame:
    """Long table: one row per (nodule, query) with the prompt and the perturbation used."""
    rows = []
    values = table[feature_names].to_numpy(dtype=np.float64)

    for i, (_, row) in enumerate(table.iterrows()):
        key = str(row[key_column])
        rng = np.random.default_rng(sample_seed(seed, key))
        for v in make_variants(feature_names, values[i], rng, n_jitter, jitter_sds, include_drops):
            rows.append({
                key_column: key,
                "label": int(row["label"]),
                "query": v["query"],
                "kind": v["kind"],
                "noise_sd": v["noise_sd"],
                "dropped_feature": None if v["dropped"] is None else feature_names[v["dropped"]],
                "prompt": v["prompt"],
            })

    return pd.DataFrame(rows)
