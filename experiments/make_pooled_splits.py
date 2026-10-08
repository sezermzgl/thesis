"""Build the pooled train/validation/calibration/test split used to restore exchangeability.

The official TN3K test set comes from a different distribution than the official trainval set
(report Section 5), so conformal guarantees calibrated on trainval do not transfer to it. This
script pools all 3,493 images and splits them at random, stratified by label and by official
source, so that every split contains the same mix of both distributions.

    python experiments/make_pooled_splits.py            # writes splits/pooled_seed42.csv

Proportions: train 60%, validation 12.5%, calibration 15%, test 12.5%.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
OFFICIAL_SPLITS = ["train", "validation", "calibration", "test"]
FRACTIONS = {"train": 0.60, "validation": 0.125, "calibration": 0.15, "test": 0.125}


def pooled_split(seed: int) -> pd.DataFrame:
    pool = pd.concat([pd.read_csv(REPO / "features" / f"radiomics_{s}.csv",
                                  usecols=["id", "official_split", "label"]) for s in OFFICIAL_SPLITS],
                     ignore_index=True)
    pool["sample_id"] = pool["official_split"].astype(str) + "/" + pool["id"].astype(str)
    assert pool["sample_id"].is_unique

    def strata(df):
        return df["official_split"] + "_" + df["label"].astype(str)

    train, rest = train_test_split(pool, train_size=FRACTIONS["train"], stratify=strata(pool),
                                   random_state=seed)
    val, rest = train_test_split(rest, train_size=FRACTIONS["validation"] / (1 - FRACTIONS["train"]),
                                 stratify=strata(rest), random_state=seed)
    cal, test = train_test_split(
        rest, train_size=FRACTIONS["calibration"] / (FRACTIONS["calibration"] + FRACTIONS["test"]),
        stratify=strata(rest), random_state=seed)

    parts = [d.assign(split=name) for name, d in
             zip(OFFICIAL_SPLITS, [train, val, cal, test])]
    return (pd.concat(parts)[["sample_id", "official_split", "label", "split"]]
            .sort_values("sample_id").reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits = pooled_split(args.seed)
    out = REPO / "splits" / f"pooled_seed{args.seed}.csv"
    out.parent.mkdir(exist_ok=True)
    splits.to_csv(out, index=False)
    print(splits.groupby("split").agg(n=("label", "size"), malignant_share=("label", "mean"),
                                      from_official_test=("official_split", lambda s: (s == "test").mean()))
          .loc[OFFICIAL_SPLITS].round(3).to_string())
    print("Saved:", out)


if __name__ == "__main__":
    main()
