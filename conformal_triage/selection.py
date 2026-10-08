"""Feature pre-filtering before mRMR."""

import numpy as np
import pandas as pd


def drop_correlated(X: pd.DataFrame, y, max_corr: float = 0.95) -> tuple[list[str], dict[str, str]]:
    """Remove near-duplicate features (|Pearson r| above `max_corr`); fit on the train split only.

    mRMR (mrmr-selection) penalises a candidate by its mean correlation with all features chosen so
    far, so a near-copy of one chosen feature is diluted by the others and can still be picked:
    on the v2 table it selected both PixelSurfaceRelative and MeshSurfaceRelative, which are the
    same number in 2D. Features are visited from most to least related to the label (|r| with y);
    a feature is kept only if it is within `max_corr` of every feature kept so far, so of two
    near-copies the more label-related one stays.

    Returns the kept features in their original column order, and a dict mapping each dropped
    feature to the kept feature it duplicates.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y, dtype=np.float64)
    corr = X.corr().abs()
    relevance = X.apply(lambda col: abs(np.corrcoef(col.to_numpy(dtype=np.float64), y)[0, 1]))
    order = relevance.fillna(0).sort_values(ascending=False, kind="mergesort").index

    kept, dropped = [], {}
    for f in order:
        twin = next((k for k in kept if corr.at[f, k] > max_corr), None)
        if twin is None:
            kept.append(f)
        else:
            dropped[f] = twin
    kept_set = set(kept)
    return [c for c in X.columns if c in kept_set], dropped
