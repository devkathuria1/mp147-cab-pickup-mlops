import numpy as np
import pandas as pd
from .data import NUM_FEATURES, TARGET, GROUP_COL

FEATURES = NUM_FEATURES + ["car_type", "zone"]


def psi(expected, actual, bins=10, method="quantile"):
    """Population Stability Index of `actual` (later) against `expected` (train)."""
    e = pd.Series(expected).dropna().to_numpy(float)
    a = pd.Series(actual).dropna().to_numpy(float)
    if method == "quantile":
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.linspace(e.min(), e.max(), bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.clip(np.histogram(e, edges)[0] / len(e), 1e-4, None)
    a_pct = np.clip(np.histogram(a, edges)[0] / len(a), 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_table(train_df, later_df, bins=10, method="quantile"):
    return {c: round(psi(train_df[c], later_df[c], bins, method), 4) for c in NUM_FEATURES}


def mae_by_group(model, df, group_col=GROUP_COL):
    pred = model.predict(df[FEATURES])
    err = (df[TARGET].to_numpy() - pred)
    out = {}
    for g in sorted(df[group_col].unique()):
        m = (df[group_col] == g).to_numpy()
        out[g] = round(float(np.abs(err[m]).mean()), 4)
    return out