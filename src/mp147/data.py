import hashlib
import numpy as np
import pandas as pd

TARGET = "pickup_eta_min"
LEAK_COL = "actual_pickup_after_min"
GROUP_COL = "zone"
NUM_FEATURES = ["drivers_nearby", "distance_to_driver_km", "hour",
                "rain_mm", "surge", "road_closures"]
TEXT_COLS = ["car_type", "zone"]
SENTINEL_CANDIDATES = [-1, -99, -999, -9999, 99, 999, 9999]
KNOWN_FACTORS = [10, 12, 60, 100, 1000]
HOLDOUT_PATH = "data/raw/holdout_features.csv"


def load_raw(path):
    return pd.read_csv(path)


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------- helpers that detect each problem ----------
def _collapse_levels(series):
    """Strip, lower-case, drop a trailing dot, then map short forms
    (like 'aut') to the longer level they start (like 'auto')."""
    s = series.map(lambda v: v if pd.isna(v) else str(v).strip().lower().rstrip("."))
    levels = sorted(s.dropna().unique(), key=len, reverse=True)
    mapping = {}
    for v in levels:
        mapping[v] = v
        for w in levels:
            if len(w) > len(v) and w.startswith(v):
                mapping[v] = w
                break
    return s.map(lambda v: v if pd.isna(v) else mapping[v])


def _detect_sentinels(df):
    found = []
    for c in NUM_FEATURES:
        if c not in df.columns:
            continue
        vc = df[c].value_counts()
        for v in SENTINEL_CANDIDATES:
            if v in vc.index and vc[v] >= 5:
                rest = df.loc[df[c] != v, c]
                if v < rest.min() or v > rest.max():
                    found.append({"column": c, "value": int(v), "count": int(vc[v])})
    return found


def _mask_sentinels(df):
    d = df.copy()
    d[NUM_FEATURES] = d[NUM_FEATURES].astype(float)
    for s in _detect_sentinels(d):
        d.loc[d[s["column"]] == s["value"], s["column"]] = np.nan
    return d


def _detect_unit_mixup(df):
    d = _mask_sentinels(df).drop_duplicates()
    for c in NUM_FEATURES:
        med = d.groupby(GROUP_COL)[c].median()
        for g in med.index:
            others = med.drop(g).median()
            if others > 0 and med[g] > 0:
                ratio = med[g] / others
                for f in KNOWN_FACTORS:
                    if abs(ratio / f - 1) < 0.25:
                        return {"column": c, "group_value": g, "factor": float(f)}
                    if abs(ratio * f - 1) < 0.25:
                        return {"column": c, "group_value": g, "factor": 1.0 / f}
    return None


def _detect_stuck(df, min_days=7, min_rows=10):
    d = _mask_sentinels(df).drop_duplicates().copy()
    d["_date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["_date", "record_id"]).reset_index(drop=True)
    for c in NUM_FEATURES:
        run_id = (d[c] != d[c].shift()).cumsum()
        runs = d.groupby(run_id).agg(start=("_date", "min"), end=("_date", "max"),
                                     rows=(c, "size"), value=(c, "first"))
        runs["days"] = (runs["end"] - runs["start"]).dt.days + 1
        good = runs[(runs["days"] >= min_days) & (runs["rows"] >= min_rows)]
        if len(good):
            b = good.sort_values("rows", ascending=False).iloc[0]
            return {"column": c, "start_date": b["start"].strftime("%Y-%m-%d"),
                    "end_date": b["end"].strftime("%Y-%m-%d"), "value": float(b["value"])}
    return None


def _typo_mask(df):
    return df[TARGET] > 5 * df[TARGET].median()


def _detect_spelling(df):
    for c in TEXT_COLS:
        raw = df[c].nunique()
        clean = _collapse_levels(df[c]).nunique()
        if raw != clean:
            return {"column": c, "true_levels": int(clean), "raw_levels": int(raw)}
    return None


def _detect_leak(df, holdout_df):
    out = []
    for c in df.columns:
        if c not in holdout_df.columns and c != TARGET and pd.api.types.is_numeric_dtype(df[c]):
            gap = float((df[c] - df[TARGET]).abs().median())
            out.append({"column": c, "median_abs_diff": round(gap, 3)})
    return sorted(out, key=lambda x: x["median_abs_diff"])


# ---------- the report that fills findings.yaml ----------
def _block(present, **kw):
    return {"present": bool(present), **{k: (v if present else None) for k, v in kw.items()}}


def find_issues(df, holdout_df=None, verbose=True):
    if holdout_df is None:
        holdout_df = pd.read_csv(HOLDOUT_PATH)
    leak = _detect_leak(df, holdout_df)
    leak_hit = leak[0] if leak and leak[0]["median_abs_diff"] < 2 else None
    unit = _detect_unit_mixup(df)
    sent = _detect_sentinels(df)
    sent = sent[0] if sent else None
    dups = int(df.duplicated().sum())
    stuck = _detect_stuck(df)
    typos = int(_typo_mask(df).sum())
    spell = _detect_spelling(df)

    issues = {
        "leaking_column": _block(leak_hit, column=leak_hit["column"] if leak_hit else None),
        "unit_mixup": _block(unit, column=unit and unit["column"],
                             group_value=unit and unit["group_value"], factor=unit and unit["factor"]),
        "sentinel_value": _block(sent, column=sent and sent["column"],
                                 value=sent and sent["value"], count=sent and sent["count"]),
        "duplicate_rows": _block(dups > 0, count=dups),
        "stuck_sensor": _block(stuck, column=stuck and stuck["column"],
                               start_date=stuck and stuck["start_date"], end_date=stuck and stuck["end_date"]),
        "target_typos": _block(typos > 0, count=typos),
        "spelling_variants": _block(spell, column=spell and spell["column"],
                                    true_levels=spell and spell["true_levels"]),
    }
    if verbose:
        print("leak candidates (median |col - target|):", leak)
        print("unit mixup:", unit)
        print("sentinels:", _detect_sentinels(df))
        print("extra duplicate rows:", dups)
        print("stuck sensor:", stuck)
        print("typo targets (> 5 x median):", typos, "| median target:", df[TARGET].median())
        print("spelling:", spell)
    return issues


# ---------- cleaning ----------
def clean_train(df):
    d = df.drop_duplicates().reset_index(drop=True).copy()
    d[NUM_FEATURES] = d[NUM_FEATURES].astype(float)

    for s in _detect_sentinels(d):                       # sentinel -> missing
        d.loc[d[s["column"]] == s["value"], s["column"]] = np.nan

    u = _detect_unit_mixup(d)                            # unit mixup -> divide
    if u:
        m = d[GROUP_COL] == u["group_value"]
        d.loc[m, u["column"]] = d.loc[m, u["column"]] / u["factor"]

    st = _detect_stuck(d)                                # stuck sensor -> missing
    if st:
        dates = pd.to_datetime(d["date"])
        m = (dates >= st["start_date"]) & (dates <= st["end_date"])
        d.loc[m, st["column"]] = np.nan

    m = _typo_mask(d)                                    # target typed 10x too big
    d.loc[m, TARGET] = d.loc[m, TARGET] / 10

    for c in TEXT_COLS:                                  # spelling variants
        d[c] = _collapse_levels(d[c])

    d = d.drop(columns=[LEAK_COL], errors="ignore")      # leaking column
    return d.reset_index(drop=True)


def load_later(path):
    d = pd.read_csv(path)
    d["_date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["_date", "record_id"]).drop(columns="_date").reset_index(drop=True)
    return d.drop(columns=[LEAK_COL], errors="ignore")