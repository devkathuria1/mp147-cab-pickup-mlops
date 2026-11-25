import sys
sys.path.insert(0, "src")
import json
from mp147.data import load_raw, find_issues, clean_train, load_later, sha256_of, TARGET

tr = load_raw("data/raw/train.csv")
print("sha256:", sha256_of("data/raw/train.csv"))
print(json.dumps(find_issues(tr), indent=2, default=str))

c = clean_train(tr)
print("\ncleaned shape:", c.shape)
print("missing per column:\n", c.isna().sum())
print("car_type:", c["car_type"].value_counts().to_dict())
print("target describe:\n", c[TARGET].describe().round(2))

la = load_later("data/raw/later.csv")
print("\nlater shape:", la.shape, "| first date", la["date"].iloc[0], "| last date", la["date"].iloc[-1])
print("later columns:", list(la.columns))