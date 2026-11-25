import sys
sys.path.insert(0, "src")
from mp147.data import load_raw, clean_train, load_later
from mp147.monitor import psi_table

train = clean_train(load_raw("data/raw/train.csv"))
later = load_later("data/raw/later.csv")

for method in ("quantile", "equal"):
    t = psi_table(train, later, bins=10, method=method)
    print(method, dict(sorted(t.items(), key=lambda kv: -kv[1])))
    print("  most drifted:", max(t, key=t.get))