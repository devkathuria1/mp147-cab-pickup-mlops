import sys
sys.path.insert(0, "src")
import joblib
import numpy as np
from mp147.data import load_later, TARGET
from mp147.monitor import mae_by_group, FEATURES

model = joblib.load("models/model.joblib")
later = load_later("data/raw/later.csv")

pred = model.predict(later[FEATURES])
overall = float(np.abs(later[TARGET].to_numpy() - pred).mean())
by_zone = mae_by_group(model, later)

print("model steps:", [name for name, _ in model.steps], "| regressor:", type(model.steps[-1][1]).__name__)
print("overall MAE of v1 on later:", round(overall, 4))
print("MAE per zone:", by_zone)
print("worst served zone:", max(by_zone, key=by_zone.get))