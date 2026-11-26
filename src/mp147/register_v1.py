"""register_v1.py - register the BEST run as version 1, set alias, save joblib (Vedika)."""

import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from mp147.data import clean_train, load_raw  # noqa: E402
from mp147.features import FEATURES  # noqa: E402

params = yaml.safe_load(open("params.yaml"))
mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
client = MlflowClient()

exp_name = params["mlflow"]["experiment_name"]
reg_name = params["mlflow"]["registered_model_name"]
model_path = params["model"]["model_path"]

# 1. Find the run with the LOWEST mae_valid
exp = client.get_experiment_by_name(exp_name)
best = client.search_runs(
    [exp.experiment_id], order_by=["metrics.mae_valid ASC"], max_results=1
)[0]
best_run_id = best.info.run_id
print("best_run_id:", best_run_id)
print("model_name :", best.data.params["model_name"])
print("mae_valid  :", round(best.data.metrics["mae_valid"], 4))

# 2. Safety: v1 must be the first version ever registered
existing = client.search_model_versions(f"name='{reg_name}'")
if existing:
    sys.exit(f"STOP: '{reg_name}' already has {len(existing)} version(s). Tell me before going on.")

# 3. Register ONLY the best run, then set the alias
mv = mlflow.register_model(f"runs:/{best_run_id}/model", reg_name)
print("registered version:", mv.version)
client.set_registered_model_alias(reg_name, "champion", mv.version)

# 4. Save the same pipeline to models/model.joblib
pipe = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
os.makedirs(Path(model_path).parent, exist_ok=True)
joblib.dump(pipe, model_path)
print("saved:", model_path)

# 5. Verify: joblib file on 5 raw rows (8 columns only)
rows = clean_train(load_raw(params["data"]["train_path"]))[FEATURES].head(5)
print("joblib predictions :", pipe.__class__.__name__, joblib.load(model_path).predict(rows).round(2))

# 6. Verify: load through the alias
champ = mlflow.sklearn.load_model(f"models:/{reg_name}@champion")
print("champion predictions:", champ.predict(rows).round(2))

# 7. Proof: versions and alias
for v in client.search_model_versions(f"name='{reg_name}'"):
    print("version", v.version, "| run_id", v.run_id, "| aliases", v.aliases)
print("alias champion -> version", client.get_model_version_by_alias(reg_name, "champion").version)