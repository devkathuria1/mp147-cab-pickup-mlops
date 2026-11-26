"""retrain.py - v2 = train + first 70% of later; compare with v1 on the last 30%; 12% rule (Devanshu).

Dry run (default): only prints MAEs and the decision. Touches nothing.
--register: logs v2 to MLflow, registers it as the next version, and if the rule says
promote, moves alias 'champion' to v2 and overwrites models/model.joblib.
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from mp147.data import clean_train, load_later, load_raw, sha256_of  # noqa: E402
from mp147.features import FEATURES, TARGET, build_pipeline  # noqa: E402
from mp147.monitor import decide, mae_by_group, split_later  # noqa: E402

V2_MODEL = "ridge"                 # same model type and settings as v1's best run
V2_PARAMS = {"alpha": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    args = ap.parse_args()

    params = yaml.safe_load(open("params.yaml"))
    train = clean_train(load_raw(params["data"]["train_path"]))
    later = load_later(params["data"]["later_path"])          # sorted by date
    first70, last30 = split_later(later)                      # 210 rows / 90 rows
    print(f"train {len(train)} | later first 70%: {len(first70)} "
          f"({first70['date'].iloc[0]} to {first70['date'].iloc[-1]}) | "
          f"check set: {len(last30)} ({last30['date'].iloc[0]} to {last30['date'].iloc[-1]})")

    v1 = joblib.load(params["model"]["model_path"])
    mae_v1 = mean_absolute_error(last30[TARGET], v1.predict(last30[FEATURES]))

    fit_df = pd.concat([train[FEATURES + [TARGET]], first70[FEATURES + [TARGET]]],
                       ignore_index=True)
    v2 = build_pipeline(V2_MODEL, **V2_PARAMS).fit(fit_df[FEATURES], fit_df[TARGET])
    mae_v2 = mean_absolute_error(last30[TARGET], v2.predict(last30[FEATURES]))

    decision, gain = decide(mae_v1, mae_v2)
    print(f"\nmae_v1_on_check = {mae_v1:.4f}")
    print(f"mae_v2_on_check = {mae_v2:.4f}")
    print(f"gain = {gain*100:.1f}% (need >= 12%) -> decision: {decision}")
    print("v1 per zone on check set:", mae_by_group(v1, last30))
    print("v2 per zone on check set:", mae_by_group(v2, last30))

    if not args.register:
        print("\nDRY RUN: nothing was saved. Run with --register to log, register and apply.")
        return

    reg_name = params["mlflow"]["registered_model_name"]
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"] + "-v2")  # v1's artifact path is on Vedika's laptop
    client = MlflowClient()
    existing = client.search_model_versions(f"name='{reg_name}'")
    if len(existing) != 1:
        sys.exit(f"STOP: expected exactly 1 registered version (v1), found {len(existing)}.")

    with mlflow.start_run(run_name="v2_retrain") as run:
        mlflow.log_param("data_sha256", sha256_of(params["data"]["train_path"]))
        mlflow.log_param("model_name", V2_MODEL)
        mlflow.log_params(V2_PARAMS)
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("later_rows_used", len(first70))
        mlflow.log_metric("mae_check_v1", mae_v1)
        mlflow.log_metric("mae_check_v2", mae_v2)
        mlflow.log_metric("mae_gain_pct", gain * 100)
        mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle",
                                 input_example=fit_df[FEATURES].head(3))
        run_id = run.info.run_id

    mv = mlflow.register_model(f"runs:/{run_id}/model", reg_name)
    print(f"\nRegistered {reg_name} version {mv.version} (run {run_id})")

    if decision == "promote":
        client.set_registered_model_alias(reg_name, "champion", mv.version)
        joblib.dump(v2, params["model"]["model_path"])
        print(f"PROMOTED: alias champion -> version {mv.version}; models/model.joblib replaced by v2")
    else:
        print("KEEP: alias champion stays on version 1; models/model.joblib unchanged")


if __name__ == "__main__":
    main()