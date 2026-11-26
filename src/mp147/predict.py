"""Predict the September holdout with the champion model."""
import os

import joblib
import pandas as pd
import yaml

from mp147.features import FEATURES

EXPECTED_ROWS = 250


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_champion(params):
    # Only try MLflow if the database file really exists. MLflow would
    # otherwise create an empty mlflow.db, which clashes with Vedika's file.
    db_file = params["mlflow"]["tracking_uri"].replace("sqlite:///", "")
    if os.path.exists(db_file):
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
            name = params["mlflow"]["registered_model_name"]
            model = mlflow.sklearn.load_model(f"models:/{name}@champion")
            print("Loaded champion from the MLflow registry")
            return model
        except Exception as error:
            print(f"MLflow load failed ({error}). Using the joblib file instead.")
    model_path = params["model"]["model_path"]
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return model


def main():
    params = load_params()
    model = load_champion(params)

    holdout = pd.read_csv(params["data"]["holdout_path"])
    predictions = model.predict(holdout[FEATURES])

    result = pd.DataFrame(
        {"record_id": holdout["record_id"], "prediction": predictions}
    )

    if len(result) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(result)}")
    if result["prediction"].isna().any():
        raise ValueError("Some predictions are empty")
    if not result["record_id"].is_unique:
        raise ValueError("Duplicate record_id found")

    out_path = params["output"]["predictions_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Saved {len(result)} predictions to {out_path}")


if __name__ == "__main__":
    main()