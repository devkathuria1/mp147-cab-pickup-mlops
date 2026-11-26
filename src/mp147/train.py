"""train.py - train models and track them in MLflow (Vedika).

Each run logs: data_sha256, model_name, model params, mae_valid, and the model.
Models are NOT registered here. Registration of the best run comes later.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # so relative paths in params.yaml always work

from mp147.data import clean_train, load_raw  # noqa: E402
from mp147.features import FEATURES, TARGET, build_pipeline  # noqa: E402


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def sha256_of_file(path):
    """SHA-256 of the ORIGINAL file bytes (before any cleaning)."""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def get_training_data(params):
    """Load train.csv and clean it with Devanshu's clean_train()."""
    raw = load_raw(params["data"]["train_path"])
    return clean_train(raw)


def train_one(model_name, model_params, params, data, data_sha):
    """Train one model, log it to MLflow, return (run_id, mae_valid)."""
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
    )

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("data_sha256", data_sha)
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model_params)

        pipe = build_pipeline(model_name, **model_params)
        pipe.fit(X_train, y_train)

        mae_valid = mean_absolute_error(y_valid, pipe.predict(X_valid))
        mlflow.log_metric("mae_valid", mae_valid)

        mlflow.sklearn.log_model(
            pipe,
            name="model",
            serialization_format="cloudpickle",
            input_example=X_train.head(3),
        )
        return run.info.run_id, mae_valid


def get_configs(random_state):
    """The settings we compare. Each is (model_name, params)."""
    return [
        ("random_forest", {"n_estimators": 100, "max_depth": 8, "random_state": random_state}),
        ("random_forest", {"n_estimators": 300, "max_depth": None, "random_state": random_state}),
        ("gradient_boosting", {"learning_rate": 0.1, "n_estimators": 200, "random_state": random_state}),
        ("ridge", {"alpha": 1.0}),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N configs (use 1 for a test run)")
    args = parser.parse_args()

    params = load_params()
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    data_sha = sha256_of_file(params["data"]["train_path"])
    data = get_training_data(params)
    print(f"Clean rows: {len(data)} | data_sha256: {data_sha}")

    configs = get_configs(params["model"]["random_state"])
    if args.limit:
        configs = configs[: args.limit]

    for model_name, model_params in configs:
        run_id, mae = train_one(model_name, model_params, params, data, data_sha)
        print(f"{model_name:18s} {model_params} -> mae_valid={mae:.3f} run_id={run_id}")


if __name__ == "__main__":
    main()