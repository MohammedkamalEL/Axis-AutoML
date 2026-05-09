import mlflow
import mlflow.sklearn
import os

os.makedirs("spaceship_experiments", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///spaceship_experiments/mlflow.db")
mlflow.set_experiment("spaceship_automl")


def log_run(model_name: str, params: dict, metrics: dict, pipeline) -> None:
    """Logs a single AutoML trial result to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Sanitize params — MLflow only accepts int, float, str, bool
        safe_params = {
            k: str(v) if not isinstance(v, (int, float, str, bool)) else v
            for k, v in params.items()
        }
        mlflow.log_params(safe_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="model")


def log_champion(champion_name: str, metrics: dict, pipeline) -> None:
    """Logs the final champion model with a special tag."""
    with mlflow.start_run(run_name=f"CHAMPION_{champion_name}"):
        mlflow.set_tag("role", "champion")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="champion_model")
