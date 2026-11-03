import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config.ct_config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PRIMARY_METRIC,
    PRODUCTION_ALIAS,
)


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_champion_version():
    configure_mlflow()
    client = MlflowClient()

    try:
        return client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=PRODUCTION_ALIAS,
        )
    except MlflowException as exc:
        if "alias champion not found" in str(exc).lower():
            return None
        if "registered model" in str(exc).lower() and "not found" in str(exc).lower():
            return None
        raise


def get_champion_metric() -> float | None:
    champion = get_champion_version()

    if champion is None:
        return None

    client = MlflowClient()
    run = client.get_run(champion.run_id)

    return run.data.metrics.get(PRIMARY_METRIC)


def promote_model(version: str) -> None:
    configure_mlflow()
    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=version,
    )


def load_champion_model():
    configure_mlflow()

    model_uri = f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"

    return mlflow.pyfunc.load_model(model_uri)