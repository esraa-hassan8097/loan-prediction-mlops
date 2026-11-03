import json
import os
import pickle
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.config.ct_config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
)
from src.logger import logging
from src.utils.model_utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl",
    )

    metrics_file_path: str = os.path.join(
        "artifacts",
        "metrics.json",
    )

    data_metadata_path: str = os.path.join(
        "artifacts",
        "data_metadata.json",
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def load_data_metadata(self):
        metadata_path = (
            self.model_trainer_config.data_metadata_path
        )

        if not os.path.exists(metadata_path):
            logging.warning(
                "Data metadata file not found."
            )

            return {
                "source": "unknown",
                "table": "unknown",
                "data_snapshot": "unknown",
                "row_count": 0,
            }

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
    ):
        logging.info(
            "Entered model training method."
        )

        X_train = train_array[:, :-1]
        y_train = train_array[:, -1]

        X_test = test_array[:, :-1]
        y_test = test_array[:, -1]

        models = {
            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000,
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42,
                ),

            "Random Forest":
                RandomForestClassifier(
                    random_state=42,
                ),
        }

        best_f1 = -1.0
        best_model = None
        best_model_name = None
        best_metrics = None
        best_run_id = None
        best_model_uri = None

        data_metadata = (
            self.load_data_metadata()
        )

        print(
            "\nTraining data metadata:"
        )

        print(
            "Source:",
            data_metadata.get("source"),
        )

        print(
            "Fabric Table:",
            data_metadata.get("table"),
        )

        print(
            "Data Snapshot:",
            data_metadata.get(
                "data_snapshot"
            ),
        )

        print(
            "Row Count:",
            data_metadata.get(
                "row_count"
            ),
        )

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            "Loan_Prediction_Registry"
        )

        for model_name, model in models.items():
            print(
                f"\nTraining model: {model_name}"
            )

            with mlflow.start_run(
                run_name=model_name,
            ) as run:
                metrics = evaluate_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )

                mlflow.log_param(
                    "model_name",
                    model_name,
                )

                mlflow.log_param(
                    "fabric_source",
                    data_metadata.get(
                        "source",
                        "unknown",
                    ),
                )

                mlflow.log_param(
                    "fabric_table",
                    data_metadata.get(
                        "table",
                        "unknown",
                    ),
                )

                mlflow.log_param(
                    "data_snapshot",
                    data_metadata.get(
                        "data_snapshot",
                        "unknown",
                    ),
                )

                mlflow.log_param(
                    "data_row_count",
                    data_metadata.get(
                        "row_count",
                        0,
                    ),
                )

                for (
                    parameter_name,
                    parameter_value,
                ) in model.get_params().items():
                    mlflow.log_param(
                        parameter_name,
                        parameter_value,
                    )

                for (
                    metric_name,
                    metric_value,
                ) in metrics.items():
                    mlflow.log_metric(
                        metric_name,
                        metric_value,
                    )

                model_info = (
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name=(
                            "loan_prediction_model"
                        ),
                    )
                )

                print(
                    f"{model_name} "
                    f"Accuracy: "
                    f"{metrics['accuracy']:.4f}"
                )

                print(
                    f"{model_name} "
                    f"F1: "
                    f"{metrics['f1']:.4f}"
                )

                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]

                    best_model = model
                    best_model_name = (
                        model_name
                    )

                    best_metrics = metrics

                    best_run_id = (
                        run.info.run_id
                    )

                    best_model_uri = (
                        model_info.model_uri
                    )

        if best_model is None:
            raise ValueError(
                "No model was successfully trained."
            )

        os.makedirs(
            "artifacts",
            exist_ok=True,
        )

        with open(
            self.model_trainer_config
            .trained_model_file_path,
            "wb",
        ) as file:
            pickle.dump(
                best_model,
                file,
            )

        with open(
            self.model_trainer_config
            .metrics_file_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                best_metrics,
                file,
                indent=4,
            )

        logging.info(
            "Registering candidate model "
            "in MLflow Model Registry."
        )

        model_version = mlflow.register_model(
            model_uri=best_model_uri,
            name=MODEL_NAME,
        )

        client = MlflowClient()

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=model_version.version,
            key="status",
            value="candidate",
        )

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=model_version.version,
            key="selected_by",
            value="f1",
        )

        print(
            "\nBest candidate model:"
        )

        print(
            "Model:",
            best_model_name,
        )

        print(
            "F1:",
            f"{best_metrics['f1']:.4f}",
        )

        print(
            "MLflow Run ID:",
            best_run_id,
        )

        print(
            "Registered Model:",
            MODEL_NAME,
        )

        print(
            "Candidate Version:",
            model_version.version,
        )

        return {
            "model_name": best_model_name,
            "version": str(
                model_version.version
            ),
            "run_id": best_run_id,
            "model_uri": best_model_uri,
            "accuracy": best_metrics[
                "accuracy"
            ],
            "precision": best_metrics[
                "precision"
            ],
            "recall": best_metrics[
                "recall"
            ],
            "f1": best_metrics["f1"],
        }