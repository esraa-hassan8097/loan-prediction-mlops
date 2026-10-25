import json
import os
import pickle
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.logger import logging
from src.utils.model_utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )

    metrics_file_path: str = os.path.join(
        "artifacts",
        "metrics.json"
    )

    data_metadata_path: str = os.path.join(
        "artifacts",
        "data_metadata.json"
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
                "row_count": 0
            }

        with open(
            metadata_path,
            "r"
        ) as file:
            metadata = json.load(file)

        return metadata

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):
        logging.info("Entered model training method.")

        X_train = train_array[:, :-1]
        y_train = train_array[:, -1]

        X_test = test_array[:, :-1]
        y_test = test_array[:, -1]

        models = {
            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    random_state=42
                )
        }

        best_accuracy = 0
        best_model = None
        best_model_name = None
        best_metrics = None
        best_run_id = None
        best_model_uri = None

        data_metadata = self.load_data_metadata()

        print(
            "\nTraining data metadata:"
        )

        print(
            "Source:",
            data_metadata.get("source")
        )

        print(
            "Fabric Table:",
            data_metadata.get("table")
        )

        print(
            "Data Snapshot:",
            data_metadata.get("data_snapshot")
        )

        print(
            "Row Count:",
            data_metadata.get("row_count")
        )

        mlflow.set_tracking_uri(
            "http://localhost:5001"
        )

        mlflow.set_experiment(
            "Loan_Prediction_Registery"
        )

        for model_name, model in models.items():

            print(
                f"\nTraining model: {model_name}"
            )

            with mlflow.start_run(
                run_name=model_name
            ) as run:

                metrics = evaluate_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )

                mlflow.log_param(
                    "model_name",
                    model_name
                )

                mlflow.log_param(
                    "fabric_source",
                    data_metadata.get(
                        "source",
                        "unknown"
                    )
                )

                mlflow.log_param(
                    "fabric_table",
                    data_metadata.get(
                        "table",
                        "unknown"
                    )
                )

                mlflow.log_param(
                    "data_snapshot",
                    data_metadata.get(
                        "data_snapshot",
                        "unknown"
                    )
                )

                mlflow.log_param(
                    "data_row_count",
                    data_metadata.get(
                        "row_count",
                        0
                    )
                )

                for parameter_name, parameter_value \
                        in model.get_params().items():

                    mlflow.log_param(
                        parameter_name,
                        parameter_value
                    )

                mlflow.log_metric(
                    "accuracy",
                    metrics["accuracy"]
                )

                mlflow.log_metric(
                    "precision",
                    metrics["precision"]
                )

                mlflow.log_metric(
                    "recall",
                    metrics["recall"]
                )

                mlflow.log_metric(
                    "f1",
                    metrics["f1"]
                )

                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="loan_prediction_model"
                )

                print(
                    f"{model_name} Accuracy: "
                    f"{metrics['accuracy']:.4f}"
                )

                if metrics["accuracy"] > best_accuracy:

                    best_accuracy = (
                        metrics["accuracy"]
                    )

                    best_model = model

                    best_model_name = model_name

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
            exist_ok=True
        )

        with open(
            self.model_trainer_config
                .trained_model_file_path,
            "wb"
        ) as file:

            pickle.dump(
                best_model,
                file
            )

        with open(
            self.model_trainer_config
                .metrics_file_path,
            "w"
        ) as file:

            json.dump(
                best_metrics,
                file,
                indent=4
            )

        registered_model = (
            mlflow.register_model(
                model_uri=best_model_uri,
                name="LoanApprovalModel"
            )
        )

        print(
            "\nBest Model:",
            best_model_name
        )

        print(
            "Accuracy:",
            round(
                best_metrics["accuracy"],
                4
            )
        )

        print(
            "Precision:",
            round(
                best_metrics["precision"],
                4
            )
        )

        print(
            "Recall:",
            round(
                best_metrics["recall"],
                4
            )
        )

        print(
            "F1:",
            round(
                best_metrics["f1"],
                4
            )
        )

        print(
            "\nRegistered Model:",
            registered_model.name
        )

        print(
            "Version:",
            registered_model.version
        )

        print(
            "Run ID:",
            best_run_id
        )

        print(
            "Model URI:",
            best_model_uri
        )

        print(
            "\nTraining Data:"
        )

        print(
            "Fabric Table:",
            data_metadata.get("table")
        )

        print(
            "Snapshot:",
            data_metadata.get(
                "data_snapshot"
            )
        )

        print(
            "Row Count:",
            data_metadata.get(
                "row_count"
            )
        )

        logging.info(
            "Model training completed successfully."
        )

        return best_metrics


if __name__ == "__main__":

    from src.components.data_transformation \
        import DataTransformation

    transformation = DataTransformation()

    train_array, test_array, _ = (
        transformation
        .initiate_data_transformation(
            "artifacts/train.csv",
            "artifacts/test.csv"
        )
    )

    trainer = ModelTrainer()

    trainer.initiate_model_trainer(
        train_array,
        test_array
    )