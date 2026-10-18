import os
import json
import pickle
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        X_train = train_array[:, :-1]
        y_train = train_array[:, -1]

        X_test = test_array[:, :-1]
        y_test = test_array[:, -1]

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                random_state=42
            )
        }

        best_accuracy = 0
        best_model = None
        best_model_name = None
        best_metrics = None
        best_run_id = None
        best_model_uri = None

        # ==================================================
        # MLflow Configuration
        # ==================================================
        mlflow.set_tracking_uri(
            "http://localhost:5001"
        )

        mlflow.set_experiment(
            "Loan_Prediction_Registery"
        )

        # ==================================================
        # Train and Evaluate Every Candidate Model
        # ==================================================
        for model_name, model in models.items():
            with mlflow.start_run(
                run_name=model_name
            ) as run:

                # ------------------------------------------
                # Train and Evaluate
                # ------------------------------------------
                metrics = evaluate_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )

                print(
                    f"{model_name} Accuracy: "
                    f"{metrics['accuracy']:.4f}"
                )

                # ==================================================
                # Log Parameters
                # ==================================================
                mlflow.log_param(
                    "model",
                    model_name
                )

                if hasattr(model, "max_iter"):
                    mlflow.log_param(
                        "max_iter",
                        model.max_iter
                    )

                if hasattr(model, "random_state"):
                    mlflow.log_param(
                        "random_state",
                        model.random_state
                    )

                # ==================================================
                # Log Metrics
                # ==================================================
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
                    "f1_score",
                    metrics["f1"]
                )

                # ==================================================
                # Log Candidate Model to MLflow
                # ==================================================
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="loan_prediction_model"
                )

                # ==================================================
                # Check Whether This Is the Best Model
                # ==================================================
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    best_model = model
                    best_model_name = model_name
                    best_metrics = metrics
                    best_run_id = run.info.run_id
                    best_model_uri = model_info.model_uri

        if best_model is None:
            raise ValueError(
                "No model was successfully selected."
            )

        # ==================================================
        # Save Best Model Locally
        # ==================================================
        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        with open(
            self.model_trainer_config.trained_model_file_path,
            "wb"
        ) as file:
            pickle.dump(
                best_model,
                file
            )

        # ==================================================
        # Save Best Model Metrics
        # ==================================================
        with open(
            self.model_trainer_config.metrics_file_path,
            "w"
        ) as file:
            json.dump(
                best_metrics,
                file,
                indent=4
            )

        print(
            "\nTraining metrics saved at: "
            f"{self.model_trainer_config.metrics_file_path}"
        )

        # ==================================================
        # Register Only the Best Model
        # ==================================================
        registered_model = mlflow.register_model(
            model_uri=best_model_uri,
            name="LoanApprovalModel"
        )

        print(
            "\n========== REGISTERED MODEL =========="
        )
        print(
            "Model Name : LoanApprovalModel"
        )
        print(
            f"Version    : {registered_model.version}"
        )
        print(
            f"Run ID     : {best_run_id}"
        )
        print(
            f"Model URI  : {best_model_uri}"
        )

        # ==================================================
        # Display Best Model
        # ==================================================
        print(
            "\n========== BEST MODEL =========="
        )
        print(
            f"Model      : {best_model_name}"
        )
        print(
            f"Accuracy   : {best_metrics['accuracy']:.4f}"
        )
        print(
            f"Precision  : {best_metrics['precision']:.4f}"
        )
        print(
            f"Recall     : {best_metrics['recall']:.4f}"
        )
        print(
            f"F1 Score   : {best_metrics['f1']:.4f}"
        )

        # ==================================================
        # Return Results
        # ==================================================
        return (
            best_metrics,
            self.model_trainer_config.trained_model_file_path
        )


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation

    transformation = DataTransformation()

    train_array, test_array, _ = (
        transformation.initiate_data_transformation(
            "artifacts/train.csv",
            "artifacts/test.csv"
        )
    )

    trainer = ModelTrainer()

    metrics, model_path = (
        trainer.initiate_model_trainer(
            train_array,
            test_array
        )
    )

    print(
        "\nModel saved at:",
        model_path
    )
    print(
        "Metrics:",
        metrics
    )