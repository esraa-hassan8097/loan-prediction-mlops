import os
import pickle
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.components.data_transformation import DataTransformation
from src.utils.model_utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        # Separate features and target
        X_train = train_array[:, :-1]
        y_train = train_array[:, -1]

        X_test = test_array[:, :-1]
        y_test = test_array[:, -1]

        # Candidate models
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        best_model = None
        best_model_name = None
        best_metrics = None
        best_accuracy = 0

        mlflow.set_tracking_uri("http://localhost:5001")

        mlflow.set_experiment("Loan_Prediction_Experiment")

        for model_name, model in models.items():

            with mlflow.start_run(run_name=model_name):

                # Train and evaluate
                metrics = evaluate_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )

                print(
                    f"{model_name} Accuracy: {metrics['accuracy']:.4f}"
                )

                # Log parameters
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

                # Log metrics
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

                # Log model artifact
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="loan_prediction_model"
                )

                # Update best model
                if metrics["accuracy"] > best_accuracy:

                    best_accuracy = metrics["accuracy"]
                    best_model = model
                    best_model_name = model_name
                    best_metrics = metrics

        # Save the best model locally
        with open(
            self.model_trainer_config.trained_model_file_path,
            "wb"
        ) as file:

            pickle.dump(
                best_model,
                file
            )

        print("\n========== BEST MODEL ==========")
        print(f"Model      : {best_model_name}")
        print(f"Accuracy   : {best_metrics['accuracy']:.4f}")
        print(f"Precision  : {best_metrics['precision']:.4f}")
        print(f"Recall     : {best_metrics['recall']:.4f}")
        print(f"F1 Score   : {best_metrics['f1']:.4f}")

        return (
            best_metrics,
            self.model_trainer_config.trained_model_file_path
        )


if __name__ == "__main__":

    data_transformation = DataTransformation()

    train_arr, test_arr, _ = (
        data_transformation.initiate_data_transformation(
            "artifacts/train.csv",
            "artifacts/test.csv"
        )
    )

    model_trainer = ModelTrainer()

    metrics, model_path = (
        model_trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )
    )

    print("\n========== RESULT ==========")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1 Score   : {metrics['f1']:.4f}")
    print(f"Model saved at : {model_path}")