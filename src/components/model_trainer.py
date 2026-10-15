import os
import json
import pickle
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.components.data_transformation import DataTransformation
from src.utils.model_utils import evaluate_model


# ==========================================================
# Configuration
# ==========================================================

@dataclass
class ModelTrainerConfig:

    # Path where the final/best trained model will be saved
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )

    # Path where the evaluation metrics of the
    # best model will be saved.
    #
    # These metrics will later be used by
    # Performance Monitoring as the training baseline.
    metrics_file_path: str = os.path.join(
        "artifacts",
        "metrics.json"
    )


# ==========================================================
# Model Trainer
# ==========================================================

class ModelTrainer:

    def __init__(self):

        # Create configuration object
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        # ==================================================
        # Separate Features and Target
        # ==================================================

        # All columns except the last one are features
        X_train = train_array[:, :-1]

        # Last column is the target/label
        y_train = train_array[:, -1]

        # Same separation for testing data
        X_test = test_array[:, :-1]

        y_test = test_array[:, -1]


        # ==================================================
        # Candidate Models
        # ==================================================

        # We train several candidate models and compare
        # their performance.
        models = {

            "Logistic Regression":
                LogisticRegression(),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    random_state=42
                )
        }


        # ==================================================
        # Variables Used to Track the Best Model
        # ==================================================

        best_model = None

        best_model_name = None

        best_metrics = None

        # Start with zero so the first model can
        # become the best model.
        best_accuracy = 0


        # ==================================================
        # MLflow Configuration
        # ==================================================

        # MLflow tracking server
        mlflow.set_tracking_uri(
            "http://localhost:5001"
        )

        # Experiment name
        mlflow.set_experiment(
            "Loan_Prediction_Fresh"
        )


        # ==================================================
        # Train and Evaluate Every Candidate Model
        # ==================================================

        for model_name, model in models.items():

            # Create an MLflow run for this model
            with mlflow.start_run(
                run_name=model_name
            ):

                # ------------------------------------------
                # Train and Evaluate
                # ------------------------------------------

                # evaluate_model:
                #
                # 1. Trains the model
                # 2. Generates predictions
                # 3. Calculates:
                #    - Accuracy
                #    - Precision
                #    - Recall
                #    - F1
                #
                # It returns these metrics as a dictionary.

                metrics = evaluate_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )


                # ------------------------------------------
                # Display Accuracy
                # ------------------------------------------

                print(
                    f"{model_name} Accuracy: "
                    f"{metrics['accuracy']:.4f}"
                )


                # ==================================================
                # Log Model Parameters to MLflow
                # ==================================================

                mlflow.log_param(
                    "model",
                    model_name
                )

                # Logistic Regression has max_iter
                if hasattr(model, "max_iter"):

                    mlflow.log_param(
                        "max_iter",
                        model.max_iter
                    )

                # Decision Tree and Random Forest
                # have random_state
                if hasattr(model, "random_state"):

                    mlflow.log_param(
                        "random_state",
                        model.random_state
                    )


                # ==================================================
                # Log Metrics to MLflow
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
                # Log Model Artifact to MLflow
                # ==================================================

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="loan_prediction_model"
                )


                # ==================================================
                # Check Whether This Is the Best Model
                # ==================================================

                # We are currently selecting the best model
                # based on Accuracy.

                if metrics["accuracy"] > best_accuracy:

                    # Store its accuracy
                    best_accuracy = (
                        metrics["accuracy"]
                    )

                    # Store the actual trained model
                    best_model = model

                    # Store its name
                    best_model_name = model_name

                    # Store ALL metrics of the best model
                    best_metrics = metrics


        # ==================================================
        # Save the Best Model Locally
        # ==================================================

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

        # At this point the loop has finished.
        #
        # best_metrics contains the metrics of the
        # BEST model, not the metrics of every model.
        #
        # Example:
        #
        # {
        #     "accuracy": 0.91,
        #     "precision": 0.89,
        #     "recall": 0.93,
        #     "f1": 0.91
        # }
        #
        # We save these values because Performance
        # Monitoring will later use them as the
        # TRAINING BASELINE.

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
        # Display Best Model
        # ==================================================

        print(
            "\n========== BEST MODEL =========="
        )

        print(
            f"Model      : "
            f"{best_model_name}"
        )

        print(
            f"Accuracy   : "
            f"{best_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision  : "
            f"{best_metrics['precision']:.4f}"
        )

        print(
            f"Recall     : "
            f"{best_metrics['recall']:.4f}"
        )

        print(
            f"F1 Score   : "
            f"{best_metrics['f1']:.4f}"
        )


        # ==================================================
        # Return Results
        # ==================================================

        return (
            best_metrics,
            self.model_trainer_config.trained_model_file_path
        )


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # Create Data Transformation object
    # ------------------------------------------------------

    data_transformation = DataTransformation()


    # ------------------------------------------------------
    # Transform the training and testing datasets
    # ------------------------------------------------------

    train_arr, test_arr, _ = (
        data_transformation.initiate_data_transformation(
            "artifacts/train.csv",
            "artifacts/test.csv"
        )
    )


    # ------------------------------------------------------
    # Create Model Trainer
    # ------------------------------------------------------

    model_trainer = ModelTrainer()


    # ------------------------------------------------------
    # Train and evaluate the candidate models
    # ------------------------------------------------------

    metrics, model_path = (
        model_trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )
    )


    # ------------------------------------------------------
    # Display Final Results
    # ------------------------------------------------------

    print(
        "\n========== RESULT =========="
    )

    print(
        f"Accuracy   : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision  : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score   : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Model saved at : "
        f"{model_path}"
    )