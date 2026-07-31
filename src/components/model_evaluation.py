import os
import pickle
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation


@dataclass
class ModelEvaluationConfig:

    best_model_file_path: str = os.path.join(
        "artifacts",
        "best_model.pkl"
    )


class ModelEvaluation:

    def __init__(self):

        self.model_evaluation_config = ModelEvaluationConfig()

    def initiate_model_evaluation(
        self,
        train_array,
        test_array
    ):

        X_train = train_array[:, :-1]
        y_train = train_array[:, -1]

        X_test = test_array[:, :-1]
        y_test = test_array[:, -1]

        models = {

            "Logistic Regression": LogisticRegression(),

            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                random_state=42
            )

        }

        model_report = {}

        best_model = None
        best_model_name = None
        best_model_score = -1

        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                pos_label="Y"
            )

            recall = recall_score(
                y_test,
                predictions,
                pos_label="Y"
            )

            f1 = f1_score(
                y_test,
                predictions,
                pos_label="Y"
            )

            model_report[name] = {

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1": f1

            }

            if f1 > best_model_score:

                best_model_score = f1

                best_model = model

                best_model_name = name

        with open(
            self.model_evaluation_config.best_model_file_path,
            "wb"
        ) as file:

            pickle.dump(
                best_model,
                file
            )

        print("\n========== Model Report ==========\n")

        for name, metrics in model_report.items():

            print(name)

            print(f"Accuracy : {metrics['accuracy']:.4f}")

            print(f"Precision: {metrics['precision']:.4f}")

            print(f"Recall   : {metrics['recall']:.4f}")

            print(f"F1 Score : {metrics['f1']:.4f}")

            print("-" * 35)

        print(f"\nBest Model : {best_model_name}")

        print(f"Best F1 Score : {best_model_score:.4f}")

        return (
            best_model_name,
            best_model_score
        )


if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    validation = DataValidation()

    validation_status = (
        validation.initiate_data_validation(
            train_path,
            test_path
        )
    )

    if not validation_status:

        print("Data Validation Failed!")

        exit()

    transformation = DataTransformation()

    train_array, test_array, preprocessor_path = (
        transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    evaluation = ModelEvaluation()

    best_model_name, best_model_score = (
        evaluation.initiate_model_evaluation(
            train_array,
            test_array
        )
    )

    print("\n========== Final Result ==========\n")

    print(f"Best Model: {best_model_name}")

    print(f"Best F1 Score: {best_model_score:.4f}")

    print(f"Preprocessor: {preprocessor_path}")

    print(
        f"Best Model File: {evaluation.model_evaluation_config.best_model_file_path}"
    )