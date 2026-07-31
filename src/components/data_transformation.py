import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import pickle


@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        # Creates and returns the preprocessing pipeline.

        numerical_columns = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History"
        ]

        categorical_columns = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area"
        ]

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num_pipeline",
                    numerical_pipeline,
                    numerical_columns
                ),
                (
                    "cat_pipeline",
                    categorical_pipeline,
                    categorical_columns
                )
            ]
        )

        return preprocessor

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        train_df = pd.read_csv(train_path)

        test_df = pd.read_csv(test_path)

        preprocessor = self.get_data_transformer_object()

        target_column = "Loan_Status"

        X_train = train_df.drop(
            columns=[target_column]
        )

        y_train = train_df[target_column]

        X_test = test_df.drop(
            columns=[target_column]
        )

        y_test = test_df[target_column]

        X_train = preprocessor.fit_transform(X_train)

        X_test = preprocessor.transform(X_test)

        print("Original train shape:", train_df.shape)
        print("Original test shape:", test_df.shape)

        print("Transformed X_train shape:", X_train.shape)
        print("Transformed X_test shape:", X_test.shape)

        with open(
            self.data_transformation_config.preprocessor_obj_file_path,
            "wb"
        ) as file:

            pickle.dump(
                preprocessor,
                file
            )

        train_arr = np.c_[
            X_train,
            y_train
        ]

        test_arr = np.c_[
            X_test,
            y_test
        ]

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )


if __name__ == "__main__":

    obj = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        obj.initiate_data_transformation(
            "artifacts/train.csv",
            "artifacts/test.csv"
        )
    )

    print("Train Array Shape:", train_arr.shape)
    print("Test Array Shape :", test_arr.shape)
    print("Preprocessor Path:", preprocessor_path)