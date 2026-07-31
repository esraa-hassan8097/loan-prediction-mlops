import os  # use it to build file paths
import json

import pandas as pd  # We'll use it to read: artifacts/train.csv
from dataclasses import dataclass


@dataclass
class DataValidationConfig:

    validation_report_path: str = os.path.join(
        "artifacts",
        "validation_report.json"
    )


class DataValidation:

    def __init__(self):

        self.validation_config = DataValidationConfig()

    def initiate_data_validation(
        self,
        train_path,
        test_path
    ):

        train_df = pd.read_csv(train_path)

        test_df = pd.read_csv(test_path)

        required_columns = [
            "Loan_ID",
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
            "Loan_Status"
        ]

        missing_columns = []

        for column in required_columns:

            if column not in train_df.columns:

                missing_columns.append(column)

        if train_df.empty:

            validation_status = False

            message = "Training dataset is empty."

        elif test_df.empty:

            validation_status = False

            message = "Testing dataset is empty."

        elif len(missing_columns) > 0:

            validation_status = False

            message = f"Missing columns: {missing_columns}"

        else:

            validation_status = True

            message = "Validation successful."

        missing_values = train_df.isnull().sum().to_dict()
        # isnull() returns a DataFrame of the same shape as train_df,
        # but with True for missing values and False for non-missing values.
        # sum() counts the number of True values in each column.
        # to_dict() converts the result into a dictionary where keys are
        # column names and values are counts of missing values.

        # Find columns that actually have missing values

        columns_with_missing_values = {}

        for column, count in missing_values.items():

            if count > 0:

                columns_with_missing_values[column] = int(count)

        validation_report = {

            "validation_status": validation_status,

            "message": message,

            "missing_columns": missing_columns,

            "columns_with_missing_values": columns_with_missing_values

        }

        with open(
            self.validation_config.validation_report_path,
            "w"
        ) as report_file:

            json.dump(
                validation_report,
                report_file,
                indent=4
            )

        # open() opens a file so Python can read from it or write to it.
        # We're writing a new validation report every time the pipeline runs,
        # so "w" is the right choice.
        # "with" is used because when the block finishes,
        # Python automatically closes the file,
        # even if an error occurs.
        # report_file is a variable that represents the file object.
        # We can use it to write to the file.
        # json.dump() converts the Python dictionary called
        # validation_report into JSON and writes it to the file.

        return validation_status


if __name__ == "__main__":  # Only execute this code if this file is run directly.

    obj = DataValidation()  # Python creates a DataValidation object and automatically calls __init__(), which creates self.validation_config.

    status = obj.initiate_data_validation(
        "artifacts/train.csv",
        "artifacts/test.csv"
    )

    print(f"Validation Status: {status}")