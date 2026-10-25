"""
data_drift.py

Purpose
-------
This component is responsible for detecting Data Drift.

Data Drift occurs when the statistical distribution of the
input features in production differs from the distribution
of the features used during model training.

Current Version
---------------
This implementation compares the mean of every numerical
feature between the training dataset and the production dataset.

Future Improvements
-------------------
- Compare categorical features
- Use statistical tests (KS Test / PSI)
- Integrate Evidently AI
- Generate HTML reports
- Trigger monitoring alerts
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging

# ==========================================================
# Configuration
# ==========================================================

@dataclass
class DataDriftConfig:
    """
    Stores all configuration related to Data Drift.

    Centralizing configuration makes the code easier
    to maintain and modify.
    """

    train_data_path = os.path.join(
        "artifacts",
        "train.csv"
    )

    production_data_path = os.path.join(
        "artifacts",
        "production_data.csv"
    )

    drift_report_path = os.path.join(
        "artifacts",
        "drift_report.csv"
    )

    # Percentage threshold used to decide
    # whether a feature has drifted.
    drift_threshold = 20


# ==========================================================
# Data Drift Component
# ==========================================================

class DataDrift:
    """
    Compares the training dataset with the production dataset
    to determine whether Data Drift exists.
    """

    def __init__(self):
        self.config = DataDriftConfig()

    def detect_data_drift(self):
        """
        Detect Data Drift by comparing the mean of numerical
        features between the training and production datasets.

        Returns
        -------
        DataFrame
            Drift analysis results.
        """

        try:

            # ==================================================
            # Read datasets
            # ==================================================

            logging.info("Reading training dataset...")

            train_df = pd.read_csv(
                self.config.train_data_path
            )

            logging.info("Reading production dataset...")

            production_df = pd.read_csv(
                self.config.production_data_path
            )

            logging.info(
                f"Training Dataset Shape: {train_df.shape}"
            )

            logging.info(
                f"Production Dataset Shape: {production_df.shape}"
            )

            # ==================================================
            # Remove non-feature columns
            # ==================================================
            # These columns are useful for monitoring,
            # but they are NOT model input features,
            # so they should not be included in
            # Data Drift calculations.

            columns_to_drop = [
                "Timestamp",
                "Prediction"
            ]

            production_df = production_df.drop(
                columns=columns_to_drop,
                errors="ignore"
            )

            # ==================================================
            # Verify dataset columns
            # ==================================================

            logging.info(
                f"Training Columns: {train_df.columns.tolist()}"
            )

            logging.info(
                f"Production Columns: {production_df.columns.tolist()}"
            )

            # ==================================================
            # Select numerical features
            # ==================================================
            # Numerical features can be compared
            # using statistics like mean and median.

            numerical_columns = train_df.select_dtypes(
                include=["number"]
            ).columns

            logging.info(
                f"Numerical Columns: {list(numerical_columns)}"
            )

            # ==================================================
            # Store drift analysis results
            # ==================================================

            drift_results = []

            # ==================================================
            # Compare every numerical feature
            # ==================================================

            for column in numerical_columns:

                training_mean = train_df[column].mean()

                production_mean = production_df[column].mean()

                # ----------------------------------------------
                # Prevent division by zero
                # ----------------------------------------------

                if training_mean == 0:
                    difference_percentage = 0

                else:

                    difference_percentage = abs(
                        (
                            production_mean
                            - training_mean
                        )
                        / training_mean
                    ) * 100

                # ----------------------------------------------
                # Determine whether drift exists
                # ----------------------------------------------

                drift_detected = (
                    difference_percentage >
                    self.config.drift_threshold
                )

                # ----------------------------------------------
                # Store result
                # ----------------------------------------------

                drift_results.append(
                    {
                        "Feature": column,
                        "Training Mean": round(
                            training_mean,
                            2
                        ),
                        "Production Mean": round(
                            production_mean,
                            2
                        ),
                        "Difference (%)": round(
                            difference_percentage,
                            2
                        ),
                        "Drift Detected": drift_detected
                    }
                )

            # ==================================================
            # Convert results into DataFrame
            # ==================================================

            drift_report = pd.DataFrame(
                drift_results
            )

            # ==================================================
            # Save Drift Report
            # ==================================================

            drift_report.to_csv(
                self.config.drift_report_path,
                index=False
            )

            logging.info(
                f"Drift report saved to "
                f"{self.config.drift_report_path}"
            )

            # ==================================================
            # Return results
            # ==================================================

            return drift_report

        except Exception as e:  # noqa: BLE001

            logging.error(
                "Error occurred while detecting Data Drift."
            )

            raise CustomException(e, sys)

            # ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    data_drift = DataDrift()

    report = data_drift.detect_data_drift()

    print(report)