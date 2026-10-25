"""
ks_drift.py

Purpose
-------
Detect numerical Data Drift using the
Kolmogorov-Smirnov (KS) statistical test.

The KS test compares the distributions of
training and production numerical features.

Interpretation
--------------
H0:
    Training and production come from the
    same underlying distribution.

If p-value < 0.05:
    Evidence of Data Drift.

If p-value >= 0.05:
    No statistically significant drift detected.

IMPORTANT
---------
A result of "no statistically significant drift"
does NOT prove that the distributions are identical.
It means that the available evidence is not strong
enough to reject the null hypothesis.
"""

import os
import sys

import pandas as pd
from scipy.stats import ks_2samp

from src.exception import CustomException
from src.logger import logging

# ==========================================================
# Configuration
# ==========================================================

class KSDiffConfig:

    train_data_path = os.path.join(
        "artifacts",
        "train.csv"
    )

    production_data_path = os.path.join(
        "artifacts",
        "production_data.csv"
    )

    ks_report_path = os.path.join(
        "artifacts",
        "ks_drift_report.csv"
    )

    # Statistical significance level
    significance_level = 0.05


# ==========================================================
# KS Drift Component
# ==========================================================

class KSDiff:

    def __init__(self):

        self.config = KSDiffConfig()


    def detect_drift(self):

        """
        Detect numerical Data Drift using KS Test.

        Returns
        -------
        DataFrame
            KS Drift report.
        """

        try:

            # ==================================================
            # Read datasets
            # ==================================================

            logging.info(
                "Reading training dataset..."
            )

            train_df = pd.read_csv(
                self.config.train_data_path
            )


            logging.info(
                "Reading production dataset..."
            )

            production_df = pd.read_csv(
                self.config.production_data_path
            )


            # ==================================================
            # Remove monitoring-only columns
            # ==================================================

            production_df = production_df.drop(
                columns=[
                    "Timestamp",
                    "Prediction_ID",
                    "Prediction",
                    "Actual"
                ],
                errors="ignore"
            )


            # ==================================================
            # Select numerical features
            # ==================================================

            numerical_columns = (
                train_df
                .select_dtypes(
                    include=["number"]
                )
                .columns
            )


            # ==================================================
            # Store results
            # ==================================================

            drift_results = []


            # ==================================================
            # Run KS Test for every numerical feature
            # ==================================================

            for column in numerical_columns:

                # ----------------------------------------------
                # Get training values
                # ----------------------------------------------

                training_values = (
                    train_df[column]
                    .dropna()
                )


                # ----------------------------------------------
                # Get production values
                # ----------------------------------------------

                production_values = (
                    production_df[column]
                    .dropna()
                )


                # ----------------------------------------------
                # Run KS Test
                # ----------------------------------------------

                ks_statistic, p_value = ks_2samp(
                    training_values,
                    production_values
                )


                # ----------------------------------------------
                # Determine drift
                # ----------------------------------------------

                drift_detected = (
                    p_value <
                    self.config.significance_level
                )


                # ----------------------------------------------
                # Store result
                # ----------------------------------------------

                drift_results.append(
                    {
                        "Feature": column,

                        "KS Statistic": round(
                            ks_statistic,
                            4
                        ),

                        "P-Value": round(
                            p_value,
                            6
                        ),

                        "Drift Detected":
                            drift_detected
                    }
                )


            # ==================================================
            # Convert results to DataFrame
            # ==================================================

            ks_report = pd.DataFrame(
                drift_results
            )


            # ==================================================
            # Save report
            # ==================================================

            ks_report.to_csv(
                self.config.ks_report_path,
                index=False
            )


            logging.info(
                "KS Drift report saved to "
                f"{self.config.ks_report_path}"
            )


            # ==================================================
            # Return report
            # ==================================================

            return ks_report


        except Exception as e:  # noqa: BLE001

            logging.error(
                "KS Drift detection failed."
            )

            raise CustomException(
                e,
                sys
            )


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    monitor = KSDiff()

    report = monitor.detect_drift()

    print(
        "\n========== KS DRIFT REPORT =========="
    )

    print(report)