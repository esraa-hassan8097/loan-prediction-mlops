"""
categorical_drift.py

Purpose
-------
Detect Data Drift in categorical features by comparing
category distributions between training and production data.

This component monitors INPUT features only.

It does NOT monitor:
- IDs
- timestamps
- predictions
- actual outcomes
- target labels

Those belong to other parts of the monitoring system.
"""

import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging

# ==========================================================
# Configuration
# ==========================================================

class CategoricalDriftConfig:

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
        "categorical_drift_report.csv"
    )

    # Difference in percentage points
    # required to flag a category.
    drift_threshold = 20


# ==========================================================
# Categorical Drift Component
# ==========================================================

class CategoricalDrift:

    def __init__(self):

        self.config = CategoricalDriftConfig()


    def detect_drift(self):

        """
        Detect categorical Data Drift.

        Returns
        -------
        DataFrame
            Detailed categorical drift report.
        """

        try:

            # ==================================================
            # 1. Read training data
            # ==================================================

            logging.info(
                "Reading training dataset..."
            )

            train_df = pd.read_csv(
                self.config.train_data_path
            )


            # ==================================================
            # 2. Read production data
            # ==================================================

            logging.info(
                "Reading production dataset..."
            )

            production_df = pd.read_csv(
                self.config.production_data_path
            )


            # ==================================================
            # 3. Remove non-feature columns
            # ==================================================
            #
            # These columns should NOT participate
            # in input Data Drift monitoring.
            #
            # Loan_ID:
            #   Identifier only
            #
            # Loan_Status:
            #   Original training target
            #
            # Timestamp:
            #   Monitoring metadata
            #
            # Prediction_ID:
            #   Production identifier
            #
            # Prediction:
            #   Model output
            #
            # Actual:
            #   Ground truth / outcome
            #
            # These belong to other monitoring components.
            # ==================================================

            columns_to_ignore = [
                "Loan_ID",
                "Loan_Status",
                "Timestamp",
                "Prediction_ID",
                "Prediction",
                "Actual"
            ]

            train_df = train_df.drop(
                columns=columns_to_ignore,
                errors="ignore"
            )

            production_df = production_df.drop(
                columns=columns_to_ignore,
                errors="ignore"
            )


            # ==================================================
            # 4. Find categorical features
            # ==================================================

            categorical_columns = (
                train_df
                .select_dtypes(
                    include=["object"]
                )
                .columns
            )

            logging.info(
                f"Categorical columns detected: "
                f"{list(categorical_columns)}"
            )


            # ==================================================
            # 5. Store monitoring results
            # ==================================================

            drift_results = []


            # ==================================================
            # 6. Analyze every categorical feature
            # ==================================================

            for column in categorical_columns:

                logging.info(
                    f"Analyzing categorical feature: "
                    f"{column}"
                )


                # ==================================================
                # 7. Calculate TRAINING distribution
                # ==================================================
                #
                # Example:
                #
                # Graduate       80%
                # Not Graduate   20%
                #
                # normalize=True converts counts
                # into proportions.
                # Multiplying by 100 converts them to percentages.
                # ==================================================

                training_distribution = (
                    train_df[column]
                    .value_counts(
                        normalize=True,
                        dropna=False
                    )
                    * 100
                )


                # ==================================================
                # 8. Calculate PRODUCTION distribution
                # ==================================================
                #
                # Example:
                #
                # Graduate       40%
                # Not Graduate   60%
                # ==================================================

                production_distribution = (
                    production_df[column]
                    .value_counts(
                        normalize=True,
                        dropna=False
                    )
                    * 100
                )


                # ==================================================
                # 9. Convert distributions to dictionaries
                # ==================================================
                #
                # This makes category lookup explicit
                # and avoids the pandas FutureWarning.
                # ==================================================

                training_distribution = (
                    training_distribution
                    .to_dict()
                )

                production_distribution = (
                    production_distribution
                    .to_dict()
                )


                # ==================================================
                # 10. Find all categories
                # ==================================================
                #
                # A category might:
                #
                # - exist in training only
                # - exist in production only
                # - exist in both
                #
                # We need to monitor all of them.
                # ==================================================

                all_categories = set(
                    training_distribution.keys()
                ).union(
                    production_distribution.keys()
                )


                # ==================================================
                # 11. Compare every category
                # ==================================================

                for category in all_categories:


                    # --------------------------------------------------
                    # Training percentage
                    # --------------------------------------------------

                    training_percentage = (
                        training_distribution
                        .get(
                            category,
                            0
                        )
                    )


                    # --------------------------------------------------
                    # Production percentage
                    # --------------------------------------------------

                    production_percentage = (
                        production_distribution
                        .get(
                            category,
                            0
                        )
                    )


                    # --------------------------------------------------
                    # Difference
                    # --------------------------------------------------
                    #
                    # Example:
                    #
                    # Training = 80%
                    # Production = 40%
                    #
                    # Difference = 40 percentage points
                    # --------------------------------------------------

                    difference = abs(
                        production_percentage
                        - training_percentage
                    )


                    # --------------------------------------------------
                    # Drift decision
                    # --------------------------------------------------

                    drift_detected = (
                        difference >
                        self.config.drift_threshold
                    )


                    # --------------------------------------------------
                    # Store result
                    # --------------------------------------------------

                    drift_results.append(
                        {
                            "Feature": column,

                            "Category": category,

                            "Training %": round(
                                training_percentage,
                                2
                            ),

                            "Production %": round(
                                production_percentage,
                                2
                            ),

                            "Difference (%)": round(
                                difference,
                                2
                            ),

                            "Drift Detected":
                                drift_detected
                        }
                    )


            # ==================================================
            # 12. Convert results to DataFrame
            # ==================================================

            drift_report = pd.DataFrame(
                drift_results
            )


            # ==================================================
            # 13. Save report
            # ==================================================

            drift_report.to_csv(
                self.config.drift_report_path,
                index=False
            )

            logging.info(
                "Categorical drift report saved to "
                f"{self.config.drift_report_path}"
            )


            # ==================================================
            # 14. Return report
            # ==================================================

            return drift_report


        except Exception as e:  # noqa: BLE001

            logging.error(
                "Categorical Drift detection failed."
            )

            raise CustomException(
                e,
                sys
            )


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    monitor = CategoricalDrift()

    report = monitor.detect_drift()

    print(
        "\n========== CATEGORICAL DRIFT REPORT =========="
    )

    print(
        report.to_string(
            index=False
        )
    )