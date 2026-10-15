"""
concept_drift.py

Purpose
-------
Investigate possible Concept Drift by comparing the relationship
between important categorical features and the actual outcome.

Concept Drift means that the relationship:

    Feature -> Outcome

has changed over time.

This component compares:

    Training data
        vs
    Production data

using outcome rates for each category.

Important
---------
This is an investigation signal.

A detected difference does NOT mathematically prove Concept Drift.

It tells us:

    "The relationship between this feature and the outcome
     changed enough that we should investigate."
"""

import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging


# ==========================================================
# Configuration
# ==========================================================

class ConceptDriftConfig:

    train_data_path = os.path.join(
        "artifacts",
        "train.csv"
    )

    production_data_path = os.path.join(
        "artifacts",
        "production_data.csv"
    )

    concept_drift_report_path = os.path.join(
        "artifacts",
        "concept_drift_report.csv"
    )

    # Difference in outcome rate required
    # before we recommend investigation.
    drift_threshold = 20

    # Features we want to investigate.
    monitored_features = [
        "Credit_History",
        "Education",
        "Self_Employed",
        "Property_Area",
        "Married",
        "Gender",
        "Dependents"
    ]


# ==========================================================
# Concept Drift Component
# ==========================================================

class ConceptDrift:

    def __init__(self):

        self.config = ConceptDriftConfig()


    def detect_drift(self):

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
            # 3. Check required target columns
            # ==================================================
            #
            # Training uses:
            #
            #     Loan_Status
            #
            # Production uses:
            #
            #     Actual
            #
            # They represent the same business outcome.
            # ==================================================

            if "Loan_Status" not in train_df.columns:

                raise ValueError(
                    "Loan_Status column not found in training data."
                )


            if "Actual" not in production_df.columns:

                raise ValueError(
                    "Actual column not found in production data."
                )


            # ==================================================
            # 4. Normalize target values
            # ==================================================
            #
            # Training target:
            #
            #     Loan_Status
            #
            # Production target:
            #
            #     Actual
            #
            # We rename both to the same name so the rest
            # of the code can use one target column.
            # ==================================================

            train_df = train_df.rename(
                columns={
                    "Loan_Status": "Outcome"
                }
            )

            production_df = production_df.rename(
                columns={
                    "Actual": "Outcome"
                }
            )


            # ==================================================
            # 5. Keep only records with known outcomes
            # ==================================================
            #
            # Concept Drift cannot be investigated without
            # the actual outcome.
            #
            # Predictions without Actual values are ignored.
            # ==================================================

            train_df = train_df[
                train_df["Outcome"].notna()
            ].copy()

            production_df = production_df[
                production_df["Outcome"].notna()
            ].copy()


            # ==================================================
            # 6. Store results
            # ==================================================

            results = []


            # ==================================================
            # 7. Analyze each monitored feature
            # ==================================================

            for feature in self.config.monitored_features:

                # --------------------------------------------------
                # Make sure the feature exists
                # --------------------------------------------------

                if feature not in train_df.columns:

                    logging.warning(
                        f"{feature} not found in training data."
                    )

                    continue


                if feature not in production_df.columns:

                    logging.warning(
                        f"{feature} not found in production data."
                    )

                    continue


                # ==================================================
                # 8. Normalize categorical values
                # ==================================================
                #
                # This prevents issues such as:
                #
                #     "0"
                #
                # versus
                #
                #     0
                #
                # being treated as different categories.
                # ==================================================

                train_feature = (
                    train_df[feature]
                    .astype(str)
                    .str.strip()
                )

                production_feature = (
                    production_df[feature]
                    .astype(str)
                    .str.strip()
                )


                # ==================================================
                # 9. Find all categories
                # ==================================================

                categories = set(
                    train_feature.unique()
                ).union(
                    production_feature.unique()
                )


                # ==================================================
                # 10. Analyze each category
                # ==================================================

                for category in categories:

                    # --------------------------------------------------
                    # Training records for this category
                    # --------------------------------------------------

                    training_mask = (
                        train_feature == category
                    )

                    training_group = (
                        train_df[
                            training_mask
                        ]
                    )


                    # --------------------------------------------------
                    # Production records for this category
                    # --------------------------------------------------

                    production_mask = (
                        production_feature == category
                    )

                    production_group = (
                        production_df[
                            production_mask
                        ]
                    )


                    # --------------------------------------------------
                    # Calculate training approval rate
                    # --------------------------------------------------

                    if len(training_group) > 0:

                        training_outcome_rate = (
                            (
                                training_group["Outcome"] == "Y"
                            ).mean()
                            * 100
                        )

                    else:

                        training_outcome_rate = 0


                    # --------------------------------------------------
                    # Calculate production approval rate
                    # --------------------------------------------------

                    if len(production_group) > 0:

                        production_outcome_rate = (
                            (
                                production_group["Outcome"] == "Y"
                            ).mean()
                            * 100
                        )

                    else:

                        production_outcome_rate = 0


                    # ==================================================
                    # 11. Calculate relationship change
                    # ==================================================

                    difference = abs(
                        production_outcome_rate
                        - training_outcome_rate
                    )


                    # ==================================================
                    # 12. Decide whether investigation is recommended
                    # ==================================================

                    investigation_recommended = (
                        difference >
                        self.config.drift_threshold
                    )


                    # ==================================================
                    # 13. Save result
                    # ==================================================

                    results.append(
                        {
                            "Feature": feature,

                            "Category": category,

                            "Training Outcome Rate (%)":
                                round(
                                    training_outcome_rate,
                                    2
                                ),

                            "Production Outcome Rate (%)":
                                round(
                                    production_outcome_rate,
                                    2
                                ),

                            "Difference (%)":
                                round(
                                    difference,
                                    2
                                ),

                            "Investigation Recommended":
                                investigation_recommended
                        }
                    )


            # ==================================================
            # 14. Convert to DataFrame
            # ==================================================

            report = pd.DataFrame(
                results
            )


            # ==================================================
            # 15. Save report
            # ==================================================

            report.to_csv(
                self.config.concept_drift_report_path,
                index=False
            )


            logging.info(
                "Concept Drift report saved to "
                f"{self.config.concept_drift_report_path}"
            )


            # ==================================================
            # 16. Return report
            # ==================================================

            return report


        except Exception as e:

            logging.error(
                "Concept Drift detection failed."
            )

            raise CustomException(
                e,
                sys
            )


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    monitor = ConceptDrift()

    report = monitor.detect_drift()

    print(
        "\n========== CONCEPT DRIFT REPORT =========="
    )

    print(
        report.to_string(
            index=False
        )
    )