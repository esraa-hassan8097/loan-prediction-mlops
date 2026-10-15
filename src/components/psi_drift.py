"""
psi_drift.py

Purpose
-------
Detect numerical Data Drift using the
Population Stability Index (PSI).

PSI compares the percentage distribution of
a feature between a reference dataset and
a production dataset.

Reference:
    Training data

Current:
    Production data

Interpretation
--------------
PSI < 0.10
    Little or no significant shift

0.10 <= PSI < 0.25
    Moderate shift - investigate

PSI >= 0.25
    Significant shift

IMPORTANT
---------
These thresholds are practical rules of thumb.
They should be calibrated for a real production
system using historical behavior and business risk.
"""

import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging


# ==========================================================
# Configuration
# ==========================================================

class PSIDriftConfig:

    train_data_path = os.path.join(
        "artifacts",
        "train.csv"
    )

    production_data_path = os.path.join(
        "artifacts",
        "production_data.csv"
    )

    # Summary report
    psi_report_path = os.path.join(
        "artifacts",
        "psi_drift_report.csv"
    )

    # Detailed bin-level report
    psi_detail_report_path = os.path.join(
        "artifacts",
        "psi_drift_details.csv"
    )

    # Number of bins used for numerical features
    number_of_bins = 10

    # Prevent division by zero and log(0)
    epsilon = 1e-6

    # PSI thresholds
    moderate_threshold = 0.10

    significant_threshold = 0.25


# ==========================================================
# PSI Drift Component
# ==========================================================

class PSIDrift:

    def __init__(self):

        self.config = PSIDriftConfig()


    # ======================================================
    # Calculate PSI
    # ======================================================

    def calculate_psi(
        self,
        expected,
        actual
    ):

        """
        Calculate PSI between two distributions.

        Parameters
        ----------
        expected : array-like
            Reference/training percentages.

        actual : array-like
            Production percentages.

        Returns
        -------
        float
            PSI value.
        """

        # --------------------------------------------------
        # Convert to numpy arrays
        # --------------------------------------------------

        expected = np.asarray(
            expected,
            dtype=float
        )

        actual = np.asarray(
            actual,
            dtype=float
        )

        # --------------------------------------------------
        # Replace zero percentages
        # --------------------------------------------------
        #
        # PSI contains:
        #
        # log(actual / expected)
        #
        # Division by zero or log(0) would cause problems.
        #
        # epsilon gives us a very small safe value.
        # --------------------------------------------------

        expected = np.where(
            expected == 0,
            self.config.epsilon,
            expected
        )

        actual = np.where(
            actual == 0,
            self.config.epsilon,
            actual
        )

        # --------------------------------------------------
        # PSI formula
        # --------------------------------------------------

        psi_values = (
            actual - expected
        ) * np.log(
            actual / expected
        )

        # --------------------------------------------------
        # Total PSI
        # --------------------------------------------------

        return np.sum(
            psi_values
        )


    # ======================================================
    # Detect Drift
    # ======================================================

    def detect_drift(self):

        """
        Detect numerical Data Drift using PSI.

        Returns
        -------
        DataFrame
            PSI summary report.
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
            #
            # These columns are not model input features.
            #
            # Timestamp
            # Prediction_ID
            # Prediction
            # Actual
            #
            # should not participate in numerical
            # feature drift calculations.
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

            logging.info(
                f"Numerical columns: "
                f"{list(numerical_columns)}"
            )


            # ==================================================
            # Store results
            # ==================================================

            psi_results = []

            psi_details = []


            # ==================================================
            # Calculate PSI for every numerical feature
            # ==================================================

            for column in numerical_columns:

                logging.info(
                    f"Calculating PSI for {column}..."
                )


                # --------------------------------------------------
                # Get non-null training values
                # --------------------------------------------------

                training_values = (
                    train_df[column]
                    .dropna()
                )


                # --------------------------------------------------
                # Get non-null production values
                # --------------------------------------------------

                production_values = (
                    production_df[column]
                    .dropna()
                )


                # --------------------------------------------------
                # Create bins from TRAINING data
                # --------------------------------------------------
                #
                # Training is our reference population.
                #
                # The same bin boundaries must be used
                # for production.
                # --------------------------------------------------

                try:

                    _, bin_edges = pd.qcut(
                        training_values,
                        q=self.config.number_of_bins,
                        retbins=True,
                        duplicates="drop"
                    )

                except ValueError:

                    logging.warning(
                        f"Could not create bins for "
                        f"{column}. Skipping feature."
                    )

                    continue


                # --------------------------------------------------
                # Extend first and last boundaries
                # --------------------------------------------------

                bin_edges[0] = -np.inf

                bin_edges[-1] = np.inf


                # --------------------------------------------------
                # Assign training values to bins
                # --------------------------------------------------

                training_bins = pd.cut(
                    training_values,
                    bins=bin_edges,
                    include_lowest=True
                )


                # --------------------------------------------------
                # Assign production values to the SAME bins
                # --------------------------------------------------

                production_bins = pd.cut(
                    production_values,
                    bins=bin_edges,
                    include_lowest=True
                )


                # --------------------------------------------------
                # Calculate training distribution
                # --------------------------------------------------
                #
                # normalize=True converts counts into
                # percentages/proportions.
                # --------------------------------------------------

                training_distribution = (
                    training_bins
                    .value_counts(
                        normalize=True,
                        sort=False
                    )
                )


                # --------------------------------------------------
                # Calculate production distribution
                # --------------------------------------------------

                production_distribution = (
                    production_bins
                    .value_counts(
                        normalize=True,
                        sort=False
                    )
                )


                # --------------------------------------------------
                # Make sure production uses exactly
                # the same bins as training
                # --------------------------------------------------

                production_distribution = (
                    production_distribution
                    .reindex(
                        training_distribution.index,
                        fill_value=0
                    )
                )


                # --------------------------------------------------
                # Convert percentages to numpy arrays
                # --------------------------------------------------

                expected = np.asarray(
                    training_distribution.values,
                    dtype=float
                )

                actual = np.asarray(
                    production_distribution.values,
                    dtype=float
                )


                # --------------------------------------------------
                # Safe values for PSI calculation
                # --------------------------------------------------

                expected_safe = np.where(
                    expected == 0,
                    self.config.epsilon,
                    expected
                )

                actual_safe = np.where(
                    actual == 0,
                    self.config.epsilon,
                    actual
                )


                # --------------------------------------------------
                # Calculate PSI contribution per bin
                # --------------------------------------------------
                #
                # Formula:
                #
                # (Actual - Expected)
                # *
                # log(Actual / Expected)
                #
                # Each bin contributes part of the
                # total PSI.
                # --------------------------------------------------

                psi_contributions = (
                    actual_safe - expected_safe
                ) * np.log(
                    actual_safe / expected_safe
                )


                # --------------------------------------------------
                # Calculate total PSI
                # --------------------------------------------------

                psi = np.sum(
                    psi_contributions
                )


                # --------------------------------------------------
                # Store detailed information
                # --------------------------------------------------

                for bin_index, bin_name in enumerate(
                    training_distribution.index
                ):

                    psi_details.append(
                        {
                            "Feature": column,

                            "Bin": str(
                                bin_name
                            ),

                            "Training %": round(
                                expected[bin_index] * 100,
                                2
                            ),

                            "Production %": round(
                                actual[bin_index] * 100,
                                2
                            ),

                            "PSI Contribution": round(
                                psi_contributions[
                                    bin_index
                                ],
                                4
                            )
                        }
                    )


                # --------------------------------------------------
                # Determine severity
                # --------------------------------------------------

                if (
                    psi >=
                    self.config.significant_threshold
                ):

                    status = "SIGNIFICANT"

                elif (
                    psi >=
                    self.config.moderate_threshold
                ):

                    status = "MODERATE"

                else:

                    status = "STABLE"


                # --------------------------------------------------
                # Store summary result
                # --------------------------------------------------

                psi_results.append(
                    {
                        "Feature": column,

                        "PSI": round(
                            psi,
                            4
                        ),

                        "Status": status
                    }
                )


            # ==================================================
            # Create summary DataFrame
            # ==================================================

            psi_report = pd.DataFrame(
                psi_results
            )


            # ==================================================
            # Create detailed DataFrame
            # ==================================================

            psi_details_report = pd.DataFrame(
                psi_details
            )


            # ==================================================
            # Save summary report
            # ==================================================

            psi_report.to_csv(
                self.config.psi_report_path,
                index=False
            )

            logging.info(
                "PSI summary report saved to "
                f"{self.config.psi_report_path}"
            )


            # ==================================================
            # Save detailed report
            # ==================================================

            psi_details_report.to_csv(
                self.config.psi_detail_report_path,
                index=False
            )

            logging.info(
                "PSI detailed report saved to "
                f"{self.config.psi_detail_report_path}"
            )


            # ==================================================
            # Return summary report
            # ==================================================

            return psi_report


        except Exception as e:

            logging.error(
                "PSI Drift detection failed."
            )

            raise CustomException(
                e,
                sys
            )


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    monitor = PSIDrift()

    report = monitor.detect_drift()

    print(
        "\n========== PSI DRIFT REPORT =========="
    )

    print(report)

    print(
        "\nDetailed report saved to:"
    )

    print(
        "artifacts/psi_drift_details.csv"
    )