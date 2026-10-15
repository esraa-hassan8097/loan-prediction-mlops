"""
actual_outcome.py

Purpose
-------
Simulates the arrival of the real-world outcome (Actual)
after a model prediction has already been made.

In a real production system, the Actual outcome would come
from a business system, database, API, or another data source.

For our project, we simulate that process.

Example
-------
Day 1:
    Prediction_ID = 1001
    Prediction = Y

Later:
    Prediction_ID = 1001
    Actual = N

The Prediction_ID allows us to connect the delayed outcome
to the original prediction.
"""

import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging


# ==========================================================
# Configuration
# ==========================================================

class ActualOutcomeConfig:

    # Production prediction records
    production_data_path = os.path.join(
        "artifacts",
        "production_data.csv"
    )


# ==========================================================
# Actual Outcome Component
# ==========================================================

class ActualOutcome:

    def __init__(self):

        self.config = ActualOutcomeConfig()


    def add_actual_outcome(
        self,
        prediction_id,
        actual
    ):
        """
        Add the real-world outcome to an existing
        production prediction.

        Parameters
        ----------
        prediction_id : str
            Unique ID of the original prediction.

        actual : str
            Real-world outcome.
            Expected values: Y or N.

        Returns
        -------
        DataFrame
            Updated production dataset.
        """

        try:

            # ==================================================
            # Read production predictions
            # ==================================================

            logging.info(
                "Reading production prediction data..."
            )

            production_df = pd.read_csv(
                self.config.production_data_path
            )


            # ==================================================
            # Validate Prediction_ID column
            # ==================================================

            if "Prediction_ID" not in production_df.columns:

                raise ValueError(
                    "Prediction_ID column not found."
                )


            # ==================================================
            # Validate Actual value
            # ==================================================

            if actual not in ["Y", "N"]:

                raise ValueError(
                    "Actual outcome must be 'Y' or 'N'."
                )


            # ==================================================
            # Find the prediction
            # ==================================================

            matching_rows = (
                production_df["Prediction_ID"]
                == prediction_id
            )


            if not matching_rows.any():

                raise ValueError(
                    f"Prediction_ID not found: "
                    f"{prediction_id}"
                )


            # ==================================================
            # Add the Actual outcome
            # ==================================================

            production_df.loc[
                matching_rows,
                "Actual"
            ] = actual


            # ==================================================
            # Save updated production data
            # ==================================================

            production_df.to_csv(
                self.config.production_data_path,
                index=False
            )


            logging.info(
                f"Actual outcome added for "
                f"Prediction_ID: {prediction_id}"
            )


            # ==================================================
            # Return updated dataset
            # ==================================================

            return production_df


        except Exception as e:

            logging.error(
                "Error occurred while adding "
                "actual outcome."
            )

            raise CustomException(e, sys)


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    outcome_manager = ActualOutcome()

    # ------------------------------------------------------
    # Read existing production data
    # ------------------------------------------------------

    production_df = pd.read_csv(
        outcome_manager.config.production_data_path
    )

    # ------------------------------------------------------
    # Get the first prediction ID
    # ------------------------------------------------------

    prediction_id = production_df[
        "Prediction_ID"
    ].iloc[0]

    # ------------------------------------------------------
    # Simulate the actual outcome
    #
    # Our current model prediction is Y.
    #
    # We intentionally use N here so we can demonstrate
    # an incorrect production prediction.
    # ------------------------------------------------------

    report = outcome_manager.add_actual_outcome(
        prediction_id=prediction_id,
        actual="N"
    )

    print("\n========== UPDATED PRODUCTION DATA ==========")

    print(report)