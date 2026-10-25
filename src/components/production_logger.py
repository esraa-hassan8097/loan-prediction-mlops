"""
production_logger.py

Purpose
-------
Logs every production prediction request.

Each prediction receives a unique Prediction_ID.

Why do we need Prediction_ID?
-----------------------------
The model prediction and the real-world outcome may happen
at different times.

For example:

Day 1:
    Prediction_ID = 1001
    Prediction = Y

Day 7:
    Prediction_ID = 1001
    Actual = N

The ID allows us to connect the delayed actual outcome
to the original prediction.
"""

import os
import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.logger import logging

# ==========================================================
# Configuration
# ==========================================================

@dataclass
class ProductionLoggerConfig:

    # File where production prediction records are stored
    production_data_path = os.path.join(
        "artifacts",
        "production_data.csv"
    )


# ==========================================================
# Production Logger
# ==========================================================

class ProductionLogger:

    def __init__(self):

        self.config = ProductionLoggerConfig()

    def log_prediction_request(
        self,
        input_data,
        prediction
    ):
        """
        Save one production prediction request.

        Parameters
        ----------
        input_data : dict
            Customer input features received by the API.

        prediction : str
            Model prediction.

        Returns
        -------
        str
            Unique Prediction_ID.
        """

        # --------------------------------------------------
        # Generate a unique ID for this prediction
        # --------------------------------------------------
        #
        # This ID will later allow us to connect the
        # prediction with the delayed Actual outcome.

        prediction_id = str(
            uuid.uuid4()
        )

        # --------------------------------------------------
        # Make a copy of the input data
        # --------------------------------------------------

        record = input_data.copy()

        # --------------------------------------------------
        # Add Prediction_ID
        # --------------------------------------------------

        record["Prediction_ID"] = prediction_id

        # --------------------------------------------------
        # Add timestamp
        # --------------------------------------------------

        record["Timestamp"] = datetime.now()

        # --------------------------------------------------
        # Add model prediction
        # --------------------------------------------------

        record["Prediction"] = prediction

        # --------------------------------------------------
        # Convert dictionary to one-row DataFrame
        # --------------------------------------------------

        new_record = pd.DataFrame(
            [record]
        )

        # --------------------------------------------------
        # Append to existing production data
        # --------------------------------------------------

        if os.path.exists(
            self.config.production_data_path
        ):

            new_record.to_csv(
                self.config.production_data_path,
                mode="a",
                header=False,
                index=False
            )

        # --------------------------------------------------
        # Create new production file
        # --------------------------------------------------

        else:

            new_record.to_csv(
                self.config.production_data_path,
                index=False
            )

        # --------------------------------------------------
        # Log successful operation
        # --------------------------------------------------

        logging.info(
            f"Production prediction logged. "
            f"Prediction_ID: {prediction_id}"
        )

        # --------------------------------------------------
        # Return the ID
        # --------------------------------------------------

        return prediction_id