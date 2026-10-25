"""
prediction_monitor.py

Purpose
-------
This component monitors the model's prediction distribution
in production.

Unlike Data Drift, Prediction Monitoring does NOT compare
training data with production data.

Instead, it monitors how the model's predictions change
over time.

Example
-------

Week 1

Approved : 60%
Rejected : 40%

Week 4

Approved : 98%
Rejected : 2%

This may indicate that something unusual has happened,
so an investigation should be performed.

Current Version
---------------
- Reads production_data.csv
- Calculates prediction distribution
- Raises alerts if one prediction class exceeds
  a configurable threshold
- Generates a prediction monitoring report

Future Improvements
-------------------
- Compare prediction distributions across time windows
- Compare today's predictions with last week's predictions
- Integrate monitoring dashboards
- Trigger automated notifications
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
class PredictionMonitorConfig:

    production_data_path = os.path.join(
        "artifacts",
        "production_data.csv"
    )

    prediction_report_path = os.path.join(
        "artifacts",
        "prediction_report.csv"
    )

    prediction_threshold = 80


# ==========================================================
# Prediction Monitoring Component
# ==========================================================

class PredictionMonitor:

    def __init__(self):
        self.config = PredictionMonitorConfig()

    def monitor_predictions(self):

        """
        Monitor prediction distribution.

        Returns
        -------
        DataFrame
            Prediction monitoring report.
        """

        try:

            # ----------------------------------------------
            # Read production dataset
            # ----------------------------------------------

            logging.info(
                "Reading production dataset..."
            )

            production_df = pd.read_csv(
                self.config.production_data_path
            )

            # ----------------------------------------------
            # Verify Prediction column exists
            # ----------------------------------------------

            if "Prediction" not in production_df.columns:

                raise ValueError(
                    "Prediction column not found."
                )

            # ----------------------------------------------
            # Calculate prediction percentages
            # ----------------------------------------------

            prediction_distribution = (
                production_df["Prediction"]
                .value_counts(normalize=True)
                * 100
            )

            monitoring_results = []

            # ----------------------------------------------
            # Analyze every prediction class
            # ----------------------------------------------

            for prediction, percentage in (
                prediction_distribution.items()
            ):

                alert = (
                    percentage >
                    self.config.prediction_threshold
                )

                monitoring_results.append(
                    {
                        "Prediction": prediction,
                        "Percentage": round(
                            percentage,
                            2
                        ),
                        "Alert": alert
                    }
                )

            # ----------------------------------------------
            # Convert to DataFrame
            # ----------------------------------------------

            prediction_report = pd.DataFrame(
                monitoring_results
            )

            # ----------------------------------------------
            # Save report
            # ----------------------------------------------

            prediction_report.to_csv(
                self.config.prediction_report_path,
                index=False
            )

            logging.info(
                "Prediction monitoring report saved."
            )

            return prediction_report

        except Exception as e:  # noqa: BLE001

            logging.error(
                "Prediction Monitoring failed."
            )

            raise CustomException(e, sys)
# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    monitor = PredictionMonitor()

    report = monitor.monitor_predictions()

    print(report)