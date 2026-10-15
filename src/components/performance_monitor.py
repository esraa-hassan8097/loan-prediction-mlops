"""
performance_monitor.py

Purpose
-------
This component is responsible for monitoring the model's
performance after deployment.

Unlike Data Drift and Prediction Monitoring,
Performance Monitoring requires the actual outcome
(Ground Truth).

Why?
----
Because we cannot determine whether a prediction is
correct until the real business outcome becomes available.

Example
-------
Prediction : Approved

10 days later...

Actual : Rejected

Now we know the prediction was incorrect.

Responsibilities
----------------
- Read production predictions
- Read actual labels
- Calculate production metrics
- Compare production metrics with training metrics
- Generate a performance report

Non-Responsibilities
--------------------
- Retraining the model
- Detecting Data Drift
- Prediction Monitoring
- Sending alerts automatically
"""

import json
import os
import sys
from dataclasses import dataclass

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.exception import CustomException
from src.logger import logging


# ==========================================================
# Configuration
# ==========================================================

@dataclass
class PerformanceMonitorConfig:
    """
    Stores configuration used by the Performance
    Monitoring component.
    """

    # Production predictions and actual outcomes
    production_data_path = os.path.join(
        "artifacts",
        "production_data.csv"
    )

    # Training baseline metrics
    training_metrics_path = os.path.join(
        "artifacts",
        "metrics.json"
    )

    # Where the performance comparison will be saved
    performance_report_path = os.path.join(
        "artifacts",
        "performance_report.csv"
    )

    # Percentage-point threshold used to determine
    # whether investigation is recommended.
    performance_drop_threshold = 5


# ==========================================================
# Performance Monitoring Component
# ==========================================================

class PerformanceMonitor:
    """
    Monitors the performance of the deployed model.

    It compares production metrics with the metrics
    obtained during model evaluation.
    """

    def __init__(self):

        # Create configuration object
        self.config = PerformanceMonitorConfig()


    def monitor_performance(self):
        """
        Monitor production model performance.

        Returns
        -------
        DataFrame
            Performance comparison report.
        """

        try:

            # ==================================================
            # Read production dataset
            # ==================================================

            logging.info(
                "Reading production dataset..."
            )

            production_df = pd.read_csv(
                self.config.production_data_path
            )


            # ==================================================
            # Verify required columns
            # ==================================================

            # We need both:
            #
            # Prediction = what the model predicted
            #
            # Actual = what really happened
            #
            # Without both values, we cannot calculate
            # performance metrics.

            required_columns = [
                "Prediction",
                "Actual"
            ]

            for column in required_columns:

                if column not in production_df.columns:

                    raise ValueError(
                        f"{column} column not found."
                    )


            # ==================================================
            # Remove records without Actual outcome
            # ==================================================

            # In a real production system, some predictions
            # may still be waiting for their real-world outcome.
            #
            # Example:
            #
            # Prediction = Y
            # Actual = NaN
            #
            # We cannot use these records for performance
            # calculations yet.

            completed_predictions = (
                production_df.dropna(
                    subset=[
                        "Prediction",
                        "Actual"
                    ]
                )
            )


            # ==================================================
            # Verify that completed predictions exist
            # ==================================================

            if completed_predictions.empty:

                raise ValueError(
                    "No completed predictions with "
                    "Actual outcomes are available."
                )


            # ==================================================
            # Calculate Production Metrics
            # ==================================================

            # These are the same metrics we used during
            # model evaluation.

            accuracy = accuracy_score(
                completed_predictions["Actual"],
                completed_predictions["Prediction"]
            )


            precision = precision_score(
                completed_predictions["Actual"],
                completed_predictions["Prediction"],
                pos_label="Y",
                zero_division=0
            )


            recall = recall_score(
                completed_predictions["Actual"],
                completed_predictions["Prediction"],
                pos_label="Y",
                zero_division=0
            )


            f1 = f1_score(
                completed_predictions["Actual"],
                completed_predictions["Prediction"],
                pos_label="Y",
                zero_division=0
            )


            # ==================================================
            # Load Training Metrics
            # ==================================================

            # These metrics represent the baseline performance
            # obtained during model evaluation.

            logging.info(
                "Loading training metrics..."
            )

            with open(
                self.config.training_metrics_path,
                "r"
            ) as file:

                training_metrics = json.load(file)


            # ==================================================
            # Store Production Metrics
            # ==================================================

            production_metrics = {

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1": f1
            }


            # ==================================================
            # Compare Training vs Production
            # ==================================================

            comparison_results = []


            for metric_name, production_value in (
                production_metrics.items()
            ):

                # ----------------------------------------------
                # Get corresponding training metric
                # ----------------------------------------------

                training_value = training_metrics[
                    metric_name
                ]


                # ----------------------------------------------
                # Calculate performance difference
                # ----------------------------------------------

                # Both values are between 0 and 1.
                #
                # Example:
                #
                # Training accuracy = 0.7886
                # Production accuracy = 0.4000
                #
                # Difference:
                #
                # |0.4000 - 0.7886| × 100
                #
                # = 38.86 percentage points

                difference = abs(
                    production_value -
                    training_value
                ) * 100


                # ----------------------------------------------
                # Determine whether investigation is needed
                # ----------------------------------------------

                investigation_required = (
                    difference >
                    self.config.performance_drop_threshold
                )


                # ----------------------------------------------
                # Store comparison result
                # ----------------------------------------------

                comparison_results.append(
                    {

                        "Metric": metric_name,

                        "Training": round(
                            training_value,
                            4
                        ),

                        "Production": round(
                            production_value,
                            4
                        ),

                        "Difference (%)": round(
                            difference,
                            2
                        ),

                        "Investigation Recommended":
                            investigation_required
                    }
                )


            # ==================================================
            # Convert Results to DataFrame
            # ==================================================

            performance_report = pd.DataFrame(
                comparison_results
            )


            # ==================================================
            # Save Performance Report
            # ==================================================

            performance_report.to_csv(
                self.config.performance_report_path,
                index=False
            )


            logging.info(
                "Performance report saved to "
                f"{self.config.performance_report_path}"
            )


            # ==================================================
            # Return Report
            # ==================================================

            return performance_report


        except Exception as e:

            logging.error(
                "Performance Monitoring failed."
            )

            raise CustomException(
                e,
                sys
            )


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    # Create Performance Monitor
    monitor = PerformanceMonitor()


    # Run Performance Monitoring
    report = monitor.monitor_performance()


    # Display result
    print(
        "\n========== PERFORMANCE MONITORING REPORT =========="
    )

    print(report)