"""
evidently_monitor.py

Purpose
-------
Generate a visual monitoring report using Evidently.

The report compares:

Reference data
    Training dataset

Current data
    Production dataset

The first version focuses on:
- Data quality
- Data drift
"""

import os
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.exception import CustomException
from src.logger import logging

# ==========================================================
# Configuration
# ==========================================================

class EvidentlyMonitorConfig:

    reference_data_path = os.path.join(
        "artifacts",
        "train.csv"
    )

    current_data_path = os.path.join(
        "artifacts",
        "production_data.csv"
    )

    report_path = os.path.join(
        "artifacts",
        "evidently_monitoring_report.html"
    )


# ==========================================================
# Evidently Monitoring
# ==========================================================

class EvidentlyMonitor:

    def __init__(self):

        self.config = EvidentlyMonitorConfig()


    def generate_report(self):

        try:

            # ==================================================
            # Read reference data
            # ==================================================

            logging.info(
                "Reading reference/training data..."
            )

            reference_df = pd.read_csv(
                self.config.reference_data_path
            )


            # ==================================================
            # Read current/production data
            # ==================================================

            logging.info(
                "Reading current/production data..."
            )

            current_df = pd.read_csv(
                self.config.current_data_path
            )


            # ==================================================
            # Remove columns that are not model inputs
            # ==================================================

            columns_to_ignore = [
                "Loan_ID",
                "Loan_Status",
                "Timestamp",
                "Prediction_ID",
                "Prediction",
                "Actual"
            ]


            reference_df = reference_df.drop(
                columns=columns_to_ignore,
                errors="ignore"
            )


            current_df = current_df.drop(
                columns=columns_to_ignore,
                errors="ignore"
            )


            # ==================================================
            # Create Evidently report
            # ==================================================

            logging.info(
                "Creating Evidently monitoring report..."
            )

            report = Report(
                metrics=[
                    DataSummaryPreset(),
                    DataDriftPreset()
                ]
            )


            # ==================================================
            # Run report
            # ==================================================

            snapshot = report.run(
                reference_data=reference_df,
                current_data=current_df
            )


            # ==================================================
            # Save HTML report
            # ==================================================

            snapshot.save_html(
                self.config.report_path
            )


            logging.info(
                "Evidently report saved to "
                f"{self.config.report_path}"
            )


            print(
                "\n========== EVIDENTLY MONITORING =========="
            )

            print(
                "Report generated successfully."
            )

            print(
                f"Location: "
                f"{self.config.report_path}"
            )


            return snapshot


        except Exception as e:  # noqa: BLE001

            logging.error(
                "Evidently monitoring failed."
            )

            raise CustomException(
                e,
                sys
            )


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    monitor = EvidentlyMonitor()

    monitor.generate_report()