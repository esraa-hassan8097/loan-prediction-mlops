"""
production_simulator.py

Purpose
-------
Creates controlled production data for testing the
Performance Monitoring component.

IMPORTANT
---------
This is ONLY a simulation for our MLOps project.

In a real system:
    Prediction -> real business process -> Actual

Here we simulate the arrival of those Actual outcomes
so that we can test our monitoring system.
"""

import os
import uuid

import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

production_data_path = os.path.join(
    "artifacts",
    "production_data.csv"
)


# ==========================================================
# Load Existing Production Data
# ==========================================================

production_df = pd.read_csv(
    production_data_path
)


# ==========================================================
# Create Simulated Production Records
# ==========================================================

simulated_records = [

    {
        "Gender": "Female",
        "Married": "No",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 4000.0,
        "CoapplicantIncome": 1500.0,
        "LoanAmount": 120.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
        "Prediction": "Y",
        "Actual": "Y"
    },

    {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000.0,
        "CoapplicantIncome": 1000.0,
        "LoanAmount": 150.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban",
        "Prediction": "Y",
        "Actual": "N"
    },

    {
        "Gender": "Male",
        "Married": "No",
        "Dependents": "0",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "ApplicantIncome": 2500.0,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 90.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 0.0,
        "Property_Area": "Rural",
        "Prediction": "N",
        "Actual": "N"
    },

    {
        "Gender": "Female",
        "Married": "Yes",
        "Dependents": "2",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 3500.0,
        "CoapplicantIncome": 1200.0,
        "LoanAmount": 100.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
        "Prediction": "N",
        "Actual": "Y"
    }
]


# ==========================================================
# Add IDs and Timestamp
# ==========================================================

for record in simulated_records:

    record["Prediction_ID"] = str(
        uuid.uuid4()
    )

    record["Timestamp"] = pd.Timestamp.now()


# ==========================================================
# Convert Simulation Records to DataFrame
# ==========================================================

simulation_df = pd.DataFrame(
    simulated_records
)


# ==========================================================
# Combine Existing and Simulated Data
# ==========================================================

combined_df = pd.concat(
    [
        production_df,
        simulation_df
    ],
    ignore_index=True,
    sort=False
)


# ==========================================================
# Save Updated Production Data
# ==========================================================

combined_df.to_csv(
    production_data_path,
    index=False
)


# ==========================================================
# Display Result
# ==========================================================

print(
    "\n========== PRODUCTION SIMULATION =========="
)

print(
    combined_df[
        [
            "Prediction_ID",
            "Prediction",
            "Actual"
        ]
    ]
)