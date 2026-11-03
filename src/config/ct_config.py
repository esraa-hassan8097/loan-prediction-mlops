import os

MODEL_NAME = "LoanApprovalModel"
PRODUCTION_ALIAS = "champion"

PRIMARY_METRIC = "f1"
MIN_IMPROVEMENT = 0.01

DRIFT_THRESHOLD = 0.20

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)
