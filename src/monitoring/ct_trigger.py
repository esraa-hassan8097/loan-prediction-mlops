import hashlib
import subprocess
from pathlib import Path

import pandas as pd

from src.config.ct_config import DRIFT_THRESHOLD


DRIFT_REPORT_PATH = Path(
    "artifacts/drift_report.csv"
)

STATE_FILE = Path(
    "artifacts/last_ct_drift_hash.txt"
)


def calculate_drift_score() -> float:
    if not DRIFT_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Drift report not found: "
            f"{DRIFT_REPORT_PATH}"
        )

    report = pd.read_csv(
        DRIFT_REPORT_PATH
    )

    if "Drift Detected" not in report.columns:
        raise ValueError(
            "Column 'Drift Detected' "
            "not found in drift report."
        )

    total_features = len(report)

    if total_features == 0:
        return 0.0

    drifted_features = (
        report["Drift Detected"]
        .astype(bool)
        .sum()
    )

    return (
        drifted_features
        / total_features
    )


def get_report_hash() -> str:
    content = (
        DRIFT_REPORT_PATH
        .read_bytes()
    )

    return hashlib.sha256(
        content
    ).hexdigest()


def already_processed(
    report_hash: str,
) -> bool:
    if not STATE_FILE.exists():
        return False

    previous_hash = (
        STATE_FILE
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    return previous_hash == report_hash


def save_processed_hash(
    report_hash: str,
) -> None:
    STATE_FILE.write_text(
        report_hash,
        encoding="utf-8",
    )


def trigger_ct_if_needed() -> bool:
    drift_score = calculate_drift_score()

    print(
        f"Real drift score: "
        f"{drift_score:.4f}"
    )

    print(
        f"CT threshold    : "
        f"{DRIFT_THRESHOLD:.4f}"
    )

    if drift_score < DRIFT_THRESHOLD:
        print(
            "Drift below threshold. "
            "CT not triggered."
        )

        return False

    report_hash = get_report_hash()

    if already_processed(
        report_hash
    ):
        print(
            "This drift report already "
            "triggered CT."
        )

        return False

    print(
        "Drift threshold exceeded."
    )

    print(
        "Triggering CT automatically..."
    )

    subprocess.run(
        [
            "python",
            "-m",
            "src.pipeline.ct_pipeline",
            "--trigger",
            "drift",
        ],
        check=True,
    )

    save_processed_hash(
        report_hash
    )

    return True


if __name__ == "__main__":
    trigger_ct_if_needed()