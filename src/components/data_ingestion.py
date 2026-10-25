import json
import os
import re
import struct
import subprocess
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import pyodbc
from sklearn.model_selection import train_test_split

from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv",
    )
    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv",
    )
    metadata_path: str = os.path.join(
        "artifacts",
        "data_metadata.json",
    )

    fabric_server: str = (
        "m3sh5gudvcvejaykp2k6pq3axa-"
        "m4e3rofohg5urbtbsz2zpkv5vq."
        "datawarehouse.fabric.microsoft.com"
    )
    fabric_database: str = "loan_mlops_lakehouse"
    fabric_table: str = "dbo.loan_raw"

    azure_config_dir: str = os.path.expanduser(
        "~/.azure-fabric"
    )

    fabric_data_timestamp: str = os.getenv(
        "FABRIC_DATA_TIMESTAMP",
        "latest",
    )


class DataIngestion:
    SQL_COPT_SS_ACCESS_TOKEN = 1256

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def _get_access_token(self):
        """
        Get a Microsoft Fabric SQL access token
        from the Azure CLI.
        """
        env = os.environ.copy()
        env["AZURE_CONFIG_DIR"] = (
            self.ingestion_config.azure_config_dir
        )

        command = [
            "az",
            "account",
            "get-access-token",
            "--resource",
            "https://database.windows.net/",
            "--query",
            "accessToken",
            "-o",
            "tsv",
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

        token = result.stdout.strip()

        if not token:
            raise RuntimeError(
                "Azure CLI returned an empty Fabric access token."
            )

        return token

    def _get_connection(self):
        """
        Create an authenticated ODBC connection
        to the Microsoft Fabric SQL endpoint.
        """
        token = self._get_access_token()

        token_bytes = token.encode("utf-16-le")

        token_struct = struct.pack(
            f"<I{len(token_bytes)}s",
            len(token_bytes),
            token_bytes,
        )

        connection_string = (
            "Driver={ODBC Driver 18 for SQL Server};"
            f"Server={self.ingestion_config.fabric_server};"
            f"Database={self.ingestion_config.fabric_database};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
        )

        return pyodbc.connect(
            connection_string,
            attrs_before={
                self.SQL_COPT_SS_ACCESS_TOKEN: token_struct
            },
        )

    def _validate_table_name(self):
        """
        Validate the configured Fabric table identifier
        before inserting it into a SQL statement.
        """
        table = self.ingestion_config.fabric_table

        if not re.fullmatch(
            r"[A-Za-z0-9_.]+",
            table,
        ):
            raise ValueError(
                f"Invalid Fabric table name: {table}"
            )

        return table

    def _normalize_timestamp(self, timestamp):
        """
        Validate and normalize a historical Fabric timestamp.
        """
        normalized_timestamp = timestamp.replace(
            "Z",
            "+00:00",
        )

        try:
            parsed_timestamp = datetime.fromisoformat(
                normalized_timestamp
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid Fabric timestamp: {timestamp}"
            ) from exc

        return parsed_timestamp.isoformat()

    def _build_query(self):
        """
        Build either:
        - a query for the latest Fabric table data, or
        - a query for a historical Fabric data version.
        """
        table = self._validate_table_name()

        timestamp = (
            self.ingestion_config.fabric_data_timestamp
        )

        if timestamp.lower() == "latest":
            return f"SELECT * FROM {table}"  # nosec B608

        safe_timestamp = self._normalize_timestamp(
            timestamp
        )

        return (
            f"SELECT * FROM {table} "  # nosec B608
            f"OPTION (FOR TIMESTAMP AS OF '{safe_timestamp}')"
        )

    def _save_metadata(self, df):
        """
        Save metadata identifying which data version
        was used during this pipeline run.
        """
        metadata = {
            "source": "Microsoft Fabric",
            "database": self.ingestion_config.fabric_database,
            "table": self.ingestion_config.fabric_table,
            "requested_timestamp": (
                self.ingestion_config.fabric_data_timestamp
            ),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "generated_at": datetime.now().isoformat(),
        }

        with open(
            self.ingestion_config.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
            )

        logging.info(
            "Fabric data metadata saved successfully."
        )

    def initiate_data_ingestion(self):
        """
        Read versioned data from Microsoft Fabric,
        split it into training and test datasets,
        save the datasets locally and record metadata.
        """
        logging.info(
            "Entered the Fabric data ingestion method."
        )

        os.makedirs(
            "artifacts",
            exist_ok=True,
        )

        query = self._build_query()

        logging.info(
            "Connecting to Microsoft Fabric."
        )

        connection = self._get_connection()

        try:
            logging.info(
                "Reading dataset from Microsoft Fabric."
            )

            df = pd.read_sql(
                query,
                connection,
            )
        finally:
            connection.close()

        logging.info(
            "Fabric dataset read successfully."
        )

        if df.empty:
            raise ValueError(
                "Fabric returned an empty dataset."
            )

        logging.info(
            "Fabric dataset contains %s rows and %s columns.",
            len(df),
            len(df.columns),
        )

        self._save_metadata(df)

        logging.info(
            "Splitting dataset into train and test sets."
        )

        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
        )

        logging.info(
            "Train and test datasets created."
        )

        train_set.to_csv(
            self.ingestion_config.train_data_path,
            index=False,
            header=True,
        )

        test_set.to_csv(
            self.ingestion_config.test_data_path,
            index=False,
            header=True,
        )

        logging.info(
            "Fabric data ingestion completed successfully."
        )

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path,
        )


if __name__ == "__main__":
    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    print("Train Path:", train_path)
    print("Test Path :", test_path)
