import os
import json
import struct
import subprocess
from dataclasses import dataclass

import pandas as pd
import pyodbc
from sklearn.model_selection import train_test_split

from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    metadata_path: str = os.path.join("artifacts", "data_metadata.json")

    fabric_server: str = (
        "m3sh5gudvcvejaykp2k6pq3axa-"
        "m4e3rofohg5urbtbsz2zpkv5vq.datawarehouse.fabric.microsoft.com"
    )

    fabric_database: str = "loan_mlops_lakehouse"
    fabric_table: str = "dbo.loan_raw"

    azure_config_dir: str = os.path.expanduser("~/.azure-fabric")

    fabric_data_timestamp: str = os.getenv(
        "FABRIC_DATA_TIMESTAMP",
        "latest"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def get_fabric_connection(self):
        logging.info("Getting Microsoft Entra access token.")

        env = os.environ.copy()
        env["AZURE_CONFIG_DIR"] = self.ingestion_config.azure_config_dir

        token = subprocess.check_output(
            [
                "az",
                "account",
                "get-access-token",
                "--resource",
                "https://database.windows.net/",
                "--query",
                "accessToken",
                "-o",
                "tsv"
            ],
            env=env,
            text=True
        ).strip()

        token_bytes = token.encode("utf-16-le")

        token_struct = struct.pack(
            f"<I{len(token_bytes)}s",
            len(token_bytes),
            token_bytes
        )

        connection_string = (
            "DRIVER={ODBC Driver 18 for SQL Server};"
            f"SERVER={self.ingestion_config.fabric_server};"
            f"DATABASE={self.ingestion_config.fabric_database};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
        )

        SQL_COPT_SS_ACCESS_TOKEN = 1256

        connection = pyodbc.connect(
            connection_string,
            attrs_before={
                SQL_COPT_SS_ACCESS_TOKEN: token_struct
            }
        )

        logging.info("Connected to Microsoft Fabric successfully.")

        return connection

    def build_query(self):
        timestamp = self.ingestion_config.fabric_data_timestamp

        if timestamp.lower() == "latest":
            return (
                f"SELECT * "
                f"FROM {self.ingestion_config.fabric_table}"
            )

        return (
            f"SELECT * "
            f"FROM {self.ingestion_config.fabric_table} "
            f"OPTION (FOR TIMESTAMP AS OF '{timestamp}')"
        )

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method.")

        os.makedirs("artifacts", exist_ok=True)

        connection = self.get_fabric_connection()

        query = self.build_query()

        logging.info(
            f"Reading Fabric data using snapshot: "
            f"{self.ingestion_config.fabric_data_timestamp}"
        )

        cursor = connection.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()

        columns = [
            column[0]
            for column in cursor.description
        ]

        df = pd.DataFrame.from_records(
            rows,
            columns=columns
        )

        connection.close()

        logging.info(
            f"Dataset loaded successfully. Shape: {df.shape}"
        )

        metadata = {
            "source": "Microsoft Fabric",
            "table": self.ingestion_config.fabric_table,
            "data_snapshot":
                self.ingestion_config.fabric_data_timestamp,
            "row_count": len(df)
        }

        with open(
            self.ingestion_config.metadata_path,
            "w"
        ) as file:
            json.dump(metadata, file, indent=4)

        logging.info(
            f"Data metadata saved: {metadata}"
        )

        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        train_set.to_csv(
            self.ingestion_config.train_data_path,
            index=False,
            header=True
        )

        test_set.to_csv(
            self.ingestion_config.test_data_path,
            index=False,
            header=True
        )

        logging.info("Data ingestion completed successfully.")

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )


if __name__ == "__main__":
    obj = DataIngestion()

    train_path, test_path = (
        obj.initiate_data_ingestion()
    )

    print("Train Path:", train_path)
    print("Test Path :", test_path)