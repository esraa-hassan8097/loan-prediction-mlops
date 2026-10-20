import os
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
    fabric_server: str = "m3sh5gudvcvejaykp2k6pq3axa-m4e3rofohg5urbtbsz2zpkv5vq.datawarehouse.fabric.microsoft.com"
    fabric_database: str = "loan_mlops_lakehouse"
    fabric_table: str = "dbo.loan_raw"
    azure_config_dir: str = os.path.expanduser("~/.azure-fabric")


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

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method.")

        os.makedirs("artifacts", exist_ok=True)

        connection = self.get_fabric_connection()

        logging.info("Reading loan dataset from Microsoft Fabric.")

        cursor = connection.cursor()

        cursor.execute(
            f"SELECT * FROM {self.ingestion_config.fabric_table}"
        )

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
            f"Dataset loaded from Fabric successfully. Shape: {df.shape}"
        )

        logging.info("Splitting dataset into train and test sets.")

        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        logging.info("Train and test sets created.")

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