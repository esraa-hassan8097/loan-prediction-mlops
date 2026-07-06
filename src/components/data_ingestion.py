import os
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join(
        "data",
        "raw",
        "train_u6lujuX_CVtuZ9i.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method.")

        os.makedirs("artifacts", exist_ok=True)

        logging.info("Reading the raw dataset.")
        df = pd.read_csv(self.ingestion_config.raw_data_path)

        logging.info("Dataset read successfully.")

        logging.info("Splitting the dataset into train and test sets.")
        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        logging.info("Train and test sets created.")

        logging.info("Saving the train and test datasets.")

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


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()