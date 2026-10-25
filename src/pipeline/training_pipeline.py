"""
training_pipeline.py

Complete ML training workflow:

Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
"""

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def run_pipeline(self):

        print("\n")
        print("========================================")
        print("        ML TRAINING PIPELINE")
        print("========================================")


        # ==================================================
        # STEP 1 — DATA INGESTION
        # ==================================================

        print("\n[1/4] Starting Data Ingestion...")

        data_ingestion = DataIngestion()

        train_path, test_path = (
            data_ingestion.initiate_data_ingestion()
        )

        print("Training data:", train_path)
        print("Testing data :", test_path)


        # ==================================================
        # STEP 2 — DATA VALIDATION
        # ==================================================

        print("\n[2/4] Starting Data Validation...")

        data_validation = DataValidation()

        validation_status = (
            data_validation.initiate_data_validation(
                train_path,
                test_path
            )
        )

        print(
            "Validation status:",
            validation_status
        )

        # STOP THE PIPELINE IF VALIDATION FAILS

        if not validation_status:

            raise ValueError(
                "Data validation failed. "
                "Training pipeline stopped."
            )

        print("Data validation passed.")


        # ==================================================
        # STEP 3 — DATA TRANSFORMATION
        # ==================================================

        print("\n[3/4] Starting Data Transformation...")

        data_transformation = DataTransformation()

        train_array, test_array, preprocessor_path = (
            data_transformation.initiate_data_transformation(
                train_path,
                test_path
            )
        )

        print(
            "Preprocessor saved at:",
            preprocessor_path
        )


        # ==================================================
        # STEP 4 — MODEL TRAINING
        # ==================================================

        print("\n[4/4] Starting Model Training...")

        model_trainer = ModelTrainer()

        metrics, model_path = (
            model_trainer.initiate_model_trainer(
                train_array,
                test_array
            )
        )


        # ==================================================
        # FINAL RESULT
        # ==================================================

        print("\n")
        print("========================================")
        print("       TRAINING PIPELINE COMPLETE")
        print("========================================")

        print(
            f"Accuracy   : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision  : {metrics['precision']:.4f}"
        )

        print(
            f"Recall     : {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score   : {metrics['f1']:.4f}"
        )

        print(
            f"Model saved at: {model_path}"
        )

        print(
            f"Preprocessor saved at: "
            f"{preprocessor_path}"
        )

        return metrics, model_path


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.run_pipeline()