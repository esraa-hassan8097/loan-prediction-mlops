from src.components.data_validation import DataValidation


def test_data_validation():

    validator = DataValidation()

    status = validator.initiate_data_validation(
        "artifacts/train.csv",
        "artifacts/test.csv"
    )

    assert status is True