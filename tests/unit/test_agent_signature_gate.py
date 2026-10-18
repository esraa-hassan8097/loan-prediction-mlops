import pytest

from src.agent.graph import prediction_node


def test_prediction_blocked_without_signature():

    state = {
        "applicant": {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 0,
            "LoanAmount": 150,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Property_Area": "Urban",
        },
        "signature_status": "not_signed",
    }

    with pytest.raises(
        ValueError,
        match="Prediction cannot run before the applicant signs the application."
    ):
        prediction_node(state)
