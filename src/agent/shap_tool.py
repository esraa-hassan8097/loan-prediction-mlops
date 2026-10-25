import pickle

import pandas as pd
import shap

from src.pipeline.prediction_pipeline import CustomData


def explain_loan(
    Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area,
):
    """
    Generate SHAP feature contributions for a loan prediction.
    """

    # ---------------------------------------------------------
    # 1. Load the trained preprocessing pipeline
    # ---------------------------------------------------------

    with open("artifacts/preprocessor.pkl", "rb") as file:
        preprocessor = pickle.load(file)  # nosec B301 - trusted internal model artifact

    # ---------------------------------------------------------
    # 2. Load the trained ML model
    # ---------------------------------------------------------

    with open("artifacts/model.pkl", "rb") as file:
        model = pickle.load(file) # nosec B301 - trusted internal model artifact

    # ---------------------------------------------------------
    # 3. Load training data to use as SHAP background
    # ---------------------------------------------------------

    training_data = pd.read_csv("artifacts/train.csv")

    # Remove columns that are not model input features
    background_data = training_data.drop(
        columns=["Loan_ID", "Loan_Status"],
        errors="ignore"
    )

    # ---------------------------------------------------------
    # 4. Transform the background data
    # ---------------------------------------------------------

    transformed_background = preprocessor.transform(
        background_data
    )

    # ---------------------------------------------------------
    # 5. Create the applicant we want to explain
    # ---------------------------------------------------------

    customer = CustomData(
        Gender=Gender,
        Married=Married,
        Dependents=Dependents,
        Education=Education,
        Self_Employed=Self_Employed,
        ApplicantIncome=ApplicantIncome,
        CoapplicantIncome=CoapplicantIncome,
        LoanAmount=LoanAmount,
        Loan_Amount_Term=Loan_Amount_Term,
        Credit_History=Credit_History,
        Property_Area=Property_Area,
    )

    features = customer.get_data_as_dataframe()

    # ---------------------------------------------------------
    # 6. Transform the applicant using the SAME preprocessor
    # ---------------------------------------------------------

    transformed_features = preprocessor.transform(features)

    # ---------------------------------------------------------
    # 7. Create the SHAP explainer
    # ---------------------------------------------------------

    explainer = shap.LinearExplainer(
        model,
        transformed_background
    )

    # ---------------------------------------------------------
    # 8. Calculate SHAP values for this applicant
    # ---------------------------------------------------------

    shap_values = explainer(transformed_features)

    # ---------------------------------------------------------
    # 9. Get the names of the transformed features
    # ---------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    # ---------------------------------------------------------
    # 10. Connect feature names with SHAP values
    # ---------------------------------------------------------

    contributions = []

    for name, value in zip(
        feature_names,
        shap_values.values[0]
    ):
        contributions.append(
            {
                "feature": name,
                "shap_value": float(value),
            }
        )

    # ---------------------------------------------------------
    # 11. Sort by absolute impact
    # ---------------------------------------------------------

    contributions.sort(
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    return contributions


if __name__ == "__main__":

    result = explain_loan(
        Gender="Male",
        Married="Yes",
        Dependents="1",
        Education="Graduate",
        Self_Employed="No",
        ApplicantIncome=5000,
        CoapplicantIncome=1500,
        LoanAmount=120,
        Loan_Amount_Term=360,
        Credit_History=1,
        Property_Area="Urban",
    )

    print("\nSHAP EXPLANATION\n")

    for item in result[:5]:
        print(
            f"{item['feature']}: "
            f"{item['shap_value']:+.6f}"
        )