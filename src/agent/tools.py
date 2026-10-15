from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


def predict_loan(
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
    Run the existing ML loan prediction model.

    Returns the raw model prediction:
    Y = approved
    N = rejected
    """

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

    pipeline = PredictPipeline()

    prediction = pipeline.predict(features)

    return str(prediction[0])


if __name__ == "__main__":

    result = predict_loan(
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

    print("Prediction:", result)