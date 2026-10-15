import pickle
import shap
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData


# 1. Load the trained preprocessing object and model
with open("artifacts/preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

with open("artifacts/model.pkl", "rb") as file:
    model = pickle.load(file)


# 2. Create one loan applicant
customer = CustomData(
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
    Property_Area="Urban"
)


# 3. Convert applicant to DataFrame
features = customer.get_data_as_dataframe()


# 4. Apply the SAME preprocessing used during training
transformed_features = preprocessor.transform(features)


# 5. Convert transformed data to DataFrame
transformed_df = pd.DataFrame(
    transformed_features,
    columns=preprocessor.get_feature_names_out()
)


# 6. Create SHAP explainer for the Logistic Regression model
explainer = shap.LinearExplainer(
    model,
    transformed_df
)


# 7. Calculate SHAP values for this applicant
shap_values = explainer(transformed_df)


# 8. Print prediction
prediction = model.predict(transformed_features)

print("Prediction:", prediction[0])


# 9. Print SHAP contributions
print("\nSHAP CONTRIBUTIONS:")

for feature, value in zip(
    transformed_df.columns,
    shap_values.values[0]
):
    print(f"{feature}: {value:.4f}")
