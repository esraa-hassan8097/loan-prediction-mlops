import pickle

import pandas as pd
import shap

from src.pipeline.prediction_pipeline import CustomData

# Load preprocessor and model
with open("artifacts/preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

with open("artifacts/model.pkl", "rb") as file:
    model = pickle.load(file)


# Create one loan application
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

features = customer.get_data_as_dataframe()


# Transform the raw data exactly as the ML model expects
transformed_features = preprocessor.transform(features)

feature_names = preprocessor.get_feature_names_out()

transformed_df = pd.DataFrame(
    transformed_features,
    columns=feature_names
)


# Use training data as SHAP background
train_df = pd.read_csv("artifacts/train.csv")

X_train = train_df.drop(
    columns=["Loan_ID", "Loan_Status"]
)

X_train_transformed = preprocessor.transform(X_train)

X_train_transformed_df = pd.DataFrame(
    X_train_transformed,
    columns=feature_names
)


# Create SHAP explainer
explainer = shap.LinearExplainer(
    model,
    X_train_transformed_df
)


# Explain this applicant
shap_values = explainer(transformed_df)


# Prediction
prediction = model.predict(transformed_features)[0]

print("\n==============================")
print("LOAN PREDICTION")
print("==============================")
print("Prediction:", prediction)

print("\n==============================")
print("SHAP CONTRIBUTIONS")
print("==============================")

contributions = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values.values[0]
})

contributions["absolute_shap"] = contributions["shap_value"].abs()

contributions = contributions.sort_values(
    "absolute_shap",
    ascending=False
)

print(
    contributions[
        ["feature", "shap_value"]
    ].head(10).to_string(index=False)
)
