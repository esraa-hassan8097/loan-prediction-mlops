import pickle
import shap
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData
from src.genai.llm_client import LLMClient


# -----------------------------
# 1. Load model and preprocessor
# -----------------------------

with open("artifacts/preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

with open("artifacts/model.pkl", "rb") as file:
    model = pickle.load(file)


# -----------------------------
# 2. Create loan application
# -----------------------------

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


# -----------------------------
# 3. Transform application
# -----------------------------

transformed_features = preprocessor.transform(features)

feature_names = preprocessor.get_feature_names_out()

transformed_df = pd.DataFrame(
    transformed_features,
    columns=feature_names
)


# -----------------------------
# 4. Prepare SHAP background
# -----------------------------

train_df = pd.read_csv("artifacts/train.csv")

X_train = train_df.drop(
    columns=["Loan_ID", "Loan_Status"]
)

X_train_transformed = preprocessor.transform(X_train)

X_train_transformed_df = pd.DataFrame(
    X_train_transformed,
    columns=feature_names
)


# -----------------------------
# 5. Calculate SHAP
# -----------------------------

explainer = shap.LinearExplainer(
    model,
    X_train_transformed_df
)

shap_values = explainer(transformed_df)


# -----------------------------
# 6. Get ML prediction
# -----------------------------

prediction = model.predict(transformed_features)[0]


# -----------------------------
# 7. Create SHAP DataFrame
# -----------------------------

contributions = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values.values[0]
})


# -----------------------------
# 8. Map transformed features
#    back to original features
# -----------------------------

def get_original_feature(feature_name):

    if "ApplicantIncome" in feature_name:
        return "Applicant Income"

    if "CoapplicantIncome" in feature_name:
        return "Coapplicant Income"

    if "LoanAmount" in feature_name:
        return "Loan Amount"

    if "Loan_Amount_Term" in feature_name:
        return "Loan Amount Term"

    if "Credit_History" in feature_name:
        return "Credit History"

    if "Gender_" in feature_name:
        return "Gender"

    if "Married_" in feature_name:
        return "Married"

    if "Dependents_" in feature_name:
        return "Dependents"

    if "Education_" in feature_name:
        return "Education"

    if "Self_Employed_" in feature_name:
        return "Self Employed"

    if "Property_Area_" in feature_name:
        return "Property Area"

    return feature_name


contributions["original_feature"] = (
    contributions["feature"].apply(get_original_feature)
)


# -----------------------------
# 9. Combine one-hot features
# -----------------------------

grouped_contributions = (
    contributions
    .groupby("original_feature")["shap_value"]
    .sum()
    .reset_index()
)


grouped_contributions["absolute_shap"] = (
    grouped_contributions["shap_value"].abs()
)


grouped_contributions = grouped_contributions.sort_values(
    "absolute_shap",
    ascending=False
)


top_contributions = grouped_contributions.head(5)


# -----------------------------
# 10. Build readable SHAP text
# -----------------------------

shap_text = "\n".join(
    f"- {row.original_feature}: "
    f"{row.shap_value:+.4f}"
    for row in top_contributions.itertuples()
)


# -----------------------------
# 11. Build LLM prompt
# -----------------------------

prompt = f"""
You are explaining a machine learning loan prediction.

The machine learning model predicted:

Prediction: {prediction}

The applicant's information is:

{features.to_string(index=False)}

The top SHAP contributions are:

{shap_text}

Explain the model prediction in simple, professional language.

Important rules:

- The ML model makes the prediction.
- Do not change or override the prediction.
- SHAP values describe how features contributed to the model output.
- A positive value means the feature pushed the model output higher.
- A negative value means the feature pushed the model output lower.
- The magnitude indicates the strength of the contribution.
- Do not claim that a feature caused the loan decision.
- Do not invent information.
- Do not claim that SHAP values are probabilities.
"""


# -----------------------------
# 12. Call Azure LLM
# -----------------------------

llm = LLMClient()

explanation = llm.generate(prompt)


# -----------------------------
# 13. Display results
# -----------------------------

print("\n==============================")
print("ML MODEL PREDICTION")
print("==============================")

print(prediction)


print("\n==============================")
print("APPLICANT")
print("==============================")

print(features.to_string(index=False))


print("\n==============================")
print("READABLE SHAP CONTRIBUTIONS")
print("==============================")

print(
    top_contributions[
        ["original_feature", "shap_value"]
    ].to_string(index=False)
)


print("\n==============================")
print("LLM EXPLANATION")
print("==============================")

print(explanation)
