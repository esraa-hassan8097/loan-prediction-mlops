import pickle

import pandas as pd
import shap

# ============================================================
# 1. Load the exact training data
# ============================================================

train_df = pd.read_csv("artifacts/train.csv")


# ============================================================
# 2. Separate input features from the target
# ============================================================

X_train = train_df.drop(
    columns=["Loan_ID", "Loan_Status"]
)


# ============================================================
# 3. Load the SAME preprocessor used during model training
# ============================================================

with open("artifacts/preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)


# ============================================================
# 4. Load the SAME trained Logistic Regression model
# ============================================================

with open("artifacts/model.pkl", "rb") as file:
    model = pickle.load(file)


# ============================================================
# 5. Apply the SAME preprocessing used during training
# ============================================================

X_train_transformed = preprocessor.transform(X_train)


# ============================================================
# 6. Get the names of the 20 transformed features
# ============================================================

feature_names = preprocessor.get_feature_names_out()


# Convert transformed data into a DataFrame
# so the feature names stay attached to the values.

X_train_transformed_df = pd.DataFrame(
    X_train_transformed,
    columns=feature_names
)


# ============================================================
# 7. Create SHAP LinearExplainer
# ============================================================

explainer = shap.LinearExplainer(
    model,
    X_train_transformed_df
)


# ============================================================
# 8. Calculate SHAP values for ALL training applicants
# ============================================================

shap_values = explainer(
    X_train_transformed_df
)


# ============================================================
# 9. Basic verification
# ============================================================

print("Training rows:", X_train.shape[0])

print(
    "Transformed features:",
    X_train_transformed_df.shape[1]
)

print(
    "\nSHAP values shape:"
)

print(
    shap_values.values.shape
)


# ============================================================
# 10. Calculate GLOBAL SHAP importance
# ============================================================

# Take the absolute value because we care about
# the magnitude of contribution, regardless of direction.

mean_abs_shap = (
    abs(shap_values.values)
    .mean(axis=0)
)


# Create a table containing:
# - feature name
# - average absolute SHAP contribution

importance = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }
)


# Sort from largest contribution to smallest.

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)


# ============================================================
# 11. Display global feature importance
# ============================================================

print(
    "\nGLOBAL SHAP IMPORTANCE:"
)

print(
    importance.to_string(index=False)
)
