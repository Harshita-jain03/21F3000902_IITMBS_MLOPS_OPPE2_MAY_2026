import pandas as pd
import joblib
import shap

# Load data
df = pd.read_csv("data/data.csv")

# Same preprocessing as training
df["gender"] = pd.factorize(df["gender"])[0]
cleaned_df = df.dropna()

# Features and target
X = cleaned_df.drop("target", axis=1)

# Load trained model
model = joblib.load("models/heart_disease_model.pkl")

# Use the best estimator from RandomizedSearchCV
best_model = model.best_estimator_

# SHAP explainer for Logistic Regression
explainer = shap.LinearExplainer(best_model, X)

# Calculate SHAP values
shap_values = explainer(X)

# Mean absolute SHAP impact
importance = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": abs(shap_values.values).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

print("\nFeature importance based on SHAP:")
print(importance.to_string(index=False))

print("\nLeast impactful factors:")
print(importance.tail(3).to_string(index=False))

# Save results
importance.to_csv("shap_feature_importance.csv", index=False)

print("\nSHAP results saved to shap_feature_importance.csv")
