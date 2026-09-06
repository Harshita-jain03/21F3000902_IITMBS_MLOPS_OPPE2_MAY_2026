import pandas as pd
import joblib

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score


# ========================================
# 1. LOAD DATA
# ========================================

df = pd.read_csv("data/data.csv")


# ========================================
# 2. SAME PREPROCESSING AS TRAINING
# ========================================

df["gender"] = pd.factorize(df["gender"])[0]

cleaned_df = df.dropna().copy()


# ========================================
# 3. CONVERT TARGET TO 0/1
# ========================================

cleaned_df["target"] = cleaned_df["target"].map({
    "no": 0,
    "yes": 1
})


# ========================================
# 4. FEATURES AND TARGET
# ========================================

X = cleaned_df.drop("target", axis=1)
y = cleaned_df["target"]


# ========================================
# 5. LOAD TRAINED MODEL
# ========================================

model = joblib.load("models/heart_disease_model.pkl")

best_model = model.best_estimator_


# ========================================
# 6. MAKE PREDICTIONS
# ========================================

raw_predictions = best_model.predict(X)

y_pred = pd.Series(raw_predictions).map({
    "no": 0,
    "yes": 1
}).to_numpy()


# ========================================
# 7. PREDICTION COUNTS
# ========================================

print("\nPrediction counts:")
print(pd.Series(y_pred).value_counts().sort_index())


# ========================================
# 8. CREATE AGE GROUPS
# ========================================

age_groups = pd.cut(
    cleaned_df["age"],
    bins=[0, 40, 50, 60, 100],
    labels=["<=40", "41-50", "51-60", ">60"],
    include_lowest=True
)


# ========================================
# 9. FAIRNESS METRICS BY AGE GROUP
# ========================================

metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate
    },
    y_true=y,
    y_pred=y_pred,
    sensitive_features=age_groups
)


print("\n========================================")
print("FAIRNESS METRICS BY AGE GROUP")
print("========================================")

print(metric_frame.by_group)


# ========================================
# 10. OVERALL METRICS
# ========================================

print("\n========================================")
print("OVERALL METRICS")
print("========================================")

print(metric_frame.overall)


# ========================================
# 11. DEMOGRAPHIC PARITY DIFFERENCE
# ========================================

dp_difference = demographic_parity_difference(
    y_true=y,
    y_pred=y_pred,
    sensitive_features=age_groups,
    method="between_groups"
)


print("\n========================================")
print("DEMOGRAPHIC PARITY DIFFERENCE")
print("========================================")

print(dp_difference)


# ========================================
# 12. EQUALIZED ODDS DIFFERENCE
# ========================================

eo_difference = equalized_odds_difference(
    y_true=y,
    y_pred=y_pred,
    sensitive_features=age_groups,
    method="between_groups"
)


print("\n========================================")
print("EQUALIZED ODDS DIFFERENCE")
print("========================================")

print(eo_difference)


# ========================================
# 13. SAVE FAIRNESS RESULTS
# ========================================

metric_frame.by_group.to_csv("fairness_by_age.csv")

print("\nFairness results saved to fairness_by_age.csv")
