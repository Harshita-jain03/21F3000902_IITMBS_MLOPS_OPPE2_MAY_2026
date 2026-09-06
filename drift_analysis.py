import pandas as pd
import numpy as np

training = pd.read_csv("data/data.csv")
prediction = pd.read_csv("data/random_100.csv")

features = [
    "age", "gender", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

results = []

for feature in features:

    train = training[feature].dropna()
    new = prediction[feature].dropna()

    # Convert categorical/string values to numeric where possible
    train = pd.to_numeric(train, errors="coerce").dropna()
    new = pd.to_numeric(new, errors="coerce").dropna()

    train_mean = train.mean()
    new_mean = new.mean()

    train_std = train.std()
    new_std = new.std()

    if train_mean != 0:
        mean_difference = abs(new_mean - train_mean) / abs(train_mean) * 100
    else:
        mean_difference = 0

    # Flag if mean changes by more than 10%
    drift = "Yes" if mean_difference > 10 else "No"

    results.append({
        "feature": feature,
        "training_mean": round(train_mean, 3),
        "new_data_mean": round(new_mean, 3),
        "training_std": round(train_std, 3),
        "new_data_std": round(new_std, 3),
        "mean_difference_percent": round(mean_difference, 2),
        "drift": drift
    })

result_df = pd.DataFrame(results)

result_df.to_csv("drift_analysis.csv", index=False)

print("\n===== INPUT DRIFT ANALYSIS =====\n")
print(result_df.to_string(index=False))

print("\n===== FEATURES WITH DRIFT =====")
drift_features = result_df[result_df["drift"] == "Yes"]["feature"].tolist()

if drift_features:
    print(drift_features)
else:
    print("No significant drift detected.")

print("\nReport saved to: drift_analysis.csv")
