import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data/data.csv")

# Convert gender to numeric
df["gender"] = pd.factorize(df["gender"])[0]

# Remove missing values
cleaned_df = df.dropna()

# Separate features and target
X = cleaned_df.drop("target", axis=1)
y = cleaned_df["target"]

# Train-test split
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Hyperparameter search
log_reg_grid = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["liblinear"]
}

model = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    verbose=True
)

# Train
model.fit(X_train, y_train)

# Results
print("Best params:", model.best_params_)
print("Test score:", model.score(X_test, y_test))

# Save trained model
joblib.dump(model, "models/heart_disease_model.pkl")

print("Model saved to models/heart_disease_model.pkl")
