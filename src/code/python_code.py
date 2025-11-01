"""
Titanic Survival Prediction - python code version
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================================
# CONFIGURATION
# =========================================================
DATA_DIR = "src/data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "results", "predictions_v2.csv")

DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Pclass"]
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]

# =========================================================
# UTILITIES
# =========================================================
def log_section(title: str):
    """Helper to print a clean section divider."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# =========================================================
# DATA PREPROCESSING
# =========================================================
def preprocess_dataframe(df: pd.DataFrame, scaler=None, fit_scaler=True, 
                         age_median=None, fare_median=None, is_train=True):
    """Generalized preprocessing for both training and test datasets."""
    df_proc = df.copy()

    # Drop irrelevant columns
    df_proc.drop(columns=DROP_COLUMNS, inplace=True, errors="ignore")

    # Fill missing categorical values with 'Missing' label
    for cat_col in CATEGORICAL_FEATURES:
        df_proc[cat_col] = df_proc[cat_col].fillna("Missing")

    # Encode categorical columns via one-hot encoding
    df_proc = pd.get_dummies(df_proc, prefix_sep="_", columns=CATEGORICAL_FEATURES)

    # Fill missing numerical values
    if is_train:
        age_median = df_proc["Age"].median()
        fare_median = df_proc["Fare"].median()
    df_proc["Age"].fillna(age_median, inplace=True)
    df_proc["Fare"].fillna(fare_median, inplace=True)

    # Standardize numeric columns
    if fit_scaler:
        scaler = StandardScaler()
        df_proc[NUMERIC_FEATURES] = scaler.fit_transform(df_proc[NUMERIC_FEATURES])
    else:
        df_proc[NUMERIC_FEATURES] = scaler.transform(df_proc[NUMERIC_FEATURES])

    return df_proc, scaler, age_median, fare_median


# =========================================================
# MODEL PIPELINE
# =========================================================
def train_model(train_df: pd.DataFrame):
    """Train logistic regression and return the fitted model."""
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]

    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f" Training Accuracy: {acc:.4f}")
    return model


def predict_and_export(model, test_df: pd.DataFrame, passenger_ids, output_path: str):
    """Generate test predictions and save them to a CSV file."""
    preds = model.predict(test_df)
    output_df = pd.DataFrame({"PassengerId": passenger_ids, "Predicted": preds})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"Total predictions: {len(preds)} | Survived=1: {(preds == 1).sum()}")


# =========================================================
# MAIN WORKFLOW
# =========================================================
def main():
    log_section("LOAD TRAINING DATA")
    train_raw = pd.read_csv(TRAIN_PATH)
    print(f"Train shape: {train_raw.shape}")

    log_section("PREPROCESS TRAINING DATA")
    train_proc, scaler, age_med, fare_med = preprocess_dataframe(train_raw, is_train=True)
    print(f"Processed training shape: {train_proc.shape}")

    log_section("TRAIN MODEL")
    model = train_model(train_proc)

    log_section("LOAD TEST DATA")
    test_raw = pd.read_csv(TEST_PATH)
    passenger_ids = test_raw["PassengerId"].copy() if "PassengerId" in test_raw.columns else test_raw.index

    log_section("PREPROCESS TEST DATA")
    test_proc, _, _, _ = preprocess_dataframe(
        test_raw,
        scaler=scaler,
        fit_scaler=False,
        age_median=age_med,
        fare_median=fare_med,
        is_train=False
    )

    # Align columns to match training set
    missing_cols = [c for c in train_proc.columns if c not in test_proc.columns and c != "Survived"]
    for col in missing_cols:
        test_proc[col] = 0
    test_proc = test_proc.reindex(columns=[c for c in train_proc.columns if c != "Survived"], fill_value=0)

    log_section("PREDICT & EXPORT RESULTS")
    predict_and_export(model, test_proc, passenger_ids, OUTPUT_PATH)

    log_section("SUMMARY")
    print(f"- Dropped Columns: {DROP_COLUMNS}")
    print(f"- One-Hot Encoded: {CATEGORICAL_FEATURES}")
    print(f"- Standardized Numeric Features: {NUMERIC_FEATURES}")
    print(f"- Missing values imputed (medians from train): Age={age_med:.2f}, Fare={fare_med:.2f}")
    print(f"- Scaler trained only on training data to prevent leakage.")


# =========================================================
# FINAL EXECUTION TO SEE THE OUTPUT
# =========================================================
if __name__ == "__main__":
    main()
