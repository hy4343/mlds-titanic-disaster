"""
Titanic Survival Prediction Script
---------------------------------
Performs data preprocessing, feature engineering, model training, and prediction output.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==============================
# Data Preprocessing
# ==============================
def process_data(path: str) -> pd.DataFrame:
    """Load and preprocess Titanic dataset."""
    print("=" * 40)
    print(f"Reading dataset: {path}")
    df = pd.read_csv(path)
    print(df.head(), "\n", "=" * 40)

    # Drop irrelevant features
    drop_cols = ["Name", "Ticket", "Cabin", "Embarked"]
    print(f"Dropping columns: {drop_cols}")
    df.drop(columns=drop_cols, inplace=True)
    print(df.head(), "\n", "=" * 40)

    # One-hot encode 'Pclass'
    print("Encoding 'Pclass' feature with OneHotEncoder...")
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded = encoder.fit_transform(df[["Pclass"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Pclass"]))
    df = pd.concat([df.drop(columns="Pclass"), encoded_df], axis=1)
    print(df.head(), "\n", "=" * 40)

    # Fill missing values
    print("Filling missing values (Age, Fare) with median...")
    for col in ["Age", "Fare"]:
        df[col] = df[col].fillna(df[col].median())
    print(df.head(), "\n", "=" * 40)

    # Feature engineering: passenger alone flag
    print("Creating 'NotAlone' feature...")
    df["NotAlone"] = ((df["SibSp"] + df["Parch"]) > 0).astype(float)
    df.drop(columns=["SibSp", "Parch"], inplace=True)
    print(df.head(), "\n", "=" * 40)

    # Encode gender
    print("Encoding 'Sex' into numeric 'IsMale'...")
    df["IsMale"] = (df["Sex"] == "male").astype(float)
    df.drop(columns="Sex", inplace=True)
    print(df.head(), "\n", "=" * 40)

    # Normalize Age and Fare
    print("Normalizing 'Age' and 'Fare' features...")
    scaler = MinMaxScaler()
    df[["Age_Normalized", "Fare_Normalized"]] = scaler.fit_transform(df[["Age", "Fare"]])
    df.drop(columns=["Age", "Fare"], inplace=True)
    print(df.head(), "\n", "=" * 40)

    return df


# ==============================
# Model Training
# ==============================
def train_model(train_df: pd.DataFrame) -> LogisticRegression:
    """Train Logistic Regression model."""
    print("=" * 40)
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    X = train_df.iloc[:, 2:]   # features
    y = train_df["Survived"]   # target
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Training Accuracy: {acc:.4f}")
    return model


# ==============================
# Model Testing
# ==============================
def test_model(model: LogisticRegression, test_path: str):
    """Apply trained model to test data and export predictions."""
    print("=" * 40)
    print("Testing model on test dataset...")
    test_df = process_data(test_path)
    X_test = test_df.iloc[:, 1:]
    predictions = model.predict(X_test)

    test_df["Prediction"] = predictions
    print("Preview of prediction output:")
    print(test_df[["PassengerId", "Prediction"]].head())

    output_path = "prediction.csv"
    test_df[["PassengerId", "Prediction"]].to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# ==============================
# Main Execution
# ==============================
def main():
    train_df = process_data("train.csv")
    model = train_model(train_df)
    test_model(model, "test.csv")


if __name__ == "__main__":
    main()
