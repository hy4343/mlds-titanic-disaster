# Titanic Logistic Regression — Part 3

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# STEP 13 – Load train.csv
# =====================================================
print("\n[STEP 13] Loading training dataset...")
train_df = pd.read_csv("src/data/train.csv")
print(" Train data loaded successfully!")
print("Shape:", train_df.shape)
print(train_df.head())

# =====================================================
# STEP 14 – Explore and clean data
# =====================================================
print("\n[STEP 14] Exploring and cleaning data...")
print(train_df.isnull().sum())

# Fill missing values (no inplace warnings)
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])

# Define features
numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

# =====================================================
# STEP 15 – Build logistic regression model
# =====================================================
print("\n[STEP 15] Building model pipeline...")
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# =====================================================
# STEP 16 – Train and evaluate model on training set
# =====================================================
print("\n[STEP 16] Training model...")
X = train_df[numeric_features + categorical_features]
y = train_df["Survived"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_valid)

print(" Model training complete!")
print("Training accuracy:", round(accuracy_score(y_valid, y_pred), 4))
print(classification_report(y_valid, y_pred))

# =====================================================
# STEP 17 – Predict on test.csv
# =====================================================
print("\n[STEP 17] Predicting on test dataset...")
test_df = pd.read_csv("src/data/test.csv")

# Clean test set — same logic, no inplace
test_df["Age"] = test_df["Age"].fillna(train_df["Age"].median())
test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0])
test_df["Fare"] = test_df["Fare"].fillna(train_df["Fare"].median())

X_test = test_df[numeric_features + categorical_features]
predictions = model.predict(X_test)

print(" Predictions generated successfully!")
print("First 10 predictions:", predictions[:10])

# =====================================================
# STEP 18 – Save predictions to CSV
# =====================================================
import os

print("\n[STEP 18] Saving predictions to file...")

# Get absolute project root (no matter where script is run from)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_root, "src", "data")
output_path = os.path.join(output_dir, "predictions.csv")

# Ensure directory exists
os.makedirs(output_dir, exist_ok=True)

# Create and save output DataFrame
output_df = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived_Prediction": predictions
})
output_df.to_csv(output_path, index=False)

# Confirm output path
print(f"✅ Predictions saved to: {output_path}")
print(f"📂 Current working directory: {os.getcwd()}")
print("[STEP 18] Finished Part 3 successfully!")



