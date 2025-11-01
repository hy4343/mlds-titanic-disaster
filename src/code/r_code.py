#!/usr/bin/env Rscript

# ======================================================
# Titanic Survival Prediction — R code version
# ======================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

# ------------------------------------------------------
# 1. Utility Functions
# ------------------------------------------------------

# Get current script directory (compatible with Docker or local execution)
get_script_directory <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  file_path <- sub(file_arg, "", args[grep(file_arg, args)])
  if (length(file_path) == 0) {
    # Fallback path for interactive use
    return(normalizePath("src/R_code/main.R"))
  }
  normalizePath(file_path)
}

# Safely read a CSV file and check if it exists
load_csv_safely <- function(file_path) {
  if (!file.exists(file_path)) {
    stop(paste("File not found:", file_path))
  }
  read_csv(file_path, show_col_types = FALSE)
}

# Create additional engineered features related to family size
create_family_features <- function(df) {
  df %>%
    mutate(
      FamilySize = SibSp + Parch + 1,
      IsAlone = as.integer(FamilySize == 1)
    )
}

# ------------------------------------------------------
# 2. Define Project Paths
# ------------------------------------------------------

# Determine the root project structure relative to the script
script_directory <- dirname(get_script_directory())
project_root     <- normalizePath(file.path(script_directory, "..", ".."))
data_directory   <- file.path(project_root, "src", "data")

# Define expected dataset paths
train_file <- file.path(data_directory, "train.csv")
test_file  <- file.path(data_directory, "test.csv")

# ------------------------------------------------------
# 3. Load Data
# ------------------------------------------------------

# Read both training and test CSV files
train <- load_csv_safely(train_file)
test  <- load_csv_safely(test_file)

# Display a few rows of the raw data for reference
cat("\n--- Training Data (Head) ---\n")
print(head(train))
cat("\n--- Test Data (Head) ---\n")
print(head(test))

# ------------------------------------------------------
# 4. Data Cleaning and Preprocessing
# ------------------------------------------------------

# Remove columns that are text-heavy or not useful for modeling
columns_to_drop <- c("Name", "Ticket", "Cabin")
train <- select(train, -any_of(columns_to_drop))
test  <- select(test,  -any_of(columns_to_drop))

# Fill missing numeric values with the median of the training data
if ("Age" %in% names(train)) {
  age_median <- median(train$Age, na.rm = TRUE)
  train$Age[is.na(train$Age)] <- age_median
  test$Age[is.na(test$Age)]   <- age_median
}

if ("Fare" %in% names(test)) {
  fare_median <- median(train$Fare, na.rm = TRUE)
  test$Fare[is.na(test$Fare)] <- fare_median
}

# Convert categorical variable 'Sex' into numeric representation
# (1 for female, 0 for male)
train$Sex <- as.integer(train$Sex == "female")
test$Sex  <- as.integer(test$Sex == "female")

# Add new engineered features related to family composition
train <- create_family_features(train)
test  <- create_family_features(test)

# Handle missing Embarked values by replacing them with the mode
if ("Embarked" %in% names(train)) {
  mode_embarked <- names(sort(table(train$Embarked), decreasing = TRUE))[1]
  train$Embarked[is.na(train$Embarked)] <- mode_embarked
  test$Embarked[is.na(test$Embarked)]   <- mode_embarked
  train$Embarked <- factor(train$Embarked)
  test$Embarked  <- factor(test$Embarked, levels = levels(train$Embarked))
}

# Display a sample of the cleaned data
cat("\n--- Cleaned Training Data (Head) ---\n")
print(head(train))

# ------------------------------------------------------
# 5. Model Training (Logistic Regression)
# ------------------------------------------------------

# Convert the Survived column to integer type for modeling
train$Survived <- as.integer(train$Survived)

# Define model formula with selected predictors
formula_model <- Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + FamilySize + IsAlone

# Fit a logistic regression model using glm() with binomial family
logit_model <- glm(formula_model, data = train, family = binomial(link = "logit"))

# Display model coefficients summary
cat("\n--- Logistic Regression Coefficients ---\n")
print(coef(summary(logit_model)))

# Evaluate training accuracy
predicted_train <- ifelse(predict(logit_model, newdata = train, type = "response") >= 0.5, 1L, 0L)
training_accuracy <- mean(predicted_train == train$Survived)
cat("\nTraining Accuracy:", round(training_accuracy, 4), "\n")

# ------------------------------------------------------
# 6. Generate Predictions on Test Data
# ------------------------------------------------------

# Apply trained model to test data
test_predictions <- predict(logit_model, newdata = test, type = "response")
test_labels <- ifelse(test_predictions >= 0.5, 1L, 0L)

# Combine PassengerId and predicted survival outcomes
predicted_output <- data.frame(
  PassengerId = test$PassengerId,
  Survived = test_labels
)

# Display first few predicted results
cat("\n--- First Few Predicted Records ---\n")
print(head(predicted_output))

# Compute basic survival rate statistics for test predictions
survival_rate <- mean(predicted_output$Survived) * 100
cat("\nPredicted Survival Rate (Test Set):", sprintf("%.2f%%", survival_rate), "\n")

# ------------------------------------------------------
# 7. Save Model Output
# ------------------------------------------------------

# Write predictions to CSV file in the data folder
output_file <- file.path(data_directory, "survival_predictions_r.csv")
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
write_csv(predicted_output, output_file)

cat("\nPredictions successfully saved to:", output_file, "\n")
cat("===== Titanic R Pipeline Completed Successfully =====\n")
