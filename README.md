# 🧭 Titanic Logistic Regression — Part 3

This project implements a **logistic regression model** to predict passenger survival on the **Titanic dataset**.  
It is designed as Part 3 of the MLDS course homework and demonstrates data preprocessing, model training, and Docker containerization.

---

## 📁 Project Structure

# 🛳️ MLDS Titanic Disaster Prediction

A data science project for predicting passenger survival on the Titanic using machine learning techniques.  
This project follows the classic [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic) and is organized for clarity, reproducibility, and containerization.

---

## 📁 Project Structure



---

## ⚙️ Environment Setup

### 1. Install Dependencies (Local Python)

If running locally (without Docker):

```bash
pip install -r src/requirements.txt
```
2. Build Docker Image
If running inside a container:
docker build -t titanic-python .


3. Run Docker Container
docker run titanic-python

| Step      | Description                                                                                  |
| --------- | -------------------------------------------------------------------------------------------- |
| **13**    | Load and inspect the Titanic training data                                                   |
| **14**    | Clean and preprocess features (handle missing values, encode categories)                     |
| **15–16** | Train a logistic regression model using scikit-learn                                         |
| **17**    | Predict survival outcomes on the test dataset                                                |
| **18**    | Save predictions to `src/data/predictions.csv` (instead of calculating accuracy on test set) |


🧩 Notes
Survived_Prediction = 1 → Passenger survived
Survived_Prediction = 0 → Passenger did not survive
Do not calculate accuracy on the test set; only save predictions as instructed.
The baseline gender_submission.csv file is provided for comparison.
