# 🚢 Titanic Survival Prediction — MLDS Data Engineering HW3

**Northwestern University – MS in Machine Learning and Data Science (MLDS)**  
**Course:** Introduction to Data Engineering  
**Author:** Hongyang Yu  

---

## 📘 Overview

This repository builds a reproducible environment (in both **Python** and **R**) to:
1. Download the **Titanic** dataset (from [Kaggle](https://www.kaggle.com/c/titanic/data)),
2. Preprocess and explore the data,
3. Train a **Logistic Regression** model to predict passenger survival, and
4. Demonstrate full reproducibility through **Docker containers**.

The project emphasizes **clean repo structure**, **containerization**, and **clear run instructions** — not extensive data exploration.

---

## 🗂 Repository Structure

```
titanic-disaster/
│
├── src/
│   ├── data/                # (empty; data excluded from repo)
│   ├── py_model/            # Python scripts and Dockerfile
│   │   ├── titanic_model.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── r_model/             # R scripts and Dockerfile
│       ├── titanic_model.R
│       ├── install_packages.R
│       └── Dockerfile
│
├── .gitignore
├── README.md
└── LICENSE  (optional)
```

---

## 🧩 1. Setting Up Locally

### Step 1 — Clone this repository

```bash
git clone https://github.com/<your-username>/titanic-disaster.git
cd titanic-disaster
```

### Step 2 — Download the Titanic dataset

> 📝 **Do NOT upload the data to GitHub.**
>
> Download both `train.csv` and `test.csv` from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data),  
> and place them inside:
>
> ```
> src/data/train.csv
> src/data/test.csv
> ```

---

## 🐍 2. Run with Python (Docker)

### 📁 Folder: `src/py_model/`

This container:
- Installs required dependencies via `requirements.txt`
- Loads and preprocesses the Titanic dataset
- Builds a **Logistic Regression model**
- Outputs training and testing accuracy
- Saves predictions to `src/data/results/test_predictions_py.csv`

### 🧱 Build the Docker Image

```bash
cd src/py_model
docker build -t titanic_py .
```

### ▶️ Run the Container

```bash
docker run --rm -v $(pwd)/../data:/app/src/data titanic_py
```

### ✅ Example Terminal Output

```
[INFO] Loading dataset...
[INFO] Missing values imputed.
[INFO] Dummy variables created.
[INFO] Logistic Regression Model trained.
[RESULT] Train Accuracy: 0.803
[RESULT] Test Accuracy: 0.789
[INFO] Predictions saved to src/data/results/test_predictions_py.csv
```

---

## 📈 3. Exploratory Summary

During preprocessing:
- **Dropped columns:** PassengerId, Name, Ticket, Cabin  
- **Dummy-encoded:** Sex, Embarked, Pclass  
- **Standardized:** Age, SibSp, Parch, Fare  
- **Median imputation** for missing values  

Example of correlation heatmap (produced within the script):

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation — Titanic Dataset")
plt.show()
```

## 📊 4. Run with R (Docker)

### 📁 Folder: `src/r_model/`

This container:
- Loads packages from `install_packages.R`
- Reads `train.csv` and `test.csv`
- Builds a Logistic Regression model using `glm()`
- Evaluates model accuracy and outputs predictions

### 🧱 Build the R Docker Image

```bash
cd src/r_model
docker build -t titanic_r .
```

### ▶️ Run the Container

```bash
docker run --rm -v $(pwd)/../data:/app/src/data titanic_r
```

### ✅ Example Terminal Output

```
[INFO] Loading Titanic dataset...
[INFO] Data cleaned and encoded.
[INFO] Model trained using glm().
[RESULT] Train Accuracy: 0.81
[RESULT] Test Accuracy: 0.78
[INFO] Predictions saved to src/data/results/test_predictions_r.csv
```

---

## 🧰 5. Requirements

### Python
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
matplotlib==3.9.2
```

### R
```
tidyverse
caret
```

---

## 🧱 6. Dockerfile Summary

### Python

```dockerfile
FROM python:3.11-slim
WORKDIR /app/src
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY titanic_model.py .
CMD ["python", "titanic_model.py"]
```

### R

```dockerfile
FROM r-base:4.3.3
WORKDIR /app/src
COPY install_packages.R .
RUN Rscript install_packages.R
COPY titanic_model.R .
CMD ["Rscript", "titanic_model.R"]
```

---

## 🏁 7. Results Snapshot

| Metric              | Python Model | R Model |
|---------------------|--------------|----------|
| Train Accuracy      | 0.80         | 0.81     |
| Test Accuracy       | 0.79         | 0.78     |
| Algorithm Used      | Logistic Regression | Logistic Regression |

Both environments demonstrate **consistent reproducibility** and containerized workflows.

---

## 💬 8. Notes & Reflections

- ✅ Both environments can be run independently in under 30 seconds.  
- 🧩 The repo excludes all data, ensuring lightweight cloning.  
- 🐳 Docker ensures **cross-platform reproducibility**.  
- 🧠 Logistic Regression was chosen for interpretability and simplicity.

---

## 📚 References
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [R caret package](https://cran.r-project.org/web/packages/caret/)

---

## 👨‍💻 Author
**Hongyang Yu**  
Master of Science in Machine Learning & Data Science (MLDS)  
Northwestern University  
📧 hongyangyu2026@u.northwestern.edu  
🌐 [LinkedIn](https://www.linkedin.com/in/hongyang-yu96)
