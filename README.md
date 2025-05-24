# Disease-Prediction-Using-Machine-Learning-Logistic-Regression-
 Predicts disease risk using logistic regression on health data. Includes preprocessing, EDA, training, evaluation, and saves the model as a .pkl file for further use in deployment or analysis.
# 🧠 Disease Prediction Using Machine Learning (Logistic Regression)

This project aims to predict the risk of disease (e.g., stroke) using patient health data by applying Logistic Regression, a supervised machine learning algorithm. The final model is saved as a `.pkl` file for future use or integration into healthcare systems.

---

## 📌 Problem Statement

Timely prediction of disease risk can assist healthcare professionals in early intervention. This project uses historical health data to build a logistic regression model that predicts the likelihood of a patient having a disease based on several clinical parameters.

---

## 📊 Dataset

The dataset contains various patient-level features such as:
- Age
- Gender
- BMI
- Hypertension
- Heart Disease
- Smoking Status
- Average Glucose Level
- Stroke (target variable)

> Dataset Source: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

---

## 🛠️ Technologies Used

| Tool            | Purpose                           |
|------------------|-----------------------------------|
| Python           | Programming language              |
| Jupyter Notebook | Development environment           |
| Pandas           | Data manipulation                 |
| NumPy            | Numerical operations              |
| Matplotlib       | Data visualization                |
| Seaborn          | Correlation analysis              |
| Scikit-learn     | Model building and evaluation     |
| Joblib           | Model serialization (.pkl file)   |

---

## 🔄 Workflow

### 1. Exploratory Data Analysis (EDA)
- Checked for missing values
- Studied feature distributions
- Plotted correlation heatmaps

### 2. Data Preprocessing
- Encoded categorical variables
- Filled or dropped missing values
- Normalized numeric features if needed
- Split data into train and test sets

### 3. Model Training
- Logistic Regression applied using Scikit-learn
- Evaluated using Accuracy, Precision, Recall, F1 Score

### 4. Model Saving
- Trained model saved using `joblib`:
```python
import joblib
joblib.dump(model, 'stroke_model.pkl')

