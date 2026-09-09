# Diabetes Data Preprocessing & Feature Selection

A machine learning project exploring **data preprocessing, dimensionality reduction, feature selection, and classification** using two diabetes datasets.

The project builds a complete data-processing pipeline with Python and compares different feature-selection strategies using a Logistic Regression classifier.

## Project Overview

The project combines two diabetes datasets:

* **PIMA Diabetes Dataset** — 768 samples
* **Frankfurt Diabetes Dataset** — 2,000 samples

After integration, the final dataset contains **2,768 samples** with 8 predictive features.

The goal is to investigate how preprocessing and feature-selection techniques affect diabetes classification performance.

## Workflow

```text
Raw Data
   ↓
Missing Value Handling
   ↓
Dataset Integration
   ↓
Log Transformation
   ↓
Feature Standardization
   ↓
PCA Analysis
   ↓
Feature Selection
   ├── Mutual Information
   └── Recursive Feature Elimination (RFE)
   ↓
Logistic Regression
   ↓
5-Fold Cross Validation
   ↓
Performance Comparison
```

## Data Preprocessing

Several preprocessing steps are applied before model training.

### 1. Missing Value Handling

Zero values in the following medical variables are treated as missing values:

* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI

These values are converted to `NaN` and replaced using the **median value of each feature**.

### 2. Dataset Integration

The PIMA and Frankfurt datasets are combined into a single dataset containing:

* **2,768 records**
* **8 input features**
* **1 target variable (`Outcome`)**

### 3. Log Transformation

Highly skewed variables are transformed using:

```python
np.log1p(x)
```

The transformed features are:

* Insulin
* Diabetes Pedigree Function

### 4. Feature Standardization

All input features are standardized using:

```python
StandardScaler
```

This transforms the data to approximately:

```text
Mean = 0
Standard Deviation = 1
```

## Principal Component Analysis

Principal Component Analysis (**PCA**) is used to analyze dimensionality and variance distribution.

The analysis shows that **7 principal components** are required to explain at least **95% of the total variance**.

## Feature Selection

Two feature-selection approaches are compared.

### Mutual Information

Mutual Information measures the dependency between each feature and the diabetes outcome.

Selected features:

```text
DiabetesPedigreeFunction
BMI
Glucose
Insulin
```

### Recursive Feature Elimination

Recursive Feature Elimination (**RFE**) repeatedly trains a Logistic Regression model and removes less important features.

Selected features:

```text
Age
BMI
DiabetesPedigreeFunction
Glucose
Pregnancies
```

## Model Evaluation

A **Logistic Regression** classifier is evaluated using **5-fold Stratified Cross Validation**.

Four evaluation metrics are used:

* Accuracy
* Precision
* Recall
* F1 Score

### Results

| Feature Set        |   Accuracy | Precision |     Recall |         F1 |
| ------------------ | ---------: | --------: | ---------: | ---------: |
| All Features       |     0.7673 |    0.7059 |     0.5557 |     0.6216 |
| Mutual Information |     0.7706 |    0.7222 |     0.5410 |     0.6180 |
| RFE                | **0.7713** |    0.7163 | **0.5557** | **0.6257** |

The **RFE-selected feature set** achieves the best overall Accuracy and F1 Score while using fewer features than the full baseline model.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Structure

```text
.
├── data/
│   ├── pima.csv
│   └── frankfurt.csv
│
├── output/
│   ├── figures/
│   └── preprocessed_dataset.csv
│
├── notebook.ipynb
└── README.md
```

## Key Techniques

This project demonstrates:

* Data cleaning
* Missing-value imputation
* Data integration
* Log transformation
* Feature scaling
* Exploratory data visualization
* Principal Component Analysis
* Mutual Information
* Recursive Feature Elimination
* Logistic Regression
* Stratified K-Fold Cross Validation
* Classification metric evaluation

## Key Takeaway

Feature selection can reduce model complexity without sacrificing predictive performance.

In this experiment, RFE reduced the feature set from **8 features to 5 features** while slightly improving classification performance compared with using all available features.


