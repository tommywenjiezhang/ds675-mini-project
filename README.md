# Diabetes Prediction Mini-Project

This project studies supervised machine learning models for predicting diabetes from demographic and clinical features in the Kaggle **Diabetes Prediction Dataset**. The work builds on prior Kaggle notebooks and focuses on comparing classification approaches, handling class imbalance, and choosing evaluation metrics that are appropriate for a medical screening setting.

## Project Summary

- **Author:** Wenjie Zhang
- **Course:** DS675, Spring 2026
- **Task:** Binary classification
- **Target:** `diabetes` (`1` = diabetic, `0` = non-diabetic)
- **Dataset:** [Diabetes Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

The core goal is to predict whether a patient has diabetes using eight health-related input features. Because the dataset is strongly imbalanced, the project emphasizes metrics such as recall, precision, F1-score, and ROC-AUC instead of relying on accuracy alone.

## Dataset Description

The dataset contains **100,000 patient records** in a single CSV file. Each record includes eight predictor variables and one binary target label.

### Features

| Feature | Type | Description |
| --- | --- | --- |
| `gender` | Categorical | Male, Female, Other |
| `age` | Continuous | Patient age in years |
| `hypertension` | Binary | Whether the patient has hypertension |
| `heart_disease` | Binary | Whether the patient has heart disease |
| `smoking_history` | Categorical | Smoking history category |
| `bmi` | Continuous | Body Mass Index |
| `HbA1c_level` | Continuous | Average blood glucose marker over about 3 months |
| `blood_glucose_level` | Continuous | Blood glucose measurement |
| `diabetes` | Binary | Target label |

### Key Characteristics

- No missing values were reported in the source dataset.
- There are **3,854 exact duplicate rows** that should be removed before modeling.
- The dataset is **class imbalanced**: about **91% non-diabetic** and **9% diabetic**.
- Mixed data types require preprocessing and encoding for categorical variables.
- `HbA1c_level` and `blood_glucose_level` are medically important predictors and consistently appear as the strongest features.

## Prediction Task

This is a **supervised binary classification** problem. The model learns from labeled patient records and predicts whether a new patient is likely to have diabetes.

### Evaluation Metrics

Because of class imbalance, a model that always predicts the majority class could still appear accurate while being clinically useless. For that reason, this project tracks:

- **Accuracy** for overall performance
- **Precision** for reliability of positive predictions
- **Recall** for detecting diabetic patients
- **F1-score** for balancing precision and recall
- **ROC-AUC** for threshold-independent discrimination

Recall is especially important in this setting because missing a diabetic patient is more costly than raising some false alarms.

## Prior Work Reviewed

This project is informed by three existing Kaggle notebooks:

### 1. EDA, SMOTE, and Random Forest with GridSearchCV

- Strong exploratory data analysis
- Removes duplicates and rare `gender == "Other"` entries
- Recategorizes smoking history to reduce sparsity
- Uses a preprocessing pipeline with scaling and one-hot encoding
- Handles class imbalance with **SMOTE + RandomUnderSampler**
- Trains a tuned **RandomForestClassifier**

Reported result:

- Accuracy: **95.1%**
- Recall for diabetic class: **0.80**

Main takeaway:

- `HbA1c_level` and `blood_glucose_level` dominate feature importance

### 2. XGBoost with PCA

- Removes duplicates
- Applies label encoding
- Scales features and applies **PCA**
- Trains an **XGBoost** classifier
- Also extends the workflow to hypertension prediction

Reported result:

- Accuracy: **96.6%**
- Precision for diabetic class: **0.93**
- Recall for diabetic class: **0.67**

Main takeaway:

- XGBoost produced fewer false positives but missed more diabetic cases than the Random Forest approach

### 3. Multi-Model Comparison on the Pima Indians Dataset

- Uses a different diabetes dataset, not the main 100k-record dataset
- Compares multiple classifiers side by side
- Applies exploratory analysis and outlier capping
- Shows that model rankings can change depending on the dataset and tuning setup

Main takeaway:

- A systematic multi-model comparison is valuable and should be repeated on the full Kaggle diabetes dataset

## Techniques Incorporated

The reviewed notebooks introduced several methods that shape this project:

- **SMOTE** for minority-class oversampling
- **PCA** for feature decorrelation
- **Leak-proof pipelines** with preprocessing inside cross-validation
- **GridSearchCV** for hyperparameter tuning
- **Model serialization** for saving trained models

## Planned Project Contributions

The mini-project extends earlier work in two main directions.

### 1. Systematic Multi-Model Comparison

The project will compare several classifiers on the full dataset under a unified evaluation pipeline:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine

All models will be evaluated using consistent preprocessing, stratified cross-validation, and the same set of performance metrics.

### 2. Recall-Oriented Threshold Selection

Rather than using the default `0.5` decision threshold only, the project will:

- Plot a precision-recall curve for the best-performing model
- Select a threshold targeting at least **90% recall** for diabetic cases
- Report the precision at that operating point

This makes the evaluation more clinically meaningful by explicitly prioritizing detection of diabetic patients.

## Main Insights So Far

- `HbA1c_level` and `blood_glucose_level` are the strongest predictors of diabetes.
- Class imbalance is a central challenge and must be handled carefully.
- High accuracy alone is not enough for evaluating medical screening models.
- Hyperparameter tuning substantially improves generalization.
- No single algorithm can be assumed to perform best without controlled comparison.

## References

- Mustafa, I. (2023). *Diabetes Prediction Dataset*. Kaggle. https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
- @pannmie. (2023). *Diabetes EDA Random Forest HP*. Kaggle. https://www.kaggle.com/code/tumpanjawat/diabetes-eda-random-forest-hp
- Mubashar, M. D. (2024). *Diabetes | Hypertension Prediction (Acc 97%)*. Kaggle. https://www.kaggle.com/code/muhammaddanishmubashar/diabetes-hypertension-predict-acc-97
- Zabihullah. (2023). *Diabetes Prediction for Pima Women*. Kaggle. https://www.kaggle.com/code/zabihullah18/diabetes-prediction
