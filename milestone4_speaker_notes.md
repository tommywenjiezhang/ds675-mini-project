# Milestone 4 Video Speaker Notes
## Predictive Learning Models for the Diabetes Prediction Dataset

### Slide 1: Title
Hello everyone. We are Wenjie Zhang and Livia Li from DS675.

In this project, we built and compared predictive learning models for diabetes risk classification. Our goal is to support early screening decisions by identifying high-risk patients more effectively.

### Slide 2: Problem Motivation
This is a binary classification problem, but in healthcare screening, not all errors are equal.

If we predict non-diabetic for someone who is actually diabetic, that is a false negative. That can delay care and follow-up testing.

If we predict diabetic for someone who is not, that is a false positive. That still has a cost, but usually the patient can be cleared with confirmatory tests.

So our evaluation prioritizes diabetic recall, because missing true cases is the bigger risk in a screening scenario.

### Slide 3: Dataset Overview
Our raw dataset has 100,000 records and 9 total columns.

That includes 8 predictor features and one binary target called `diabetes`.

We found 3,854 duplicate rows and removed them, leaving 96,146 cleaned rows.

After stratified splitting, the class distribution is still imbalanced, about 91.18% non-diabetic and 8.82% diabetic.

That imbalance is exactly why accuracy alone can be misleading. A model can look accurate but still miss many diabetic patients.

### Slide 4: Exploratory Data Analysis
In data quality checks, we found zero missing values.

For predictor strength, the strongest target correlations were:
- blood_glucose_level: 0.419558
- HbA1c_level: 0.400660
- age: 0.258008
- bmi: 0.214357
- hypertension: 0.197823
- heart_disease: 0.171727
- smoking_history: 0.094290
- gender: 0.037411

Blood glucose and HbA1c being strongest makes clinical sense. They are direct indicators of glycemic status and longer-term glucose regulation.

So the feature relationships look plausible and useful for prediction.

### Slide 5: Preprocessing
Our preprocessing pipeline had six key steps.

First, we removed duplicates before splitting.

Second, we encoded categorical variables.

Third, we used a stratified 80/20 split. Train size was 76,916 and test size was 19,230.

Fourth, we scaled features only for Logistic Regression.

Fifth, we handled class imbalance with SMOTE on training data only.
- Before SMOTE: class 0 = 70,130, class 1 = 6,786.
- After SMOTE: class 0 = 70,130, class 1 = 70,130.

Sixth, for XGBoost we used native class weighting with `scale_pos_weight=10`.

Most importantly, we prevented data leakage by fitting transforms on training data only and keeping the test set untouched.

### Slide 6: Model Selection
We selected five model families on purpose.

- Logistic Regression: simple interpretable linear baseline.
- Decision Tree: transparent non-linear baseline.
- Random Forest: stable bagging ensemble that reduces variance.
- Gradient Boosting: sequential boosting that is strong for tabular data.
- XGBoost: regularized boosting with strong hyperparameter control and class imbalance handling.

We selected models to compare learning strategies, not just to increase the model count.

### Slide 7: Cross-Validation Results
In 5-fold stratified cross-validation, ensemble models were strongest.

Key values:
- Random Forest ROC-AUC: 0.9971.
- Gradient Boosting ROC-AUC: 0.9963.
- XGBoost ROC-AUC: 0.9725.
- Logistic Regression recall: 0.8851.
- Decision Tree recall: 0.9730.

These results estimate generalization before final test evaluation.

We also note that CV on SMOTE-balanced training data can look stronger than final held-out test performance.

### Slide 8: Held-Out Test Results
On the untouched test set at threshold 0.50:

- Gradient Boosting: ROC-AUC 0.9758, precision 0.9077, recall 0.7188.
- XGBoost: ROC-AUC 0.9720, precision 0.5717, recall 0.8325.
- Logistic Regression: recall 0.8732, precision 0.4231.
- Random Forest: recall 0.7488.
- Decision Tree: ROC-AUC 0.8585.

Interpretation:
Gradient Boosting ranked patients best overall by ROC-AUC.

Logistic Regression caught more diabetics than most models, but with many false positives.

Gradient Boosting was stricter and very precise, but it missed too many diabetic cases at the default threshold.

XGBoost gave a balanced default tradeoff.

Most importantly, no model reached 90% recall at threshold 0.50.

### Slide 9: Threshold Optimization
Because Gradient Boosting had the best held-out ROC-AUC, we selected it for threshold optimization.

We lowered threshold from 0.50 to 0.21 using this rule: choose the highest threshold that still gives at least 90% diabetic recall.

Results:
- Recall: 0.7188 to 0.9057.
- Precision: 0.9077 to 0.4829.
- Accuracy: 0.9687 to 0.9061.
- F1: 0.8022 to 0.6299.

So yes, false positives increase. But for screening, this is acceptable because catching more true diabetic cases is the priority.

This is one of our strongest contributions because recall improved without retraining the model.

### Slide 10: Precision-Recall Tradeoff
In simple terms, precision and recall often move in opposite directions.

If we lower threshold, we flag more patients as positive. Recall usually rises, precision usually falls.

In medical screening, recall usually has higher priority because missed cases can be more harmful.

The exact operating threshold should be chosen with clinical stakeholders based on cost and care capacity.

### Slide 11: Hyperparameter Tuning
We tuned XGBoost with RandomizedSearchCV.

We selected XGBoost for tuning because it has regularization, strong tabular performance, and many meaningful parameters.

Important clarification: XGBoost was not the best held-out ROC-AUC model. Gradient Boosting was best on the held-out test set.

From the notebook output, randomized search reported best CV ROC-AUC 0.9795, and tuned XGBoost output reported ROC-AUC 0.9781 with very high recall and lower precision under an optimal-threshold-applied setup.

Tuning was used to explore whether a regularized boosting model could improve discrimination further.

### Slide 12: Key Contributions
Our key contributions are four points.

1. A systematic comparison across model families.
2. Recall-focused threshold optimization for screening goals.
3. Fairer evaluation using both stratified CV and a held-out test set.
4. Explicit medical interpretation of precision versus recall tradeoffs.

### Slide 13: Limitations
This work has several limitations.

We did not perform clinical validation.

Dataset representativeness across populations is uncertain.

SMOTE creates synthetic minority examples that may not reflect all real-world variability.

The chosen threshold should be refined with domain experts and real deployment constraints.

### Slide 14: Future Work
Next steps include:
- probability calibration,
- cost-sensitive learning,
- SHAP-based explanations,
- external validation on independent cohorts,
- and deployment as an API or decision-support prototype.

### Slide 15: Conclusion
To conclude, Gradient Boosting had the best held-out test ROC-AUC.

After threshold tuning, recall increased above 90%, which better matches a screening objective.

Our final takeaway is this: in imbalanced medical prediction, model selection matters, and threshold selection is equally important.
