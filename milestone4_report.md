# Milestone 4 Report
## Predictive Learning Models for the Diabetes Prediction Dataset

**Authors:**
- Wenjie Zhang
- Livia Li

**Course:**
- DS675, Spring 2026

**Dataset:**
- Diabetes Prediction Dataset, Kaggle

## Author Contributions

| Team Member | Contributions |
|---|---|
| Wenjie Zhang | Designed and implemented the preprocessing pipeline, trained and evaluated all baseline models, performed recall-focused threshold tuning, ran XGBoost hyperparameter tuning, and drafted core technical sections of the report. |
| Livia Li | Developed the introduction and literature context, reviewed Kaggle baseline approaches and identified gaps, wrote discussion and conclusion sections, and prepared the video presentation narrative. |

Both team members jointly reviewed the final analysis for consistency with project objectives and grading rubric.

## 1. Introduction
Diabetes prediction in this project is framed as a **supervised binary classification** task, where the target is whether a patient is diabetic (`1`) or non-diabetic (`0`). Early risk detection matters because delayed intervention can increase complications and treatment burden.

In a screening context, **false negatives are typically more costly than false positives**. Missing a high-risk patient may delay follow-up testing and treatment. A false positive usually triggers additional confirmatory testing, which is less harmful than missing a true diabetic case. For this reason, recall for the diabetic class is prioritized in model selection and threshold design.

Because the dataset is imbalanced, overall accuracy can be misleading. A model can score high accuracy by predicting mostly non-diabetic cases while still missing many diabetic patients. This motivates our use of ROC-AUC, recall, precision, and F1 rather than relying on accuracy alone.

## 2. Dataset Description and Learning Task
The source is the **Diabetes Prediction Dataset** from Kaggle. The raw dataset contains **100,000 rows and 9 columns**, with **8 predictors** and one binary target `diabetes`.

Predictor features include demographic, medical-history, and clinical measurement variables such as gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, and blood glucose level. The target is whether the patient is diabetic.

The cleaned dataset remains imbalanced, with approximately **91.18% non-diabetic** and **8.82% diabetic** records after splitting. The original imbalance ratio is about **10.8:1**.

Formally, the learning task is to estimate the conditional probability of diabetes and map that probability to a class label using a decision threshold. The intended use is **screening support**, not clinical diagnosis replacement.

## 3. Exploratory Data Analysis
### 3.1 Data quality summary
- Missing values: **0** across all variables.
- Duplicate rows: **3,854**.
- Rows after deduplication: **96,146**.

These checks showed that the dataset has complete fields but contains substantial exact duplicates that must be removed before splitting.

### 3.2 Class balance
`[Figure: Class distribution chart]`

- **What it shows:** The diabetic class is much smaller than the non-diabetic class, with an approximate 10.8:1 majority-to-minority ratio.
- **Why it matters:** Imbalance can bias learning toward majority predictions and inflate accuracy.
- **What conclusion it supports:** Imbalance-aware training and recall-focused evaluation are necessary.

### 3.3 Feature distributions by target
`[Figure: Feature distributions by diabetes status]`

- **What it shows:** Distribution shifts between diabetic and non-diabetic groups, especially on glucose-related measures.
- **Why it matters:** Separation patterns indicate which variables likely contribute most to predictive discrimination.
- **What conclusion it supports:** Non-linear and ensemble models are justified for capturing heterogeneous patterns.

### 3.4 Correlation with target
`[Figure: Correlation heatmap]`

Target correlations:
- `blood_glucose_level`: **0.419558**
- `HbA1c_level`: **0.400660**
- `age`: **0.258008**
- `bmi`: **0.214357**
- `hypertension`: **0.197823**
- `heart_disease`: **0.171727**
- `smoking_history`: **0.094290**
- `gender`: **0.037411**

- **What it shows:** Blood glucose and HbA1c are the strongest positively correlated predictors in this dataset.
- **Why it matters:** These features are clinically aligned with glycemic status and long-term glucose control.
- **What conclusion it supports:** The learned patterns are medically plausible and not arbitrary artifacts.

## 4. Data Preprocessing
Preprocessing was designed to support fair comparison and prevent leakage.

| Step | Implementation | Rationale |
|---|---|---|
| Duplicate handling | Removed 3,854 exact duplicate records before split. | Prevents duplicate leakage across train and test partitions. |
| Categorical encoding | Encoded `gender` and mapped ordered `smoking_history` categories to integers. | Ensures compatibility with scikit-learn and XGBoost estimators. |
| Train/test split | Stratified 80/20 split. Train: **76,916**. Test: **19,230**. | Preserves class ratio in both partitions. |
| Scaling | Applied StandardScaler for Logistic Regression only, fitted on training data only. | LR is scale-sensitive; tree models are not. |
| Class imbalance handling (main pipeline) | Applied SMOTE on training set only: before SMOTE `{0: 70,130, 1: 6,786}`, after SMOTE `{0: 70,130, 1: 70,130}`. | Improves minority signal without contaminating test data. |
| XGBoost imbalance handling | Used `scale_pos_weight=10` in XGBoost pipeline. | Native imbalance weighting for boosted trees. |

**Leakage prevention:** All learned transforms were fit on training data only. SMOTE was never applied to test data.

## 5. Model Selection Rationale
We selected models to compare different learning families rather than increasing model count.

- **Logistic Regression:** Linear and interpretable baseline for probability estimation.
- **Decision Tree:** Transparent non-linear baseline that captures interactions.
- **Random Forest:** Bagging ensemble to reduce variance and improve stability.
- **Gradient Boosting:** Sequential boosting that often performs strongly on tabular data.
- **XGBoost:** Regularized boosting with flexible hyperparameters and native class-imbalance handling.

This design enables family-level comparison of linear, single-tree, bagging, and boosting strategies under a shared evaluation framework.

## 6. Model Evaluation Results
### 6.1 Cross-validation results (5-fold stratified)

| Model | ROC-AUC | Recall | F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.9628 ± 0.0009 | 0.8851 ± 0.0026 | 0.8859 ± 0.0024 |
| Decision Tree | 0.9705 ± 0.0006 | 0.9730 ± 0.0010 | 0.9703 ± 0.0007 |
| Random Forest | 0.9971 ± 0.0001 | 0.9725 ± 0.0006 | 0.9722 ± 0.0005 |
| Gradient Boosting | 0.9963 ± 0.0002 | 0.9473 ± 0.0006 | 0.9692 ± 0.0007 |
| XGBoost | 0.9725 ± 0.0014 | 0.8229 ± 0.0062 | 0.6927 ± 0.0075 |

**Interpretation:**
- **What this shows:** Ensemble methods, especially Random Forest and Gradient Boosting, achieved very strong CV ROC-AUC.
- **Why it matters:** CV provides a stability-aware estimate of performance before final hold-out testing.
- **Conclusion supported:** Ensemble families are promising, but final model choice must still be validated on untouched test data.

### 6.2 Held-out test results (default threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8838 | 0.4231 | 0.8732 | 0.5701 | 0.9597 |
| Decision Tree | 0.9474 | 0.6844 | 0.7494 | 0.7155 | 0.8585 |
| Random Forest | 0.9555 | 0.7475 | 0.7488 | 0.7482 | 0.9653 |
| Gradient Boosting | 0.9687 | 0.9077 | 0.7188 | 0.8022 | 0.9758 |
| XGBoost | 0.9302 | 0.5717 | 0.8325 | 0.6779 | 0.9720 |

**Interpretation:**
- **What this shows:** Gradient Boosting achieved the highest held-out ROC-AUC (**0.9758**) and very high precision (**0.9077**), but recall at 0.50 threshold was only **0.7188**.
- **Why it matters:** The project objective is screening-sensitive detection, so recall performance is critical.
- **Conclusion supported:** At default thresholds, no model reached the 90% recall screening target. Logistic Regression had the highest recall (**0.8732**) but low precision (**0.4231**). XGBoost provided a useful middle ground (recall **0.8325**, ROC-AUC **0.9720**). Accuracy is secondary in this imbalanced setting.

## 7. Recall-Optimized Threshold Selection
Default binary classification uses threshold **0.50**. We tuned the decision threshold for the best discrimination model, Gradient Boosting, because it had the highest test ROC-AUC.

Selection rule: choose the **highest threshold** that still achieves at least **90% recall** on diabetic cases.

The selected threshold was **0.21**.

| Metric | Default Threshold 0.50 | Optimized Threshold 0.21 |
|---|---:|---:|
| Recall | 0.7188 | 0.9057 |
| Precision | 0.9077 | 0.4829 |
| F1 | 0.8022 | 0.6299 |
| Accuracy | 0.9687 | 0.9061 |

**Interpretation:**
- **What this shows:** Lowering the threshold increased recall by about **18.7 percentage points**.
- **Why it matters:** More diabetic patients are flagged for follow-up, reducing missed high-risk cases.
- **Conclusion supported:** For screening, the recall gain is clinically meaningful and justifies lower precision and accuracy. Threshold tuning met the main objective without retraining the model.

## 8. Hyperparameter Tuning
XGBoost was chosen for hyperparameter tuning using **RandomizedSearchCV**, not because it had the highest held-out ROC-AUC, but because it is a strong regularized boosting method with many impactful hyperparameters and native imbalance controls.

Randomized search was preferred over exhaustive grid search for computational efficiency while still exploring a broad parameter space.

Observed notebook outputs:
- Best CV ROC-AUC from randomized search: **0.9795**.
- Best parameters: `subsample=0.8`, `reg_lambda=1.0`, `reg_alpha=0.1`, `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `colsample_bytree=0.7`.
- Reported comparison output: Default XGBoost ROC-AUC **0.9720** vs Tuned XGBoost ROC-AUC **0.9781**, with tuned recall **0.9835** and precision **0.3234** under an optimal-threshold-applied setup.

### Fair-comparison note
To separate tuning effects from threshold effects, the following 2x2 comparison should be reported before final submission:

| Configuration | ROC-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| Default XGBoost @ 0.50 | 0.9720 | 0.8325 | 0.5717 | 0.6779 |
| Default XGBoost @ 0.21 | TODO | TODO | TODO | TODO |
| Tuned XGBoost @ 0.50 | TODO | TODO | TODO | TODO |
| Tuned XGBoost @ 0.21 | 0.9781* | 0.9835* | 0.3234* | 0.4867* |

\*Values shown are from notebook output where optimal threshold was applied; confirm exact threshold and matched baseline rows for strictly fair interpretation.

**TODO:** Tuned XGBoost results should be inserted from the executed notebook output before final submission if any of the missing entries remain unavailable.

## 9. Comparison with Prior Kaggle Work
Based on reviewed Kaggle notebooks, prior approaches often report strong accuracy but provide limited screening-oriented threshold analysis.

| Prior notebook pattern (based on reviewed notebooks) | Reported metric emphasis | Common limitation | Improvement in this project |
|---|---|---|---|
| Single-model baseline with default threshold | Often accuracy-focused | Limited recall-first framing for screening | Multi-family comparison with explicit recall priority |
| Boosting model benchmark | ROC-AUC or accuracy | Threshold fixed at 0.50 | Threshold optimization to satisfy ≥90% recall target |
| Basic train/test evaluation | One-shot test metrics | Minimal CV stability analysis | Stratified 5-fold CV plus held-out evaluation |
| Generic model report | Limited clinical interpretation | Weak discussion of false-negative cost | Explicit screening interpretation of precision-recall tradeoff |

## 10. Limitations
- The dataset source and collection process may limit representativeness across populations.
- Class imbalance remains a challenge even with resampling and weighting.
- SMOTE introduces synthetic minority samples that may not capture full clinical diversity.
- The project has no external clinical validation cohort.
- Threshold selection depends on operational cost tolerance and care pathway design.
- The model should support screening triage, not replace clinical diagnosis.

## 11. Future Work
- Probability calibration for better risk communication.
- Cost-sensitive learning aligned with clinical utility.
- Additional feature engineering and interaction modeling.
- SHAP or other explainability methods for patient-level interpretation.
- External validation on independent cohorts.
- Deployment as an API or decision-support prototype with monitoring.

## 12. Conclusion
This milestone compared major model families for imbalanced diabetes prediction and evaluated both ranking quality and classification behavior. Ensemble methods performed strongly, and **Gradient Boosting achieved the best held-out ROC-AUC (0.9758)**. However, default-threshold recall was below screening needs.

By tuning the decision threshold from **0.50 to 0.21**, recall increased above **90%** on diabetic cases, with expected reductions in precision and accuracy. The key takeaway is that in imbalanced medical prediction, effective design requires both model selection and threshold selection.

## References
1. Kaggle. *Diabetes Prediction Dataset*. https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
2. Pedregosa et al. *Scikit-learn: Machine Learning in Python*. JMLR, 2011. https://scikit-learn.org/
3. Chen, T., Guestrin, C. *XGBoost: A Scalable Tree Boosting System*. KDD, 2016.
4. Chawla et al. *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 2002.
5. Bergstra, J., Bengio, Y. *Random Search for Hyper-Parameter Optimization*. JMLR, 2012.
6. Reviewed Kaggle notebooks on diabetes prediction (links to be inserted).
7. DS675 Milestone 2 proposal (link placeholder to be inserted if unavailable).
