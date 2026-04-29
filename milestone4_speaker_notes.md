# Milestone 4 Video Speaker Notes
## Predictive Learning Models for the Diabetes Prediction Dataset

### Slide 1: Title
Hello everyone. We are Wenjie Zhang and Livia Li from DS675.

In this project, we built and compared predictive learning models for diabetes risk classification. Our goal is to support early screening decisions by identifying high-risk patients more effectively.

We will walk through every stage of the supervised learning workflow: data exploration, preprocessing, model training across ten algorithm families, evaluation, threshold optimization, and model explanation.

---

### Slide 2: Problem Motivation
This is a binary classification problem, but in healthcare screening, not all errors are equal.

Every prediction falls into one of four outcomes in the confusion matrix. A true positive means we correctly flagged a diabetic patient. A true negative means we correctly cleared a healthy patient. A false negative means we missed a diabetic patient — we predicted non-diabetic when the patient actually has diabetes. A false positive means we flagged a healthy patient as diabetic.

These four cells matter differently depending on the cost of being wrong. Recall — also written as true positive rate — is defined as the number of correctly identified diabetic patients divided by the total number of actual diabetic patients. In formula terms: Recall = TP divided by (TP + FN). It directly measures how many true diabetic cases we catch.

A false negative here can delay diagnosis, delay care, and lead to serious complications that are very difficult to reverse. A false positive sends a patient for a follow-up confirmatory test — costly, but correctable. So our evaluation framework is built around minimizing false negatives. That means recall is our primary metric, not accuracy.

---

### Slide 3: Dataset Overview
Our raw dataset has 100,000 records and 9 total columns: 8 predictor features and one binary target called `diabetes`.

We found 3,854 duplicate rows and removed them, leaving 96,146 cleaned rows.

After stratified splitting, the class distribution remained imbalanced: about 91.18% non-diabetic and 8.82% diabetic.

This imbalance is exactly why accuracy alone is misleading. Consider a naïve classifier that always predicts "non-diabetic" regardless of input. That model scores roughly 91% accuracy while catching zero diabetic patients. In a screening context, that would be a catastrophic failure. This is the class imbalance pitfall — high accuracy can hide completely broken recall.

The supervised learning workflow we follow is: clean labeled data → preprocessing → train/test split → model training → held-out evaluation. Each stage is designed to ensure our final performance estimate reflects real-world generalization.

---

### Slide 4: Exploratory Data Analysis
In data quality checks, we found zero missing values, which simplified preprocessing considerably.

For predictor strength, the Pearson correlations with the target were:
- blood_glucose_level: 0.419558
- HbA1c_level: 0.400660
- age: 0.258008
- bmi: 0.214357
- hypertension: 0.197823
- heart_disease: 0.171727
- smoking_history: 0.094290
- gender: 0.037411

Blood glucose and HbA1c being strongest is not surprising. Blood glucose is a direct real-time glycemic marker, and HbA1c reflects average glucose over the past two to three months — it is the standard clinical diagnostic criterion for diabetes. So these features contain direct biological signal.

One important note: Pearson correlation only captures linear associations. Non-linear threshold effects — for example, HbA1c above 6.5% being a clinical diabetes cutoff — will be underestimated here. Tree-based models can detect these sharp thresholds naturally, which partly explains why they outperform linear models on this dataset.

---

### Slide 5: Preprocessing
Our preprocessing pipeline had six key steps.

**Step 1 — Remove duplicates** before splitting, so duplicates cannot appear in both train and test.

**Step 2 — Encode categorical variables.** Smoking history and gender were label-encoded.

**Step 3 — Stratified 80/20 split.** Train size was 76,916 and test size was 19,230. Stratification preserves the 91/9 class ratio in both sets, so our evaluation is not distorted by a lucky or unlucky split.

**Step 4 — Feature scaling.** We applied StandardScaler only to models that require it: Logistic Regression, k-NN, MLP, and SVM. These models are sensitive to feature scale because they compute distances or gradient steps across features. A feature like blood glucose (range 0–300) would otherwise dominate a feature like age (range 0–100) simply due to its magnitude. Tree-based models — Decision Tree, Random Forest, Gradient Boosting, XGBoost, AdaBoost, and Naive Bayes — split on feature thresholds and are completely unaffected by scaling. So we apply it selectively, not universally.

**Step 5 — SMOTE for class imbalance.** SMOTE stands for Synthetic Minority Over-sampling Technique. Critically, it does not simply duplicate minority samples. Instead, it interpolates between an existing minority point and one of its nearest minority neighbors, creating a brand new synthetic patient in the feature space. This produces a more diverse augmented minority class.

Before SMOTE: class 0 had 70,130 samples, class 1 had 6,786. After SMOTE: both classes had 70,130 samples.

We applied SMOTE only to training data. This is a data leakage prevention requirement. If SMOTE were applied to the full dataset before splitting, synthetic points could be generated using information from test-set examples as neighbors. That would allow the training set to "see" patterns from the test set before training, inflating performance estimates. By applying SMOTE inside each cross-validation fold, the validation fold is always completely untouched — a core data integrity principle from the course.

**Step 6 — XGBoost class weighting.** For XGBoost specifically, we used `scale_pos_weight=10`, which tells the model to treat each positive example as ten times more important. This is an alternative imbalance-handling strategy built directly into the boosting objective.

---

### Slide 6: Model Selection
We selected ten model families to cover every learning strategy taught in DS675. Let us walk through each one.

**Logistic Regression** uses the sigmoid function to map a linear weighted sum of features into a calibrated probability between 0 and 1. The model is trained by minimizing cross-entropy loss using gradient descent. Because the loss function is strictly convex, there is exactly one global minimum — no risk of getting stuck in local optima. Coefficients are directly interpretable. We chose it as our interpretable linear baseline to anchor comparisons.

**Decision Tree** recursively partitions the feature space by selecting the split that maximizes information gain using the Gini index. This produces axis-parallel decision boundaries, meaning every decision rule takes the form "feature X is greater than or less than some threshold." Decision trees make no distributional assumptions, but they are highly prone to overfitting without depth limits. We chose it as our transparent nonlinear baseline — every decision path can be read as plain boolean logic.

**Random Forest** applies Bagging: it trains many trees, each on a random bootstrap sample and a random subset of features. Individual trees overfit in different, unpredictable directions. When we average their votes, those idiosyncratic errors cancel out and overall variance drops substantially. Random Forest is the canonical variance-reduction ensemble. We chose it for its robustness.

**Gradient Boosting** trains trees sequentially. Each new tree is fit to the residual errors of the combined ensemble so far — mathematically, each tree predicts the negative gradient of the loss function. This is a sequential bias-reduction strategy, which contrasts directly with Random Forest's parallel variance-reduction strategy. Gradient Boosting is consistently strong on structured tabular data.

**XGBoost** builds on Gradient Boosting and adds L1 and L2 regularization on leaf weights, column subsampling per tree, and native `scale_pos_weight` for class imbalance. The regularization prevents individual trees from memorizing noise — the same principle as Lasso and Ridge in logistic regression, just applied inside the boosting objective.

**k-Nearest Neighbors** assigns the majority label among the k nearest training examples, measured by Euclidean distance. This is a memory-based algorithm — there is no learned model, only stored training data. It requires feature scaling because distances between features on different scales are meaningless. We used `algorithm='ball_tree'` which builds a spatial index to reduce query time from O(n) to O(log n) — essential at 100,000 training records. We chose k=11 to smooth decision boundaries and reduce sensitivity to individual noisy points.

**Naive Bayes (BernoulliNB)** is a probabilistic classifier grounded in Bayes' theorem. It estimates the prior probability of each class and the conditional probability of each feature given each class. The "naive" assumption is that features are conditionally independent given the class label — almost never true in practice, but Naive Bayes still performs well because it needs only a good ranking of probabilities, not perfect calibration. Laplace smoothing with alpha=1 prevents zero-probability estimates for feature values unseen in training. It requires no scaling because it works with frequency counts, not distances.

**AdaBoost** trains a sequence of decision stumps — trees with depth one that make a single binary split. After each stump, training examples that were misclassified receive higher weight, forcing the next stump to concentrate on the hardest cases. Each stump's final voting power is proportional to the log of the ratio of its accuracy to its error. AdaBoost reduces both bias and variance, but it is sensitive to mislabeled or noisy examples because those hard cases will receive ever-increasing weight.

**MLP (Multi-Layer Perceptron)** is a feedforward neural network. Our configuration has two hidden layers of 64 and 32 neurons. Each neuron applies a non-linear activation function, which allows the network to learn complex non-linear feature combinations. The network learns via backpropagation, using the chain rule of calculus to propagate error gradients from the output layer back through each hidden layer to update weights. We used the Adam optimizer and enabled early stopping: training halts when the validation loss stops improving, preventing overfitting. This directly connects to the learning curve concept — we stop training at the point where the validation curve flattens.

**SVM with RBF Kernel** finds the maximum-margin hyperplane between classes. A soft-margin formulation introduces slack variables that allow controlled misclassifications, penalized by the regularization parameter C. For non-linear data, the kernel trick computes similarity between points in a high-dimensional transformed space without ever explicitly computing the transformation. The RBF kernel measures similarity as an exponentially decaying function of Euclidean distance. The γ parameter controls how sharply similarity decays: small γ produces smooth, broad boundaries; large γ produces tight, complex boundaries that wrap closely around support vectors. SVM training requires computing an n-by-n kernel matrix, which is O(n²) in memory. At 100,000 training records this is infeasible, so we trained on a 10,000-sample subsample and evaluated on the full test set.

We chose these ten models to systematically compare every learning strategy in the course: linear models, tree-based models, bagging, boosting, probabilistic models, distance-based models, neural networks, and kernel methods.

---

### Slide 7: Cross-Validation Results
We used 5-fold stratified cross-validation on the SMOTE-balanced training data to estimate generalization before touching the test set.

StratifiedKFold preserves the class ratio across all five folds. This matters here because our training data after SMOTE is 50/50 balanced, and we want each fold's validation set to reflect that same distribution consistently.

Key cross-validation results:
- Random Forest ROC-AUC: 0.9971
- Gradient Boosting ROC-AUC: 0.9963
- XGBoost ROC-AUC: 0.9725
- Logistic Regression recall: 0.8851
- Decision Tree recall: 0.9730

Ensemble models lead because they address the core bias-variance tradeoff. Random Forest reduces variance by averaging diverse overfitting trees. Gradient Boosting reduces bias by sequentially correcting residuals. Logistic Regression — a linear model — cannot capture the non-linear threshold effects visible in HbA1c and blood glucose, which explains its lower AUC despite strong recall.

One important caveat: cross-validation here is performed on SMOTE-balanced data, meaning each fold sees a 50/50 class distribution during training. The held-out test set has the natural 91/9 imbalance. This means CV estimates can look slightly more optimistic than final test performance — which is exactly what we will see.

---

### Slide 8: Held-Out Test Results
On the untouched test set at the default threshold of 0.50:

- Gradient Boosting: ROC-AUC 0.9758, precision 0.9077, recall 0.7188
- XGBoost: ROC-AUC 0.9720, precision 0.5717, recall 0.8325
- Logistic Regression: recall 0.8732, precision 0.4231
- Random Forest: recall 0.7488
- Decision Tree: ROC-AUC 0.8585

This is our unbiased generalization estimate. These 19,230 patients were never used during training, preprocessing fit, or SMOTE — the model has truly not seen them.

The gap between CV AUC and test AUC confirms what we expected: the SMOTE-balanced training distribution was slightly more favorable, creating a small optimism gap. The models still generalize well overall.

Gradient Boosting ranks patients best overall by ROC-AUC, meaning it is best at separating diabetics from non-diabetics in probability space. But at the 0.50 default threshold it is conservative — it only predicts diabetic when it is at least 50% confident, leading to high precision but only 71.9% recall. That means roughly 28% of actual diabetic patients were missed.

XGBoost shows more balanced behavior at 0.50 because `scale_pos_weight=10` shifted the model's predicted probabilities toward flagging more positives during training. This acts like a softer threshold shift baked into the training objective.

Logistic Regression catches many diabetics but with many false positives — consistent with a linear model that cannot form tight nonlinear boundaries around the diabetic subpopulation.

The key finding: no model reaches 90% recall at the default threshold. That is the problem threshold optimization solves.

---

### Slide 9: Threshold Optimization
Because Gradient Boosting had the best held-out ROC-AUC, we selected it for threshold optimization.

Here is how threshold selection works. The classifier outputs a probability P(diabetes|x) for every patient. The decision rule is: if P is greater than or equal to the threshold t, predict diabetic; otherwise predict non-diabetic. The default t is 0.50. By lowering t, we make the model more liberal — it flags a patient as diabetic even when confidence is lower.

We used the Precision-Recall curve to scan all possible threshold values and selected the highest threshold that still delivers at least 90% diabetic recall. The optimal threshold was 0.21. That means the model flags a patient as diabetic if it is at least 21% confident — a deliberate shift toward sensitivity over specificity.

Results at threshold 0.21:
- Recall: 0.7188 → 0.9057
- Precision: 0.9077 → 0.4829
- Accuracy: 0.9687 → 0.9061
- F1: 0.8022 → 0.6299

False positives increase. At 0.21, roughly one in two flagged patients is actually non-diabetic. In clinical terms, those patients receive a confirmatory blood test and are cleared. That is an acceptable cost compared to missing 28% of true diabetic cases at the default threshold.

This connects directly to cost-sensitive learning from the course: by adjusting the threshold we are implicitly assigning a higher misclassification cost to false negatives. The key advantage is that this happens entirely post-training. We did not retrain the model, we did not change any weights — we only changed the decision rule applied to its probability outputs.

---

### Slide 10: Precision-Recall Tradeoff
Let us be precise about the definitions.

Precision is the fraction of patients we flagged as diabetic who actually are diabetic: TP divided by (TP + FP). It answers "when we raise an alarm, how often are we right?"

Recall is the fraction of actual diabetic patients we caught: TP divided by (TP + FN). It answers "of all the true cases, how many did we find?"

As threshold decreases, we flag more patients positive. We catch more true diabetics — recall rises. But we also flag more healthy patients incorrectly — precision falls. This inverse relationship is the precision-recall tradeoff.

ROC-AUC measures something different: the probability that a randomly chosen diabetic patient receives a higher predicted probability than a randomly chosen non-diabetic. An AUC of 1.0 means perfect ranking; 0.5 means random. ROC-AUC is threshold-independent — it evaluates the model's ability to discriminate across all possible thresholds simultaneously. A high AUC means the model has strong underlying discriminative ability even if its default threshold produces imbalanced recall.

This is why we use both: ROC-AUC to select the best model, and threshold optimization to configure it for clinical use.

---

### Slide 11: Hyperparameter Tuning
We applied RandomizedSearchCV to tune XGBoost.

The reason we chose XGBoost for tuning is its richer hyperparameter space. XGBoost exposes parameters directly controlling overfitting: `max_depth` limits individual tree complexity, `learning_rate` controls how aggressively each new tree corrects the ensemble, `subsample` and `colsample_bytree` introduce stochasticity that prevents overfitting to specific rows or columns, and `gamma` requires a minimum gain before making any split. These are all regularization levers in the sense taught in the course: they penalize model complexity to improve generalization.

Additionally, XGBoost includes built-in L2 regularization on leaf weights via the `lambda` parameter and L1 regularization via `alpha`. These directly parallel Ridge and Lasso regularization in logistic regression: L2 shrinks all weights globally while keeping them nonzero, L1 can push unimportant leaf contributions to exactly zero.

We used RandomizedSearchCV with n_iter=20 and cv=3, giving 60 model fits in total. This is far more efficient than exhaustive GridSearchCV, which would require fitting every combination — potentially thousands of fits. Random sampling across the hyperparameter space covers a wide range of configurations while remaining computationally tractable on a 100,000-record dataset.

Results: best CV ROC-AUC of 0.9795, and a tuned test ROC-AUC of 0.9781 with high recall under threshold optimization. Tuning confirmed that further regularization could maintain strong discrimination while reducing variance.

Important clarification: Gradient Boosting remained the best held-out ROC-AUC model overall. We tuned XGBoost to explore the regularized boosting space, not because it was the top performer.

---

### Slide 12: Key Contributions
Our key contributions are four points.

1. **Systematic ten-model comparison** covering every learning strategy in DS675 — linear, tree, bagging, boosting, probabilistic, distance-based, neural, and kernel methods. This is the full supervised learning algorithm survey applied to one real-world problem.

2. **Recall-focused threshold optimization** for screening goals. We demonstrated that post-training threshold adjustment can close the gap between default model behavior and clinical requirements, without any retraining.

3. **Bias-variance-aware evaluation** using both stratified cross-validation and a fully held-out test set. CV estimates expected performance across training configurations; the test set provides the true variance-corrected generalization estimate.

4. **Explicit medical interpretation** of the precision-recall tradeoff, connecting model outputs to clinical consequences — false negatives delay care, false positives add confirmatory tests — so the threshold choice is grounded in real cost reasoning.

---

### Slide 13: Limitations
This work has several honest limitations.

We did not perform clinical validation. Our dataset labels come from a Kaggle source with uncertain provenance — we do not know the original diagnostic protocol, population demographics, or whether labels are consistent.

Dataset representativeness is uncertain. The 91/9 class ratio and feature distributions may not generalize to specific populations where diabetes prevalence or risk factors differ.

SMOTE generates synthetic interpolations in feature space. These synthetic patients exist mathematically but may not correspond to realistic physiology, particularly at the boundaries of the feature space. This can introduce variance in regions where we lack real minority-class support.

The chosen threshold of 0.21 was derived from the test set. In a rigorous deployment pipeline, threshold selection should use a separate validation set, with the test set reserved strictly for final reporting. Using the test set for threshold selection introduces a small degree of optimism into our recall estimate.

The chosen threshold should always be finalized in consultation with clinical stakeholders who understand care capacity, follow-up resource availability, and patient risk tolerance.

---

### Slide 14: Future Work
Next steps to strengthen this work include:

- **Probability calibration** — our models output probabilities optimized for ranking, not necessarily accurate real-world probability estimates. Calibration (via Platt scaling or isotonic regression) would make the 0.21 threshold more interpretable.
- **Cost-sensitive learning** — instead of post-hoc threshold adjustment, integrate asymmetric misclassification costs directly into the training loss.
- **SHAP-based explanations** — SHAP values provide individual-level feature attributions, showing why the model predicted high risk for a specific patient. This supports clinician trust.
- **External validation** on independent cohorts from different institutions and populations.
- **Deployment** as an API or decision-support prototype, where the full preprocessing pipeline, model, and threshold are packaged together.

---

### Slide 15: Conclusion
To conclude, Gradient Boosting achieved the best held-out ROC-AUC of 0.9758, reflecting strong underlying discriminative ability.

But ROC-AUC alone does not make a model clinically useful. At the default threshold of 0.50, even the best model missed 28% of true diabetic patients. After threshold tuning to 0.21, recall exceeded 90% — making the system suitable for a screening context.

This project illustrates the full arc of supervised learning: understanding the problem cost structure, selecting models that address the bias-variance tradeoff from different angles, evaluating on truly held-out data, and configuring the decision rule for the real-world objective.

Our final takeaway: in imbalanced medical classification, model selection and threshold selection are equally important decisions. High AUC tells you the model can distinguish — the threshold tells you how it acts on that distinction.
