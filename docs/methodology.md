# Methodology

## 1. Problem definition

This project studies two supervised-learning tasks using UCI Student Performance (Mathematics): predict the numeric final grade `G3` (regression), and classify it into Low (0–9), Medium (10–14), or High (15–20) (classification).

## 2. Data and leakage controls

The UCI CSV is semicolon-delimited and contains 395 rows. The prediction inputs omit `G1`, `G2`, and `G3`. This is a deliberate, conservative leakage policy: earlier-period grades are highly predictive of `G3`, but including them would make the demo easier while limiting its use to cases where those grades are already known. `G3` remains the target; the class label is deterministically derived from `G3`. The derived class is never included in X.

Categorical features are most-frequent imputed and one-hot encoded. Numeric features are median imputed; scale-sensitive models additionally standardize them. All transformations are inside Scikit-learn pipelines, so each CV fold learns preprocessing from that fold's training partition only.

## 3. Experimental design

A deterministic 80/20 split (seed 42) is used in the training script for final holdout comparisons and saved-model selection. Model selection is also supported through shuffled 5-fold CV (stratified for classification), which uses the complete dataset for the CV experiment. Do not select models based on the test set if reporting an unbiased final score; select using CV then evaluate once on held-out test data in a formal report.

Regression candidates: linear regression, degree-2 polynomial regression, Ridge, Lasso, decision tree, and random forest. Classification candidates: logistic regression, SVM linear/polynomial/RBF, decision tree, and random forest. Optional grid search tunes Random Forest estimator count, maximum depth and minimum leaf size.

## 4. Metrics

- Regression: MAE is average absolute grade error; RMSE penalizes larger errors more; R² compares against a constant-mean baseline (can be negative).
- Classification: accuracy, macro precision, macro recall, macro F1, and confusion matrix. Macro averages give equal importance to each class and help surface class imbalance hidden by accuracy.
- Residual analysis: residual = actual G3 − predicted G3. Plot residuals against predicted grades; random scatter near zero is desirable. Curvature suggests misspecification; changing spread suggests non-constant variance; extreme residuals merit data/measurement review.

## 5. Concepts demonstrated

- **Underfitting / high bias:** model is too constrained; training and validation scores are both poor.
- **Overfitting / high variance:** training performance is much better than validation performance.
- **Learning curves:** compare train and validation performance as training size increases. A persistent gap suggests variance; both curves plateauing poorly suggests bias or limited signal.
- **Regularization:** Ridge penalizes squared coefficient magnitude (L2); Lasso penalizes absolute magnitude (L1) and can shrink coefficients to zero. Larger alpha increases regularization.
- **SVM:** a separating hyperplane maximizes margin; support vectors define the boundary. C controls penalty for margin violations. Polynomial and RBF kernels allow nonlinear boundaries; gamma controls the locality of RBF influence.
- **Trees / forests:** tree splitting chooses partitions to reduce impurity (Gini/entropy for classification; squared error for regression). Random forests bootstrap samples and aggregate many trees; feature importance can describe model reliance but is not causal evidence.

## 6. Limitations and responsible use

The dataset is small and dated, contains self-reported variables, and does not represent every school or population. Grade thresholds are project-defined. Estimates have sampling uncertainty. Do not use predictions to label, rank, admit, discipline, or otherwise make high-impact decisions about real students. Explain uncertainty and data limitations in any presentation.
