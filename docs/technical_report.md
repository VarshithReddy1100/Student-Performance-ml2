# Technical Report: Student Academic Performance Prediction

## Abstract

This project implements reproducible regression and classification experiments using UCI's Student Performance Mathematics data. It compares linear and polynomial regression, L1/L2 regularized regression, support-vector machines with three kernels, decision trees, and random forests. A Streamlit interface exposes the dataset, cross-validation, learning curves, and predictions. The input design excludes prior grades G1 and G2 to avoid leakage.

## Objectives

1. Estimate final mathematics grade G3 from demographic, family, school, and lifestyle variables.
2. Classify G3 into Low, Medium, and High bands.
3. Compare model families using suitable metrics and discuss bias-variance behavior.
4. Provide an understandable interactive application and reproducible local workflow.

## Dataset and preparation

The UCI Student Performance dataset includes Portuguese and Mathematics CSVs. This project uses only `student-mat.csv` (395 student records). `G3` is numeric 0–20. Classes are Low: 0–9, Medium: 10–14, High: 15–20. G1/G2/G3 are explicitly removed from features. Numeric and categorical preprocessing occurs inside each estimator pipeline. Numeric imputation uses medians; categorical imputation uses the most frequent value; one-hot encoding safely ignores unseen categories. Standardization is applied to linear and SVM pipelines.

## Algorithms

- **Linear regression:** estimates an additive linear relationship; the same all-feature design is multiple regression. Restricting the design to one input would give simple linear regression.
- **Polynomial regression:** degree-two feature expansion followed by linear regression models selected nonlinear interactions.
- **Ridge / Lasso:** regularized linear models reduce coefficient variance. Lasso can perform embedded feature selection.
- **SVM:** linear, polynomial, and RBF kernels expose kernel choice, C and gamma concepts.
- **Decision tree:** interpretable nonlinear split model with bounded depth.
- **Random forest:** bootstrap-based ensemble averaging many trees; often reduces variance relative to a single tree.

## Experimental approach

The training command creates a seeded 80/20 holdout, fits every candidate and stores the holdout comparison plus the selected model artifacts. The dashboard separately supports shuffled five-fold cross-validation (StratifiedKFold for classification). Its learning curves use training and validation scores over increasing sample sizes. Optional GridSearchCV searches Random Forest hyperparameters. Because the available dataset is small, CV scores and holdout scores can vary; preserve a truly untouched test set for a final academic evaluation after selecting candidates by CV.

## Evaluation and interpretation

Regression metrics are MAE, RMSE, and R². Classification metrics are accuracy, macro-precision, macro-recall, macro-F1, and a confusion matrix with fixed class ordering. Residual = actual minus predicted grade; inspect residuals against fitted values for non-random structure, bias, or heteroscedasticity. Learning curves help diagnose underfitting (both scores weak) versus overfitting (large train-validation gap). Compare macro-F1 rather than accuracy alone if class frequencies are uneven.

## Software architecture

`data.py` owns download and target preparation; `models.py` defines fitted-in-place Scikit-learn pipelines; `evaluation.py` contains metrics, CV, learning curves and tuning; `training.py` trains, saves, loads, and invokes artifacts. `app.py` consumes these modules rather than duplicating model logic. Unit tests cover leakage prevention, grade-band mapping, pipeline fit/predict, and metric outputs.

## Conclusion and limitations

This is a coursework demonstration rather than a validated educational assessment. Sample count, dated context, self-reported variables, class imbalance and subjective thresholds limit generalization. Feature importance is model-dependent and non-causal. Predictions must not be used for high-impact decisions about students. Add external validation and fairness analysis before any real-world research use.

## Result reporting template

After running the experiments, report actual observed values here rather than assuming expected results:

| Task | Selected model | Metrics | Notes |
|---|---|---|---|
| Regression | Fill from generated training summary | MAE, RMSE, R² | Include holdout/CV protocol |
| Classification | Fill from generated training summary | Accuracy, macro-F1 | Include confusion matrix and class balance |

No performance values are pre-filled because they must be measured in the user's local environment.
