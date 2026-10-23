# Student Academic Performance Prediction and Analysis System Using Machine Learning

An educational, modular machine-learning project for predicting the final mathematics grade (`G3`) and classifying performance using the UCI Student Performance dataset. It demonstrates regression, classification, cross-validation, learning curves, model comparison and basic tuning.

## Important data design

- Uses UCI's `student-mat.csv` (mathematics subject, 395 students).
- Regression target: `G3` (0–20).
- Classification target: `Low` (0–9), `Medium` (10–14), `High` (15–20).
- `G1`, `G2`, and `G3` are excluded from predictor columns to prevent grade leakage. This means the model predicts from demographic, family, school, and lifestyle inputs rather than prior grades.
- Dataset licensing/usage details are published by UCI. Keep attribution if redistributing the dataset.

## Windows 11 / PowerShell setup

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_data.py
python -m pytest -q
python scripts/train_models.py
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once in a normal user PowerShell window, or invoke `.\.venv\Scripts\python.exe` directly for each command. Python packages are not installed automatically by this project setup.

## Project structure

- `src/student_performance/data.py`: download, load, validate, and prepare leakage-safe targets.
- `src/student_performance/models.py`: reusable regression/classification preprocessing pipelines and model registries.
- `src/student_performance/evaluation.py`: metrics, shuffled K-fold comparison, learning curves, and Random Forest grid search.
- `src/student_performance/training.py`: reproducible holdout training, artifact persistence, and prediction interface.
- `scripts/download_data.py`: dataset download/extraction command.
- `scripts/train_models.py`: trains all model families and saves the selected models and summary to `artifacts/`.
- `app.py`: Streamlit dashboard for exploration, model evaluation, and interactive prediction.
- `tests/`: unit and smoke tests.
- `docs/`: methodology, technical report, and viva preparation.

## Models and syllabus mapping

**Regression:** simple/multiple linear regression are both represented by the linear pipeline (one predictor vs all predictors); degree-2 polynomial regression; Ridge and Lasso regularization; decision tree and random forest regressors. Metrics include MAE, RMSE and R². The evaluation module also exposes residual analysis through the residuals `y_true - y_pred`; inspect predictions against actual values to identify patterns and changing error spread.

**Classification:** logistic baseline, SVM with linear, polynomial, and RBF kernels, decision tree and random forest. The SVM pipeline scales numeric inputs. Metrics include accuracy, macro precision, macro recall, macro F1 and confusion matrix.

**Model selection:** deterministic 80/20 train/test split for final holdout comparison; shuffled 5-fold CV for model comparison; learning curves to diagnose bias/variance; `tune_random_forest()` conducts a compact `GridSearchCV`.

## Running evaluations

The dashboard's **Model evaluation** page runs five-fold CV. The reusable API is:

```python
from student_performance.data import load_data, prepare_targets
from student_performance.models import regression_models
from student_performance.evaluation import compare_models

frame = load_data()
X, y, _ = prepare_targets(frame)
results = compare_models(regression_models(X), X, y, task="regression")
print(results)
```

To tune: import `tune_random_forest` from `student_performance.evaluation` and call it with `X`, the target, and `task="regression"` or `task="classification"`. Grid search can take a few minutes.

## Reproducibility and limitations

All splitters and estimator seeds use 42. The dataset is small, observational, and the grade categories can be imbalanced. Holdout/CV values are estimates, not guarantees. The project is for coursework and must not be used for consequential decisions about students. The tuned search is provided as an optional experiment and is not automatically run during the basic training command.
