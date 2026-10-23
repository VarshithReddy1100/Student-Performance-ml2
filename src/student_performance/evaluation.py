"""Evaluation, cross-validation, learning-curve, and tuning helpers."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate, learning_curve


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calculate common regression metrics without external dependencies."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def residual_table(y_true: Any, y_pred: Any) -> pd.DataFrame:
    """Return actual, predicted, and residual values for diagnostic plots."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return pd.DataFrame({"Actual": actual, "Predicted": predicted, "Residual": actual - predicted})


def get_feature_importances(fitted_pipeline: Any) -> pd.DataFrame:
    """Extract impurity-based feature importances from a fitted tree pipeline."""
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError("The fitted estimator does not expose feature_importances_")
    preprocessor = fitted_pipeline.named_steps["prep"]
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame({"Feature": names, "Importance": model.feature_importances_}).sort_values(
        "Importance", ascending=False, ignore_index=True
    )


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Return accuracy, macro precision/recall/F1, and confusion matrix."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    labels = ["Low", "Medium", "High"]
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro precision": float(precision),
        "Macro recall": float(recall),
        "Macro F1": float(f1),
        "Confusion matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "Labels": labels,
    }


def compare_models(models: dict[str, Any], X: Any, y: Any, task: str, cv_splits: int = 5) -> pd.DataFrame:
    """Fit each pipeline and report holdout-free CV mean/std for appropriate metrics."""
    scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"] if task == "regression" else ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    splitter = KFold(n_splits=cv_splits, shuffle=True, random_state=42) if task == "regression" else StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        result = cross_validate(model, X, y, cv=splitter, scoring=scoring, n_jobs=-1, return_train_score=True)
        row: dict[str, Any] = {"Model": name}
        for metric in scoring:
            key = metric.removeprefix("neg_")
            values = -result[f"test_{metric}"] if metric.startswith("neg_") else result[f"test_{metric}"]
            row[f"CV {key} mean"] = float(np.mean(values))
            row[f"CV {key} std"] = float(np.std(values))
        train_values = result[f"train_{scoring[0]}"]
        if scoring[0].startswith("neg_"):
            train_values = -train_values
        row["Train score mean"] = float(np.mean(train_values))
        rows.append(row)
    return pd.DataFrame(rows)


def make_learning_curve(estimator: Any, X: Any, y: Any, task: str = "regression") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute learning-curve sizes and train/validation score means."""
    scoring = "r2" if task == "regression" else "f1_macro"
    cv = KFold(5, shuffle=True, random_state=42) if task == "regression" else StratifiedKFold(5, shuffle=True, random_state=42)
    sizes, train_scores, validation_scores = learning_curve(
        clone(estimator), X, y, train_sizes=np.linspace(0.2, 1.0, 5), cv=cv,
        scoring=scoring, n_jobs=-1,
    )
    return sizes, train_scores.mean(axis=1), validation_scores.mean(axis=1)


def tune_random_forest(X: Any, y: Any, task: str, cv_splits: int = 5) -> GridSearchCV:
    """Tune compact Random Forest hyperparameters with cross-validation."""
    from .models import classification_models, regression_models

    candidates = classification_models(X) if task == "classification" else regression_models(X)
    name = "Random Forest Classifier" if task == "classification" else "Random Forest Regressor"
    estimator = candidates[name]
    params = {
        "model__n_estimators": [100, 250],
        "model__max_depth": [None, 8],
        "model__min_samples_leaf": [1, 3],
    }
    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=42) if task == "classification" else KFold(cv_splits, shuffle=True, random_state=42)
    search = GridSearchCV(estimator, params, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
    search.fit(X, y)
    return search
