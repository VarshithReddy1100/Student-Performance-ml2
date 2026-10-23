"""Training, artifact persistence, and single-student predictions."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import DEFAULT_DATA_PATH, load_data, prepare_targets
from .evaluation import classification_metrics, regression_metrics
from .models import classification_models, regression_models

ARTIFACT_DIR = Path("artifacts")


def train_experiments(data_path: Path = DEFAULT_DATA_PATH, output_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    """Train all syllabus models, compare CV and holdout performance, and save artifacts."""
    frame = load_data(data_path)
    X, y_reg, y_clf = prepare_targets(frame)
    indices = pd.RangeIndex(len(frame))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"dataset_rows": len(frame), "feature_columns": X.columns.tolist(), "target_leakage_excluded": ["G1", "G2", "G3"]}

    for task, y, registry, filename in [
        ("regression", y_reg, regression_models(X), "best_regression.joblib"),
        ("classification", y_clf, classification_models(X), "best_classification.joblib"),
    ]:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        results = []
        for model_name, estimator in registry.items():
            estimator.fit(X_train, y_train)
            prediction = estimator.predict(X_test)
            scores = regression_metrics(y_test, prediction) if task == "regression" else classification_metrics(y_test, prediction)
            results.append({"Model": model_name, **scores})
        sort_key = "RMSE" if task == "regression" else "Macro F1"
        best_index = min(range(len(results)), key=lambda i: results[i][sort_key]) if task == "regression" else max(range(len(results)), key=lambda i: results[i][sort_key])
        best_name = results[best_index]["Model"]
        best_model = registry[best_name].fit(X_train, y_train)
        joblib.dump({"model": best_model, "features": X.columns.tolist(), "target": "G3" if task == "regression" else "G3_category", "task": task, "best_model_name": best_name}, output_dir / filename)
        summary[task] = {"holdout_results": results, "best_model": best_name, "best_metrics": results[best_index]}
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_artifact(task: str, artifact_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    """Load the saved best regression or classification artifact."""
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")
    path = Path(artifact_dir) / f"best_{task}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Run python scripts/train_models.py")
    return joblib.load(path)


def predict_student(student: dict[str, Any] | pd.DataFrame, task: str, artifact_dir: Path = ARTIFACT_DIR) -> Any:
    """Predict G3 or category for a row/dict containing the original student features."""
    artifact = load_artifact(task, artifact_dir)
    row = student.copy() if isinstance(student, pd.DataFrame) else pd.DataFrame([student])
    missing = set(artifact["features"]) - set(row.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    return artifact["model"].predict(row[artifact["features"]])
