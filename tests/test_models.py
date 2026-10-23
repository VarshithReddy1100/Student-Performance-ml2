import pandas as pd
from sklearn.model_selection import train_test_split

from student_performance.evaluation import classification_metrics, regression_metrics
from student_performance.models import classification_models, regression_models


def features_and_targets():
    X = pd.DataFrame({"age": [15, 16, 17, 18, 16, 15, 17, 18, 16, 17, 18, 15], "studytime": [1, 2, 3, 4] * 3, "school": ["GP", "MS"] * 6})
    y_reg = pd.Series([5, 8, 10, 15, 9, 6, 12, 18, 11, 13, 16, 7])
    y_clf = pd.Series(["Low", "Low", "Medium", "High", "Low", "Low", "Medium", "High", "Medium", "Medium", "High", "Low"])
    return X, y_reg, y_clf


def test_all_model_families_fit_and_predict():
    X, y_reg, y_clf = features_and_targets()
    for registry, target in [(regression_models(X), y_reg), (classification_models(X), y_clf)]:
        name, model = next(iter(registry.items()))
        model.fit(X, target)
        assert len(model.predict(X.iloc[:2])) == 2


def test_metric_helpers_return_expected_keys():
    assert set(regression_metrics([1, 2], [1, 3])) == {"MAE", "RMSE", "R2"}
    scores = classification_metrics(["Low", "High"], ["Low", "Medium"])
    assert scores["Accuracy"] == 0.5
    assert len(scores["Confusion matrix"]) == 3
