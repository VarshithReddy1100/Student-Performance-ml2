"""Model pipelines for regression and classification experiments."""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def build_preprocessor(X: Any, scale: bool = True) -> ColumnTransformer:
    """Impute numeric columns and impute/one-hot encode categorical columns."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def regression_models(X: Any) -> dict[str, Pipeline]:
    """Return comparable regression pipelines, including polynomial and regularized models."""
    return {
        "Linear Regression": Pipeline([("prep", build_preprocessor(X)), ("model", LinearRegression())]),
        "Polynomial Regression (degree 2)": Pipeline([
            ("prep", build_preprocessor(X)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([("prep", build_preprocessor(X)), ("model", Ridge(alpha=1.0))]),
        "Lasso": Pipeline([("prep", build_preprocessor(X)), ("model", Lasso(alpha=0.01, max_iter=10000))]),
        "Decision Tree Regressor": Pipeline([
            ("prep", build_preprocessor(X, scale=False)),
            ("model", DecisionTreeRegressor(max_depth=5, random_state=42)),
        ]),
        "Random Forest Regressor": Pipeline([
            ("prep", build_preprocessor(X, scale=False)),
            ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ]),
    }


def classification_models(X: Any) -> dict[str, Pipeline]:
    """Return linear, polynomial/RBF SVM, tree, forest, and logistic baselines."""
    return {
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor(X)),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]),
        "SVM Linear": Pipeline([("prep", build_preprocessor(X)), ("model", SVC(kernel="linear", C=1.0))]),
        "SVM Polynomial": Pipeline([("prep", build_preprocessor(X)), ("model", SVC(kernel="poly", degree=3, C=1.0, gamma="scale"))]),
        "SVM RBF": Pipeline([("prep", build_preprocessor(X)), ("model", SVC(kernel="rbf", C=1.0, gamma="scale"))]),
        "Decision Tree Classifier": Pipeline([
            ("prep", build_preprocessor(X, scale=False)),
            ("model", DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42, class_weight="balanced")),
        ]),
        "Random Forest Classifier": Pipeline([
            ("prep", build_preprocessor(X, scale=False)),
            ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced")),
        ]),
    }
