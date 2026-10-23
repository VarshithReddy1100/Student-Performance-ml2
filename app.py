"""Streamlit dashboard for Student Academic Performance Prediction."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from student_performance.data import DEFAULT_DATA_PATH, load_data, prepare_targets
from student_performance.evaluation import compare_models, get_feature_importances, make_learning_curve, regression_metrics, residual_table, tune_random_forest
from student_performance.models import classification_models, regression_models
from student_performance.training import load_artifact, predict_student

st.set_page_config(page_title="Student Performance ML", page_icon="🎓", layout="wide")
st.title("🎓 Student Academic Performance Prediction")
st.caption("Explore leakage-safe regression and classification experiments on the UCI student-mat dataset.")

if not DEFAULT_DATA_PATH.exists():
    st.warning("Dataset is not downloaded. In PowerShell run: python scripts/download_data.py")
    st.stop()

frame = load_data(DEFAULT_DATA_PATH)
X, y_reg, y_clf = prepare_targets(frame)
with st.sidebar:
    st.header("Project overview")
    st.write(f"**Students:** {len(frame)}")
    st.write(f"**Predictors:** {len(X.columns)} (G1/G2/G3 excluded)")
    page = st.radio("Choose a view", ["Overview", "Model evaluation", "Predict a student", "Methodology"])

if page == "Overview":
    a, b, c = st.columns(3)
    a.metric("Records", len(frame))
    b.metric("Input features", X.shape[1])
    c.metric("Average final grade", f"{y_reg.mean():.2f} / 20")
    left, right = st.columns(2)
    with left:
        st.subheader("Final grade distribution")
        fig, ax = plt.subplots()
        sns.histplot(y_reg, discrete=True, ax=ax, color="#4c78a8")
        ax.set(xlabel="G3 final grade", ylabel="Students")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Performance categories")
        counts = y_clf.value_counts().reindex(["Low", "Medium", "High"])
        st.bar_chart(counts)
    st.info("Categories: Low = G3 0–9, Medium = 10–14, High = 15–20. The model intentionally excludes G1 and G2 to avoid leakage.")

elif page == "Model evaluation":
    task_label = st.selectbox("Prediction task", ["Regression: predict G3", "Classification: performance band"])
    task = "regression" if task_label.startswith("Regression") else "classification"
    target = y_reg if task == "regression" else y_clf
    registry = regression_models(X) if task == "regression" else classification_models(X)
    if st.button("Run 5-fold cross-validation", type="primary"):
        with st.spinner("Evaluating models across folds..."):
            st.session_state[f"cv_{task}"] = compare_models(registry, X, target, task)
    results = st.session_state.get(f"cv_{task}")
    if results is not None:
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.caption("Regression reports MAE, RMSE and R²; classification reports accuracy, macro-F1, precision and recall. Train score vs validation score helps identify overfitting/underfitting.")
    else:
        st.write("Choose a task, then run cross-validation to compare model families.")
    chosen = st.selectbox("Learning curve model", list(registry))
    if st.button("Draw learning curve"):
        with st.spinner("Calculating learning curve..."):
            sizes, train_scores, val_scores = make_learning_curve(registry[chosen], X, target, task)
        curve = pd.DataFrame({"Training examples": sizes, "Train score": train_scores, "Validation score": val_scores}).set_index("Training examples")
        st.line_chart(curve)
        st.caption("A large persistent train/validation gap suggests high variance (overfitting); both scores remaining low can indicate high bias (underfitting).")
    if task == "regression" and st.button("Show residual analysis"):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        fitted = registry[chosen].fit(X_train, y_train)
        residuals = residual_table(y_test, fitted.predict(X_test))
        st.write(regression_metrics(residuals["Actual"], residuals["Predicted"]))
        fig, ax = plt.subplots()
        sns.scatterplot(data=residuals, x="Predicted", y="Residual", ax=ax)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title(f"Residual analysis: {chosen}")
        st.pyplot(fig)
        plt.close(fig)
    tree_names = [name for name in registry if "Tree" in name or "Forest" in name]
    if tree_names:
        importance_model = st.selectbox("Tree feature-importance model", tree_names)
        if st.button("Calculate feature importances"):
            fitted_tree = registry[importance_model].fit(X, target)
            importance = get_feature_importances(fitted_tree).head(15).set_index("Feature")
            st.bar_chart(importance)
            st.caption("Impurity-based importance indicates predictive split usage; it is not causal and may favor high-cardinality features.")
    with st.expander("Hyperparameter tuning"):
        st.write("GridSearchCV compares Random Forest depth, leaf size and estimator count. This may take a few minutes.")
        if st.button("Tune Random Forest with 5-fold CV"):
            with st.spinner("Searching Random Forest hyperparameters..."):
                search = tune_random_forest(X, target, task, cv_splits=5)
            st.write({"Best parameters": search.best_params_, "CV score": float(search.best_score_)})

elif page == "Predict a student":
    task = st.radio("Output", ["Regression (G3 grade)", "Classification (performance band)"], horizontal=True)
    artifact_task = "regression" if task.startswith("Regression") else "classification"
    try:
        artifact = load_artifact(artifact_task)
    except FileNotFoundError as error:
        st.warning(str(error))
        st.stop()
    st.write(f"Using trained **{artifact['best_model_name']}**. Enter student details; grades G1/G2 are not used.")
    values = {}
    with st.form("student_input"):
        columns = st.columns(3)
        for index, feature in enumerate(X.columns):
            sample = X[feature].dropna()
            with columns[index % 3]:
                if pd.api.types.is_numeric_dtype(X[feature]):
                    values[feature] = st.number_input(feature, value=float(sample.median()), step=1.0)
                else:
                    options = sorted(sample.astype(str).unique().tolist())
                    values[feature] = st.selectbox(feature, options)
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        result = predict_student(values, artifact_task)
        if artifact_task == "regression":
            st.success(f"Predicted final grade: **{float(result[0]):.2f} / 20**")
        else:
            st.success(f"Predicted performance category: **{result[0]}**")

else:
    st.header("Methodology at a glance")
    st.markdown("""
    1. **Prepare:** UCI Mathematics records; categorical variables one-hot encoded; missing numeric values median-imputed.
    2. **Prevent leakage:** Features exclude G1, G2 and G3. G3 remains the regression target and generates Low/Medium/High classification labels.
    3. **Evaluate:** fixed 80/20 train/test split for final comparisons plus shuffled 5-fold CV; keep the test set for a final unbiased assessment.
    4. **Compare:** linear, polynomial, Ridge, Lasso, SVM kernels, decision trees and random forests.
    5. **Interpret:** MAE/RMSE/R² and residuals for regression; accuracy, macro precision/recall/F1 and confusion matrix for classification. Tree-based models expose feature importance through Scikit-learn.

    **Important:** This is a small educational dataset. Cross-validation scores may vary, the classes may be imbalanced, and grade prediction should not be treated as a consequential student assessment.
    """)
