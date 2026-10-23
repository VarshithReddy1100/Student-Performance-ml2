# 🎓 Student Academic Performance Prediction & Analysis System

### An End-to-End Machine Learning Project | Python · Scikit-learn · Streamlit

<p align="center">
  <a href="https://student-performance-tsafgd6wn2dkqsvyprhkn.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://github.com/VarshithReddy1100/Student-Performance-ml2">
    <img src="https://img.shields.io/badge/GitHub-View_Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

<p align="center">
  An educational machine-learning application that explores student academic performance, compares predictive models, and demonstrates essential machine-learning concepts through an interactive dashboard.
</p>

---

## 🌐 Live Demo

**Try the deployed application:**  
👉 https://student-performance-tsafgd6wn2dkqsvyprhkn.streamlit.app/

**Explore the source code:**  
👉 https://github.com/VarshithReddy1100/Student-Performance-ml2

The application is designed to demonstrate how machine-learning workflows can be applied to an educational dataset, from data preparation and model training to evaluation and prediction.

> Note: The availability and functionality of the live deployment should be verified independently. Available dashboard features depend on the implementation in the repository.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Machine Learning Algorithms](#-machine-learning-algorithms)
- [Evaluation Metrics](#-evaluation-metrics)
- [Dataset](#-dataset)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [Testing](#-testing)
- [Applications](#-applications)
- [Limitations and Ethical Considerations](#-limitations-and-ethical-considerations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [References](#-references)

---

## 📖 Project Overview

Student academic performance is influenced by multiple factors, including personal circumstances, study habits, family background, and school-related conditions. Machine learning provides a way to investigate patterns in educational data and compare different approaches to prediction.

This project develops an educational machine-learning system using the **UCI Student Performance dataset**. It focuses on predicting final mathematics grades and exploring performance categories using regression and classification techniques.

The project brings together data preprocessing, feature engineering, supervised learning, model evaluation, and an interactive web interface.

### 🎯 Prediction Tasks

| Task | Target | Output |
|---|---|---|
| Regression | `G3` | Predicted final mathematics grade from 0 to 20 |
| Classification | Derived from `G3` | Low, Medium, or High performance |

The classification categories are defined as:

- **Low:** 0–9
- **Medium:** 10–14
- **High:** 15–20

These categories are project-defined labels for educational experimentation.

---

## ❓ Problem Statement

Educational datasets contain a mixture of numerical and categorical information. Understanding such data and evaluating predictive models requires appropriate preprocessing, model selection, and performance measurement.

The objective of this project is to build a structured machine-learning workflow that investigates how different supervised learning algorithms perform on student academic data.

The system also provides a practical environment for studying the strengths and limitations of regression and classification algorithms.

---

## 🎯 Project Objectives

- Analyze and explore an educational dataset.
- Prepare numerical and categorical features for machine learning.
- Predict final mathematics grades using regression algorithms.
- Classify students into predefined performance categories.
- Compare linear models, regularized regression, SVMs, decision trees, and random forests.
- Demonstrate model evaluation and cross-validation.
- Investigate overfitting, underfitting, and the bias-variance trade-off.
- Present machine-learning workflows through a web-based interface.
- Document the methodology, implementation, limitations, and possible extensions.

---

## ✨ Key Features

The following are the intended project capabilities; consult the source code to confirm which are implemented in the current version.

| Feature | Description |
|---|---|
| 📊 Dataset exploration | Inspect student records and feature distributions |
| 🧹 Data preprocessing | Prepare numerical and categorical variables |
| 📈 Grade prediction | Estimate the final grade using regression |
| 🏷️ Performance classification | Predict a Low, Medium, or High category |
| 🧠 Model comparison | Compare different supervised learning approaches |
| 📉 Regression evaluation | Analyze prediction errors and goodness of fit |
| 🎯 Classification evaluation | Examine classification performance |
| 🔬 Model validation | Explore train/test evaluation and cross-validation |
| 🌐 Interactive dashboard | Access the application through Streamlit |

---

## 🤖 Machine Learning Algorithms

The project scope covers the following algorithms. Their actual availability depends on the implemented code.

### 1. Linear Regression

Models the relationship between input features and a numerical target.

### 2. Polynomial Regression

Extends linear regression with polynomial features to represent nonlinear relationships.

### 3. Ridge Regression

Uses L2 regularization to penalize large coefficients and help control overfitting.

### 4. Lasso Regression

Uses L1 regularization, which can shrink some coefficients to zero.

### 5. Support Vector Machine (SVM)

Explores classification using different kernel functions:

- Linear kernel
- Polynomial kernel
- Radial Basis Function (RBF) kernel

Important concepts include hyperplanes, margins, support vectors, the regularization parameter `C`, and the RBF parameter `gamma`.

### 6. Decision Trees

Learn feature-based decision rules for classification or regression. Tree depth and splitting criteria influence model complexity.

### 7. Random Forests

Combine multiple decision trees to improve generalization and support feature-importance analysis.

---

## 📏 Evaluation Metrics

Appropriate metrics help compare models and understand their limitations.

### Regression Metrics

| Metric | Purpose |
|---|---|
| MAE | Average absolute prediction error |
| MSE | Average squared prediction error |
| RMSE | Error measure in the target's original units |
| R² | Measures goodness of fit relative to a baseline |

### Classification Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Fraction of correctly classified observations |
| Precision | Fraction of predicted positives that are correct |
| Recall | Fraction of actual positives that are identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion matrix | Shows correct and incorrect class predictions |
| ROC-AUC | Measures discrimination when an appropriate score and evaluation setup are available |

The choice of metric should reflect the task, class balance, and consequences of prediction errors.

### Model Validation Concepts

The project also addresses these ML-2 concepts where implemented:

- Train/test split
- K-fold cross-validation
- Hyperparameter tuning
- Overfitting and underfitting
- Bias-variance trade-off
- Learning curves
- Validation curves
- Residual analysis

No model is assumed to be best before its results are measured.

---

## 📂 Dataset

**Source:** UCI Machine Learning Repository — Student Performance

🔗 https://archive.ics.uci.edu/dataset/320/student+performance

The dataset contains information about secondary-school students in Portugal, including demographic, family, social, and school-related attributes.

This project uses the mathematics dataset, `student-mat.csv`.

### Target Variables

- `G3`: final mathematics grade, used for regression.
- A derived performance label: used for classification.

### Target Leakage Prevention

For the main prediction tasks, `G1` and `G2`—the first and second period grades—should be excluded from the input features, along with `G3` itself.

This reduces the risk of overly optimistic predictions caused by using earlier grades that are closely related to the final grade.

All learned preprocessing transformations should be fitted on training data only, then applied to validation, test, and prediction data.

### Dataset Attribution

Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance.*

Please consult the original UCI dataset page for the complete citation, dataset documentation, and usage information.

---

## 🛠️ Technology Stack

| Technology | Role |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Scikit-learn | Machine-learning pipelines and algorithms |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |
| Joblib | Model persistence |
| Pytest | Automated testing |

---

## 🏗️ Project Architecture

The repository separates the user interface, machine-learning logic, tests, scripts, and documentation.

```text
Student-Performance-ml2/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── pytest.ini
│
├── src/
│   └── student_performance/
│       ├── __init__.py
│       ├── data.py
│       ├── evaluation.py
│       ├── models.py
│       └── training.py
│
├── scripts/
│   ├── download_data.py
│   └── train_models.py
│
├── tests/
│   ├── test_data.py
│   └── test_models.py
│
└── docs/
    ├── methodology.md
    ├── technical_report.md
    └── viva_questions.md
```

This tree reflects the expected core structure. If the repository changes, update this section to match the actual files.

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11 recommended
- Git
- Visual Studio Code or another Python IDE
- Internet access for downloading the dataset and dependencies

### Step 1: Clone the Repository

```bash
git clone https://github.com/VarshithReddy1100/Student-Performance-ml2.git
cd Student-Performance-ml2
```

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download the Dataset

```bash
python scripts/download_data.py
```

Follow the script's output if manual dataset placement is required.

### Step 5: Train the Models

```bash
python scripts/train_models.py
```

Run this command only after confirming that the dataset is available and that the training script supports your local setup.

---

## 🚀 Running the Application

Launch the Streamlit dashboard from the repository root:

```bash
streamlit run app.py
```

Streamlit should display a local URL in your terminal, usually:

`http://localhost:8501`

Open that address in your browser.

For the hosted version, use the [Live Demo](https://student-performance-tsafgd6wn2dkqsvyprhkn.streamlit.app/).

---

## 🧪 Testing

Run the project's automated tests:

```bash
pytest -q
```

The test suite should be used to check data handling and model behavior. Actual test results must be obtained by running the tests in the current environment.

---

## 🏫 Potential Applications

This project is primarily intended for education, coursework, and machine-learning experimentation.

Potential uses include:

- **Teaching:** Demonstrating supervised learning and evaluation.
- **Research:** Comparing algorithms on a common educational dataset.
- **Data analysis:** Exploring relationships between recorded student attributes and academic outcomes.
- **Prototype development:** Providing a foundation for investigating educational analytics using suitable, responsibly collected data.

These are potential applications, not evidence that the model has been validated for operational university use.

---

## ⚠️ Limitations and Ethical Considerations

1. **Secondary-school dataset:** The data represents secondary-school students in Portugal, not university students.
2. **Limited generalization:** Model performance on this dataset does not establish performance on other schools, countries, or populations.
3. **Prediction uncertainty:** Estimated grades and categories may be incorrect and should not be treated as facts.
4. **Bias and fairness:** Student attributes may reflect social and economic inequalities. A predictive model can reproduce or amplify such patterns.
5. **Data leakage:** Earlier grades and target-related information can inflate evaluation results if incorrectly included.
6. **Privacy:** Any future real-world deployment should use appropriately authorized, minimized, and protected data.
7. **Human oversight:** Predictions should not independently determine admissions, grades, scholarships, disciplinary action, or access to educational opportunities.

**This is an educational prototype, not a validated university student assessment system.** Its predictions should be used for learning and experimentation, not consequential decisions about individual students.

---

## 🔮 Future Improvements

Possible future extensions include:

- Add more robust experiment tracking and model comparison.
- Improve the interface with clearer explanations of individual predictions.
- Add reproducible experiment reports and saved evaluation results.
- Evaluate fairness and calibration where appropriate.
- Validate models on independent datasets and different student cohorts.
- Improve automated testing and continuous integration.
- Add model monitoring and versioning.
- Investigate privacy-preserving approaches for responsibly collected university data.

---

## 👨‍💻 Author

**Varshith Reddy Surakanti**

B.Tech — Computer Science and Engineering  
Lovely Professional University, Phagwara, Punjab, India

- **GitHub:** https://github.com/VarshithReddy1100
- **Project Repository:** https://github.com/VarshithReddy1100/Student-Performance-ml2
- **Live Application:** https://student-performance-tsafgd6wn2dkqsvyprhkn.streamlit.app/

---

## 📚 References

1. UCI Machine Learning Repository — [Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
2. Scikit-learn documentation — [Machine Learning in Python](https://scikit-learn.org/stable/)
3. Streamlit documentation — [Build Data Apps](https://docs.streamlit.io/)
4. Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance.*

---

<p align="center">
  <b>Built for learning, experimentation, and exploring machine learning in education.</b>
  <br>
  ⭐ If you find this project useful, consider starring the repository.
</p>
