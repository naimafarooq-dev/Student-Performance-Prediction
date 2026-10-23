# Student Performance Analysis and Prediction Using Machine Learning

**Machine Learning | Python | Data Analysis | Regression | Classification**

## 1. Project Overview

This project investigates student academic performance using machine learning techniques and survey-based educational data.

The primary objective is to explore relationships between students' academic backgrounds and undergraduate performance, predict undergraduate CGPA using regression models, and classify students into academic-risk categories using logistic regression.

The project covers the complete machine learning workflow, including data cleaning, exploratory data analysis (EDA), feature preprocessing, model training, hyperparameter tuning, evaluation, and interpretation.

It also includes a manual implementation of Logistic Regression using gradient descent to demonstrate the underlying mathematical concepts rather than relying entirely on prebuilt machine learning libraries.

---

## 2. Project Objectives

The main objectives of this project are:

- Analyze student academic performance using survey data.
- Identify patterns and relationships among educational features.
- Handle missing values, duplicate records, and inconsistent data.
- Perform exploratory data analysis and visualize feature distributions.
- Predict undergraduate CGPA using regression algorithms.
- Apply L1 and L2 regularization to machine learning models.
- Classify students using a CGPA-based academic-risk indicator.
- Implement Logistic Regression manually using sigmoid, cost function, and gradient descent.
- Evaluate and compare different machine learning models.
- Investigate how reducing training data affects model performance.
- Interpret model coefficients and discuss limitations.

---

## 3. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning algorithms and evaluation |
| Jupyter Notebook | Interactive development and experimentation |
| OpenPyXL | Reading Excel datasets |

---

## 4. Dataset Description

The dataset was collected through a student survey containing academic and educational background information.

### Input Features

The machine learning models use the following features:

| Feature | Description |
|---|---|
| Gender | Student gender |
| Province | Student's province |
| Parental Education Group | Educational background of parents |
| Matric Percentage | Academic percentage obtained in Matric |
| FSc Percentage | Academic percentage obtained in Intermediate/FSc |

### Target Variable

**Undergraduate CGPA (UG CGPA)**

This is the continuous target variable used for regression.

### Academic-Risk Classification

For classification, a binary academic-risk indicator is derived from undergraduate CGPA:

| Condition | Class | Interpretation |
|---|---|---|
| CGPA >= 2.5 | 0 | Not At Risk |
| CGPA < 2.5 | 1 | At Risk |

**Important:** This label is a CGPA-based academic-risk proxy. It does not represent observed student dropout or independently verified academic failure.

### Dataset Privacy

The original dataset is not included in this public repository to protect the privacy of survey participants.

The notebook demonstrates the implementation and analysis workflow, but reproducing the original results requires authorized access to the dataset.

---

## 5. Machine Learning Workflow

The project follows a structured machine learning pipeline.

### Step 1: Data Loading and Inspection

- Load survey data from an Excel file.
- Inspect column names and dataset structure.
- Identify missing values and duplicate records.
- Examine the target variable.

### Step 2: Data Cleaning

- Standardize column names.
- Normalize missing-value representations.
- Convert percentage values into consistent units.
- Normalize parental education categories.
- Remove records without valid target values.
- Exclude variables that could introduce target leakage.

### Step 3: Exploratory Data Analysis

Exploratory data analysis is performed to understand the distribution and relationships of academic features.

The analysis includes:

- Missing-value analysis
- Outlier detection using the Interquartile Range (IQR) method
- Correlation analysis
- Feature distributions
- CGPA distribution
- Academic-risk class distribution

Outliers are investigated rather than automatically removed.

### Step 4: Data Preprocessing

The preprocessing pipeline includes:

**Numerical features:**
- Missing-value imputation
- Standardization using StandardScaler

**Categorical features:**
- Missing-value imputation
- One-hot encoding

Scikit-learn pipelines and ColumnTransformer are used to apply preprocessing consistently.

### Step 5: Train-Test Split

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

Preprocessing transformations are fitted using training data to prevent data leakage.

The held-out test set is used for final model evaluation.

---

## 6. Regression Models

Regression models are trained to predict undergraduate CGPA.

### 6.1 Linear Regression

Linear Regression establishes a baseline relationship between input features and undergraduate CGPA.

The general equation is:

`ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ`

Where:

- `ŷ` represents predicted CGPA.
- `θ₀` represents the intercept.
- `θ₁ ... θₙ` represent model coefficients.
- `x₁ ... xₙ` represent input features.

### 6.2 Ridge Regression (L2 Regularization)

Ridge Regression introduces an L2 penalty to reduce excessive coefficient magnitudes.

**Objective function:**

`J(θ) = MSE + λ Σ θⱼ²`

Ridge Regression is useful for controlling model complexity and improving stability when features are correlated.

### 6.3 Lasso Regression (L1 Regularization)

Lasso Regression introduces an L1 penalty.

**Objective function:**

`J(θ) = MSE + λ Σ |θⱼ|`

Lasso can shrink some coefficients to zero, providing a form of automatic feature selection.

### 6.4 Hyperparameter Tuning

The project uses GridSearchCV with five-fold cross-validation to tune Ridge and Lasso regularization parameters.

The selected models are evaluated using the same held-out test set.

---

## 7. Classification Models

Classification models predict the derived academic-risk category.

### 7.1 Logistic Regression

Logistic Regression estimates the probability that a student belongs to the academic-risk class.

The sigmoid function is:

`σ(z) = 1 / (1 + e^(-z))`

Where:

`z = θ₀ + θ₁x₁ + ... + θₙxₙ`

The sigmoid function transforms the linear model output into a probability between 0 and 1.

### 7.2 L1-Regularized Logistic Regression

L1 regularization adds an absolute-value penalty to the Logistic Regression objective.

It can encourage sparse coefficients and reduce model complexity.

### 7.3 L2-Regularized Logistic Regression

L2 regularization penalizes large coefficients while generally retaining all model features.

### 7.4 Classification Hyperparameter Tuning

The project uses stratified cross-validation where possible to tune Logistic Regression models.

ROC-AUC is used as a model-selection metric during hyperparameter tuning.

Class imbalance is considered when interpreting classification performance.

---

## 8. Logistic Regression From Scratch

One of the key components of this project is the manual implementation of Logistic Regression without using `sklearn.LogisticRegression` for the model itself.

The implementation demonstrates the mathematical foundations of binary classification.

### 8.1 Linear Combination

`z = Xθ`

### 8.2 Sigmoid Function

`hθ(x) = 1 / (1 + e^(-z))`

### 8.3 Binary Cross-Entropy Cost Function

`J(θ) = -(1/m) Σ [y log(hθ(x)) + (1-y) log(1-hθ(x))]`

Where:

- `m` is the number of training examples.
- `y` is the actual class.
- `hθ(x)` is the predicted probability.

### 8.4 Gradient Descent

Model parameters are updated iteratively using:

`θ := θ - α∇J(θ)`

Where:

- `α` is the learning rate.
- `∇J(θ)` is the gradient of the cost function.

### 8.5 Model Evaluation

The manually implemented model is evaluated using the held-out test set and compared with Scikit-learn Logistic Regression.

The notebook also examines predicted probabilities, predicted classes, and classification errors.

---

## 9. Model Evaluation

Different metrics are used for regression and classification because they solve different types of prediction problems.

### Regression Evaluation

| Metric | Purpose |
|---|---|
| Mean Squared Error (MSE) | Measures average squared prediction error |
| Root Mean Squared Error (RMSE) | Measures prediction error in CGPA units |
| R² Score | Measures variance explained relative to a baseline |

### Classification Evaluation

| Metric | Purpose |
|---|---|
| Accuracy | Proportion of correct predictions |
| Precision | Proportion of predicted positives that are correct |
| Recall | Proportion of actual positives correctly identified |
| F1 Score | Harmonic mean of precision and recall |
| Balanced Accuracy | Average recall across classes |
| ROC-AUC | Measures ranking ability across classification thresholds |
| Confusion Matrix | Displays correct and incorrect classifications |

The notebook generates comparison tables and diagnostic visualizations to examine model performance.

---

## 10. Training Data Reduction Experiments

The project investigates how model performance changes when less training data is available.

Experiments are performed using:

- 100% of the original training data
- 75% of the original training data
- 50% of the original training data

### Ridge Regression

Ridge Regression is evaluated using reduced training subsets.

Repeated training-data truncation experiments are also performed to examine variability across different samples.

### Logistic Regression

Logistic Regression is evaluated using stratified training subsets to approximately preserve the original class distribution.

The held-out test set remains unchanged during these experiments.

**Purpose:** To investigate model sensitivity to training-data availability and assess performance stability.

---

## 11. Model Interpretation

The notebook includes coefficient analysis for both regression and classification models.

### Regression Coefficients

The coefficients of Linear Regression, Ridge Regression, and Lasso Regression are compared to examine the influence of different input features.

### Logistic Regression Coefficients

Logistic Regression coefficients are analyzed to understand their association with the predicted academic-risk class.

Positive coefficients indicate increased log-odds of the derived CGPA-below-2.5 class, subject to the model's feature encoding and other variables.

These coefficients represent statistical associations rather than causal effects.

---

## 12. Prediction Diagnostics

The project examines predictions on the held-out test set.

This includes:

- Actual versus predicted CGPA
- Prediction errors
- Classification probabilities
- Actual versus predicted academic-risk labels
- Confusion matrices
- ROC analysis
- Misclassified cases

These diagnostics help identify model weaknesses and understand prediction behavior.

---

## 13. Repository Structure

```text
Student-Performance-ML/
│
├── Student Performance.ipynb
├── README.md
└── requirements.txt
```

**Note:** The original Excel dataset is intentionally excluded from the public repository.

---

## 14. Installation and Setup

### Prerequisites

- Python 3
- Jupyter Notebook or Google Colab
- Required Python libraries
- Authorized access to the original dataset for execution

### Clone the Repository

```bash
git clone https://github.com/naimafarooq-dev/Student-Performance-ML.git
```

```bash
cd Student-Performance-ML
```

Replace the repository URL if a different repository name is used.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

`Student Performance.ipynb`

### Run the Notebook

1. Make the authorized dataset available locally or in Google Colab.
2. Update the dataset path in the loading cell.
3. Execute the notebook cells sequentially.
4. Review the generated analysis, evaluation tables, and visualizations.

**Reproducibility Note:** Since the original dataset is private, the notebook cannot reproduce the original experimental results using only the publicly available repository files.

---

## 15. Limitations

This project has several important limitations:

- The dataset is based on self-reported survey responses.
- A small dataset may produce unstable model estimates.
- Class imbalance can affect classification evaluation.
- The academic-risk label is derived from CGPA rather than observed dropout outcomes.
- Correlations and model coefficients do not establish causation.
- Model performance may vary when trained on different samples.
- Results should not be interpreted as validated predictions for the wider student population.

The project is intended for academic experimentation and machine learning practice rather than high-stakes educational decision-making.

---

## 16. Learning Outcomes

Through this project, the following machine learning concepts are explored:

- End-to-end machine learning workflow
- Data cleaning and exploratory analysis
- Numerical and categorical preprocessing
- Feature engineering
- Regression and classification
- L1 and L2 regularization
- Cross-validation and hyperparameter tuning
- Logistic Regression mathematics
- Gradient descent implementation
- Model evaluation and comparison
- Training-data sensitivity
- Model interpretability
- Data leakage prevention
- Responsible interpretation of educational data

---

## 17. Future Improvements

Potential future extensions include:

- Evaluating additional machine learning algorithms.
- Collecting a larger and more representative dataset with appropriate consent.
- Applying more robust model validation techniques.
- Investigating feature importance and explainability.
- Developing an interactive dashboard for visualizing academic performance patterns.
- Testing model generalization using independent datasets.
- Exploring fairness and performance differences across demographic groups.

These are proposed improvements and are not part of the current implementation.

---

## 19. Acknowledgment

Developed as part of the CS-871 Machine Learning coursework to gain practical experience in data preprocessing, exploratory data analysis, regression, classification, regularization, and model evaluation.

The project emphasizes both the practical implementation of machine learning algorithms and an understanding of their mathematical foundations.
