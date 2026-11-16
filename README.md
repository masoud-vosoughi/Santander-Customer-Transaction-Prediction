# Santander Customer Transaction Prediction

## Project Overview

This project addresses a binary classification problem: predicting whether a Santander customer will make a specific transaction.

The analysis focuses on building and comparing machine learning models for structured numerical data, with particular attention to validation methodology, generalization, overfitting, and reproducibility.

The primary evaluation metric is **ROC-AUC**, which measures the ability of a model to distinguish positive from negative instances across classification thresholds.

## Dataset

**Source:** [Santander Customer Transaction Prediction — Kaggle Competition](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)

**Alternative download:** [Zenodo Dataset Archive](https://zenodo.org/records/7980290)

The dataset contains:

| Property | Description |
|---|---|
| Training samples | 200,000 |
| Test samples | 200,000 |
| Predictive features | 200 |
| Target | `target` |
| Customer identifier | `ID_code` |
| Positive class prevalence | Approximately 10.05% |

The predictors are anonymized numerical features named `var_0` through `var_199`.

The original CSV files are excluded from version control.

## Methodology

### 1. Data Understanding and Quality Assessment

Initial analysis established that:

- All 200 predictors are numerical.
- No missing values were found in the training or test datasets.
- No duplicate feature vectors were detected within either dataset.
- Customer identifiers are unique within each dataset.
- The target distribution is imbalanced, with approximately 10% positive observations.

No missing-value imputation, categorical encoding, or duplicate removal was required.

### 2. Validation Strategy

The labeled dataset was divided into training and validation subsets using an 80/20 stratified split with `random_state=42`.

Stratification was selected to preserve the positive-class proportion across both subsets.

Three-fold stratified cross-validation was subsequently used on the training partition to assess performance consistency.

### 3. Baseline Modeling

Logistic Regression was selected as the initial baseline because it offers a relatively simple and interpretable reference for binary classification.

StandardScaler was integrated into a scikit-learn Pipeline to ensure that scaling parameters were learned exclusively from the corresponding training data.

### 4. Nonlinear Modeling

LightGBM was evaluated to investigate whether nonlinear relationships and feature interactions could improve predictive performance.

The experiments included:

- An initial LightGBM classifier with 15 leaves per tree.
- A reduced-complexity configuration with 7 leaves.
- Early stopping based on validation ROC-AUC.
- An extended boosting experiment to avoid prematurely limiting the number of iterations.

The selected provisional LightGBM configuration uses 7 leaves per tree and 1,726 boosting iterations.

## Experimental Results

### Holdout Validation

| Model | Training ROC-AUC | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.8613 | 0.8599 |
| Initial LightGBM | 0.9608 | 0.8928 |
| Reduced-complexity LightGBM | 0.9372 | 0.8939 |
| Extended LightGBM | 0.9417 | **0.8945** |

### Three-Fold Cross-Validation

| Model | Mean ROC-AUC | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.8590 | 0.0010 |
| LightGBM | **0.8931** | 0.0010 |

### Model Selection

LightGBM was selected as the provisional preferred model because it demonstrated higher ROC-AUC than Logistic Regression on both holdout validation and cross-validation.

Reducing the number of leaves improved holdout performance while decreasing the training-validation gap relative to the initial LightGBM configuration.

However, LightGBM continued to exhibit a larger generalization gap than Logistic Regression.

The three-fold cross-validation results showed relatively low variability, providing preliminary evidence of performance stability.

## Project Structure

```text
Santander-Customer-Transaction-Prediction/
├── data/
│   └── raw/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
├── notebooks/
│   └── 01_data_understanding.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

The `data/` directory and local Python virtual environment are excluded from Git.

## Environment Setup

The project was developed using Python 3.12.

Create and activate a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and extract the dataset, placing the CSV files in `data/raw/`.

Open the project in VS Code, select the `.venv` Python environment as the notebook kernel, and open:

`notebooks/01_data_understanding.ipynb`

The current notebook uses a relative data path that assumes its working directory is `notebooks/`. Confirm this with `Path.cwd()` before execution.

Run the notebook cells in sequence to reproduce the analysis and modeling experiments.

## Limitations

- The predictors are anonymized, limiting domain-level interpretability.
- LightGBM hyperparameters were selected using holdout validation; therefore, the reported results are not a fully independent estimate of deployment performance.
- Cross-validation was performed using a fixed configuration and only three folds.
- The unlabeled competition test set cannot provide an independently measurable ROC-AUC.
- Final model serialization and independent inference verification have not yet been completed.
- Reproduction in a newly created environment has not yet been independently verified.

## Current Status

Exploratory analysis, baseline modeling, LightGBM experiments, and cross-validation comparison have been completed.

The preferred LightGBM configuration is provisional, pending final model preparation and inference validation.