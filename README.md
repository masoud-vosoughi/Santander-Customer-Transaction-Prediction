# Santander Customer Transaction Prediction

An end-to-end binary classification project predicting customer transaction behavior using Logistic Regression and LightGBM.

The workflow includes exploratory data analysis, baseline modeling, model complexity experiments, cross-validation, final model training, serialization, batch inference, and external evaluation on Kaggle.

## Key Results

| Evaluation | ROC-AUC |
|---|---:|
| Logistic Regression — Holdout | 0.8599 |
| LightGBM — Holdout | **0.8945** |
| LightGBM — 3-Fold CV Mean | 0.8931 |
| Kaggle Public Leaderboard | 0.89522 |
| **Kaggle Private Leaderboard** | **0.89217** |

**Final model:** LightGBM with 1,726 boosting iterations and 7 leaves per tree.

## 1. Problem and Dataset

The objective is to predict whether a customer will make a specific transaction using 200 anonymized numerical features.

The dataset originates from the [Santander Customer Transaction Prediction competition](https://www.kaggle.com/competitions/santander-customer-transaction-prediction). The data files are also available through [Zenodo](https://zenodo.org/records/7980290).

| Property | Value |
|---|---|
| Training observations | 200,000 |
| Test observations | 200,000 |
| Predictive features | 200 |
| Target | `target` |
| Customer identifier | `ID_code` |
| Positive class proportion | 10.05% |
| Evaluation metric | ROC-AUC |

The dataset contains 179,902 negative observations and 20,098 positive observations.

Initial quality checks confirmed that the training and test datasets contain no missing values, all predictive features are numerical, and no duplicate feature vectors were detected within either dataset.

The customer identifier was excluded from model training.

## 2. Methodology

### Validation Strategy

The labeled dataset was divided into training and validation partitions using an 80/20 stratified split with `random_state=42`.

This produced 160,000 training observations and 40,000 validation observations.

ROC-AUC was selected because the objective is to evaluate how effectively the model ranks positive observations above negative observations, independently of a fixed classification threshold.

### Baseline: Logistic Regression

A scikit-learn Pipeline combined `StandardScaler` and `LogisticRegression`.

Using a Pipeline ensures that feature scaling is fitted only on the training portion of each evaluation split, preventing preprocessing leakage.

The baseline achieved:

- Training ROC-AUC: **0.8613**
- Validation ROC-AUC: **0.8599**

The small generalization gap indicated limited overfitting on the selected split, but the model's predictive performance left room for improvement.

### LightGBM Experiments

LightGBM was evaluated to capture nonlinear relationships and feature interactions.

Three configurations were compared:

| Configuration | Best Iteration | Train AUC | Validation AUC |
|---|---:|---:|---:|
| Initial LightGBM (`num_leaves=15`) | 1079 | 0.9608 | 0.8928 |
| Reduced complexity (`num_leaves=7`) | 1500 | 0.9372 | 0.8939 |
| Extended boosting (`num_leaves=7`) | 1726 | 0.9417 | **0.8945** |

Reducing tree complexity improved validation performance and reduced the training-validation gap relative to the initial configuration.

Extending the boosting limit to 3,000 allowed early stopping to identify an improved model at iteration 1,726.

### Cross-Validation

Three-fold Stratified Cross-Validation was performed on the 160,000-observation training partition.

| Model | Mean ROC-AUC | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.8590 | 0.0010 |
| LightGBM | **0.8931** | 0.0010 |

LightGBM showed consistently stronger ranking performance across the evaluated folds.

Because its configuration had already been selected using holdout validation, these internal results should not be treated as a fully independent estimate of generalization.

## 3. Final Model

The selected LightGBM configuration was retrained on all 200,000 labeled training observations.

| Hyperparameter | Value |
|---|---|
| `objective` | `binary` |
| `n_estimators` | 1726 |
| `learning_rate` | 0.05 |
| `num_leaves` | 7 |
| `min_child_samples` | 50 |
| `random_state` | 42 |
| `n_jobs` | -1 |

The model was serialized with Joblib and successfully reloaded.

Prediction consistency was verified on five training observations using `numpy.allclose()` with an absolute tolerance of `1e-12` and zero relative tolerance.

Batch inference generated 200,000 positive-class probabilities for the competition test dataset. All predictions were finite and within the interval `[0, 1]`.

## 4. External Evaluation

The final predictions were submitted through Kaggle's late-submission mechanism and evaluated against hidden test labels.

| Leaderboard | ROC-AUC |
|---|---:|
| Public | 0.89522 |
| **Private** | **0.89217** |

The private leaderboard result was close to the internal validation estimates, providing additional evidence of generalization on the competition dataset.

No additional hyperparameter tuning was performed using leaderboard feedback.

## 5. Repository Structure

```text
Santander-Customer-Transaction-Prediction/
├── data/                              # Not tracked by Git
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── submissions/
│       └── lightgbm_submission.csv
├── models/                            # Not tracked by Git
│   └── lightgbm_final.joblib
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_final_model.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

The raw datasets, generated predictions, serialized model, and virtual environment are intentionally excluded from version control.

## 6. Installation and Reproduction

### Requirements

- Python 3.12
- Git
- Jupyter or VS Code with the Python and Jupyter extensions

### Clone the Repository

```bash
git clone https://github.com/masoud-vosoughi/Santander-Customer-Transaction-Prediction.git
cd Santander-Customer-Transaction-Prediction
```

### Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Direct dependencies include NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, LightGBM, Joblib, Jupyter, and IPython Kernel.

### Prepare the Dataset

Download the CSV files from the [Zenodo archive](https://zenodo.org/records/7980290) and extract them into:

```text
data/raw/
```

The following files are required:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

### Execute the Notebooks

Open the project in VS Code and select the project's `.venv` Python interpreter as the Jupyter kernel.

Run the notebooks in order:

1. `notebooks/01_data_understanding.ipynb` — Data exploration, model experiments, and cross-validation.
2. `notebooks/02_final_model.ipynb` — Final model training, serialization, verification, and test inference.

**Working-directory requirement:** The current notebooks resolve the project root using `Path.cwd().parent`. They have been verified with the kernel working directory set to `notebooks/`. Other working directories may require path adjustments.

Both notebooks have been successfully executed from a restarted kernel using **Restart Kernel and Run All Cells** in the development environment.

A separate clean-environment reproduction has not yet been verified.

## 7. Limitations

**Anonymized Features:** Feature anonymity limits domain-specific interpretation and business explanations.

**Model Selection Bias:** The holdout partition was used during hyperparameter selection, so its reported performance is not entirely independent of the development process.

**Probability Calibration:** The calibration of predicted probabilities has not been assessed.

**External Validity:** Kaggle leaderboard performance does not guarantee equivalent results on future production data.

**Operational Scope:** The project includes local batch inference and model serialization, but no deployment service, automated monitoring, or dedicated automated test suite.

**Environment Portability:** Execution was verified in the development environment; independent installation and execution on a clean environment remain untested.

## 8. Conclusion

LightGBM significantly outperformed the Logistic Regression baseline on the Santander Customer Transaction Prediction dataset.

The selected model achieved a holdout ROC-AUC of **0.8945**, mean three-fold cross-validation ROC-AUC of **0.8931**, and Kaggle Private Leaderboard ROC-AUC of **0.89217**.

The project demonstrates a complete local machine learning workflow, from initial data assessment and baseline development to model selection, final training, serialization, inference, and external evaluation.

The final classifier and the associated experiments are documented in the accompanying notebooks.