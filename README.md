# Santander Customer Transaction Prediction

## 1. Project Overview

This project focuses on predicting whether a Santander customer will make a specific transaction in the future using anonymized numerical features.

The objective is to develop, evaluate, and validate a binary classification model through a reproducible machine learning workflow.

The project covers:

- Data understanding and quality assessment
- Target distribution analysis
- Baseline model development
- Nonlinear modeling with LightGBM
- Hyperparameter experimentation and early stopping
- Cross-validation and generalization analysis
- Final model training and serialization
- Batch inference and prediction consistency testing
- External evaluation using Kaggle's hidden test labels

**Primary Evaluation Metric:** ROC-AUC

**Final Selected Model:** LightGBM

**Kaggle Private Leaderboard ROC-AUC: 0.89217**

---

## 2. Dataset

**Original Competition:** [Santander Customer Transaction Prediction — Kaggle](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)

**Alternative Dataset Source:** [Zenodo Archive](https://zenodo.org/records/7980290)

The dataset contains anonymized customer information represented by 200 numerical features.

| Property | Description |
|---|---|
| Training observations | 200,000 |
| Test observations | 200,000 |
| Predictive features | 200 |
| Feature names | `var_0` to `var_199` |
| Target variable | `target` |
| Identifier | `ID_code` |
| Problem type | Binary classification |
| Positive class proportion | 10.05% |

### Target Distribution

The training dataset contains:

| Class | Count | Percentage |
|---|---:|---:|
| 0 | 179,902 | 89.95% |
| 1 | 20,098 | 10.05% |

The target distribution is imbalanced, with approximately 10% positive observations.

ROC-AUC was selected as the primary evaluation metric because it measures ranking performance across classification thresholds without relying on a fixed decision threshold.

The customer identifier (`ID_code`) was excluded from the model features.

### Data Quality Assessment

The initial data quality analysis identified:

- No missing values in the training or test datasets.
- No duplicate feature vectors within either dataset.
- All 200 predictive features were numerical.
- Unique customer identifiers within each dataset.
- No overlapping customer identifiers between the training and test datasets.

Based on these findings, missing-value imputation, categorical encoding, and duplicate removal were not required.

However, these checks do not establish the absence of outliers, distribution shifts, or data leakage.

---

## 3. Methodology

### 3.1 Validation Strategy

The labeled dataset was divided into training and validation partitions using an 80/20 stratified split.

| Partition | Observations |
|---|---:|
| Training | 160,000 |
| Validation | 40,000 |

The split was configured with:

- `test_size=0.2`
- `stratify=y`
- `random_state=42`

Stratification preserved the positive-class proportion across both partitions.

The same split was used for the initial model comparisons to reduce variability caused by different data partitions.

Three-fold Stratified Cross-Validation was subsequently performed on the training partition to assess model performance consistency.

### 3.2 Baseline Model: Logistic Regression

Logistic Regression was selected as the baseline because it provides a relatively simple and interpretable reference for binary classification.

The implementation used a scikit-learn Pipeline containing:

1. `StandardScaler`
2. `LogisticRegression`

Feature scaling was performed within the Pipeline to prevent preprocessing leakage.

The scaler was fitted exclusively on the corresponding training data, while validation observations were transformed using the learned parameters.

#### Baseline Results

| Metric | ROC-AUC |
|---|---:|
| Training | 0.8613 |
| Validation | 0.8599 |
| Generalization Gap | 0.0014 |

The small training-validation gap provided encouraging evidence of generalization on the selected holdout partition.

However, a small gap does not rule out underfitting.

This motivated evaluating a nonlinear model capable of learning more complex relationships.

### 3.3 Nonlinear Model: LightGBM

LightGBM was introduced to investigate whether nonlinear relationships and feature interactions could improve predictive performance.

Unlike Logistic Regression, tree-based LightGBM generally does not require feature standardization.

The experiments focused on controlling model complexity and monitoring validation performance.

#### Experiment 1: Initial LightGBM

The initial configuration included:

- `num_leaves=15`
- `learning_rate=0.05`
- `min_child_samples=50`
- `n_estimators=1500`
- Early stopping after 50 rounds without validation improvement

The model achieved a validation ROC-AUC of **0.8928**, outperforming Logistic Regression.

However, its training-validation gap was considerably larger.

#### Experiment 2: Reduced Tree Complexity

The maximum number of leaves was reduced from 15 to 7 while keeping the remaining parameters unchanged.

This experiment investigated whether limiting model capacity could improve generalization.

The resulting validation ROC-AUC increased to **0.8939**, while the training-validation gap decreased.

The best iteration reached the configured limit of 1,500 boosting rounds, suggesting that additional iterations could potentially improve performance.

#### Experiment 3: Extended Boosting

The maximum number of boosting iterations was increased to 3,000 while keeping the other parameters unchanged.

Early stopping identified iteration **1,726** as the best-performing iteration.

The resulting validation ROC-AUC was **0.8945**.

This configuration was retained for subsequent evaluation.

---

## 4. Experimental Results

### 4.1 Holdout Validation Comparison

| Model | Train ROC-AUC | Validation ROC-AUC | Generalization Gap |
|---|---:|---:|---:|
| Logistic Regression | 0.8613 | 0.8599 | 0.0014 |
| Initial LightGBM | 0.9608 | 0.8928 | 0.0680 |
| Reduced-complexity LightGBM | 0.9372 | 0.8939 | 0.0432 |
| Extended LightGBM | 0.9417 | **0.8945** | 0.0473 |

The experiments demonstrated that LightGBM achieved higher validation ROC-AUC than the linear baseline.

Reducing the number of leaves improved validation performance while decreasing the generalization gap relative to the initial LightGBM configuration.

Extending the boosting process produced a further improvement.

Although LightGBM exhibited a larger training-validation gap than Logistic Regression, its superior validation performance supported its selection for further evaluation.

### 4.2 Three-Fold Cross-Validation

Both models were evaluated using the same three-fold Stratified Cross-Validation procedure on the training partition.

| Model | Mean ROC-AUC | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.8590 | 0.0010 |
| LightGBM | **0.8931** | 0.0010 |

LightGBM improved mean cross-validation ROC-AUC by approximately **0.0341** compared with Logistic Regression.

Both models exhibited relatively low variability across the three folds.

These results provided encouraging evidence of performance consistency.

However, the LightGBM configuration had already been selected using holdout validation. Therefore, the internal evaluation results should not be interpreted as a completely independent estimate of future production performance.

---

## 5. Final Model Selection

LightGBM was selected as the final classifier based on:

- Superior holdout validation ROC-AUC.
- Higher mean cross-validation ROC-AUC.
- Consistent performance across the evaluated folds.
- Improved validation performance after reducing tree complexity.
- Successful external evaluation on hidden competition test labels.

### Final Hyperparameters

| Hyperparameter | Value |
|---|---|
| `objective` | `binary` |
| `n_estimators` | 1726 |
| `learning_rate` | 0.05 |
| `num_leaves` | 7 |
| `min_child_samples` | 50 |
| `random_state` | 42 |
| `n_jobs` | -1 |

The final model was retrained from scratch using all **200,000 labeled training observations**.

The previously selected number of boosting iterations was retained without additional tuning.

Because the complete labeled dataset was used for final training, no independent internal holdout remained for evaluating this final fitted model.

---

## 6. Model Serialization and Inference

### 6.1 Model Serialization

The trained LightGBM classifier was serialized using Joblib.

**Model artifact:**

`models/lightgbm_final.joblib`

**Serialized model size:** Approximately 1.53 MB.

The model was reloaded and its predictions were compared against those of the original in-memory model.

Prediction consistency was verified using `numpy.allclose()` with:

- Absolute tolerance: `1e-12`
- Relative tolerance: `0`

The verification test passed successfully on five examined training observations.

This confirms numerical prediction consistency for the tested samples after serialization and reloading.

### 6.2 Batch Inference

The reloaded model was used to generate positive-class probabilities for the complete competition test dataset.

| Check | Result |
|---|---|
| Test observations | 200,000 |
| Input features | 200 |
| Prediction shape | (200000,) |
| Feature ordering | Matched |
| Finite predictions | True |
| Probabilities within [0, 1] | True |

All observations were processed successfully.

These checks establish basic inference validity but do not independently measure predictive accuracy or probability calibration.

---

## 7. External Evaluation: Kaggle

After final model training, positive-class probabilities were generated for all 200,000 test observations.

Predictions were submitted to the original Kaggle competition through its late-submission mechanism.

The submission was accepted and evaluated against Kaggle's hidden target labels.

### Evaluation Results

| Evaluation Method | ROC-AUC |
|---|---:|
| Holdout Validation | 0.89450 |
| Three-Fold Cross-Validation (Mean) | 0.89310 |
| Kaggle Public Leaderboard | 0.89522 |
| **Kaggle Private Leaderboard** | **0.89217** |

### Interpretation

The Private Leaderboard ROC-AUC of **0.89217** was close to the internal validation and cross-validation estimates.

This provides additional evidence that the selected modeling approach generalizes beyond the labeled training observations.

The external evaluation is particularly valuable because the hidden test labels were not used during model development or hyperparameter selection.

However, the leaderboard results represent performance on competition evaluation subsets rather than a direct estimate of performance on future production data.

### Final Modeling Decision

The existing LightGBM configuration was retained after external evaluation.

No additional hyperparameter tuning was performed using leaderboard feedback.

This decision avoided further adaptation to the competition's hidden evaluation results.

---

## 8. Project Structure

```text
Santander-Customer-Transaction-Prediction/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── submissions/
│       └── lightgbm_submission.csv
├── models/
│   └── lightgbm_final.joblib
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_final_model.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

### Directory Description

| Directory / File | Purpose |
|---|---|
| `data/raw/` | Original datasets |
| `data/submissions/` | Generated competition predictions |
| `models/` | Serialized model artifacts |
| `notebooks/01_data_understanding.ipynb` | Data quality, model experiments, and validation |
| `notebooks/02_final_model.ipynb` | Final training, serialization, and inference |
| `requirements.txt` | Project dependencies |
| `README.md` | Documentation |
| `.gitignore` | Files excluded from version control |

The `data/`, `models/`, and `.venv/` directories are excluded from Git.

Model artifacts and prediction files are generated locally rather than committed to the repository.

---

## 9. Environment Setup

### Requirements

- Python 3.12
- Git
- Jupyter or VS Code with Python and Jupyter extensions

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Santander-Customer-Transaction-Prediction
```

Replace `<repository-url>` with the repository URL after publishing the project.

### Step 2: Create a Virtual Environment

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

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The project uses Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, LightGBM, and Jupyter-related packages.

Joblib is also used for model serialization and is currently available through the installed dependencies. Its explicit declaration as a direct dependency remains a repository cleanup task.

### Step 4: Download the Dataset

Download the original dataset from:

[Zenodo — Santander Customer Transaction Prediction](https://zenodo.org/records/7980290)

Extract the files and place the CSV datasets in:

`data/raw/`

The repository does not include the original datasets.

### Step 5: Run the Notebooks

Open the project in VS Code.

Select the Python interpreter from the `.venv` environment as the notebook kernel.

Run the notebooks in the following order:

**1. Data Understanding and Modeling**

`notebooks/01_data_understanding.ipynb`

This notebook contains data quality checks, baseline modeling, LightGBM experiments, and cross-validation.

**2. Final Model Training and Inference**

`notebooks/02_final_model.ipynb`

This notebook retrains the selected configuration, serializes the model, verifies the reloaded predictions, and generates competition test predictions.

**Working-directory note:** The notebooks currently contain relative-path assumptions. When executed with the working directory set to `notebooks/`, data paths should be resolved relative to the parent project directory. Verify the working directory using `Path.cwd()` before running the notebooks.

Path handling is scheduled for final reproducibility cleanup.

---

## 10. Reproducibility

The project incorporates several reproducibility practices:

- Fixed random seeds for data splitting and model training.
- Consistent train-validation partitions for model comparison.
- Stratified cross-validation.
- Explicitly recorded model hyperparameters.
- Versioned project dependencies.
- Model serialization and reloading verification.
- Documented experimental results and modeling decisions.
- Separation of source code from raw datasets and generated model artifacts.

The final model can be regenerated from the labeled training dataset using the final-model notebook.

However, a complete clean-environment reproduction has not yet been independently verified.

---

## 11. Limitations

The following limitations should be considered when interpreting the results:

**Anonymized Features**

The predictors are anonymized, limiting domain-level interpretation and the ability to explain individual feature effects in business terms.

**Validation-Based Model Selection**

LightGBM hyperparameters were selected through holdout validation experiments. The resulting internal performance estimates are not entirely independent of the model-selection process.

**Limited Cross-Validation**

Three-fold cross-validation provided preliminary evidence of stability, but it does not establish performance robustness under all possible data partitions.

**Competition-Based External Evaluation**

The Kaggle leaderboard scores are based on hidden competition evaluation subsets. They do not guarantee equivalent performance under future changes in customer behavior or data distributions.

**Probability Calibration**

Probability calibration has not been evaluated. Although the model produces valid numerical probabilities, their agreement with actual event frequencies remains unverified.

**Operational Validation**

The model has been tested for local serialization consistency and batch inference. A dedicated inference API, deployment pipeline, and automated test suite have not been implemented.

**Environment Reproducibility**

The project has been executed successfully in its development environment, but clean-environment reproduction and fully portable notebook path handling remain to be verified.

---

## 12. Conclusion

This project demonstrated that nonlinear modeling with LightGBM improved customer transaction prediction performance compared with a Logistic Regression baseline.

The selected LightGBM classifier achieved:

- **0.89450** holdout validation ROC-AUC
- **0.89310** mean three-fold cross-validation ROC-AUC
- **0.89217** Kaggle Private Leaderboard ROC-AUC

The agreement between internal and external evaluation results provides encouraging evidence of the modeling approach's generalization capability on the competition data.

The workflow also demonstrated the importance of controlling model complexity, evaluating overfitting, documenting experimental decisions, and verifying model serialization and inference.

The final result is a trained and externally evaluated classification model, with remaining engineering work focused on improving execution portability and repository reproducibility.

---

## 13. Current Project Status

**Completed:**

- Data quality assessment
- Target distribution analysis
- Baseline modeling
- LightGBM experimentation
- Model complexity analysis
- Cross-validation
- Final model training
- Model serialization and reload verification
- Batch inference validation
- Kaggle external evaluation
- Git version control and project documentation

**Remaining Engineering Tasks:**

- Explicitly register Joblib as a direct dependency
- Standardize notebook path handling
- Verify execution in a clean environment
- Complete repository hygiene checks
- Publish the finalized repository on GitHub