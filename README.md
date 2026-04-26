# AI Model Monitoring & Incident Explanation Assistant

Production-style ML monitoring project that tracks model performance, prediction drift, feature drift, data quality issues, SQL validation, SHAP explainability, and incident summaries across simulated production batches.

## Project Overview

This project simulates how a deployed fraud-risk model can be monitored after production release.

It trains a baseline XGBoost classifier, evaluates incoming batch data, detects drift and anomalies, validates batch-level patterns with SQL/SQLite, explains feature impact changes using SHAP, logs experiments with MLflow, and presents the results in an interactive Streamlit dashboard.

## Key Features

- Trained a baseline fraud-risk classification model using XGBoost
- Logged model metrics, parameters, confusion matrix, and artifacts with MLflow
- Monitored production-style batches for model performance degradation, prediction drift, feature distribution drift, missing values, and duplicate records
- Implemented automated anomaly flags for drift, risk-score movement, and data quality issues
- Loaded baseline and production batch data into SQLite using `src/sqlite_loader.py`
- Added SQL validation queries in `sql/monitoring_queries.sql` to check batch volume, fraud rate, risk-score movement, data quality, and high-risk transaction counts
- Used SHAP to compare baseline and drifted batch behavior
- Generated structured incident summaries with severity, root-cause hypothesis, impacted features, model risk, and retraining recommendations
- Built an interactive Streamlit dashboard for batch-level monitoring, SQL validation, explainability, and incident summaries

## Tech Stack

Python, SQL, SQLite, pandas, scikit-learn, XGBoost, SHAP, MLflow, Streamlit, Plotly

## Project Structure

```text
ai-model-monitoring-incident-assistant/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── baseline/
│   │   └── train_baseline.csv
│   └── batches/
│       ├── batch_01_normal.csv
│       ├── batch_02_feature_drift.csv
│       ├── batch_03_prediction_drift.csv
│       └── batch_04_quality_issue.csv
├── outputs/
│   ├── batch_monitoring_report.csv
│   ├── shap/
│   └── incident_summaries/
├── sql/
│   └── monitoring_queries.sql
├── src/
│   ├── data_loader.py
│   ├── sqlite_loader.py
│   ├── train_model.py
│   ├── monitoring_checks.py
│   ├── shap_analysis.py
│   └── incident_summary.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The project uses synthetic fraud-risk transaction data designed for ML monitoring.

It includes:

- one baseline training dataset
- one normal production batch
- one feature drift batch
- one prediction drift batch
- one data quality issue batch

Key fields include:

- transaction amount
- account age
- chargeback history
- login velocity
- device trust score
- IP risk score
- merchant risk score
- channel
- country
- payment method
- fraud label

## Workflow

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Load data into SQLite and run SQL validation

```bash
python src/sqlite_loader.py
```

This creates a local SQLite database and validates batch-level data using SQL queries.

It checks:

- baseline vs production batch row counts
- batch-level fraud rates
- average simulated fraud risk
- transaction amount patterns
- login velocity movement
- data quality checks
- high-risk transaction counts

The SQL queries are also available in:

```text
sql/monitoring_queries.sql
```

### 3. Train baseline model

```bash
python src/train_model.py
```

This trains the baseline XGBoost model and logs metrics, parameters, confusion matrix, and artifacts using MLflow.

### 4. Run batch monitoring checks

```bash
python src/monitoring_checks.py
```

This evaluates each production batch and creates:

```text
outputs/batch_monitoring_report.csv
```

### 5. Run SHAP analysis

```bash
python src/shap_analysis.py
```

This generates baseline vs drifted batch SHAP comparison files and summary plots.

### 6. Generate incident summaries

```bash
python src/incident_summary.py
```

This creates incident summaries for anomalous batches.

The incident-summary module is OpenAI GPT API-ready and also includes fallback logic, so the project can run without exposing or requiring an API key.

### 7. Launch Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

## SQL and SQLite Validation

The project uses SQLite to make SQL validation part of the monitoring workflow.

`src/sqlite_loader.py` loads the baseline and production batch CSV files into a local SQLite table named:

```text
transactions
```

The SQL validation layer checks:

- total rows by data source
- batch-level fraud rate
- average simulated fraud probability
- average transaction amount
- login velocity movement
- device trust score
- IP risk score
- merchant risk score
- missing values by batch
- high-risk transaction counts

The SQL query file is included for visibility:

```text
sql/monitoring_queries.sql
```

The SQLite database is generated locally and should not be uploaded to GitHub.

## MLflow Tracking

The training workflow logs:

- accuracy
- ROC-AUC
- F1-score
- precision
- recall
- model parameters
- confusion matrix
- trained model artifact

The monitoring workflow also logs batch-level metrics and drift signals for production-style batches.

To open MLflow locally:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

## Monitoring Results

The project evaluates four simulated production batches:

| Batch | Scenario | Result |
|---|---|---|
| `batch_01_normal.csv` | Normal batch | No anomaly detected |
| `batch_02_feature_drift.csv` | Feature drift | Anomaly detected |
| `batch_03_prediction_drift.csv` | Prediction drift | Anomaly detected |
| `batch_04_quality_issue.csv` | Data quality issue | Anomaly detected |

## Key Insights

- The monitoring system correctly identified `batch_01_normal.csv` as a stable batch with no anomaly detected.
- `batch_02_feature_drift.csv` showed the strongest drift pattern, with `login_velocity_24h` emerging as the most important changed feature in the SHAP comparison.
- SQL validation showed higher high-risk transaction concentration in drifted and quality-issue batches.
- The average predicted risk increased in drifted batches compared with the baseline, indicating prediction behavior changed after batch-level distribution shifts.
- `batch_03_prediction_drift.csv` was flagged due to risk-score movement and feature drift, showing how prediction drift can appear even when data quality is not the main issue.
- `batch_04_quality_issue.csv` was flagged because monitoring checks detected data quality issues along with drift signals.
- SHAP comparison showed that feature impact changed between baseline and drifted behavior, helping explain why the model produced different risk patterns.
- The incident-summary module converted monitoring outputs into readable investigation notes covering severity, root-cause hypothesis, impacted features, model risk, and retraining recommendation.

## SHAP Explainability

SHAP analysis compares baseline model behavior against the drifted batch.

The largest SHAP impact change was observed in:

```text
login_velocity_24h
```

This indicates that login velocity became more influential in the drifted batch compared with the baseline model behavior.

## Dashboard

The Streamlit dashboard includes:

- batch selector
- model performance KPIs
- prediction drift flags
- feature drift flags
- data quality issue flags
- anomaly status
- prediction risk comparison
- feature drift charts
- SQL validation summary
- high-risk transaction count chart
- model performance trend charts
- SHAP impact comparison
- SHAP summary plots
- incident summaries for anomalous batches

## Incident Summary Module

The incident summary module generates structured investigation notes for anomalous batches.

Each summary includes:

- incident summary
- root-cause hypothesis
- impacted features
- business / model risk
- recommended action
- retraining recommendation

The module has an OpenAI GPT API integration path and fallback logic. This makes the project reproducible without requiring an API key while still showing how GPT-based summaries can be integrated.

