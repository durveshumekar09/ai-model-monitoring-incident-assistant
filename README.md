# AI Model Monitoring & Incident Explanation Assistant

Production-style ML monitoring project that tracks model performance, prediction drift, feature drift, data quality issues, SHAP explainability, and incident summaries across simulated production batches.

## Project Overview

This project simulates how a deployed fraud-risk model can be monitored after production release.

It trains a baseline XGBoost classifier, evaluates incoming batch data, detects drift and anomalies, explains feature impact changes using SHAP, logs experiments with MLflow, and presents the results in an interactive Streamlit dashboard.

## Key Features

- Trained a baseline fraud-risk classification model using XGBoost
- Logged model metrics, parameters, confusion matrix, and artifacts with MLflow
- Monitored production-style batches for model performance degradation, prediction drift, feature distribution drift, missing values, and duplicate records
- Implemented automated anomaly flags for drift, risk-score movement, and data quality issues
- Used SHAP to compare baseline and drifted batch behavior
- Generated structured incident summaries with severity, root-cause hypothesis, impacted features, model risk, and retraining recommendations
- Built an interactive Streamlit dashboard for batch-level monitoring and explanation

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, MLflow, Streamlit, Plotly, SQLite

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
├── src/
│   ├── data_loader.py
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

### 2. Train baseline model

```bash
python src/train_model.py
```

This trains the baseline XGBoost model and logs metrics, parameters, confusion matrix, and artifacts using MLflow.

### 3. Run batch monitoring checks

```bash
python src/monitoring_checks.py
```

This evaluates each production batch and creates:

```text
outputs/batch_monitoring_report.csv
```

### 4. Run SHAP analysis

```bash
python src/shap_analysis.py
```

This generates baseline vs drifted batch SHAP comparison files and summary plots.

### 5. Generate incident summaries

```bash
python src/incident_summary.py
```

This creates incident summaries for anomalous batches.

The incident-summary module is OpenAI GPT API-ready and also includes fallback logic, so the project can run without exposing or requiring an API key.

### 6. Launch Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

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

