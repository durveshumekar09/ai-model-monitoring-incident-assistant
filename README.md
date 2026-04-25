# AI Model Monitoring & Incident Explanation Assistant

Production-style ML monitoring project that tracks model performance, prediction drift, feature drift, data quality issues, SHAP explainability, and incident summaries across simulated production batches.

## Project Overview

This project simulates how a deployed fraud-risk model can be monitored after production release.

It trains a baseline XGBoost classifier, evaluates incoming batch data, detects drift and anomalies, explains feature impact changes using SHAP, logs experiments with MLflow, and presents the results in an interactive Streamlit dashboard.

## Key Features

- Trained a baseline fraud-risk classification model using XGBoost
- Logged model metrics, parameters, confusion matrix, and artifacts with MLflow
- Monitored production-style batches for:
  - model performance degradation
  - prediction drift
  - feature distribution drift
  - missing values
  - duplicate records
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
