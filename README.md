# AI Model Monitoring & Incident Explanation Assistant

Production-style ML monitoring project that tracks model performance degradation, prediction drift, feature drift, data quality issues, SQL validation checks, SHAP explainability, and incident summaries across simulated production batches.

---

# Project Overview

This project simulates how a deployed fraud-risk machine learning model can be monitored after production release.

It trains a baseline XGBoost classifier, evaluates incoming production-style batches, detects anomalies and drift patterns, validates data quality using SQL and SQLite, compares feature behavior using SHAP explainability, logs experiments with MLflow, and visualizes monitoring insights through an interactive Streamlit dashboard.

The project demonstrates practical skills across:

- Machine Learning
- MLOps and model monitoring
- Data quality validation
- SQL analytics
- Explainable AI (XAI)
- Production-style batch evaluation
- Monitoring dashboard development

---

# Key Features

- Trained a baseline fraud-risk classification model using XGBoost
- Logged metrics, parameters, confusion matrix, and artifacts using MLflow
- Monitored production-style batches for:
  - prediction drift
  - feature distribution drift
  - model performance degradation
  - missing values
  - duplicate records
- Implemented automated anomaly detection and drift flagging
- Added SQL validation workflows using SQLite
- Compared baseline vs drifted behavior using SHAP explainability
- Generated structured incident investigation summaries
- Built an interactive Streamlit dashboard for monitoring and explainability
- Included GPT-ready incident summarization workflow with offline fallback support

---

# Tech Stack

- Python
- SQL
- SQLite
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- MLflow
- Streamlit
- Plotly

---

# Project Structure

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

---

# Dataset

The project uses synthetic fraud-risk transaction data designed for ML monitoring simulations.

The dataset includes:

- one baseline training dataset
- one normal production batch
- one feature drift batch
- one prediction drift batch
- one data quality issue batch

## Key Features in the Dataset

- transaction amount
- account age
- chargeback history
- login velocity
- device trust score
- IP risk score
- merchant risk score
- payment channel
- country
- payment method
- fraud label

---

# End-to-End Workflow

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd ai-model-monitoring-incident-assistant
```

---

## 2. Create Virtual Environment

### Windows

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Load Data into SQLite

```bash
python src/sqlite_loader.py
```

This step:

- creates a local SQLite database
- loads baseline and production batch data
- enables SQL-based validation checks

The generated database file is stored locally and should not be uploaded to GitHub.

---

## 5. Train Baseline Model

```bash
python src/train_model.py
```

This workflow:

- trains the XGBoost baseline model
- evaluates classification performance
- logs experiments and artifacts with MLflow

---

## 6. Run Monitoring Checks

```bash
python src/monitoring_checks.py
```

This generates:

```text
outputs/batch_monitoring_report.csv
```

The monitoring pipeline evaluates:

- prediction drift
- feature drift
- model performance movement
- missing values
- duplicate records
- anomaly severity

---

## 7. Run SHAP Explainability Analysis

```bash
python src/shap_analysis.py
```

This step generates:

- SHAP comparison outputs
- feature importance comparison charts
- baseline vs drifted batch explainability analysis

---

## 8. Generate Incident Summaries

```bash
python src/incident_summary.py
```

This module generates structured incident investigation summaries for anomalous batches.

The workflow is OpenAI GPT API-ready and also includes fallback logic so the project can run without requiring an API key.

---

## 9. Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

# SQL + SQLite Validation Layer

The project integrates SQL validation into the ML monitoring workflow using SQLite.

`src/sqlite_loader.py` loads all datasets into a local SQLite table:

```text
transactions
```

The SQL validation layer evaluates:

- batch row counts
- fraud-rate movement
- average risk-score changes
- transaction amount distribution shifts
- login velocity movement
- device trust score changes
- IP risk score movement
- merchant risk score movement
- missing values
- high-risk transaction concentration

The SQL queries are available in:

```text
sql/monitoring_queries.sql
```

---

# MLflow Experiment Tracking

The project logs:

- accuracy
- ROC-AUC
- F1-score
- precision
- recall
- model parameters
- confusion matrix
- trained model artifacts
- batch-level monitoring metrics

## Launch MLflow UI

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

# Monitoring Results

The monitoring workflow evaluates four simulated production batches:

| Batch | Scenario | Result |
|---|---|---|
| `batch_01_normal.csv` | Normal production batch | Stable |
| `batch_02_feature_drift.csv` | Feature distribution drift | Anomaly detected |
| `batch_03_prediction_drift.csv` | Prediction behavior drift | Anomaly detected |
| `batch_04_quality_issue.csv` | Data quality issue | Anomaly detected |

---

# Key Insights

- The monitoring system correctly identified the normal batch as stable.
- Feature drift was strongest in `batch_02_feature_drift.csv`.
- SHAP analysis showed major feature-impact movement in:
  - `login_velocity_24h`
- Drifted batches showed increased predicted fraud-risk behavior.
- SQL validation identified higher concentrations of high-risk transactions in anomalous batches.
- Prediction drift was detected even when explicit data quality issues were limited.
- Data quality monitoring successfully identified missing-value anomalies and quality degradation patterns.
- Incident summaries converted technical monitoring outputs into readable investigation notes suitable for operational review.

---

# SHAP Explainability

The project uses SHAP explainability to compare baseline vs drifted model behavior.

The largest feature-impact movement was observed in:

```text
login_velocity_24h
```

This indicates that login velocity became significantly more influential in the drifted production batch.

---

# Streamlit Dashboard

The dashboard includes:

- batch selector
- monitoring KPIs
- anomaly status indicators
- prediction drift metrics
- feature drift metrics
- data quality alerts
- risk-score comparison charts
- feature drift visualizations
- SQL validation summaries
- high-risk transaction analysis
- SHAP explainability charts
- incident investigation summaries

---

# Incident Summary Module

The incident-summary module generates structured investigation notes for anomalous production batches.

Each summary includes:

- incident description
- severity assessment
- root-cause hypothesis
- impacted features
- business / model risk
- recommended action
- retraining recommendation

The module supports:

- OpenAI GPT integration
- offline fallback generation
- reproducible local execution without API dependency

---

# Future Improvements

Potential enhancements:

- real-time streaming batch monitoring
- AWS deployment
- automated retraining pipeline
- drift-threshold configuration UI
- alerting system integration
- Docker containerization
- CI/CD pipeline integration
- feature-store integration
- monitoring API endpoints using FastAPI

---

# Author

**Durvesh Umekar**

AI/ML • Model Monitoring • MLOps • SQL • Explainable AI • Streamlit
