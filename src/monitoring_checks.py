from pathlib import Path

import joblib
import mlflow
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import load_baseline_data, load_all_batches
from train_model import prepare_features, DROP_COLUMNS, TARGET


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "fraud_xgb_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MONITORING_REPORT_PATH = OUTPUT_DIR / "batch_monitoring_report.csv"


# ---------------------------------------------------------
# Monitoring settings
# ---------------------------------------------------------

# Same fraud threshold used during training
FRAUD_THRESHOLD = 0.30

# If feature drift score crosses this value, we mark drift as detected
FEATURE_DRIFT_THRESHOLD = 0.15

# If average fraud risk changes by this much, we flag prediction drift
PREDICTION_DRIFT_THRESHOLD = 0.05

# If more than 5% values are missing, we flag data quality issue
MISSING_VALUE_THRESHOLD = 0.05


def calculate_feature_drift(baseline_df, batch_df, numeric_cols):
    """
    Compare numeric feature distributions between baseline data and one batch.

    We use KS test here:
    - Low drift score means baseline and batch look similar.
    - High drift score means the feature distribution changed.
    """

    drift_results = {}

    for col in numeric_cols:
        baseline_values = baseline_df[col].dropna()
        batch_values = batch_df[col].dropna()

        # Skip if column has no usable values
        if baseline_values.empty or batch_values.empty:
            drift_results[col] = 0
            continue

        # KS statistic is used as simple feature drift score
        ks_statistic, _ = ks_2samp(baseline_values, batch_values)

        drift_results[col] = ks_statistic

    return drift_results


def calculate_data_quality(batch_df):
    """
    Check basic data quality issues in a batch.

    We calculate:
    - missing value percentage
    - duplicate row count
    """

    total_cells = batch_df.shape[0] * batch_df.shape[1]

    missing_cells = batch_df.isna().sum().sum()
    missing_percentage = missing_cells / total_cells

    duplicate_rows = batch_df.duplicated().sum()

    return missing_percentage, duplicate_rows


def evaluate_batch(model, batch_name, batch_df, baseline_df, baseline_avg_risk, numeric_cols):
    """
    Evaluate one production batch.

    This function checks:
    1. Model performance
    2. Prediction drift
    3. Feature drift
    4. Data quality issues
    """

    # Prepare features and target for this batch
    X_batch = batch_df.drop(columns=DROP_COLUMNS + [TARGET])
    y_batch = batch_df[TARGET]

    # Predict fraud probabilities
    y_proba = model.predict_proba(X_batch)[:, 1]

    # Convert probabilities into fraud / non-fraud prediction
    y_pred = (y_proba >= FRAUD_THRESHOLD).astype(int)

    # ---------------------------------------------------------
    # Model performance metrics
    # ---------------------------------------------------------

    metrics = {
        "accuracy": accuracy_score(y_batch, y_pred),
        "roc_auc": roc_auc_score(y_batch, y_proba),
        "f1_score": f1_score(y_batch, y_pred, zero_division=0),
        "precision": precision_score(y_batch, y_pred, zero_division=0),
        "recall": recall_score(y_batch, y_pred, zero_division=0),
    }

    # ---------------------------------------------------------
    # Prediction drift check
    # ---------------------------------------------------------

    batch_avg_risk = y_proba.mean()
    prediction_drift_score = abs(batch_avg_risk - baseline_avg_risk)

    prediction_drift_detected = prediction_drift_score >= PREDICTION_DRIFT_THRESHOLD

    # ---------------------------------------------------------
    # Feature drift check
    # ---------------------------------------------------------

    feature_drift_scores = calculate_feature_drift(
        baseline_df=baseline_df,
        batch_df=batch_df,
        numeric_cols=numeric_cols,
    )

    max_feature_drift_score = max(feature_drift_scores.values())

    feature_drift_detected = max_feature_drift_score >= FEATURE_DRIFT_THRESHOLD

    # Get the feature with the highest drift
    most_drifted_feature = max(feature_drift_scores, key=feature_drift_scores.get)

    # ---------------------------------------------------------
    # Data quality check
    # ---------------------------------------------------------

    missing_percentage, duplicate_rows = calculate_data_quality(batch_df)

    data_quality_issue_detected = (
        missing_percentage >= MISSING_VALUE_THRESHOLD or duplicate_rows > 0
    )

    # ---------------------------------------------------------
    # Overall anomaly flag
    # ---------------------------------------------------------

    anomaly_detected = (
        prediction_drift_detected
        or feature_drift_detected
        or data_quality_issue_detected
    )

    # ---------------------------------------------------------
    # Create one summary row for this batch
    # ---------------------------------------------------------

    batch_report = {
        "batch_name": batch_name,
        "rows": batch_df.shape[0],
        "fraud_rate_actual": y_batch.mean(),
        "avg_predicted_risk": batch_avg_risk,
        "baseline_avg_risk": baseline_avg_risk,
        "prediction_drift_score": prediction_drift_score,
        "prediction_drift_detected": prediction_drift_detected,
        "max_feature_drift_score": max_feature_drift_score,
        "most_drifted_feature": most_drifted_feature,
        "feature_drift_detected": feature_drift_detected,
        "missing_percentage": missing_percentage,
        "duplicate_rows": duplicate_rows,
        "data_quality_issue_detected": data_quality_issue_detected,
        "anomaly_detected": anomaly_detected,
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "f1_score": metrics["f1_score"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }

    return batch_report


def run_monitoring_checks():
    """
    Run monitoring checks on all production batches.

    This function:
    1. Loads the trained model.
    2. Loads baseline data and batch data.
    3. Calculates baseline average fraud risk.
    4. Evaluates each batch.
    5. Logs batch-level metrics to MLflow.
    6. Saves final monitoring report as CSV.
    """

    # ---------------------------------------------------------
    # Load model and data
    # ---------------------------------------------------------

    model = joblib.load(MODEL_PATH)

    baseline_df = load_baseline_data()
    batches = load_all_batches()

    # Prepare baseline features
    X_baseline, y_baseline, categorical_cols, numeric_cols = prepare_features(baseline_df)

    # Calculate baseline predicted risk
    baseline_proba = model.predict_proba(X_baseline)[:, 1]
    baseline_avg_risk = baseline_proba.mean()

    print("Baseline average predicted risk:", round(baseline_avg_risk, 4))

    all_reports = []

    # MLflow experiment for monitoring runs
    mlflow.set_experiment("AI Model Monitoring - Batch Checks")

    # ---------------------------------------------------------
    # Evaluate each production batch
    # ---------------------------------------------------------

    for batch_name, batch_df in batches.items():
        print("\nEvaluating:", batch_name)

        report = evaluate_batch(
            model=model,
            batch_name=batch_name,
            batch_df=batch_df,
            baseline_df=baseline_df,
            baseline_avg_risk=baseline_avg_risk,
            numeric_cols=numeric_cols,
        )

        all_reports.append(report)

        # ---------------------------------------------------------
        # Log each batch as a separate MLflow run
        # ---------------------------------------------------------

        with mlflow.start_run(run_name=batch_name):
            mlflow.log_param("batch_name", batch_name)
            mlflow.log_param("rows", report["rows"])
            mlflow.log_param("most_drifted_feature", report["most_drifted_feature"])

            mlflow.log_metric("accuracy", report["accuracy"])
            mlflow.log_metric("roc_auc", report["roc_auc"])
            mlflow.log_metric("f1_score", report["f1_score"])
            mlflow.log_metric("precision", report["precision"])
            mlflow.log_metric("recall", report["recall"])

            mlflow.log_metric(
                "prediction_drift_score",
                report["prediction_drift_score"],
            )
            mlflow.log_metric(
                "max_feature_drift_score",
                report["max_feature_drift_score"],
            )
            mlflow.log_metric(
                "missing_percentage",
                report["missing_percentage"],
            )
            mlflow.log_metric(
                "duplicate_rows",
                report["duplicate_rows"],
            )

        print("Anomaly detected:", report["anomaly_detected"])
        print("Most drifted feature:", report["most_drifted_feature"])
        print("Prediction drift score:", round(report["prediction_drift_score"], 4))
        print("Max feature drift score:", round(report["max_feature_drift_score"], 4))

    # ---------------------------------------------------------
    # Save final monitoring report
    # ---------------------------------------------------------

    monitoring_report = pd.DataFrame(all_reports)
    monitoring_report.to_csv(MONITORING_REPORT_PATH, index=False)

    print("\nMonitoring completed successfully")
    print("Report saved at:", MONITORING_REPORT_PATH)

    return monitoring_report


# This allows the file to run directly using:
# python src/monitoring_checks.py
if __name__ == "__main__":
    run_monitoring_checks()