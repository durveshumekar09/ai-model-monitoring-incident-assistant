from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

# Main project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Output files created by earlier project scripts
MONITORING_REPORT_PATH = BASE_DIR / "outputs" / "batch_monitoring_report.csv"
SHAP_DIR = BASE_DIR / "outputs" / "shap"
SUMMARY_DIR = BASE_DIR / "outputs" / "incident_summaries"

# SQLite database created by src/sqlite_loader.py
SQLITE_DB_PATH = BASE_DIR / "database" / "monitoring_data.sqlite"


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Model Monitoring Dashboard",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

@st.cache_data
def load_monitoring_report():
    """
    Load the batch monitoring report created by monitoring_checks.py.

    This report contains:
    - model performance metrics
    - prediction drift scores
    - feature drift scores
    - data quality checks
    - anomaly flags
    """

    return pd.read_csv(MONITORING_REPORT_PATH)


@st.cache_data
def load_shap_comparison():
    """
    Load the SHAP comparison file created by shap_analysis.py.

    This file compares feature impact between:
    - baseline data
    - drifted production batch
    """

    shap_path = SHAP_DIR / "baseline_vs_drift_shap_comparison.csv"
    return pd.read_csv(shap_path)


@st.cache_data
def load_sql_validation_summary():
    """
    Load SQL validation output from the SQLite database.

    This makes the SQL + SQLite part visible in the dashboard.
    The database is created by running:
    python src/sqlite_loader.py
    """

    if not SQLITE_DB_PATH.exists():
        return None

    conn = sqlite3.connect(SQLITE_DB_PATH)

    query = """
    SELECT
        batch_file_name,
        batch_type,
        COUNT(*) AS total_transactions,
        ROUND(AVG(label_fraud), 4) AS fraud_rate,
        ROUND(AVG(fraud_probability_simulated), 4) AS avg_simulated_risk,
        ROUND(AVG(amount), 2) AS avg_transaction_amount,
        ROUND(AVG(login_velocity_24h), 2) AS avg_login_velocity,
        ROUND(AVG(device_trust_score), 4) AS avg_device_trust_score,
        ROUND(AVG(ip_risk_score), 4) AS avg_ip_risk_score,
        ROUND(AVG(merchant_risk_score), 4) AS avg_merchant_risk_score
    FROM transactions
    GROUP BY batch_file_name, batch_type
    ORDER BY batch_file_name;
    """

    sql_df = pd.read_sql_query(query, conn)
    conn.close()

    return sql_df


@st.cache_data
def load_high_risk_sql_summary():
    """
    Load high-risk transaction counts using SQL.

    This query helps validate risk concentration across batches.
    """

    if not SQLITE_DB_PATH.exists():
        return None

    conn = sqlite3.connect(SQLITE_DB_PATH)

    query = """
    SELECT
        batch_file_name,
        COUNT(*) AS high_risk_transactions,
        ROUND(AVG(fraud_probability_simulated), 4) AS avg_simulated_risk
    FROM transactions
    WHERE fraud_probability_simulated >= 0.30
    GROUP BY batch_file_name
    ORDER BY high_risk_transactions DESC;
    """

    high_risk_df = pd.read_sql_query(query, conn)
    conn.close()

    return high_risk_df


def format_flag(value):
    """
    Convert True/False values into readable status labels.
    """

    return "Detected" if value else "Not Detected"


def get_incident_summary(batch_name):
    """
    Load incident summary text for the selected batch.

    Normal batches may not have a summary because no anomaly was detected.
    """

    summary_file_name = batch_name.replace(".csv", "_incident_summary.txt")
    summary_path = SUMMARY_DIR / summary_file_name

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as file:
            return file.read()

    return None


# ---------------------------------------------------------
# Load project outputs
# ---------------------------------------------------------

try:
    monitoring_df = load_monitoring_report()
    shap_df = load_shap_comparison()
except FileNotFoundError:
    st.error(
        "Required output files were not found. Please run these scripts first: "
        "train_model.py, monitoring_checks.py, shap_analysis.py, and incident_summary.py."
    )
    st.stop()


# ---------------------------------------------------------
# Dashboard title
# ---------------------------------------------------------

st.title("AI Model Monitoring & Incident Explanation Assistant")

st.markdown(
    """
This dashboard tracks **model performance**, **prediction drift**, **feature drift**, 
**data quality changes**, **SHAP feature impact**, **SQL validation**, and **incident summaries** across simulated production batches.
"""
)


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

st.sidebar.header("Dashboard Controls")

batch_names = monitoring_df["batch_name"].tolist()

selected_batch = st.sidebar.selectbox(
    "Select production batch",
    batch_names,
)

top_n_features = st.sidebar.slider(
    "Top SHAP features to display",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
)

show_raw_tables = st.sidebar.checkbox(
    "Show full technical tables",
    value=True,
)

batch_row = monitoring_df[monitoring_df["batch_name"] == selected_batch].iloc[0]


# ---------------------------------------------------------
# Overall project health section
# ---------------------------------------------------------

st.subheader("Overall Monitoring Overview")

total_batches = len(monitoring_df)
anomalous_batches = int(monitoring_df["anomaly_detected"].sum())
normal_batches = total_batches - anomalous_batches

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

overview_col1.metric("Total Batches", total_batches)
overview_col2.metric("Normal Batches", normal_batches)
overview_col3.metric("Anomalous Batches", anomalous_batches)
overview_col4.metric(
    "Highest Drift Score",
    round(monitoring_df["max_feature_drift_score"].max(), 4),
)


# ---------------------------------------------------------
# Selected batch KPI cards
# ---------------------------------------------------------

st.subheader(f"Selected Batch: {selected_batch}")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

kpi_col1.metric("Rows", int(batch_row["rows"]))
kpi_col2.metric("Actual Fraud Rate", round(batch_row["fraud_rate_actual"], 4))
kpi_col3.metric("Avg Predicted Risk", round(batch_row["avg_predicted_risk"], 4))
kpi_col4.metric("Baseline Avg Risk", round(batch_row["baseline_avg_risk"], 4))

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

metric_col1.metric("Accuracy", round(batch_row["accuracy"], 4))
metric_col2.metric("ROC-AUC", round(batch_row["roc_auc"], 4))
metric_col3.metric("F1-score", round(batch_row["f1_score"], 4))
metric_col4.metric("Precision", round(batch_row["precision"], 4))
metric_col5.metric("Recall", round(batch_row["recall"], 4))


# ---------------------------------------------------------
# Monitoring flags section
# ---------------------------------------------------------

st.subheader("Monitoring Flags")

flag_col1, flag_col2, flag_col3, flag_col4 = st.columns(4)

with flag_col1:
    st.write("Prediction Drift")
    if batch_row["prediction_drift_detected"]:
        st.warning("Detected")
    else:
        st.success("Not Detected")

with flag_col2:
    st.write("Feature Drift")
    if batch_row["feature_drift_detected"]:
        st.warning("Detected")
    else:
        st.success("Not Detected")

with flag_col3:
    st.write("Data Quality Issue")
    if batch_row["data_quality_issue_detected"]:
        st.warning("Detected")
    else:
        st.success("Not Detected")

with flag_col4:
    st.write("Overall Anomaly")
    if batch_row["anomaly_detected"]:
        st.error("Anomaly Detected")
    else:
        st.success("Normal Batch")


# ---------------------------------------------------------
# Drift and data quality details
# ---------------------------------------------------------

st.subheader("Drift and Data Quality Details")

detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)

detail_col1.metric(
    "Prediction Drift Score",
    round(batch_row["prediction_drift_score"], 4),
)

detail_col2.metric(
    "Max Feature Drift Score",
    round(batch_row["max_feature_drift_score"], 4),
)

detail_col3.metric(
    "Missing Value %",
    round(batch_row["missing_percentage"] * 100, 2),
)

detail_col4.metric(
    "Duplicate Rows",
    int(batch_row["duplicate_rows"]),
)

st.info(
    f"Most drifted feature for this batch: **{batch_row['most_drifted_feature']}**"
)


# ---------------------------------------------------------
# Batch comparison charts
# ---------------------------------------------------------

st.subheader("Batch Comparison Charts")

chart_col1, chart_col2 = st.columns(2)

risk_chart_df = monitoring_df[
    ["batch_name", "avg_predicted_risk", "baseline_avg_risk"]
].melt(
    id_vars="batch_name",
    value_vars=["avg_predicted_risk", "baseline_avg_risk"],
    var_name="risk_type",
    value_name="risk_value",
)

fig_risk = px.bar(
    risk_chart_df,
    x="batch_name",
    y="risk_value",
    color="risk_type",
    barmode="group",
    title="Average Predicted Risk vs Baseline Risk",
)

chart_col1.plotly_chart(fig_risk, use_container_width=True)

fig_drift = px.bar(
    monitoring_df,
    x="batch_name",
    y="max_feature_drift_score",
    color="anomaly_detected",
    title="Maximum Feature Drift Score by Batch",
    hover_data=["most_drifted_feature", "prediction_drift_score"],
)

chart_col2.plotly_chart(fig_drift, use_container_width=True)


# ---------------------------------------------------------
# SQL validation section
# ---------------------------------------------------------

st.subheader("SQL Validation Summary")

sql_validation_df = load_sql_validation_summary()
high_risk_df = load_high_risk_sql_summary()

if sql_validation_df is not None:
    st.write(
        "This section uses SQLite + SQL queries to validate batch volume, fraud rate, simulated risk, and transaction behavior."
    )

    st.dataframe(
        sql_validation_df,
        use_container_width=True,
    )

    if high_risk_df is not None and not high_risk_df.empty:
        fig_high_risk = px.bar(
            high_risk_df,
            x="batch_file_name",
            y="high_risk_transactions",
            title="High-Risk Transactions by Batch from SQL Query",
            hover_data=["avg_simulated_risk"],
        )

        st.plotly_chart(fig_high_risk, use_container_width=True)

        st.dataframe(
            high_risk_df,
            use_container_width=True,
        )
else:
    st.info(
        "SQLite database not found. Run `python src/sqlite_loader.py` to generate SQL validation outputs."
    )


# ---------------------------------------------------------
# Model performance comparison
# ---------------------------------------------------------

st.subheader("Model Performance Across Production Batches")

performance_metrics = ["accuracy", "roc_auc", "f1_score", "precision", "recall"]

performance_df = monitoring_df[["batch_name"] + performance_metrics].melt(
    id_vars="batch_name",
    value_vars=performance_metrics,
    var_name="metric",
    value_name="value",
)

fig_performance = px.line(
    performance_df,
    x="batch_name",
    y="value",
    color="metric",
    markers=True,
    title="Batch-Level Model Performance Metrics",
)

st.plotly_chart(fig_performance, use_container_width=True)


# ---------------------------------------------------------
# SHAP analysis section
# ---------------------------------------------------------

st.subheader("SHAP Baseline vs Drifted Batch Comparison")

st.markdown(
    """
SHAP helps explain how much each feature influenced the model.  
This section compares feature impact between the baseline data and the drifted batch.
"""
)

top_shap_df = shap_df.head(top_n_features)

shap_col1, shap_col2 = st.columns([1.2, 1])

fig_shap = px.bar(
    top_shap_df.sort_values("shap_impact_change"),
    x="shap_impact_change",
    y="feature",
    orientation="h",
    title=f"Top {top_n_features} SHAP Impact Changes",
    hover_data=[
        "mean_abs_shap_baseline",
        "mean_abs_shap_drift_batch",
    ],
)

shap_col1.plotly_chart(fig_shap, use_container_width=True)

shap_col2.dataframe(
    top_shap_df[
        [
            "feature",
            "mean_abs_shap_baseline",
            "mean_abs_shap_drift_batch",
            "shap_impact_change",
        ]
    ],
    use_container_width=True,
)


# ---------------------------------------------------------
# SHAP summary plots
# ---------------------------------------------------------

st.subheader("SHAP Summary Plots")

plot_col1, plot_col2 = st.columns(2)

baseline_plot = SHAP_DIR / "baseline_shap_summary.png"
drift_plot = SHAP_DIR / "drift_batch_shap_summary.png"

with plot_col1:
    st.markdown("**Baseline SHAP Summary**")
    if baseline_plot.exists():
        st.image(str(baseline_plot), use_container_width=True)
    else:
        st.info("Baseline SHAP plot not found.")

with plot_col2:
    st.markdown("**Drift Batch SHAP Summary**")
    if drift_plot.exists():
        st.image(str(drift_plot), use_container_width=True)
    else:
        st.info("Drift batch SHAP plot not found.")


# ---------------------------------------------------------
# Incident summary section
# ---------------------------------------------------------

st.subheader("Incident Summary")

summary_text = get_incident_summary(selected_batch)

if summary_text:
    st.text_area(
        "Generated incident summary",
        summary_text,
        height=360,
    )
else:
    st.success(
        "No incident summary was generated for this batch because no anomaly was detected."
    )

st.caption(
    "The incident summary module is OpenAI GPT API-ready and also includes fallback logic "
    "so the project can run without exposing or requiring an API key."
)


# ---------------------------------------------------------
# Optional full technical tables
# ---------------------------------------------------------

if show_raw_tables:
    st.subheader("Full Monitoring Report")

    display_df = monitoring_df.copy()

    boolean_cols = [
        "prediction_drift_detected",
        "feature_drift_detected",
        "data_quality_issue_detected",
        "anomaly_detected",
    ]

    for col in boolean_cols:
        display_df[col] = display_df[col].apply(format_flag)

    st.dataframe(display_df, use_container_width=True)

    st.subheader("Full SHAP Comparison Table")

    st.dataframe(shap_df, use_container_width=True)


# ---------------------------------------------------------
# Project coverage checklist
# ---------------------------------------------------------

st.subheader("Project Coverage")

coverage_col1, coverage_col2 = st.columns(2)

with coverage_col1:
    st.markdown(
        """
**Monitoring coverage**
- Model performance tracking
- Prediction drift detection
- Feature drift detection
- Data quality checks
- Batch-level anomaly flagging
- SQL validation with SQLite
"""
    )

with coverage_col2:
    st.markdown(
        """
**Explainability and incident workflow**
- SHAP baseline vs drift comparison
- Incident summary generation
- Severity and root-cause explanation
- Impacted feature identification
- Retraining recommendation output
"""
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.caption(
    "Built with Python, SQL, SQLite, pandas, scikit-learn, XGBoost, SHAP, MLflow, Streamlit, and Plotly."
)