from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "outputs"
SUMMARY_DIR = OUTPUT_DIR / "incident_summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

MONITORING_REPORT_PATH = OUTPUT_DIR / "batch_monitoring_report.csv"
SHAP_COMPARISON_PATH = OUTPUT_DIR / "shap" / "baseline_vs_drift_shap_comparison.csv"


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

# This loads OPENAI_API_KEY from a .env file if available.
# The .env file should NOT be uploaded to GitHub.
#
# Example .env:
# OPENAI_API_KEY=sk-your-real-api-key-here
load_dotenv()


def load_monitoring_report():
    """
    Load the batch monitoring report created by monitoring_checks.py.
    """

    return pd.read_csv(MONITORING_REPORT_PATH)


def load_shap_comparison():
    """
    Load SHAP comparison output created by shap_analysis.py.
    """

    return pd.read_csv(SHAP_COMPARISON_PATH)


def assign_severity(row):
    """
    Assign incident severity using simple monitoring logic.

    Severity logic:
    - High: two or more issue types detected
    - Medium: one issue type detected
    - Low: no major issue detected
    """

    issue_count = 0

    if row["prediction_drift_detected"]:
        issue_count += 1

    if row["feature_drift_detected"]:
        issue_count += 1

    if row["data_quality_issue_detected"]:
        issue_count += 1

    if issue_count >= 2:
        return "High"

    if issue_count == 1:
        return "Medium"

    return "Low"


def create_incident_prompt(batch_row, top_shap_changes):
    """
    Create the prompt that would be sent to OpenAI GPT.

    This prompt converts raw monitoring signals into a structured
    incident summary format.
    """

    prompt = f"""
You are an ML monitoring analyst.

Create a concise incident summary for the following production batch.

Batch Name:
{batch_row["batch_name"]}

Severity:
{batch_row["severity"]}

Monitoring Results:
- Rows: {batch_row["rows"]}
- Actual fraud rate: {batch_row["fraud_rate_actual"]:.4f}
- Average predicted risk: {batch_row["avg_predicted_risk"]:.4f}
- Baseline average risk: {batch_row["baseline_avg_risk"]:.4f}
- Prediction drift score: {batch_row["prediction_drift_score"]:.4f}
- Prediction drift detected: {batch_row["prediction_drift_detected"]}
- Max feature drift score: {batch_row["max_feature_drift_score"]:.4f}
- Most drifted feature: {batch_row["most_drifted_feature"]}
- Feature drift detected: {batch_row["feature_drift_detected"]}
- Missing percentage: {batch_row["missing_percentage"]:.4f}
- Duplicate rows: {batch_row["duplicate_rows"]}
- Data quality issue detected: {batch_row["data_quality_issue_detected"]}
- Anomaly detected: {batch_row["anomaly_detected"]}

Model Performance:
- Accuracy: {batch_row["accuracy"]:.4f}
- ROC-AUC: {batch_row["roc_auc"]:.4f}
- F1-score: {batch_row["f1_score"]:.4f}
- Precision: {batch_row["precision"]:.4f}
- Recall: {batch_row["recall"]:.4f}

Top SHAP Impact Changes:
{top_shap_changes}

Write the output in this exact structure:

Incident Summary:
Root Cause Hypothesis:
Impacted Features:
Business / Model Risk:
Recommended Action:
Retraining Recommendation:
"""

    return prompt


def generate_fallback_summary(batch_row, top_shap_changes):
    """
    Generate a free rule-based summary.

    This is used when:
    - OPENAI_API_KEY is not available
    - API credits are not available
    - API call fails

    This keeps the project reproducible on GitHub.
    """

    issue_reasons = []

    if batch_row["prediction_drift_detected"]:
        issue_reasons.append(
            f"prediction drift detected with score {batch_row['prediction_drift_score']:.4f}"
        )

    if batch_row["feature_drift_detected"]:
        issue_reasons.append(
            f"feature drift detected, strongest in {batch_row['most_drifted_feature']}"
        )

    if batch_row["data_quality_issue_detected"]:
        issue_reasons.append(
            f"data quality issue detected with {batch_row['missing_percentage']:.4f} missing percentage "
            f"and {batch_row['duplicate_rows']} duplicate rows"
        )

    if not issue_reasons:
        issue_reasons.append("no major monitoring issue detected")

    issue_text = "; ".join(issue_reasons)

    summary = f"""
Incident Summary:
Batch {batch_row["batch_name"]} was flagged with {batch_row["severity"]} severity. The monitoring checks found {issue_text}.

Root Cause Hypothesis:
The batch may differ from the baseline production pattern due to changes in transaction behavior, risk-score movement, feature distribution shift, or data quality issues. The most drifted feature was {batch_row["most_drifted_feature"]}.

Impacted Features:
Primary monitored feature impact: {batch_row["most_drifted_feature"]}.
Top SHAP impact changes:
{top_shap_changes}

Business / Model Risk:
The model may produce less stable fraud-risk predictions on this batch compared with the baseline. Current batch performance was ROC-AUC {batch_row["roc_auc"]:.4f}, F1-score {batch_row["f1_score"]:.4f}, precision {batch_row["precision"]:.4f}, and recall {batch_row["recall"]:.4f}.

Recommended Action:
Review the drifted feature distributions, validate missing and duplicate records, compare batch predictions with known fraud patterns, and monitor whether the same issue repeats in future batches.

Retraining Recommendation:
Retraining is recommended if similar drift or performance degradation continues across future batches. If the issue is caused by data quality, fix upstream data validation before retraining.
"""

    return summary.strip()


def generate_gpt_summary(prompt, batch_row, top_shap_changes):
    """
    Generate incident summary using OpenAI GPT API.

    Important:
    - If OPENAI_API_KEY exists, the function tries the GPT API.
    - If API key is missing or API call fails, it uses the free fallback summary.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    # Free fallback path when API key is not available
    if not api_key:
        print("OPENAI_API_KEY not found. Using free fallback summary.")
        return generate_fallback_summary(batch_row, top_shap_changes)

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You create concise ML monitoring incident summaries.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as error:
        # This handles no credits, invalid key, network errors, rate limits, etc.
        print("OpenAI API call failed. Using free fallback summary.")
        print("Error:", error)

        return generate_fallback_summary(batch_row, top_shap_changes)


def generate_incident_summaries():
    """
    Generate incident summaries for anomalous production batches.

    This function:
    1. Loads monitoring report.
    2. Loads SHAP comparison output.
    3. Assigns severity.
    4. Creates an OpenAI GPT prompt.
    5. Uses GPT if API key/credits are available.
    6. Uses free fallback summary if GPT is unavailable.
    7. Saves summaries as text files.
    """

    monitoring_df = load_monitoring_report()
    shap_df = load_shap_comparison()

    # Add severity column to each batch
    monitoring_df["severity"] = monitoring_df.apply(assign_severity, axis=1)

    # Use top 10 SHAP impact changes for summary context
    top_shap_changes = shap_df.head(10).to_string(index=False)

    generated_files = []

    for _, row in monitoring_df.iterrows():

        # Create incident summaries only for anomalous batches
        if not row["anomaly_detected"]:
            continue

        prompt = create_incident_prompt(row, top_shap_changes)

        summary = generate_gpt_summary(
            prompt=prompt,
            batch_row=row,
            top_shap_changes=top_shap_changes,
        )

        # Clean batch name for file output
        batch_name_clean = row["batch_name"].replace(".csv", "")
        summary_path = SUMMARY_DIR / f"{batch_name_clean}_incident_summary.txt"

        with open(summary_path, "w", encoding="utf-8") as file:
            file.write(summary)

        generated_files.append(summary_path)

        print("\nGenerated incident summary for:", row["batch_name"])
        print("Saved at:", summary_path)

    print("\nIncident summary generation completed")

    if not generated_files:
        print("No anomalous batches found. No summaries generated.")

    return generated_files


# This allows the file to run directly using:
# python src/incident_summary.py
if __name__ == "__main__":
    generate_incident_summaries()