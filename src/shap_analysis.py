from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from data_loader import load_baseline_data, load_batch_data
from train_model import prepare_features, DROP_COLUMNS, TARGET


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "fraud_xgb_model.joblib"

OUTPUT_DIR = BASE_DIR / "outputs" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# SHAP settings
# ---------------------------------------------------------

# We use a sample to keep SHAP fast and easy to run locally.
# You can increase this later if your laptop handles it well.
SAMPLE_SIZE = 500

# Batch used to compare against baseline.
# This batch has drift, so it is useful for explaining changed model behavior.
DRIFT_BATCH_FILE = "batch_02_feature_drift.csv"


def get_transformed_feature_names(model_pipeline):
    """
    Get final feature names after preprocessing.

    The model pipeline has:
    1. OneHotEncoder for categorical columns
    2. Direct numeric columns

    SHAP needs these final transformed feature names.
    """

    preprocessor = model_pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    # Clean names like cat__country_IN into country_IN
    feature_names = [name.replace("cat__", "").replace("num__", "") for name in feature_names]

    return feature_names


def transform_features(model_pipeline, X):
    """
    Apply the same preprocessing used during model training.

    This converts categorical columns into one-hot encoded columns.
    """

    preprocessor = model_pipeline.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(X)

    # Convert sparse matrix to dense array if needed
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    return X_transformed


def calculate_mean_abs_shap(shap_values, feature_names):
    """
    Calculate average absolute SHAP value for each feature.

    Higher value = feature had stronger impact on model predictions.
    """

    mean_abs_values = abs(shap_values).mean(axis=0)

    shap_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_values,
        }
    )

    shap_importance = shap_importance.sort_values(
        by="mean_abs_shap",
        ascending=False,
    )

    return shap_importance


def run_shap_analysis():
    """
    Run SHAP comparison between baseline data and drifted batch.

    This function:
    1. Loads trained model.
    2. Loads baseline data.
    3. Loads one drifted production batch.
    4. Applies model preprocessing.
    5. Calculates SHAP values.
    6. Saves SHAP importance CSV files.
    7. Saves SHAP summary plots.
    8. Saves comparison of baseline vs drifted batch feature impact.
    """

    # ---------------------------------------------------------
    # Load model and data
    # ---------------------------------------------------------

    model_pipeline = joblib.load(MODEL_PATH)

    baseline_df = load_baseline_data()
    drift_batch_df = load_batch_data(DRIFT_BATCH_FILE)

    # Prepare features using same logic as training
    X_baseline, _, _, _ = prepare_features(baseline_df)
    X_drift = drift_batch_df.drop(columns=DROP_COLUMNS + [TARGET])

    # Use sample to make SHAP calculation faster
    X_baseline_sample = X_baseline.sample(
        n=min(SAMPLE_SIZE, len(X_baseline)),
        random_state=42,
    )

    X_drift_sample = X_drift.sample(
        n=min(SAMPLE_SIZE, len(X_drift)),
        random_state=42,
    )

    # ---------------------------------------------------------
    # Transform data using trained preprocessing pipeline
    # ---------------------------------------------------------

    X_baseline_transformed = transform_features(model_pipeline, X_baseline_sample)
    X_drift_transformed = transform_features(model_pipeline, X_drift_sample)

    feature_names = get_transformed_feature_names(model_pipeline)

    # ---------------------------------------------------------
    # Extract trained XGBoost model
    # ---------------------------------------------------------

    xgb_model = model_pipeline.named_steps["model"]

    # TreeExplainer works well for tree-based models like XGBoost
    explainer = shap.TreeExplainer(xgb_model)

    # ---------------------------------------------------------
    # Calculate SHAP values
    # ---------------------------------------------------------

    baseline_shap_values = explainer.shap_values(X_baseline_transformed)
    drift_shap_values = explainer.shap_values(X_drift_transformed)

    # ---------------------------------------------------------
    # Create SHAP importance tables
    # ---------------------------------------------------------

    baseline_importance = calculate_mean_abs_shap(
        baseline_shap_values,
        feature_names,
    )

    drift_importance = calculate_mean_abs_shap(
        drift_shap_values,
        feature_names,
    )

    baseline_importance_path = OUTPUT_DIR / "baseline_shap_importance.csv"
    drift_importance_path = OUTPUT_DIR / "drift_batch_shap_importance.csv"

    baseline_importance.to_csv(baseline_importance_path, index=False)
    drift_importance.to_csv(drift_importance_path, index=False)

    # ---------------------------------------------------------
    # Compare baseline vs drifted batch SHAP impact
    # ---------------------------------------------------------

    shap_comparison = baseline_importance.merge(
        drift_importance,
        on="feature",
        suffixes=("_baseline", "_drift_batch"),
    )

    shap_comparison["shap_impact_change"] = (
        shap_comparison["mean_abs_shap_drift_batch"]
        - shap_comparison["mean_abs_shap_baseline"]
    )

    shap_comparison = shap_comparison.sort_values(
        by="shap_impact_change",
        ascending=False,
    )

    shap_comparison_path = OUTPUT_DIR / "baseline_vs_drift_shap_comparison.csv"
    shap_comparison.to_csv(shap_comparison_path, index=False)

    # ---------------------------------------------------------
    # Save SHAP summary plots
    # ---------------------------------------------------------

    baseline_plot_path = OUTPUT_DIR / "baseline_shap_summary.png"
    drift_plot_path = OUTPUT_DIR / "drift_batch_shap_summary.png"

    # Baseline SHAP plot
    plt.figure()
    shap.summary_plot(
        baseline_shap_values,
        X_baseline_transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(baseline_plot_path, bbox_inches="tight")
    plt.close()

    # Drift batch SHAP plot
    plt.figure()
    shap.summary_plot(
        drift_shap_values,
        X_drift_transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(drift_plot_path, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # Print clear output
    # ---------------------------------------------------------

    print("SHAP analysis completed successfully")

    print("\nSaved files:")
    print(baseline_importance_path)
    print(drift_importance_path)
    print(shap_comparison_path)
    print(baseline_plot_path)
    print(drift_plot_path)

    print("\nTop baseline SHAP features:")
    print(baseline_importance.head(10))

    print("\nTop drift batch SHAP features:")
    print(drift_importance.head(10))

    print("\nTop SHAP impact changes:")
    print(shap_comparison.head(10))


# This allows the file to run directly using:
# python src/shap_analysis.py
if __name__ == "__main__":
    run_shap_analysis()