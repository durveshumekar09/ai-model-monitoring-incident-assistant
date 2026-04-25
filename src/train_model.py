from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from data_loader import load_baseline_data


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

# BASE_DIR points to the main project folder:
# ai-model-monitoring-incident-assistant/
BASE_DIR = Path(__file__).resolve().parents[1]

# Folder where trained model and artifacts will be saved
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Final trained model file path
MODEL_PATH = MODEL_DIR / "fraud_xgb_model.joblib"


# ---------------------------------------------------------
# Target and columns to remove
# ---------------------------------------------------------

# This is the column we want to predict.
TARGET = "label_fraud"

# These columns should not be used as model input features:
# - IDs are not useful for learning fraud patterns.
# - event_timestamp needs separate time-based feature engineering.
# - fraud_probability_simulated is a synthetic helper column and would cause data leakage.
DROP_COLUMNS = [
    "transaction_id",
    "batch_id",
    "batch_type",
    "event_timestamp",
    "account_id",
    "fraud_probability_simulated",
]


def prepare_features(df: pd.DataFrame):
    """
    Prepare input features and target variable.

    Steps:
    1. Remove unwanted columns.
    2. Separate X features and y target.
    3. Identify categorical and numeric columns.

    Returns:
        X: Feature dataframe
        y: Target column
        categorical_cols: List of categorical feature names
        numeric_cols: List of numeric feature names
    """

    # X contains only model input features.
    X = df.drop(columns=DROP_COLUMNS + [TARGET])

    # y contains the fraud label we want to predict.
    y = df[TARGET]

    # Object columns are categorical and need one-hot encoding.
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Non-object columns are numeric and can be passed directly to the model.
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    return X, y, categorical_cols, numeric_cols


def train_model():
    """
    Train the baseline XGBoost fraud classifier and log results with MLflow.

    This function:
    1. Loads baseline training data.
    2. Splits data into train and test sets.
    3. Handles fraud/non-fraud class imbalance.
    4. Builds a preprocessing + model pipeline.
    5. Trains an XGBoost classifier.
    6. Calculates classification metrics.
    7. Saves model and confusion matrix.
    8. Logs metrics, parameters, and artifacts to MLflow.
    """

    # ---------------------------------------------------------
    # Load baseline dataset
    # ---------------------------------------------------------
    df = load_baseline_data()

    print("Baseline data loaded successfully")
    print("Dataset shape:", df.shape)

    # Prepare X, y, and column groups.
    X, y, categorical_cols, numeric_cols = prepare_features(df)

    print("\nFeature columns prepared")
    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)

    # ---------------------------------------------------------
    # Train-test split
    # ---------------------------------------------------------

    # stratify=y keeps the fraud/non-fraud ratio similar in train and test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("\nTrain-test split completed")
    print("Training rows:", X_train.shape[0])
    print("Testing rows:", X_test.shape[0])

    # ---------------------------------------------------------
    # Handle class imbalance
    # ---------------------------------------------------------

    # Fraud cases are usually much fewer than non-fraud cases.
    # scale_pos_weight tells XGBoost to give more importance to fraud cases.
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print("\nClass balance:")
    print("Non-fraud training rows:", negative_count)
    print("Fraud training rows:", positive_count)
    print("Scale pos weight:", round(scale_pos_weight, 2))

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    # OneHotEncoder converts categorical columns like country/channel/payment_method
    # into machine-readable numeric columns.
    #
    # handle_unknown="ignore" prevents errors if production batches contain
    # a new category that was not present during training.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # ---------------------------------------------------------
    # Model configuration
    # ---------------------------------------------------------

    # XGBoost is a strong tree-based classifier commonly used for tabular data.
    # scale_pos_weight helps the model pay more attention to rare fraud cases.
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    # Pipeline combines preprocessing and model training.
    # This ensures the same preprocessing is applied during training and prediction.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # ---------------------------------------------------------
    # MLflow experiment setup
    # ---------------------------------------------------------

    # MLflow stores experiment runs, metrics, parameters, and artifacts locally.
    mlflow.set_experiment("AI Model Monitoring - Fraud Risk")

    # Start one MLflow run for this baseline model.
    with mlflow.start_run(run_name="baseline_xgboost_model"):

        # ---------------------------------------------------------
        # Train model
        # ---------------------------------------------------------
        pipeline.fit(X_train, y_train)

        # ---------------------------------------------------------
        # Generate predictions
        # ---------------------------------------------------------

        # Predicted probability of fraud.
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Default threshold is 0.50, but for fraud/anomaly use cases,
        # a lower threshold helps catch more risky cases.
        fraud_threshold = 0.30

        # Convert probability into final class prediction.
        # 1 = fraud, 0 = non-fraud.
        y_pred = (y_proba >= fraud_threshold).astype(int)

        # ---------------------------------------------------------
        # Calculate evaluation metrics
        # ---------------------------------------------------------

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
        }

        # Log metrics to MLflow.
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # ---------------------------------------------------------
        # Log model parameters to MLflow
        # ---------------------------------------------------------

        mlflow.log_param("model_type", "XGBoost Classifier")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("fraud_threshold", fraud_threshold)
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])
        mlflow.log_param("categorical_features", categorical_cols)
        mlflow.log_param("numeric_features", numeric_cols)

        # ---------------------------------------------------------
        # Create and save confusion matrix
        # ---------------------------------------------------------

        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual_0", "Actual_1"],
            columns=["Predicted_0", "Predicted_1"],
        )

        cm_path = MODEL_DIR / "confusion_matrix_baseline.csv"
        cm_df.to_csv(cm_path)

        # ---------------------------------------------------------
        # Save trained model locally
        # ---------------------------------------------------------

        joblib.dump(pipeline, MODEL_PATH)

        # ---------------------------------------------------------
        # Log artifacts to MLflow
        # ---------------------------------------------------------

        # Artifact = saved file attached to an MLflow run.
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(cm_path))

        # ---------------------------------------------------------
        # Print final training output
        # ---------------------------------------------------------

        print("\nModel trained and saved successfully")
        print("Model path:", MODEL_PATH)

        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(cm_df)

        print("\nMLflow run completed successfully")


# This allows the file to run directly using:
# python src/train_model.py
if __name__ == "__main__":
    train_model()