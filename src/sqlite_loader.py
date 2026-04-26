from pathlib import Path
import sqlite3

import pandas as pd

from data_loader import load_baseline_data, load_all_batches


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

# BASE_DIR points to the main project folder:
# ai-model-monitoring-incident-assistant/
BASE_DIR = Path(__file__).resolve().parents[1]

# Local SQLite database folder
DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(exist_ok=True)

# SQLite database file path
DB_PATH = DB_DIR / "monitoring_data.sqlite"


# ---------------------------------------------------------
# SQLite loading functions
# ---------------------------------------------------------

def load_data_to_sqlite():
    """
    Load baseline and production batch CSV files into a local SQLite database.

    This creates one main table:
    - transactions

    The table contains:
    - baseline training rows
    - all simulated production batch rows

    This makes the SQL + SQLite part of the project real and reproducible.
    """

    # Load baseline dataset
    baseline_df = load_baseline_data()

    # Add source marker so we can separate baseline from production batches
    baseline_df["data_source"] = "baseline"

    # Add file name for consistent SQL filtering
    baseline_df["batch_file_name"] = "train_baseline.csv"

    # Load all production batches
    batches = load_all_batches()

    batch_dfs = []

    for batch_name, batch_df in batches.items():
        batch_df = batch_df.copy()

        # Add source marker
        batch_df["data_source"] = "production_batch"

        # Add file name for SQL filtering and validation
        batch_df["batch_file_name"] = batch_name

        batch_dfs.append(batch_df)

    # Combine baseline and all batches into one dataframe
    transactions_df = pd.concat(
        [baseline_df] + batch_dfs,
        ignore_index=True,
    )

    # Connect to SQLite database
    conn = sqlite3.connect(DB_PATH)

    # Save dataframe as SQLite table
    transactions_df.to_sql(
        "transactions",
        conn,
        if_exists="replace",
        index=False,
    )

    conn.close()

    print("SQLite database created successfully")
    print("Database path:", DB_PATH)
    print("Table created: transactions")
    print("Total rows loaded:", len(transactions_df))


def run_validation_queries():
    """
    Run simple SQL validation queries.

    These queries prove that SQL is being used to inspect:
    - batch volume
    - fraud rate
    - risk-score movement
    - data quality issues
    - high-risk transaction count
    """

    conn = sqlite3.connect(DB_PATH)

    # ---------------------------------------------------------
    # Query 1: Validate row count by data source
    # ---------------------------------------------------------

    print("\nQuery 1: Row count by source")

    query_1 = """
    SELECT
        data_source,
        COUNT(*) AS total_rows
    FROM transactions
    GROUP BY data_source;
    """

    print(pd.read_sql_query(query_1, conn))

    # ---------------------------------------------------------
    # Query 2: Batch-level fraud rate and risk profile
    # ---------------------------------------------------------

    print("\nQuery 2: Batch-level fraud rate and risk profile")

    query_2 = """
    SELECT
        batch_file_name,
        batch_type,
        COUNT(*) AS total_transactions,
        ROUND(AVG(label_fraud), 4) AS fraud_rate,
        ROUND(AVG(fraud_probability_simulated), 4) AS avg_simulated_fraud_probability,
        ROUND(AVG(amount), 2) AS avg_transaction_amount,
        ROUND(AVG(login_velocity_24h), 2) AS avg_login_velocity,
        ROUND(AVG(device_trust_score), 4) AS avg_device_trust_score,
        ROUND(AVG(ip_risk_score), 4) AS avg_ip_risk_score,
        ROUND(AVG(merchant_risk_score), 4) AS avg_merchant_risk_score
    FROM transactions
    GROUP BY batch_file_name, batch_type
    ORDER BY batch_file_name;
    """

    print(pd.read_sql_query(query_2, conn))

    # ---------------------------------------------------------
    # Query 3: Missing value check by batch
    # ---------------------------------------------------------

    print("\nQuery 3: Missing value check by batch")

    query_3 = """
    SELECT
        batch_file_name,
        SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS missing_amount,
        SUM(CASE WHEN login_velocity_24h IS NULL THEN 1 ELSE 0 END) AS missing_login_velocity,
        SUM(CASE WHEN device_trust_score IS NULL THEN 1 ELSE 0 END) AS missing_device_trust_score,
        SUM(CASE WHEN ip_risk_score IS NULL THEN 1 ELSE 0 END) AS missing_ip_risk_score,
        SUM(CASE WHEN merchant_risk_score IS NULL THEN 1 ELSE 0 END) AS missing_merchant_risk_score
    FROM transactions
    GROUP BY batch_file_name
    ORDER BY batch_file_name;
    """

    print(pd.read_sql_query(query_3, conn))

    # ---------------------------------------------------------
    # Query 4: High-risk transactions by batch
    # ---------------------------------------------------------

    print("\nQuery 4: High-risk transactions by batch")

    # 0.30 is used as a practical risk threshold for this synthetic dataset.
    # This makes the query useful for comparing batch-level risk concentration.
    query_4 = """
    SELECT
        batch_file_name,
        COUNT(*) AS high_risk_transactions,
        ROUND(AVG(fraud_probability_simulated), 4) AS avg_simulated_risk
    FROM transactions
    WHERE fraud_probability_simulated >= 0.30
    GROUP BY batch_file_name
    ORDER BY high_risk_transactions DESC;
    """

    print(pd.read_sql_query(query_4, conn))

    conn.close()


# This allows the file to run directly using:
# python src/sqlite_loader.py
if __name__ == "__main__":
    load_data_to_sqlite()
    run_validation_queries()