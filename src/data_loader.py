from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

BASELINE_PATH = BASE_DIR / "data" / "baseline" / "train_baseline.csv"
BATCH_DIR = BASE_DIR / "data" / "batches"


def load_baseline_data():
    """Load baseline training dataset."""
    return pd.read_csv(BASELINE_PATH)


def load_batch_data(batch_file_name):
    """Load one production batch file."""
    batch_path = BATCH_DIR / batch_file_name
    return pd.read_csv(batch_path)


def load_all_batches():
    """Load all batch CSV files as a dictionary."""
    batch_files = sorted(BATCH_DIR.glob("*.csv"))

    batches = {}
    for file in batch_files:
        batches[file.name] = pd.read_csv(file)

    return batches


if __name__ == "__main__":
    baseline_df = load_baseline_data()
    batches = load_all_batches()

    print("Baseline loaded successfully")
    print("Baseline shape:", baseline_df.shape)
    print("\nBaseline columns:")
    print(baseline_df.columns.tolist())

    print("\nBatch files loaded:")
    for batch_name, batch_df in batches.items():
        print(batch_name, batch_df.shape)