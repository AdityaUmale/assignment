import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def main():
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)

    global_mean = train["duration_seconds"].mean()
    print(f"Global mean duration: {global_mean:.1f} seconds")

    print("\nCalculating zone-pair averages from training data...")
    lookup_table = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].mean().reset_index()
    lookup_table.rename(columns={"duration_seconds": "predicted_duration"}, inplace=True)

    print("Applying lookup table to dev data...")
    dev_with_preds = dev.merge(lookup_table, on=["pickup_zone", "dropoff_zone"], how="left")
    
    # Fill pairs that were not seen in training with the global mean
    dev_with_preds["predicted_duration"] = dev_with_preds["predicted_duration"].fillna(global_mean)

    y_dev = dev_with_preds["duration_seconds"].to_numpy()
    preds = dev_with_preds["predicted_duration"].to_numpy()

    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE (Zone-Pair Lookup): {mae:.1f} seconds")

if __name__ == "__main__":
    main()
