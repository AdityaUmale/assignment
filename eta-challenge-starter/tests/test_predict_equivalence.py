"""Regression tests for training/inference feature parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predict import predict
from train_route_model import add_time_columns, load_artifact, predict_artifact


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEV_PATH = DATA_DIR / "dev.parquet"
MODEL_PATH = Path(__file__).resolve().parents[1] / "model.pkl"
REQUEST_FIELDS = ["pickup_zone", "dropoff_zone", "requested_at", "passenger_count"]


def _take(df: pd.DataFrame, mask: pd.Series | np.ndarray, n: int) -> pd.DataFrame:
    rows = df.loc[mask]
    if len(rows) == 0:
        return rows
    return rows.sample(n=min(n, len(rows)), random_state=42)


def _equivalence_sample(dev: pd.DataFrame) -> pd.DataFrame:
    pair_counts = dev.groupby(["pickup_zone", "dropoff_zone"]).size().rename("pair_dev_count")
    with_counts = dev.join(pair_counts, on=["pickup_zone", "dropoff_zone"])
    pieces = [
        dev.sample(n=min(60, len(dev)), random_state=42),
        _take(dev, dev["pickup_zone"].isin([264, 265]), 10),
        _take(dev, dev["dropoff_zone"].isin([264, 265]), 10),
        _take(with_counts, with_counts["pair_dev_count"] < 5, 10).drop(columns=["pair_dev_count"], errors="ignore"),
        _take(dev, pd.to_datetime(dev["requested_at"]).dt.hour.isin([0, 23]), 10),
        _take(dev, dev["passenger_count"].isin([0, 6]), 10),
    ]
    sample = pd.concat(pieces, ignore_index=True)
    sample = sample.drop_duplicates(subset=REQUEST_FIELDS).head(120).reset_index(drop=True)
    if len(sample) < 100:
        filler = dev.sample(n=min(100 - len(sample), len(dev)), random_state=7)
        sample = pd.concat([sample, filler], ignore_index=True)
        sample = sample.drop_duplicates(subset=REQUEST_FIELDS).head(100).reset_index(drop=True)
    return sample


@pytest.mark.skipif(not DEV_PATH.exists(), reason="data/dev.parquet is not available")
def test_predict_matches_training_pipeline_on_edge_sample():
    dev = pd.read_parquet(DEV_PATH, columns=REQUEST_FIELDS + ["duration_seconds"])
    sample = _equivalence_sample(dev)

    artifact = load_artifact(MODEL_PATH)
    vectorized = predict_artifact(add_time_columns(sample), artifact)
    rowwise = np.array([predict(req) for req in sample[REQUEST_FIELDS].to_dict("records")])

    # XGBoost returns float32-scale predictions; row-wise and vectorized paths
    # can differ by a fraction of a millisecond after Python/NumPy casting.
    max_diff = float(np.max(np.abs(vectorized - rowwise)))
    assert max_diff <= 2e-4, f"predict.py drifted from training pipeline; max diff={max_diff}"
