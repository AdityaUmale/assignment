#!/usr/bin/env python
"""Train and evaluate route-aware ETA artifacts.

This script keeps the first submissions intentionally simple:

1. raw zone-pair lookup
2. smoothed zone-pair lookup
3. residual XGBoost on top of smoothed route features

It writes `model.pkl` in the format consumed by `predict.py`.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
ZONE_COUNT = 266
FEATURE_COLUMNS = [
    "pair_value",
    "pair_hour",
    "pair_dow",
    "log_pair_count",
    "pickup_value",
    "dropoff_value",
    "hour",
    "dow",
    "month",
    "passenger_count",
    "is_weekend",
    "is_rush",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def mae(preds: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - truth)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for path in (train_path, dev_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run `python data/download_data.py` first.")

    columns = [
        "pickup_zone",
        "dropoff_zone",
        "requested_at",
        "passenger_count",
        "duration_seconds",
    ]
    print("Loading train/dev parquet...")
    train = pd.read_parquet(train_path, columns=columns)
    dev = pd.read_parquet(dev_path, columns=columns)
    print(f"  train: {len(train):,}")
    print(f"  dev:   {len(dev):,}")
    return train, dev


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])
    out = df.copy()
    out["hour"] = ts.dt.hour.astype("int8")
    out["dow"] = ts.dt.dayofweek.astype("int8")
    out["month"] = ts.dt.month.astype("int8")
    return out


def group_arrays(train: pd.DataFrame) -> dict[str, np.ndarray | float]:
    global_mean = float(train["duration_seconds"].mean())

    pair = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].agg(["mean", "median", "count"])
    pickup = train.groupby("pickup_zone")["duration_seconds"].mean()
    dropoff = train.groupby("dropoff_zone")["duration_seconds"].mean()

    pair_mean = np.full((ZONE_COUNT, ZONE_COUNT), np.nan, dtype=np.float32)
    pair_median = np.full((ZONE_COUNT, ZONE_COUNT), np.nan, dtype=np.float32)
    pair_count = np.zeros((ZONE_COUNT, ZONE_COUNT), dtype=np.float32)
    pickup_value = np.full(ZONE_COUNT, global_mean, dtype=np.float32)
    dropoff_value = np.full(ZONE_COUNT, global_mean, dtype=np.float32)

    pair_index = pair.index.to_frame(index=False).to_numpy(dtype=np.int16)
    pair_mean[pair_index[:, 0], pair_index[:, 1]] = pair["mean"].to_numpy(dtype=np.float32)
    pair_median[pair_index[:, 0], pair_index[:, 1]] = pair["median"].to_numpy(dtype=np.float32)
    pair_count[pair_index[:, 0], pair_index[:, 1]] = pair["count"].to_numpy(dtype=np.float32)
    pickup_value[pickup.index.to_numpy(dtype=np.int16)] = pickup.to_numpy(dtype=np.float32)
    dropoff_value[dropoff.index.to_numpy(dtype=np.int16)] = dropoff.to_numpy(dtype=np.float32)

    return {
        "global_mean": global_mean,
        "pair_mean": pair_mean,
        "pair_median": pair_median,
        "pair_count": pair_count,
        "pickup_value": pickup_value,
        "dropoff_value": dropoff_value,
    }


def pair_predictions(df: pd.DataFrame, table: np.ndarray, fallback: float) -> np.ndarray:
    pickup = df["pickup_zone"].to_numpy(dtype=np.int16)
    dropoff = df["dropoff_zone"].to_numpy(dtype=np.int16)
    preds = table[pickup, dropoff].astype(np.float64)
    return np.where(np.isfinite(preds), preds, fallback)


def smoothed_pair_table(stats: dict[str, np.ndarray | float], m: float, statistic: str) -> np.ndarray:
    global_mean = float(stats["global_mean"])
    values = stats["pair_median" if statistic == "median" else "pair_mean"].astype(np.float64)
    counts = stats["pair_count"].astype(np.float64)
    filled = np.where(np.isfinite(values), values, global_mean)
    smoothed = (counts * filled + m * global_mean) / (counts + m)
    smoothed[counts == 0] = global_mean
    return smoothed.astype(np.float32)


def make_artifact(
    stats: dict[str, np.ndarray | float],
    pair_value: np.ndarray,
    residual_model: xgb.XGBRegressor | None = None,
    pair_hour_value: np.ndarray | None = None,
    pair_dow_value: np.ndarray | None = None,
    notes: str = "",
) -> dict:
    return {
        "artifact_type": "route_model",
        "version": 1,
        "notes": notes,
        "global_mean": float(stats["global_mean"]),
        "pair_value": pair_value.astype(np.float32),
        "pair_count": stats["pair_count"].astype(np.float32),
        "pickup_value": stats["pickup_value"].astype(np.float32),
        "dropoff_value": stats["dropoff_value"].astype(np.float32),
        "pair_hour_value": None if pair_hour_value is None else pair_hour_value.astype(np.float32),
        "pair_dow_value": None if pair_dow_value is None else pair_dow_value.astype(np.float32),
        "residual_model": residual_model,
        "features": FEATURE_COLUMNS,
    }


def save_artifact(artifact: dict, path: Path = MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved artifact to {path}")


def evaluate_lookup(train: pd.DataFrame, dev: pd.DataFrame) -> tuple[dict, np.ndarray]:
    stats = group_arrays(train)
    truth = dev["duration_seconds"].to_numpy(dtype=np.float64)

    raw_preds = pair_predictions(dev, stats["pair_mean"], float(stats["global_mean"]))
    raw_mae = mae(raw_preds, truth)
    print(f"pure_pair_mean_full_dev_mae={raw_mae:.3f}")

    best_name = "pure_pair_mean"
    best_table = stats["pair_mean"].copy()
    best_mae = raw_mae

    candidates: list[tuple[str, np.ndarray]] = []
    for statistic in ("mean", "median"):
        for m in (0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0):
            name = f"smoothed_{statistic}_m{m:g}"
            table = smoothed_pair_table(stats, m, statistic)
            candidates.append((name, table))

    print("\nLookup candidates:")
    for name, table in candidates:
        preds = pair_predictions(dev, table, float(stats["global_mean"]))
        score = mae(preds, truth)
        print(f"  {name}: {score:.3f}")
        if score < best_mae:
            best_name = name
            best_table = table
            best_mae = score

    print(f"\nbest_lookup={best_name} full_dev_mae={best_mae:.3f}")
    best_pair_hour = None
    best_pair_dow = None

    print("\nTemporal lookup candidates:")
    for m in (25.0, 50.0, 100.0, 250.0, 500.0, 1000.0):
        pair_hour = smoothed_child_table(train, best_table, "hour", m=m)
        hour_preds = pair_hour[
            dev["pickup_zone"].to_numpy(dtype=np.int16),
            dev["dropoff_zone"].to_numpy(dtype=np.int16),
            dev["hour"].to_numpy(dtype=np.int16),
        ]
        hour_score = mae(hour_preds, truth)
        print(f"  pair_hour_m{m:g}: {hour_score:.3f}")
        if hour_score < best_mae:
            best_name = f"pair_hour_m{m:g}"
            best_mae = hour_score
            best_pair_hour = pair_hour
            best_pair_dow = None

        pair_dow = smoothed_child_table(train, best_table, "dow", m=m)
        dow_preds = pair_dow[
            dev["pickup_zone"].to_numpy(dtype=np.int16),
            dev["dropoff_zone"].to_numpy(dtype=np.int16),
            dev["dow"].to_numpy(dtype=np.int16),
        ]
        dow_score = mae(dow_preds, truth)
        print(f"  pair_dow_m{m:g}: {dow_score:.3f}")
        if dow_score < best_mae:
            best_name = f"pair_dow_m{m:g}"
            best_mae = dow_score
            best_pair_hour = None
            best_pair_dow = pair_dow

        blended = 0.7 * hour_preds + 0.3 * dow_preds
        blended_score = mae(blended, truth)
        print(f"  pair_hour_dow_blend_m{m:g}: {blended_score:.3f}")
        if blended_score < best_mae:
            best_name = f"pair_hour_dow_blend_m{m:g}"
            best_mae = blended_score
            best_pair_hour = pair_hour
            best_pair_dow = pair_dow

    print(f"\nbest_lookup_with_time={best_name} full_dev_mae={best_mae:.3f}")
    artifact = make_artifact(
        stats,
        best_table,
        pair_hour_value=best_pair_hour,
        pair_dow_value=best_pair_dow,
        notes=f"lookup:{best_name}:mae={best_mae:.3f}",
    )
    return artifact, best_table


def smoothed_child_table(
    train: pd.DataFrame,
    parent_table: np.ndarray,
    child: str,
    m: float,
) -> np.ndarray:
    dim = 24 if child == "hour" else 7
    table = np.empty((ZONE_COUNT, ZONE_COUNT, dim), dtype=np.float32)
    table[:] = parent_table[:, :, None]

    grouped = train.groupby(["pickup_zone", "dropoff_zone", child])["duration_seconds"].agg(["mean", "count"])
    idx = grouped.index.to_frame(index=False).to_numpy(dtype=np.int16)
    parent = parent_table[idx[:, 0], idx[:, 1]].astype(np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    means = grouped["mean"].to_numpy(dtype=np.float64)
    values = (counts * means + m * parent) / (counts + m)
    table[idx[:, 0], idx[:, 1], idx[:, 2]] = values.astype(np.float32)
    return table


def design_matrix(
    df: pd.DataFrame,
    artifact: dict,
    sample_n: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

    pickup = df["pickup_zone"].to_numpy(dtype=np.int16)
    dropoff = df["dropoff_zone"].to_numpy(dtype=np.int16)
    hour = df["hour"].to_numpy(dtype=np.int16)
    dow = df["dow"].to_numpy(dtype=np.int16)
    month = df["month"].to_numpy(dtype=np.int16)
    passenger_count = df["passenger_count"].to_numpy(dtype=np.float32)

    pair_value = artifact["pair_value"][pickup, dropoff].astype(np.float32)
    pair_count = artifact["pair_count"][pickup, dropoff].astype(np.float32)
    pickup_value = artifact["pickup_value"][pickup].astype(np.float32)
    dropoff_value = artifact["dropoff_value"][dropoff].astype(np.float32)
    pair_hour_table = artifact.get("pair_hour_value")
    pair_dow_table = artifact.get("pair_dow_value")
    pair_hour = pair_value if pair_hour_table is None else pair_hour_table[pickup, dropoff, hour].astype(np.float32)
    pair_dow = pair_value if pair_dow_table is None else pair_dow_table[pickup, dropoff, dow].astype(np.float32)
    if pair_hour_table is not None and pair_dow_table is not None:
        base_prediction = (0.7 * pair_hour + 0.3 * pair_dow).astype(np.float32)
    elif pair_hour_table is not None:
        base_prediction = pair_hour.astype(np.float32)
    elif pair_dow_table is not None:
        base_prediction = pair_dow.astype(np.float32)
    else:
        base_prediction = pair_value.astype(np.float32)

    hour_angle = 2.0 * np.pi * hour.astype(np.float32) / 24.0
    dow_angle = 2.0 * np.pi * dow.astype(np.float32) / 7.0
    is_weekend = (dow >= 5).astype(np.float32)
    is_rush = ((dow < 5) & (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18)))).astype(np.float32)

    x = np.column_stack(
        [
            pair_value,
            pair_hour,
            pair_dow,
            np.log1p(pair_count),
            pickup_value,
            dropoff_value,
            hour,
            dow,
            month,
            passenger_count,
            is_weekend,
            is_rush,
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(dow_angle),
            np.cos(dow_angle),
        ]
    ).astype(np.float32)
    y = df["duration_seconds"].to_numpy(dtype=np.float32)
    return x, y, base_prediction.astype(np.float32)


def train_residual_model(train: pd.DataFrame, dev: pd.DataFrame, lookup_artifact: dict) -> tuple[dict, float]:
    model_artifact = dict(lookup_artifact)
    if model_artifact.get("pair_hour_value") is None:
        print("\nBuilding fallback pair-hour table...")
        model_artifact["pair_hour_value"] = smoothed_child_table(train, lookup_artifact["pair_value"], "hour", m=25.0)
    if model_artifact.get("pair_dow_value") is None:
        print("\nBuilding fallback pair-dow table...")
        model_artifact["pair_dow_value"] = smoothed_child_table(train, lookup_artifact["pair_value"], "dow", m=25.0)

    # Keep this bounded so iteration remains laptop-friendly. The route tables
    # still use every train row; the booster learns residual corrections.
    print("Building residual training matrix...")
    x_train, y_train, base_train = design_matrix(train, model_artifact, sample_n=5_000_000)
    x_dev, y_dev, base_dev = design_matrix(dev, model_artifact)
    y_residual = y_train - base_train

    print("Training residual XGBoost...")
    t0 = time.time()
    residual_model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:absoluteerror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    residual_model.fit(
        x_train,
        y_residual,
        eval_set=[(x_dev, y_dev - base_dev)],
        verbose=False,
    )
    print(f"  trained in {time.time() - t0:.0f}s")

    residual_preds = residual_model.predict(x_dev)
    preds = np.maximum(1.0, base_dev + residual_preds)
    score = mae(preds, y_dev)
    print(f"target_encoded_residual_xgb_full_dev_mae={score:.3f}")

    model_artifact["residual_model"] = residual_model
    model_artifact["notes"] = f"residual_xgb:mae={score:.3f}"
    return model_artifact, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-lookup", action="store_true", help="save the best lookup artifact and stop")
    parser.add_argument("--train-xgb", action="store_true", help="train residual XGBoost after lookup tuning")
    args = parser.parse_args()

    train, dev = load_data()
    train = add_time_columns(train)
    dev = add_time_columns(dev)

    lookup_artifact, _ = evaluate_lookup(train, dev)
    lookup_score = float(lookup_artifact["notes"].rsplit("=", 1)[-1])

    best_artifact = lookup_artifact
    best_score = lookup_score

    if args.train_xgb:
        xgb_artifact, xgb_score = train_residual_model(train, dev, lookup_artifact)
        if xgb_score < best_score:
            best_artifact = xgb_artifact
            best_score = xgb_score
            print(f"residual model wins by {lookup_score - xgb_score:.3f}s")
        else:
            print(f"lookup remains best by {xgb_score - lookup_score:.3f}s")

    if args.save_lookup or args.train_xgb:
        save_artifact(best_artifact)
        print(f"saved_best_full_dev_mae={best_score:.3f}")


if __name__ == "__main__":
    main()
