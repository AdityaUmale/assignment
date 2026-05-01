"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)
# Disable xgboost's feature-name validation so we can predict on a bare
# numpy array (skips per-call DataFrame construction overhead).
if hasattr(_MODEL, "get_booster"):
    _MODEL.get_booster().feature_names = None


def _cyclical(value: int, period: int) -> tuple[float, float]:
    arr = np.array([value], dtype=np.float32)
    angle = 2.0 * np.pi * arr / period
    return float(np.sin(angle)[0]), float(np.cos(angle)[0])


def _predict_xgboost_baseline(request: dict) -> float:
    ts = datetime.fromisoformat(request["requested_at"])
    x = np.array(
        [[
            int(request["pickup_zone"]),
            int(request["dropoff_zone"]),
            ts.hour,
            ts.weekday(),
            ts.month,
            int(request["passenger_count"]),
        ]],
        dtype=np.int32,
    )
    return float(_MODEL.predict(x)[0])


def _lookup_value(table: np.ndarray | None, pickup: int, dropoff: int, default: float) -> float:
    if table is None or not (0 <= pickup < table.shape[0]) or not (0 <= dropoff < table.shape[1]):
        return default
    value = float(table[pickup, dropoff])
    if not np.isfinite(value) or value <= 0:
        return default
    return value


def _distance_features(pickup: int, dropoff: int) -> list[float]:
    zone_lat = _MODEL.get("zone_lat")
    zone_lon = _MODEL.get("zone_lon")
    if zone_lat is None or zone_lon is None:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

    if not (0 <= pickup < len(zone_lat)) or not (0 <= dropoff < len(zone_lat)):
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

    pickup_lat = float(zone_lat[pickup])
    pickup_lon = float(zone_lon[pickup])
    dropoff_lat = float(zone_lat[dropoff])
    dropoff_lon = float(zone_lon[dropoff])
    values = [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    if not all(np.isfinite(v) for v in values):
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

    pickup_lat_arr = np.array([pickup_lat], dtype=np.float32)
    pickup_lon_arr = np.array([pickup_lon], dtype=np.float32)
    dropoff_lat_arr = np.array([dropoff_lat], dtype=np.float32)
    dropoff_lon_arr = np.array([dropoff_lon], dtype=np.float32)

    dlat = np.radians(dropoff_lat_arr - pickup_lat_arr)
    dlon = np.radians(dropoff_lon_arr - pickup_lon_arr)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(pickup_lat_arr)) * np.cos(np.radians(dropoff_lat_arr)) * np.sin(dlon / 2.0) ** 2
    )
    haversine_miles = (3958.7613 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(np.float32)
    haversine_miles = float(haversine_miles[0])
    return [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, haversine_miles]


def _bearing_features(pickup_lat: float, pickup_lon: float, dropoff_lat: float, dropoff_lon: float) -> list[float]:
    if not all(np.isfinite(v) for v in [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
        return [np.nan, np.nan]
    lat1 = np.radians(pickup_lat)
    lat2 = np.radians(dropoff_lat)
    dlon = np.radians(dropoff_lon - pickup_lon)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    return [float(np.sin(bearing)), float(np.cos(bearing))]


def _predict_route_artifact(request: dict) -> float:
    ts = datetime.fromisoformat(request["requested_at"])
    pickup = int(request["pickup_zone"])
    dropoff = int(request["dropoff_zone"])
    passenger_count = int(request["passenger_count"])

    global_mean = float(_MODEL["global_mean"])
    pair_value = _lookup_value(_MODEL["pair_value"], pickup, dropoff, global_mean)
    pair_hour_value = _MODEL.get("pair_hour_value")
    pair_dow_value = _MODEL.get("pair_dow_value")

    if pair_hour_value is not None and 0 <= pickup < pair_hour_value.shape[0] and 0 <= dropoff < pair_hour_value.shape[1]:
        pair_hour = float(pair_hour_value[pickup, dropoff, ts.hour])
    else:
        pair_hour = pair_value
    if pair_dow_value is not None and 0 <= pickup < pair_dow_value.shape[0] and 0 <= dropoff < pair_dow_value.shape[1]:
        pair_dow = float(pair_dow_value[pickup, dropoff, ts.weekday()])
    else:
        pair_dow = pair_value

    if pair_hour_value is not None and pair_dow_value is not None:
        base_prediction = float(np.array([0.7 * pair_hour + 0.3 * pair_dow], dtype=np.float32)[0])
    elif pair_hour_value is not None:
        base_prediction = float(np.array([pair_hour], dtype=np.float32)[0])
    elif pair_dow_value is not None:
        base_prediction = float(np.array([pair_dow], dtype=np.float32)[0])
    else:
        base_prediction = float(np.array([pair_value], dtype=np.float32)[0])

    residual_model = _MODEL.get("residual_model")
    if residual_model is None:
        return max(1.0, base_prediction)

    pickup_values = _MODEL["pickup_value"]
    dropoff_values = _MODEL["dropoff_value"]
    pair_counts = _MODEL["pair_count"]

    pickup_value = float(pickup_values[pickup]) if 0 <= pickup < len(pickup_values) else global_mean
    dropoff_value = float(dropoff_values[dropoff]) if 0 <= dropoff < len(dropoff_values) else global_mean
    pair_count = float(pair_counts[pickup, dropoff]) if 0 <= pickup < pair_counts.shape[0] and 0 <= dropoff < pair_counts.shape[1] else 0.0

    hour_sin, hour_cos = _cyclical(ts.hour, 24)
    dow_sin, dow_cos = _cyclical(ts.weekday(), 7)
    is_weekend = 1.0 if ts.weekday() >= 5 else 0.0
    is_rush = 1.0 if ts.weekday() < 5 and (7 <= ts.hour <= 9 or 16 <= ts.hour <= 18) else 0.0

    features = [
        pair_value,
        pair_hour,
        pair_dow,
        np.log1p(pair_count),
        pickup_value,
        dropoff_value,
        ts.hour,
        ts.weekday(),
        ts.month,
        passenger_count,
        is_weekend,
        is_rush,
        hour_sin,
        hour_cos,
        dow_sin,
        dow_cos,
    ]
    if _MODEL.get("use_distance"):
        distance_features = _distance_features(pickup, dropoff)
        features.extend(distance_features)
        if _MODEL.get("use_bearing"):
            features.extend(_bearing_features(*distance_features[:4]))

    x = np.array([features], dtype=np.float32)
    residual = float(residual_model.predict(x)[0])
    return max(1.0, base_prediction + residual)


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    if isinstance(_MODEL, dict) and _MODEL.get("artifact_type") == "route_model":
        return _predict_route_artifact(request)
    return _predict_xgboost_baseline(request)
