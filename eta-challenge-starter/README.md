# ETA Challenge Submission

This is my submission for the Gobblecube ETA challenge. The task is to predict
NYC taxi trip duration from only four request-time fields:

- pickup zone
- dropoff zone
- request time
- passenger count

The final model is a time-aware route lookup plus a small residual XGBoost
model. The main idea was simple: for this problem, the pickup/dropoff pair is
much more informative than treating zone IDs as normal numbers.

## How I Worked

I am primarily a full-stack developer, and I had not worked on this kind of
ETA/modeling problem before. I approached it as a problem to figure out step by
step rather than as something where I already knew the right ML recipe.

I used AI tools heavily, but mostly as a pair-programming and review loop. I
used Codex and opencode. My workflow was usually:
use one llm model to help implement or run an experiment, then ask the other to
review the reasoning, criticize the methodology, and point out risks before I
moved on.

Where AI helped most:

- Debugging methodology, especially target-encoding leakage and Dev overfitting
  risk.
- Reviewing the training/inference boundary, which led to the
  predict-vs-training equivalence test.
- Turning experiment ideas into small implementation steps I could measure.

Where it fell short:

- It sometimes pushed for more features before the validation setup was strong
  enough.
- It was useful for review, but I still had to decide what evidence was strong
  enough to keep or drop an experiment.

Total time spent on this challenge: 2 hours 

## Final Score

Dev MAE: **267.940 seconds**

Other validation numbers:

- `grade.py` 50k sample MAE: **270.0 seconds**
- Dev holdout MAE: **273.987 seconds**
- Docker image size: **943MB**
- Docker build time: **69s**
- `predict()` latency p99: **0.237ms**

The Dev holdout is the last few days of Dev and is harder than the full Dev
average, so I think it is the more honest number to keep in mind for Eval.

## What I Built

I started with the provided XGBoost baseline, but it performed worse than a
simple lookup table. The reason is that taxi zone IDs are categorical labels.
Zone `43` and zone `44` are not necessarily close just because their numbers
are close.

So I made the route itself the base signal:

1. Compute historical duration for each `(pickup_zone, dropoff_zone)` pair.
2. Add time-aware tables for `(pickup_zone, dropoff_zone, hour)` and
   `(pickup_zone, dropoff_zone, day_of_week)`.
3. Use this as a strong base prediction.
4. Train XGBoost to predict only the remaining error.
5. Add centroid distance between pickup/dropoff zones as one more feature.

The final prediction is:

```text
route/time lookup prediction + XGBoost residual correction
```

## Experiment Timeline

| Step                          | Dev MAE  | What I learned                                  |
|-------------------------------|---------:|-------------------------------------------------|
| Starter XGBoost baseline      |   ~351s  | Raw zone IDs underfit route geography           |
| Zone-pair mean lookup         | 301.220s | Route identity is the main signal               |
| Zone-pair median lookup       | 296.896s | Median worked better for MAE                    |
| Time-aware route lookup       | 272.666s | Hour-of-day matters a lot                       |
| First residual XGBoost        | 269.479s | Residual modeling helped, but had leakage risk  |
| OOF-clean residual XGBoost    | 269.605s | Slightly worse, but methodologically correct    |
| Final distance residual model | 267.940s | Distance gave a small additional lift           |

The biggest jump came from the time-aware route lookup. The residual model and
distance features helped after that, but the lookup was the foundation.

## Leakage Fix

My first residual model used target-encoded route features computed from all
training rows. That can leak a row's own target into its features.

I fixed this with K-fold out-of-fold encoding. For each fold, I built the route
statistics from the other folds, then encoded that fold using only those
statistics. The final inference tables are rebuilt using all training data,
which is safe because Eval rows are not part of training.

This changed the residual model from a slightly optimistic result to a cleaner
one:

```text
leaky residual XGBoost: 269.479s
OOF-clean residual XGBoost: 269.605s
```

## Distance Features

I used the NYC taxi zone shapefile to compute pickup/dropoff centroid features
and haversine distance.

One detail I was careful about: I computed polygon centroids in the shapefile's
projected CRS first, then converted those centroid points to latitude/longitude.
For missing centroid zones like `264` and `265`, I used `NaN` distance features
instead of zero distance. Zero would incorrectly mean the pickup and dropoff are
in the same place, while XGBoost can handle missing values.

I also tested bearing features, but the gain was only about `0.024s`, so I
dropped them.

## Validation

I ran both normal correctness checks and submission-path checks:

```bash
python -m pytest tests/
python grade.py
docker build -t eta-distance-model .
docker run --rm -v "$(pwd)/data:/work" eta-distance-model /work/dev.parquet /work/preds.csv
```

Final validation:

- Tests: **12 passed**
- `grade.py`: **270.0s MAE**
- Docker wrote **1,230,911** predictions
- Docker image size: **943MB**
- Docker build time: **69s**
- p99 prediction latency: **0.237ms**

I also added a test that compares row-by-row `predict.py` output against the
training-side prediction path on edge-heavy samples. That test caught a real
feature-construction mismatch before submission, which I fixed.

## What Did Not Work

- The starter XGBoost model with raw zone IDs was much worse than a simple route
  lookup.
- Plain mean lookup was worse than median lookup.
- The first residual model had target-encoding leakage risk, so I replaced it
  with OOF encoding.
- Bearing features were too small/noisy to keep.

## Known Limitations

The model is still weakest on airport, unknown-zone, and outer-borough trips.
The hidden Eval set is a 2024 winter-holiday slice, so airport and holiday
traffic could be harder than average Dev.

The model also uses straight-line centroid distance, not actual road distance.
It does not use weather, holidays, events, or live traffic.

## Next Experiments

If I kept going, I would try:

1. Pair-hour speed features, such as route duration divided by centroid distance.
2. Holiday and holiday-window features for the winter Eval shift.
3. Airport-specific correction features.
4. Road-network distance using OSRM/OpenStreetMap instead of straight-line
   distance.
5. Offline weather features from NOAA.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
python data/download_data.py

python train_route_model.py --train-clean-xgb --clean-mode oof --distance --sample-n 5000000 --oof-folds 5

python -m pytest tests/
python grade.py

docker build -t eta-distance-model .
docker run --rm -v "$(pwd)/data:/work" eta-distance-model /work/dev.parquet /work/preds.csv
```


