#!/usr/bin/env python3
"""Train an entry selection regressor on *actual BTCUSDT minute data*.

Key points (per your requirements):
- Regression (not classification): predicts oracle best forward return within next N minutes.
- No future leakage in features: uses only current/past rolling features.
- Accurate loading indication:
  - CSV read progress by bytes (tqdm)
  - Label computation progress (tqdm over horizon)
  - LightGBM training progress (tqdm over boosting iterations)
- Saves model (timestamped) + metrics (timestamped).

Outputs:
- models/entry_regressor_btcusdt_<ts>.joblib
- data/entry_regression_training/entry_regressor_metrics_<ts>.csv

Example:
  python3 scripts/train_entry_regressor_btcusdt_2025-12-20T13-35-05Z.py \
    --market-csv data/btc_profitability_analysis_filtered.csv \
    --hold-min 10 --fee 0.001 --pre-min 5 \
    --y-clip-low -1 --y-clip-high 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit("LightGBM is required: pip install lightgbm") from e

try:
    from tqdm import tqdm
except Exception as e:
    raise SystemExit("tqdm is required: pip install tqdm") from e

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame

_MKT_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class SplitSpec:
    holdout_start: pd.Timestamp | None


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def read_csv_with_progress(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with a bytes-based progress bar (accurate for wall time)."""

    total = path.stat().st_size
    with open(path, "rb") as f:
        # NOTE: tqdm.wrapattr returns a context manager.
        with tqdm.wrapattr(
            f,
            "read",
            total=total,
            desc=f"Reading {path.name}",
            unit="B",
            unit_scale=True,
            miniters=1,
        ) as wrapped:
            df = pd.read_csv(wrapped, **kwargs)
    return df


def minute_bars(csv_path: Path) -> pd.DataFrame:
    t0 = time.perf_counter()
    raw = read_csv_with_progress(
        csv_path,
        parse_dates=["timestamp"],
        usecols=_MKT_COLS,
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )

    if not raw["timestamp"].is_monotonic_increasing:
        raw = raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    raw["ts_min"] = raw["timestamp"].dt.floor("min")

    # Normalize to minute bars (fast aggregation)
    g = (
        raw.groupby("ts_min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"ts_min": "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    dt_s = time.perf_counter() - t0
    print(f"Loaded+minutized market: rows={len(g):,} in {dt_s:.1f}s")
    return g


def compute_oracle_best_return_pct(close: np.ndarray, hold_min: int, fee_side: float) -> np.ndarray:
    """For each t, best net % return exiting at t+1..t+hold_min (net of per-side fee)."""

    close = close.astype(np.float64, copy=False)
    n = int(close.size)
    hold_min = int(hold_min)
    fee_side = float(fee_side)

    if hold_min < 1:
        raise ValueError("hold_min must be >= 1")
    if fee_side < 0:
        raise ValueError("fee must be >= 0")

    entry_cost = close * (1.0 + fee_side)
    best = np.full(n, -np.inf, dtype=np.float64)

    # Progress is per-k (small, but accurate for knowing it's not stuck)
    for k in tqdm(range(1, hold_min + 1), desc=f"Labeling oracle_best_ret_pct_{hold_min}m", unit="step"):
        if n <= k:
            break
        ret_k = (close[k:] * (1.0 - fee_side)) / entry_cost[:-k] - 1.0
        better = ret_k > best[:-k]
        if np.any(better):
            idx = np.flatnonzero(better)
            best[idx] = ret_k[idx]

    y = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(best) & (best > -np.inf)
    y[valid] = best[valid] * 100.0
    return y


def load_split_spec(splits_file: Path, train_end_date: str | None) -> SplitSpec:
    holdout_start = None
    if train_end_date:
        holdout_start = pd.Timestamp(train_end_date)
        return SplitSpec(holdout_start=holdout_start)

    if splits_file.exists():
        try:
            with open(splits_file, "r") as f:
                spl = json.load(f)
            if spl.get("holdout_start"):
                holdout_start = pd.Timestamp(spl["holdout_start"])
        except Exception:
            holdout_start = None

    return SplitSpec(holdout_start=holdout_start)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eval_regression(tag: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    out = {f"{tag}_mae": mae, f"{tag}_rmse": rmse, f"{tag}_r2": r2}

    # Ranking-oriented summary: mean y in top predicted buckets
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    n = int(y_sorted.size)
    for frac in (0.001, 0.005, 0.01, 0.05):
        k = max(1, int(n * frac))
        out[f"{tag}_top{int(frac*1000)}permil_mean_y"] = float(np.mean(y_sorted[:k]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train BTCUSDT entry regressor (oracle-best-forward-return).")
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--hold-min", type=int, default=10, help="Forward window (minutes) for oracle best return label")
    ap.add_argument("--fee", type=float, default=0.001, help="Per-side fee used in oracle label")
    ap.add_argument("--pre-min", type=int, default=5, help="Rolling mean window for context features")
    ap.add_argument("--y-clip-low", type=float, default=-1.0)
    ap.add_argument("--y-clip-high", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None, help="Optional: use only last N minutes of data")

    ap.add_argument("--splits-file", default="data/splits.json")
    ap.add_argument("--train-end-date", type=str, default=None, help="Overrides splits.json holdout_start")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=4000)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-child-samples", type=int, default=200)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample-bytree", type=float, default=0.9)
    ap.add_argument("--early-stopping-rounds", type=int, default=200)
    ap.add_argument("--log-period", type=int, default=200)

    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out-stem", default="entry_regressor_btcusdt")
    args = ap.parse_args()

    ts = _now_ts()

    # 1) Load market minute bars
    mkt = minute_bars(Path(args.market_csv))
    if args.limit is not None and args.limit > 0 and len(mkt) > args.limit:
        mkt = mkt.iloc[-int(args.limit) :].reset_index(drop=True)
        print(f"Applied --limit: rows={len(mkt):,}")

    # 2) Compute features
    t0 = time.perf_counter()
    mkt2 = mkt.rename(columns={"timestamp": "ts_min"})
    feats = compute_feature_frame(mkt2)

    # Rolling mean context (past-only)
    pre_min = int(args.pre_min)
    base = feats[list(FEATURES)]
    ctx_mean = base.rolling(pre_min, min_periods=pre_min).mean().add_suffix(f"_mean_{pre_min}m")

    X_df = pd.concat([base, ctx_mean], axis=1)
    X = X_df.to_numpy(dtype=np.float32)

    dt_s = time.perf_counter() - t0
    print(f"Computed features: rows={len(feats):,} cols={X.shape[1]} in {dt_s:.1f}s")

    # 3) Labels (oracle best return)
    t0 = time.perf_counter()
    y = compute_oracle_best_return_pct(
        mkt["close"].to_numpy(dtype=np.float64, copy=False),
        hold_min=int(args.hold_min),
        fee_side=float(args.fee),
    ).astype(np.float32)

    # Clip targets to reduce tail dominance
    y = np.clip(y, float(args.y_clip_low), float(args.y_clip_high))

    dt_s = time.perf_counter() - t0
    print(f"Computed labels: hold={int(args.hold_min)}m in {dt_s:.1f}s")

    # 4) Align and filter invalid rows
    ts_arr = feats["ts_min"].to_numpy(dtype="datetime64[ns]")
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[ok]
    y = y[ok]
    ts_ok = ts_arr[ok]
    print(f"Training rows after dropna: {len(y):,}")

    # 5) Time split
    split = load_split_spec(Path(args.splits_file), args.train_end_date)

    if split.holdout_start is not None:
        hold = np.datetime64(split.holdout_start.to_datetime64())
        train_mask = ts_ok < hold
        test_mask = ~train_mask
        X_train_full, y_train_full = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Validation = last 20% of train_full by time
        cut = max(1, int(0.8 * len(y_train_full)))
        X_train, y_train = X_train_full[:cut], y_train_full[:cut]
        X_val, y_val = X_train_full[cut:], y_train_full[cut:]
        print(
            "Time split by holdout_start:",
            str(split.holdout_start),
            f"train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}",
        )
    else:
        n = int(len(y))
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        print(f"Time split 70/15/15: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}")

    # 6) Train LightGBM regressor with progress bar
    feat_names = list(base.columns) + list(ctx_mean.columns)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        random_state=int(args.seed),
        n_jobs=-1,
    )

    pbar = tqdm(total=int(args.n_estimators), desc="LightGBM training", unit="iter")

    def _tqdm_cb(env):
        # env.iteration is 0-indexed
        want = int(env.iteration) + 1
        if want > pbar.n:
            pbar.update(want - pbar.n)

    t0 = time.perf_counter()
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(int(args.early_stopping_rounds), verbose=False),
                lgb.log_evaluation(period=int(args.log_period)),
                _tqdm_cb,
            ],
        )
    finally:
        pbar.close()

    dt_s = time.perf_counter() - t0
    print(f"Training done in {dt_s:.1f}s (best_iteration={getattr(model, 'best_iteration_', None)})")

    # 7) Evaluate
    pred_tr = model.predict(X_train)
    pred_va = model.predict(X_val)
    pred_te = model.predict(X_test)

    metrics = {
        "rows_train": int(len(y_train)),
        "rows_val": int(len(y_val)),
        "rows_test": int(len(y_test)),
        "hold_min": int(args.hold_min),
        "fee_side": float(args.fee),
        "pre_min": int(args.pre_min),
        "y_clip_low": float(args.y_clip_low),
        "y_clip_high": float(args.y_clip_high),
    }
    metrics.update(eval_regression("train", y_train, pred_tr))
    metrics.update(eval_regression("val", y_val, pred_va))
    metrics.update(eval_regression("test", y_test, pred_te))

    print("Metrics:")
    for k in sorted(metrics.keys()):
        if k.startswith("rows_") or k in {"hold_min", "fee_side", "pre_min", "y_clip_low", "y_clip_high"}:
            continue
        print(f"  {k}: {metrics[k]:.6f}")

    # 8) Save model + metrics
    models_dir = REPO_ROOT / args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{args.out_stem}_{ts}.joblib"

    payload = {
        "model": model,
        "features": feat_names,
        "created_utc": ts,
        "label": {
            "kind": "oracle_best_return_pct",
            "hold_min": int(args.hold_min),
            "fee_side": float(args.fee),
            "clip": [float(args.y_clip_low), float(args.y_clip_high)],
        },
        "context": {"pre_min": int(args.pre_min), "ctx_agg": "rolling_mean"},
        "metrics": metrics,
    }
    joblib.dump(payload, model_path)

    met_dir = REPO_ROOT / "data" / "entry_regression_training"
    met_dir.mkdir(parents=True, exist_ok=True)
    met_path = met_dir / f"entry_regressor_metrics_{ts}.csv"
    pd.DataFrame([{"model_path": str(model_path), **metrics}]).to_csv(met_path, index=False)

    print("Saved model:", model_path)
    print("Saved metrics:", met_path)


if __name__ == "__main__":
    main()
