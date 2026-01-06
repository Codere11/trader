#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-02T22:09:47Z
"""Train a causal entry regressor for live timing: decide on minute close, enter next minute open.

Goal
Predict the *oracle best net return within next hold_min minutes* if we enter at next minute open.

Causality / NO LOOKAHEAD
- Features come from the signal candle i (known at close) and the prior pre_min candles.
- Label uses future candles starting at entry candle (i+1).

Dataset
- Built on-the-fly from a 1-min market CSV (dYdX BTC-USD).
- Includes all minutes (not just positives) so scores are meaningful across the whole stream.

Outputs
- Joblib artifact with model + ordered feature list.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def fee_side_from_roundtrip_pct(roundtrip_fee_pct: float) -> float:
    """Convert a desired total round-trip fee % into the per-side fee parameter used in this repo.

    Repo fee model:
      net_mult = gross_mult * (1 - fee_side) / (1 + fee_side)

    If you want fees alone to multiply by fee_mult = (1 - roundtrip_fee_pct/100), solve:
      (1 - fee_side)/(1 + fee_side) = fee_mult
      => fee_side = (1 - fee_mult) / (1 + fee_mult)
    """
    fee_mult = 1.0 - float(roundtrip_fee_pct) / 100.0
    if fee_mult <= 0.0 or fee_mult >= 1.0:
        raise ValueError("roundtrip_fee_pct must be in (0, 100)")
    return float((1.0 - fee_mult) / (1.0 + fee_mult))


def net_return_pct_vec(entry_px: np.ndarray, exit_px: np.ndarray, fee_side: float) -> np.ndarray:
    """Vectorized net return % at 1x (accounts for entry and exit fees)."""
    entry_px = np.maximum(entry_px.astype(np.float64), 1e-12)
    exit_px = exit_px.astype(np.float64)
    gross_mult = exit_px / entry_px
    net_mult = gross_mult * (1.0 - float(fee_side)) / (1.0 + float(fee_side))
    return (net_mult - 1.0) * 100.0


def progress_eta_callback(period: int = 25):
    """LightGBM callback that prints progress + ETA."""
    t0 = time.time()

    def _cb(env):
        it = int(env.iteration) + 1
        total = int(env.end_iteration)
        if it == 1:
            # reset start once training begins
            nonlocal t0
            t0 = time.time()
        if it % int(period) != 0 and it != total:
            return
        elapsed = time.time() - t0
        rate = elapsed / max(1, it)
        eta = rate * max(0, total - it)
        # Format evals
        parts = []
        for tup in env.evaluation_result_list or []:
            data_name, metric_name, val, _ = tup
            parts.append(f"{data_name} {metric_name}={float(val):.6f}")
        evals = "  ".join(parts)
        print(f"[iter {it:>4}/{total}] {evals}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m", flush=True)

    return _cb


def build_training_matrices(
    bars: pd.DataFrame,
    pre_min: int,
    hold_min: int,
    fee_side: float,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series]:
    """Return (X, y, feature_names, signal_times_utc) for signal indices i.

    Signal i:
      - features from candle i and i-1..i-pre_min
      - entry at i+1 open
      - label = oracle best net return within next hold_min minutes (exit at close)
    """
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")

    base = compute_feature_frame(bars)
    src = base[[c for c in FEATURES if c in base.columns]]

    open_arr = bars["open"].to_numpy(np.float64)
    close_arr = bars["close"].to_numpy(np.float64)

    feat_arrs: dict[str, np.ndarray] = {str(c): src[c].to_numpy(np.float64) for c in src.columns}

    n = len(bars)
    last_signal_i = n - 2 - int(hold_min)
    if last_signal_i <= int(pre_min):
        raise ValueError("Not enough bars for pre-context and hold horizon")

    idxs = np.arange(int(pre_min), last_signal_i + 1, dtype=np.int64)
    signal_times = pd.to_datetime(bars.loc[idxs, "timestamp"], utc=True)

    # Build feature list in stable order.
    feature_names: list[str] = []

    # Current candle i (known at decision time = close).
    feature_names.append("cur_open")
    feature_names.append("cur_close")
    for feat in feat_arrs.keys():
        feature_names.append(f"cur_{feat}")

    # Prior candles.
    for k in range(1, int(pre_min) + 1):
        feature_names.append(f"pre{k}_open")
        feature_names.append(f"pre{k}_close")
        for feat in feat_arrs.keys():
            feature_names.append(f"pre{k}_{feat}")

    X = np.empty((len(idxs), len(feature_names)), dtype=np.float32)

    # Fill X columns.
    col = 0

    # cur
    X[:, col] = open_arr[idxs].astype(np.float32)
    col += 1
    X[:, col] = close_arr[idxs].astype(np.float32)
    col += 1
    for feat in feat_arrs.keys():
        X[:, col] = feat_arrs[feat][idxs].astype(np.float32)
        col += 1

    # pre1..pre_min
    for k in range(1, int(pre_min) + 1):
        j = idxs - k
        X[:, col] = open_arr[j].astype(np.float32)
        col += 1
        X[:, col] = close_arr[j].astype(np.float32)
        col += 1
        for feat in feat_arrs.keys():
            X[:, col] = feat_arrs[feat][j].astype(np.float32)
            col += 1

    assert col == X.shape[1]

    # Labels: oracle best return for entry at next open (i+1)
    entry_idx = idxs + 1
    entry_px = open_arr[entry_idx]

    best_ret = np.full(len(idxs), -1e18, dtype=np.float64)

    for k in range(1, int(hold_min) + 1):
        exit_idx = entry_idx + k
        exit_px = close_arr[exit_idx]
        ret = net_return_pct_vec(entry_px, exit_px, fee_side)
        best_ret = np.maximum(best_ret, ret)

    y = best_ret.astype(np.float32)

    # Drop rows with any NaNs/infs in X or y.
    good = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[good]
    y = y[good]
    signal_times = signal_times.iloc[np.where(good)[0]]

    return X, y, feature_names, signal_times


def main() -> None:
    ap = argparse.ArgumentParser(description="Train entry regressor (decide close, enter next open) with oracle-exit labels")
    ap.add_argument("--market-csv", required=True, help="CSV with timestamp, open, high, low, close, volume")
    ap.add_argument("--pre-min", type=int, default=5, help="Number of prior minutes of context")
    ap.add_argument("--hold-min", type=int, default=10, help="Oracle horizon in minutes")
    ap.add_argument(
        "--fee-roundtrip-pct",
        type=float,
        default=0.1,
        help="TOTAL round-trip fee percent (entry+exit). Default 0.1 means 0.1% total.",
    )
    ap.add_argument(
        "--fee-side",
        type=float,
        default=None,
        help="Override per-side fee parameter used by repo net return function. If omitted, derived from --fee-roundtrip-pct.",
    )
    ap.add_argument("--test-frac", type=float, default=0.2, help="Chronological test fraction")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--progress-period", type=int, default=25, help="Print training progress every N boosting rounds")

    args = ap.parse_args()

    bars = pd.read_csv(args.market_csv, parse_dates=["timestamp"])
    print(f"Loaded {len(bars)} bars")
    print(f"Range: {bars['timestamp'].min()} → {bars['timestamp'].max()}")

    if args.fee_side is None:
        fee_side = fee_side_from_roundtrip_pct(float(args.fee_roundtrip_pct))
    else:
        fee_side = float(args.fee_side)

    # Print fee interpretation explicitly.
    fee_mult = (1.0 - fee_side) / (1.0 + fee_side)
    print(f"Fees: roundtrip_target={float(args.fee_roundtrip_pct):.6f}% -> fee_side={fee_side:.8f} (fee_mult={fee_mult:.6f})")

    print("Building training matrices (this can take a bit)...")
    X, y, feature_names, signal_times = build_training_matrices(
        bars=bars,
        pre_min=int(args.pre_min),
        hold_min=int(args.hold_min),
        fee_side=float(fee_side),
    )

    print(f"Samples after NaN drop: {len(y)}")
    print(f"Features: {len(feature_names)}")
    print(f"Label mean: {float(np.mean(y)):.4f}%  median: {float(np.median(y)):.4f}%")

    # Chronological split
    test_frac = float(args.test_frac)
    if not (0.0 < test_frac < 1.0):
        raise ValueError("--test-frac must be in (0,1)")

    split = int(len(y) * (1.0 - test_frac))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    print(f"Train n={len(y_tr)}  Test n={len(y_te)}")

    print("Training LightGBM (progress + ETA will print)...")
    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=128,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric="rmse",
        callbacks=[
            progress_eta_callback(period=int(args.progress_period)),
            lgb.early_stopping(stopping_rounds=100, verbose=False),
        ],
    )

    # Eval
    pred_te = model.predict(X_te)
    pred_tr = model.predict(X_tr)

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(np.mean(np.abs(a - b)))

    print("=== Evaluation ===")
    print(f"Train RMSE: {rmse(pred_tr, y_tr):.4f}%  MAE: {mae(pred_tr, y_tr):.4f}%")
    print(f"Test  RMSE: {rmse(pred_te, y_te):.4f}%  MAE: {mae(pred_te, y_te):.4f}%")
    if len(y_te) > 10:
        pear = float(np.corrcoef(pred_te.astype(np.float64), y_te.astype(np.float64))[0, 1])
        print(f"Test Pearson corr: {pear:.4f}")

    # Save
    ts = _now_ts()
    out_dir = REPO_ROOT / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"entry_regressor_dydx_nextopen_oracle_{ts}.joblib"

    artifact = {
        "model": model,
        "features": feature_names,
        "label": "oracle_best_ret_nextopen_pct",
        "context": {
            "market_csv": str(Path(args.market_csv).resolve()),
            "pre_min": int(args.pre_min),
            "hold_min": int(args.hold_min),
            "fee_roundtrip_pct": float(args.fee_roundtrip_pct),
            "fee_side": float(fee_side),
            "n_samples": int(len(y)),
            "train_n": int(len(y_tr)),
            "test_n": int(len(y_te)),
            "signal_time_min_utc": str(signal_times.min()),
            "signal_time_max_utc": str(signal_times.max()),
        },
        "created_utc": ts,
    }

    joblib.dump(artifact, model_path)
    print("=== Saved ===")
    print(model_path)


if __name__ == "__main__":
    main()
