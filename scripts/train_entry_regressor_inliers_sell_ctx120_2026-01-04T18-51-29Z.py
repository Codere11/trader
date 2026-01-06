#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T18:51:29Z
"""Train a SELL entry regressor on *inlier* oracle-entry contexts, augmented with up-to-120m context features.

This does NOT modify shared feature pipelines.

What it does
- Loads the existing inlier-trade aggregated 5m pre-entry dataset:
  data/analysis_entry_inliers_outliertrim_<ts>/SELL/entry_context_agg_scored.parquet
- Loads the full 1m ETH bars parquet (OHLCV + base features).
- Computes 30/60/120m context *series* features causally (using only t-1 and earlier).
- For each trade entry time t, extracts the 5-minute pre-entry window (t-5..t-1) of those context series
  and aggregates them with the same {__last, __mean5, __std5, __min5, __max5, __range5, __slope5} schema.
- Appends these aggregated context columns to the inlier-trade dataset.
- Trains a LightGBM regressor to predict trade_net_return_pct.

Outputs (timestamped)
- data/entry_regressor_inliers_sell_ctx120_<ts>/
  - metrics.json
  - feature_importance.csv
  - preds_sample.csv
  - models/entry_regressor_sell_inliers_ctx120_<ts>.joblib

Artifact
- Dict with keys {model, feature_cols, pre_min, side, ...}.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from lightgbm.callback import log_evaluation


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def pick_latest_analysis_root(default: str) -> Path:
    p = Path(default)
    if p.exists():
        return p
    cands = sorted(Path("data").glob("analysis_entry_inliers_outliertrim_*/"))
    if not cands:
        raise SystemExit(f"No analysis_entry_inliers_outliertrim_* found and default missing: {p}")
    return cands[-1]


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop_exact = {
        "trade_row",
        "trade_date",
        "trade_entry_time",
        "trade_exit_time",
        "trade_duration_min",
        "trade_side",
        "trade_net_return_pct",
        # outlier-scoring artifacts
        "outlier_score",
        "nn_dist",
        "is_outlier_dropped",
    }
    cols: list[str] = []
    for c in df.columns:
        if c in drop_exact:
            continue
        if c.startswith("trade_"):
            continue
        if df[c].dtype.kind not in "fc":
            continue
        cols.append(c)
    return cols


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _slope_5(y: np.ndarray) -> np.ndarray:
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return np.nansum(y * xc[None, :], axis=1) / 10.0


def _safe_nanstd(x: np.ndarray) -> np.ndarray:
    import warnings

    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(x, axis=1)
        ok = np.isfinite(m)
        if np.any(ok):
            out[ok] = np.nanmean((x[ok] - m[ok, None]) ** 2, axis=1) ** 0.5
    return out


def _agg_5(x: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean5 = np.nanmean(x, axis=1)
        min5 = np.nanmin(x, axis=1)
        max5 = np.nanmax(x, axis=1)

    return {
        f"{prefix}__last": x[:, -1],
        f"{prefix}__mean5": mean5,
        f"{prefix}__std5": _safe_nanstd(x),
        f"{prefix}__min5": min5,
        f"{prefix}__max5": max5,
        f"{prefix}__range5": max5 - min5,
        f"{prefix}__slope5": _slope_5(x),
    }


def _rolling_sum_nan_prefix(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    out[w - 1 :] = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    return out


def _compute_ctx_series(df_bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute causal context series at each minute using only t-1 and earlier."""
    bars = df_bars.copy()
    # Use t-1 values as the "current" point for context
    close_prev = pd.to_numeric(bars["close"], errors="coerce").shift(1)
    high_prev = pd.to_numeric(bars["high"], errors="coerce").shift(1)
    low_prev = pd.to_numeric(bars["low"], errors="coerce").shift(1)
    vol_prev = pd.to_numeric(bars["volume"], errors="coerce").shift(1)

    out = pd.DataFrame(index=bars.index)

    for w in windows:
        w = int(w)
        out[f"mom_{w}m_pct"] = close_prev.pct_change(w) * 100.0
        out[f"vol_std_{w}m"] = close_prev.rolling(w, min_periods=w).std(ddof=0)

        rng = high_prev.rolling(w, min_periods=w).max() - low_prev.rolling(w, min_periods=w).min()
        out[f"range_{w}m"] = rng
        out[f"range_norm_{w}m"] = rng / close_prev.replace(0, np.nan)

        cmax = close_prev.rolling(w, min_periods=w).max()
        cmin = close_prev.rolling(w, min_periods=w).min()
        crng = cmax - cmin
        eps = 1e-9

        out[f"close_dd_from_{w}m_max_pct"] = (cmax / close_prev.replace(0, np.nan) - 1.0) * 100.0
        out[f"close_bounce_from_{w}m_min_pct"] = (close_prev / cmin.replace(0, np.nan) - 1.0) * 100.0
        out[f"close_pos_in_{w}m_range"] = (close_prev - cmin) / (crng + eps)

        # VWAP dev (rolling)
        # Match crossref behavior: if sum_v == 0 => vwap_dev = 0.
        v = vol_prev.to_numpy(np.float64, copy=False)
        c = close_prev.to_numpy(np.float64, copy=False)
        sum_v = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0), w)
        sum_pv = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0) * np.nan_to_num(c, nan=0.0), w)
        vwap = sum_pv / np.maximum(1e-9, sum_v)
        vwap_dev = np.where(sum_v > 0.0, ((c - vwap) / np.maximum(1e-9, vwap)) * 100.0, 0.0)
        out[f"vwap_dev_{w}m"] = vwap_dev

    # downcast
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)

    return out


def _windows_for_trade_idxs(trade_idx: np.ndarray, L: int = 5) -> np.ndarray:
    trade_idx = np.asarray(trade_idx, dtype=np.int64)
    offsets = np.arange(-L, 0, dtype=np.int64)  # -5..-1
    return trade_idx[:, None] + offsets[None, :]


def _extract_5m_agg_for_trades(series: np.ndarray, win_idx: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """series shape (n,), win_idx shape (m,5)."""
    n = series.size
    idx = win_idx
    valid = (idx >= 0) & (idx < n)
    x = np.full(idx.shape, np.nan, dtype=np.float64)
    flat = idx[valid]
    x[valid] = series[flat]
    return _agg_5(x, prefix)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SELL inlier entry regressor with 30/60/120m context features")
    ap.add_argument(
        "--analysis-root",
        default="data/analysis_entry_inliers_outliertrim_2026-01-04T16-42-57Z",
        help="Root folder data/analysis_entry_inliers_outliertrim_<ts>/ (must contain SELL/entry_context_agg_scored.parquet)",
    )
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="Full 1m bars parquet (must contain timestamp, open, high, low, close, volume)",
    )
    ap.add_argument("--side", default="SELL", choices=["SELL", "sell"], help="Which side to train")
    ap.add_argument("--pre-min", type=int, default=5)

    ap.add_argument("--ctx-windows", default="30,60,120", help="Comma-separated context windows (minutes)")

    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--lgb-log-period", type=int, default=50)

    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--random-state", type=int, default=42)

    args = ap.parse_args()

    side = str(args.side).upper()
    analysis_root = pick_latest_analysis_root(str(args.analysis_root))
    scored_path = analysis_root / side / "entry_context_agg_scored.parquet"
    if not scored_path.exists():
        raise SystemExit(f"Missing: {scored_path}")

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"Missing bars: {bars_path}")

    ts = now_ts()
    out_root = Path(args.out_dir) / f"entry_regressor_inliers_{side.lower()}_ctx120_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scored inlier dataset: {scored_path}", flush=True)
    df = pd.read_parquet(scored_path)

    if "is_outlier_dropped" not in df.columns:
        raise SystemExit("Expected column is_outlier_dropped in scored dataset")
    df["is_outlier_dropped"] = pd.to_numeric(df["is_outlier_dropped"], errors="coerce").fillna(1).astype(int)
    df = df[df["is_outlier_dropped"] == 0].copy().reset_index(drop=True)
    if df.empty:
        raise SystemExit("No inlier rows found (is_outlier_dropped==0)")

    if "trade_entry_time" not in df.columns:
        raise SystemExit("Expected trade_entry_time in scored dataset")
    df["trade_entry_time"] = pd.to_datetime(df["trade_entry_time"], utc=True, errors="coerce")
    df = df.sort_values("trade_entry_time").reset_index(drop=True)

    print(f"Loading bars: {bars_path}", flush=True)
    bars = pd.read_parquet(bars_path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    # Map trade_entry_time -> bars index
    ts_sorted = bars["timestamp"].to_numpy(dtype="datetime64[ns]")
    te = df["trade_entry_time"].to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(ts_sorted, te)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == te)
    if not np.all(ok):
        miss = int((~ok).sum())
        raise SystemExit(f"Failed to map {miss} trade_entry_time values into bars timeline")
    trade_idx = pos.astype(np.int64)

    ctx_windows = [int(x.strip()) for x in str(args.ctx_windows).split(",") if x.strip()]
    print(f"Computing causal context series for windows={ctx_windows} ...", flush=True)
    ctx_series = _compute_ctx_series(bars, ctx_windows)

    # For each trade, extract t-5..t-1 window indices
    L = int(args.pre_min)
    if L != 5:
        raise SystemExit("This script currently supports --pre-min=5 only")

    win_idx = _windows_for_trade_idxs(trade_idx, L=L)

    # Build aggregated context features (same schema as other precontext features)
    new_cols: dict[str, np.ndarray] = {}
    for c in ctx_series.columns:
        arr = ctx_series[c].to_numpy(np.float64, copy=False)
        feats = _extract_5m_agg_for_trades(arr, win_idx, c)
        new_cols.update(feats)

    ctx_agg = pd.DataFrame(new_cols)

    # Merge onto df
    df2 = pd.concat([df.reset_index(drop=True), ctx_agg.reset_index(drop=True)], axis=1)

    y = pd.to_numeric(df2["trade_net_return_pct"], errors="coerce").to_numpy(np.float64)

    X_cols = feature_columns(df2)
    if not X_cols:
        raise SystemExit("No feature columns found")

    X = df2[X_cols].to_numpy(np.float64)

    # Impute NaNs with column medians
    med = np.nanmedian(X, axis=0)
    bad = ~np.isfinite(X)
    if np.any(bad):
        X[bad] = med[np.where(bad)[1]]

    n = int(len(df2))
    test_frac = float(args.test_frac)
    split_test = int(n * (1.0 - test_frac))

    train_idx = np.arange(0, split_test, dtype=np.int64)
    test_idx = np.arange(split_test, n, dtype=np.int64)

    val_frac = float(args.val_frac)
    n_tr = int(train_idx.size)
    n_val = int(n_tr * val_frac)
    if n_val < 500:
        n_val = min(3000, max(0, n_tr // 10))
    if n_val <= 0 or n_tr - n_val <= 0:
        raise SystemExit("Not enough data for train/val split")

    tr_idx = train_idx[: n_tr - n_val]
    va_idx = train_idx[n_tr - n_val :]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    print(
        f"Training {side} inlier regressor + ctx120: n={n:,} train={len(tr_idx):,} val={len(va_idx):,} test={len(test_idx):,} features={len(X_cols):,} n_estimators={int(args.n_estimators)}",
        flush=True,
    )

    model = LGBMRegressor(
        n_estimators=int(args.n_estimators),
        learning_rate=0.05,
        num_leaves=256,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=int(args.random_state),
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = []
    if int(args.lgb_log_period) > 0:
        callbacks.append(log_evaluation(period=int(args.lgb_log_period)))

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l2",
        callbacks=callbacks,
    )

    pred_tr = model.predict(X_tr)
    pred_va = model.predict(X_va)
    pred_te = model.predict(X_te)

    metrics = {
        "created_utc": ts,
        "side": side,
        "analysis_root": str(analysis_root),
        "scored_path": str(scored_path),
        "bars": str(bars_path),
        "pre_min": int(args.pre_min),
        "ctx_windows": ctx_windows,
        "n_estimators": int(args.n_estimators),
        "n": int(n),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(test_idx)),
        "rmse_train": _rmse(pred_tr, y_tr),
        "rmse_val": _rmse(pred_va, y_va),
        "rmse_test": _rmse(pred_te, y_te),
        "corr_train": _corr(pred_tr, y_tr),
        "corr_val": _corr(pred_va, y_va),
        "corr_test": _corr(pred_te, y_te),
        "y_median": float(np.nanmedian(y)),
        "y_p90": float(np.nanquantile(y, 0.9)),
        "y_p99": float(np.nanquantile(y, 0.99)),
        "n_features": int(len(X_cols)),
        "n_ctx_agg_features": int(ctx_agg.shape[1]),
    }

    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    imp = pd.DataFrame(
        {
            "feature": X_cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(out_root / "feature_importance.csv", index=False)

    sample = df2.iloc[test_idx].copy()
    sample["pred"] = pred_te
    keep = [c for c in ["trade_entry_time", "trade_exit_time", "trade_duration_min", "trade_net_return_pct", "pred"] if c in sample.columns]
    sample[keep].sample(n=min(20000, len(sample)), random_state=42).to_csv(out_root / "preds_sample.csv", index=False)

    artifact = {
        "created_utc": ts,
        "side": side,
        "pre_min": int(args.pre_min),
        "label": "trade_net_return_pct",
        "ctx_windows": ctx_windows,
        "feature_cols": list(X_cols),
        "model": model,
        "metrics": metrics,
    }

    model_path = models_dir / f"entry_regressor_{side.lower()}_inliers_ctx120_{ts}.joblib"
    joblib.dump(artifact, model_path)

    print("Saved model:", model_path, flush=True)
    print("Wrote:", out_root, flush=True)


if __name__ == "__main__":
    main()
