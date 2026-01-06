#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T19:39:00Z
"""Train a SELL entry *classifier* on inlier oracle-entry contexts + ctx120 features.

Goal
- Use the same inlier trade contexts you used for the regressor, but make the target a clean
  good-vs-bad label:
  - good: trade_net_return_pct >= ret_thresh (default 0.2)
  - bad:  trade_net_return_pct <  bad_thresh (default 0.0)
  - neutral excluded from training

Features
- Start from the per-trade aggregated 5m pre-entry descriptor dataset (already has 5m aggregates).
- Append ctx120 aggregated features computed from the full bars parquet:
  - compute 30/60/120m context series causally using only t-1 and earlier
  - for each trade entry time t, aggregate ctx series over t-5..t-1 using the same agg schema

Outputs (timestamped)
- data/entry_classifier_inliers_sell_ctx120_goodbad_<ts>/
  - metrics.json
  - feature_importance.csv
  - preds_sample.csv
  - models/entry_classifier_sell_inliers_ctx120_goodbad_<ts>.joblib

Artifact
- Dict with keys {model, feature_cols, pre_min, side, label_def, ...}
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
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
        "label_good",
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


def _rolling_sum_nan_prefix(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    out[w - 1 :] = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    return out


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


def _compute_ctx_series(df_bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute causal context series at each minute using only t-1 and earlier."""
    close_prev = pd.to_numeric(df_bars["close"], errors="coerce").shift(1)
    high_prev = pd.to_numeric(df_bars["high"], errors="coerce").shift(1)
    low_prev = pd.to_numeric(df_bars["low"], errors="coerce").shift(1)
    vol_prev = pd.to_numeric(df_bars["volume"], errors="coerce").shift(1)

    out = pd.DataFrame(index=df_bars.index)

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

        # VWAP dev
        v = vol_prev.to_numpy(np.float64, copy=False)
        c = close_prev.to_numpy(np.float64, copy=False)
        sum_v = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0), w)
        sum_pv = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0) * np.nan_to_num(c, nan=0.0), w)
        vwap = sum_pv / np.maximum(1e-9, sum_v)
        out[f"vwap_dev_{w}m"] = np.where(sum_v > 0.0, ((c - vwap) / np.maximum(1e-9, vwap)) * 100.0, 0.0)

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)

    return out


def _windows_for_trade_idxs(trade_idx: np.ndarray, L: int = 5) -> np.ndarray:
    trade_idx = np.asarray(trade_idx, dtype=np.int64)
    offsets = np.arange(-L, 0, dtype=np.int64)
    return trade_idx[:, None] + offsets[None, :]


def _extract_5m_agg_for_trades(series: np.ndarray, win_idx: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    n = series.size
    idx = win_idx
    valid = (idx >= 0) & (idx < n)
    x = np.full(idx.shape, np.nan, dtype=np.float64)
    x[valid] = series[idx[valid]]
    return _agg_5(x, prefix)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SELL entry classifier (inliers + ctx120) for good vs bad")
    ap.add_argument(
        "--analysis-root",
        default="data/analysis_entry_inliers_outliertrim_2026-01-04T16-42-57Z",
        help="Root folder data/analysis_entry_inliers_outliertrim_<ts>/ (must contain SELL/entry_context_agg_scored.parquet)",
    )
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="Full 1m bars parquet (timestamp, open, high, low, close, volume)",
    )
    ap.add_argument("--pre-min", type=int, default=5)
    ap.add_argument("--ctx-windows", default="30,60,120")

    ap.add_argument("--ret-thresh", type=float, default=0.2)
    ap.add_argument("--bad-thresh", type=float, default=0.0)

    ap.add_argument("--n-estimators", type=int, default=1200)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--lgb-log-period", type=int, default=50)

    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    if int(args.pre_min) != 5:
        raise SystemExit("This script currently supports --pre-min=5 only")

    analysis_root = pick_latest_analysis_root(str(args.analysis_root))
    scored_path = analysis_root / "SELL" / "entry_context_agg_scored.parquet"
    if not scored_path.exists():
        raise SystemExit(f"Missing: {scored_path}")

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"Missing bars: {bars_path}")

    ts = now_ts()
    out_root = Path(args.out_dir) / f"entry_classifier_inliers_sell_ctx120_goodbad_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scored inlier dataset: {scored_path}", flush=True)
    df = pd.read_parquet(scored_path)

    if "is_outlier_dropped" not in df.columns:
        raise SystemExit("Expected column is_outlier_dropped in scored dataset")
    df["is_outlier_dropped"] = pd.to_numeric(df["is_outlier_dropped"], errors="coerce").fillna(1).astype(int)
    df = df[df["is_outlier_dropped"] == 0].copy().reset_index(drop=True)

    if "trade_entry_time" not in df.columns:
        raise SystemExit("Expected trade_entry_time in scored dataset")
    df["trade_entry_time"] = pd.to_datetime(df["trade_entry_time"], utc=True, errors="coerce")
    df = df.sort_values("trade_entry_time").reset_index(drop=True)

    y_ret = pd.to_numeric(df["trade_net_return_pct"], errors="coerce").to_numpy(np.float64)
    good = y_ret >= float(args.ret_thresh)
    bad = y_ret < float(args.bad_thresh)
    keep = good | bad

    df = df.loc[keep].copy().reset_index(drop=True)
    y = (pd.to_numeric(df["trade_net_return_pct"], errors="coerce").to_numpy(np.float64) >= float(args.ret_thresh)).astype(np.int32)

    print(f"Training labels: n={len(df):,} pos(good)={int(y.sum()):,} neg(bad)={int((1-y).sum()):,}", flush=True)

    # Load bars and map trade_entry_time to bar index
    print(f"Loading bars: {bars_path}", flush=True)
    bars = pd.read_parquet(bars_path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    ts_sorted = bars["timestamp"].to_numpy(dtype="datetime64[ns]")
    te = df["trade_entry_time"].to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(ts_sorted, te)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == te)
    if not np.all(ok):
        raise SystemExit(f"Failed to map {(~ok).sum()} trade_entry_time values into bars timeline")
    trade_idx = pos.astype(np.int64)

    ctx_windows = [int(x.strip()) for x in str(args.ctx_windows).split(",") if x.strip()]
    print(f"Computing causal ctx series windows={ctx_windows} ...", flush=True)
    ctx_series = _compute_ctx_series(bars, ctx_windows)

    win_idx = _windows_for_trade_idxs(trade_idx, L=5)

    new_cols: dict[str, np.ndarray] = {}
    for c in ctx_series.columns:
        arr = ctx_series[c].to_numpy(np.float64, copy=False)
        feats = _extract_5m_agg_for_trades(arr, win_idx, c)
        new_cols.update(feats)

    ctx_agg = pd.DataFrame(new_cols)
    df2 = pd.concat([df.reset_index(drop=True), ctx_agg.reset_index(drop=True)], axis=1)

    X_cols = feature_columns(df2)
    if not X_cols:
        raise SystemExit("No feature columns found")

    X = df2[X_cols].to_numpy(np.float64)

    # Impute NaNs with column medians
    med = np.nanmedian(X, axis=0)
    badm = ~np.isfinite(X)
    if np.any(badm):
        X[badm] = med[np.where(badm)[1]]

    n = int(len(df2))
    split_test = int(n * (1.0 - float(args.test_frac)))
    train_idx = np.arange(0, split_test, dtype=np.int64)
    test_idx = np.arange(split_test, n, dtype=np.int64)

    n_tr = int(train_idx.size)
    n_val = int(n_tr * float(args.val_frac))
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
        f"Training SELL classifier (ctx120) n={n:,} train={len(tr_idx):,} val={len(va_idx):,} test={len(test_idx):,} features={len(X_cols):,} n_estimators={int(args.n_estimators)}",
        flush=True,
    )

    clf = LGBMClassifier(
        n_estimators=int(args.n_estimators),
        learning_rate=0.03,
        num_leaves=256,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    )

    callbacks = []
    if int(args.lgb_log_period) > 0:
        callbacks.append(log_evaluation(period=int(args.lgb_log_period)))

    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=callbacks,
    )

    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]

    try:
        from sklearn.metrics import roc_auc_score

        auc_tr = float(roc_auc_score(y_tr, p_tr)) if len(np.unique(y_tr)) == 2 else float("nan")
        auc_va = float(roc_auc_score(y_va, p_va)) if len(np.unique(y_va)) == 2 else float("nan")
        auc_te = float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te)) == 2 else float("nan")
    except Exception:
        auc_tr = auc_va = auc_te = float("nan")

    metrics = {
        "created_utc": ts,
        "side": "SELL",
        "analysis_root": str(analysis_root),
        "scored_path": str(scored_path),
        "bars": str(bars_path),
        "pre_min": 5,
        "ctx_windows": ctx_windows,
        "label_def": {"good": f">={float(args.ret_thresh)}", "bad": f"<{float(args.bad_thresh)}", "neutral": "excluded"},
        "n_estimators": int(args.n_estimators),
        "n": int(n),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(test_idx)),
        "auc_train": auc_tr,
        "auc_val": auc_va,
        "auc_test": auc_te,
        "pos_rate": float(np.mean(y)),
        "n_features": int(len(X_cols)),
        "n_ctx_agg_features": int(ctx_agg.shape[1]),
    }

    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    imp = pd.DataFrame(
        {
            "feature": X_cols,
            "importance_gain": clf.booster_.feature_importance(importance_type="gain"),
            "importance_split": clf.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(out_root / "feature_importance.csv", index=False)

    sample = df2.iloc[test_idx].copy()
    sample["proba_good"] = p_te
    keep_cols = [c for c in ["trade_entry_time", "trade_net_return_pct", "proba_good"] if c in sample.columns]
    sample[keep_cols].sample(n=min(20000, len(sample)), random_state=42).to_csv(out_root / "preds_sample.csv", index=False)

    artifact = {
        "created_utc": ts,
        "side": "SELL",
        "pre_min": 5,
        "ctx_windows": ctx_windows,
        "label": "good_vs_bad",
        "label_def": {"good": float(args.ret_thresh), "bad": float(args.bad_thresh)},
        "feature_cols": list(X_cols),
        "model": clf,
        "metrics": metrics,
    }

    model_path = models_dir / f"entry_classifier_sell_inliers_ctx120_goodbad_{ts}.joblib"
    joblib.dump(artifact, model_path)

    print("Saved model:", model_path, flush=True)
    print("Wrote:", out_root, flush=True)


if __name__ == "__main__":
    main()
