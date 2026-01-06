#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T16:48:10Z
"""Train a SELL entry regressor on *inlier* oracle-entry contexts.

You requested:
- Train SELL entry model on inliers (after trimming pattern-outliers)
- 500-tree entry regressor

This script trains on the per-trade aggregated 5m pre-entry descriptor dataset produced by:
- scripts/select_entry_inliers_by_outlier_trim_*.py

Inputs
- data/analysis_entry_inliers_outliertrim_<ts>/SELL/entry_context_agg_scored.parquet
  (must contain: trade_net_return_pct, is_outlier_dropped)

Outputs (timestamped)
- data/entry_regressor_inliers_sell_<ts>/
  - metrics.json
  - feature_importance.csv
  - preds_sample.csv
  - models/entry_regressor_sell_inliers_<ts>.joblib

Notes
- Label: trade_net_return_pct (oracle trade net return %).
- Features: all numeric non-trade columns except outlier score fields.
- Split: chronological by trade_entry_time (train/val/test).
- Artifact is a dict with keys {model, feature_cols, pre_min, side, ...}.
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SELL entry regressor on outlier-trimmed inliers")
    ap.add_argument(
        "--analysis-root",
        default="data/analysis_entry_inliers_outliertrim_2026-01-04T16-42-57Z",
        help="Root folder data/analysis_entry_inliers_outliertrim_<ts>/ (must contain SELL/entry_context_agg_scored.parquet)",
    )
    ap.add_argument("--side", default="SELL", choices=["SELL", "BUY", "sell", "buy"], help="Which side to train")
    ap.add_argument("--pre-min", type=int, default=5)

    ap.add_argument("--n-estimators", type=int, default=500)
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

    ts = now_ts()
    out_root = Path(args.out_dir) / f"entry_regressor_inliers_{side.lower()}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scored dataset: {scored_path}", flush=True)
    df = pd.read_parquet(scored_path)

    # Filter to inliers
    if "is_outlier_dropped" not in df.columns:
        raise SystemExit("Expected column is_outlier_dropped in scored dataset")

    df["is_outlier_dropped"] = pd.to_numeric(df["is_outlier_dropped"], errors="coerce").fillna(1).astype(int)
    df = df[df["is_outlier_dropped"] == 0].copy().reset_index(drop=True)

    if df.empty:
        raise SystemExit("No inlier rows found (is_outlier_dropped==0)")

    # Order by time for causal-ish split
    if "trade_entry_time" in df.columns:
        df["trade_entry_time"] = pd.to_datetime(df["trade_entry_time"], utc=True, errors="coerce")
        df = df.sort_values("trade_entry_time").reset_index(drop=True)

    y = pd.to_numeric(df["trade_net_return_pct"], errors="coerce").to_numpy(np.float64)

    X_cols = feature_columns(df)
    if not X_cols:
        raise SystemExit("No feature columns found")

    X = df[X_cols].to_numpy(np.float64)

    # Impute NaNs with column medians
    med = np.nanmedian(X, axis=0)
    bad = ~np.isfinite(X)
    if np.any(bad):
        X[bad] = med[np.where(bad)[1]]

    n = int(len(df))
    test_frac = float(args.test_frac)
    split_test = int(n * (1.0 - test_frac))

    # train/val split within train
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
        f"Training {side} inlier regressor: n={n:,} train={len(tr_idx):,} val={len(va_idx):,} test={len(test_idx):,} features={len(X_cols):,} n_estimators={int(args.n_estimators)}",
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
        "pre_min": int(args.pre_min),
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
    }

    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # Feature importance (gain)
    imp = pd.DataFrame({
        "feature": X_cols,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    imp.to_csv(out_root / "feature_importance.csv", index=False)

    # Small prediction sample (test)
    sample = df.iloc[test_idx].copy()
    sample["pred"] = pred_te
    keep = [c for c in ["trade_entry_time", "trade_exit_time", "trade_duration_min", "trade_net_return_pct", "pred"] if c in sample.columns]
    sample[keep].sample(n=min(20000, len(sample)), random_state=42).to_csv(out_root / "preds_sample.csv", index=False)

    artifact = {
        "created_utc": ts,
        "side": side,
        "pre_min": int(args.pre_min),
        "label": "trade_net_return_pct",
        "feature_cols": list(X_cols),
        "model": model,
        "metrics": metrics,
    }

    model_path = models_dir / f"entry_regressor_{side.lower()}_inliers_{ts}.joblib"
    joblib.dump(artifact, model_path)

    print("Saved model:", model_path, flush=True)
    print("Wrote:", out_root, flush=True)


if __name__ == "__main__":
    main()
