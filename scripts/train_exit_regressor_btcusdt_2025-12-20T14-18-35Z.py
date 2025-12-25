#!/usr/bin/env python3
"""
Train a LightGBM exit regressor on exit_reg_dataset_*.parquet.
- X: FEATURES (and optional short context);
- y: target_close_ret_pct;
- Time-based split by entry_time.
Outputs:
- models/exit_regressor_btcusdt_<ts>.joblib
- data/exit_regression/exit_regressor_metrics_<ts>.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit("LightGBM is required: pip install lightgbm") from e

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def eval_regression(tag: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    out = {f"{tag}_mae": mae, f"{tag}_rmse": rmse, f"{tag}_r2": r2}
    # Ranking utility
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    n = int(y_sorted.size)
    for frac in (0.001, 0.005, 0.01):
        k = max(1, int(n * frac))
        out[f"{tag}_top{int(frac*1000)}permil_mean_y"] = float(np.mean(y_sorted[:k]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train exit regressor on per-minute exit dataset")
    ap.add_argument("--dataset", required=True, help="Parquet from build_exit_regression_dataset_*.py")
    ap.add_argument("--splits-file", default="data/splits.json")
    ap.add_argument("--train-end-date", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=1500)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-child-samples", type=int, default=200)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample-bytree", type=float, default=0.9)
    ap.add_argument("--early-stopping-rounds", type=int, default=200)
    ap.add_argument("--log-period", type=int, default=100)
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    ts = _now_ts()

    df = pd.read_parquet(args.dataset)
    if "entry_time" not in df.columns:
        raise SystemExit("dataset must include entry_time column")

    # Sort by time; split 70/15/15 unless overridden by train_end_date
    df = df.sort_values(["entry_time", "rel_min"]).reset_index(drop=True)

    if args.train_end_date:
        hold = pd.Timestamp(args.train_end_date)
        tr = df[pd.to_datetime(df["entry_time"]) < hold]
        te = df[pd.to_datetime(df["entry_time"]) >= hold]
        if len(te) == 0:
            # Fallback: last 15% as test by time
            cut = int(0.85 * len(df))
            tr, te = df.iloc[:cut], df.iloc[cut:]
        # Build val as last 15% of train by time
        cut2 = int(0.85 * len(tr))
        tr, va = tr.iloc[:cut2], tr.iloc[cut2:]
    else:
        n = len(df)
        cut1 = int(0.70 * n)
        cut2 = int(0.85 * n)
        tr, va, te = df.iloc[:cut1], df.iloc[cut1:cut2], df.iloc[cut2:]

    feat_cols = list(FEATURES)
    Xtr = tr[feat_cols].to_numpy(dtype=np.float32)
    ytr = tr["target_close_ret_pct"].to_numpy(dtype=np.float32)
    Xva = va[feat_cols].to_numpy(dtype=np.float32)
    yva = va["target_close_ret_pct"].to_numpy(dtype=np.float32)
    Xte = te[feat_cols].to_numpy(dtype=np.float32)
    yte = te["target_close_ret_pct"].to_numpy(dtype=np.float32)

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

    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(int(args.early_stopping_rounds), verbose=False),
            lgb.log_evaluation(period=int(args.log_period)),
        ],
    )

    pred_tr = model.predict(Xtr)
    pred_va = model.predict(Xva)
    pred_te = model.predict(Xte)

    metrics = {}
    metrics.update(eval_regression("train", ytr, pred_tr))
    metrics.update(eval_regression("val", yva, pred_va))
    metrics.update(eval_regression("test", yte, pred_te))

    models_dir = REPO_ROOT / args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"exit_regressor_btcusdt_{ts}.joblib"

    payload = {
        "model": model,
        "features": feat_cols,
        "created_utc": ts,
        "label": {"kind": "close_based_return_pct", "hold_min": int(df["rel_min"].max())},
    }
    joblib.dump(payload, model_path)

    met_dir = REPO_ROOT / "data" / "exit_regression"
    met_dir.mkdir(parents=True, exist_ok=True)
    met_path = met_dir / f"exit_regressor_metrics_{ts}.csv"
    pdf = pd.DataFrame([{**metrics, "model_path": str(model_path), "rows_train": len(tr), "rows_val": len(va), "rows_test": len(te)}])
    pdf.to_csv(met_path, index=False)

    print("Saved model:", model_path)
    print("Saved metrics:", met_path)


if __name__ == "__main__":
    main()
