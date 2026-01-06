#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T13:23:12Z
"""Train an exit regressor on oracle gap.

We already have a per-minute in-trade dataset built from an entry stream:
- Each row is a decision minute k (k_rel=1..hold_min) inside a trade.
- Features include last-3-minutes of price + selected indicators + delta movement.

This script trains a regressor to predict:
    oracle_gap_pct = oracle_ret_pct - ret_if_exit_now_pct

Interpretation:
- oracle_gap_pct ~= "how much more upside (in % points) the oracle still has from now"
- 0 means: you are at the oracle exit moment (or equivalent best within horizon)

Policy evaluation on held-out trades:
- Exit at the first minute where predicted_gap_pct <= tau_gap (and k_rel >= min_exit_k).
- Otherwise exit at k=hold_min.

Outputs
- Model artifact joblib and evaluation CSV.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRegressor
except Exception as e:  # pragma: no cover
    raise SystemExit(f"lightgbm import failed: {e}")


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return float("nan")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    c = np.corrcoef(a, b)
    return float(c[0, 1])


def policy_eval_gap(
    df: pd.DataFrame,
    pred_gap: np.ndarray,
    *,
    hold_min: int,
    taus: List[float],
    min_exit_k: int,
) -> pd.DataFrame:
    """Evaluate a gap-threshold policy on a set of trades.

    Policy: for each trade, scan k=1..hold_min, exit at first k where pred_gap<=tau and k>=min_exit_k.
    Otherwise exit at hold_min.
    """

    work = df[["trade_id", "k_rel", "ret_if_exit_now_pct", "oracle_ret_pct", "oracle_k", "oracle_gap_pct"]].copy()
    work["pred_gap_pct"] = pred_gap.astype(np.float64)

    by_trade = work.groupby("trade_id", sort=False)

    def _ret_at_k(g: pd.DataFrame, k: int) -> float:
        r = g.loc[g["k_rel"] == int(k), "ret_if_exit_now_pct"]
        return float(r.iloc[0]) if len(r) else float("nan")

    # baselines (avoid groupby.apply deprecation)
    hold_rets = (
        work.loc[work["k_rel"] == int(hold_min), ["trade_id", "ret_if_exit_now_pct"]]
        .drop_duplicates("trade_id")
        .sort_values("trade_id")
        .loc[:, "ret_if_exit_now_pct"]
        .to_numpy(np.float64)
    )
    oracle_rets = by_trade["oracle_ret_pct"].first().to_numpy(np.float64)
    oracle_ks = by_trade["oracle_k"].first().to_numpy(np.float64)

    out: List[Dict[str, float]] = []
    out.append(
        {
            "tau_gap": float("nan"),
            "policy": "hold",
            "n_trades": float(len(hold_rets)),
            "mean_ret_pct": float(np.nanmean(hold_rets)),
            "median_ret_pct": float(np.nanmedian(hold_rets)),
            "win_rate_gt0_pct": float(np.nanmean(hold_rets > 0.0) * 100.0),
            "mean_exit_k": float(hold_min),
        }
    )
    out.append(
        {
            "tau_gap": float("nan"),
            "policy": "oracle",
            "n_trades": float(len(oracle_rets)),
            "mean_ret_pct": float(np.nanmean(oracle_rets)),
            "median_ret_pct": float(np.nanmedian(oracle_rets)),
            "win_rate_gt0_pct": float(np.nanmean(oracle_rets > 0.0) * 100.0),
            "mean_exit_k": float(np.nanmean(oracle_ks)),
        }
    )

    # sanity: if you could see the true oracle_gap, tau=0 exits exactly at oracle_k (should match oracle)
    # (kept as a debug baseline)
    true_rets = []
    true_ks = []
    for _, g in by_trade:
        g2 = g.sort_values("k_rel")
        pick = g2[g2["oracle_gap_pct"] <= 0.0]
        if len(pick) > 0:
            k = int(pick.iloc[0]["k_rel"])
            r = float(pick.iloc[0]["ret_if_exit_now_pct"])
        else:
            k = int(hold_min)
            r = float(_ret_at_k(g2, int(hold_min)))
        true_rets.append(r)
        true_ks.append(k)
    arr = np.asarray(true_rets, dtype=np.float64)
    out.append(
        {
            "tau_gap": 0.0,
            "policy": "true_gap<=0",
            "n_trades": float(len(arr)),
            "mean_ret_pct": float(np.mean(arr)),
            "median_ret_pct": float(np.median(arr)),
            "win_rate_gt0_pct": float(np.mean(arr > 0.0) * 100.0),
            "mean_exit_k": float(np.mean(np.asarray(true_ks, dtype=np.float64))),
        }
    )

    # learned policies
    for tau in taus:
        chosen_rets: List[float] = []
        chosen_ks: List[int] = []

        for _, g in by_trade:
            g2 = g.sort_values("k_rel")
            pick = g2[(g2["k_rel"] >= int(min_exit_k)) & (g2["pred_gap_pct"] <= float(tau))]
            if len(pick) > 0:
                k = int(pick.iloc[0]["k_rel"])
                r = float(pick.iloc[0]["ret_if_exit_now_pct"])
            else:
                k = int(hold_min)
                r = float(_ret_at_k(g2, int(hold_min)))

            chosen_rets.append(r)
            chosen_ks.append(k)

        arr2 = np.asarray(chosen_rets, dtype=np.float64)
        out.append(
            {
                "tau_gap": float(tau),
                "policy": f"pred_gap<={tau}",
                "n_trades": float(len(arr2)),
                "mean_ret_pct": float(np.mean(arr2)),
                "median_ret_pct": float(np.median(arr2)),
                "win_rate_gt0_pct": float(np.mean(arr2 > 0.0) * 100.0),
                "mean_exit_k": float(np.mean(np.asarray(chosen_ks, dtype=np.float64))),
            }
        )

    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an oracle-gap exit regressor")
    ap.add_argument(
        "--dataset-parquet",
        default="data/exit_oracle/exit_oracle_dataset_hold15_frac0p001_2026-01-03T13-20-28Z.parquet",
        help="Dataset produced by train_exit_oracle_classifier_hold15_*.py",
    )
    ap.add_argument("--hold-min", type=int, default=15)

    ap.add_argument("--test-trade-frac", type=float, default=0.2)
    ap.add_argument("--trees", type=int, default=800)
    ap.add_argument("--learning-rate", type=float, default=0.05)

    ap.add_argument("--min-exit-k", type=int, default=2, help="Disallow exits before this minute")
    ap.add_argument(
        "--taus",
        default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,1.00,1.50,2.00",
        help="Gap thresholds (pct points) to sweep",
    )

    ap.add_argument("--out-dir", default="data/exit_oracle")
    ap.add_argument("--save-dataset", action="store_true", help="Save a copy with oracle_gap_pct column")

    args = ap.parse_args()

    ds_path = Path(args.dataset_parquet)
    if not ds_path.exists():
        raise SystemExit(f"dataset not found: {ds_path}")

    hold_min = int(args.hold_min)

    print(f"Loading dataset: {ds_path}")
    df = pd.read_parquet(ds_path)

    # Label
    df["oracle_gap_pct"] = pd.to_numeric(df["oracle_ret_pct"], errors="coerce") - pd.to_numeric(df["ret_if_exit_now_pct"], errors="coerce")

    # Keep only finite rows
    df = df[np.isfinite(df["oracle_gap_pct"].to_numpy(np.float64))].reset_index(drop=True)

    # Feature columns: everything except ids/times/labels/metadata
    ignore = {
        "trade_id",
        "signal_i",
        "entry_idx",
        "entry_time",
        "entry_px",
        "oracle_k",
        "oracle_ret_pct",
        "k_rel",
        "decision_idx",
        "decision_time",
        "y_oracle_exit",
        "ret_if_exit_now_pct",
        "oracle_gap_pct",
    }
    feat_cols = [c for c in df.columns if c not in ignore]

    X_all = df[feat_cols].to_numpy(np.float32)
    y_all = df["oracle_gap_pct"].to_numpy(np.float32)

    # Keep NaN feature values (LightGBM handles missing natively). Only drop rows with non-finite label.
    good_y = np.isfinite(y_all.astype(np.float64))
    df = df.iloc[np.where(good_y)[0]].reset_index(drop=True)
    X_all = X_all[np.where(good_y)[0]]
    y_all = y_all[np.where(good_y)[0]]

    print(f"Rows: {len(df)}  Trades: {df['trade_id'].nunique()}  Features: {len(feat_cols)}")
    print(
        "oracle_gap_pct summary:",
        f"mean={float(np.mean(y_all)):.4f}",
        f"median={float(np.median(y_all)):.4f}",
        f"p90={float(np.percentile(y_all,90)):.4f}",
        f"p99={float(np.percentile(y_all,99)):.4f}",
    )

    # Split by trades (time-ordered by entry_time)
    trade_order = (
        df[["trade_id", "entry_time"]]
        .drop_duplicates("trade_id")
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    n_trades = len(trade_order)
    n_test = max(1, int(n_trades * float(args.test_trade_frac)))
    test_ids = set(trade_order.tail(n_test)["trade_id"].tolist())

    train_mask = ~df["trade_id"].isin(test_ids)
    test_mask = df["trade_id"].isin(test_ids)

    X_tr = X_all[np.where(train_mask.to_numpy())[0]]
    y_tr = y_all[np.where(train_mask.to_numpy())[0]]
    X_te = X_all[np.where(test_mask.to_numpy())[0]]
    y_te = y_all[np.where(test_mask.to_numpy())[0]]

    print(f"Train rows: {len(y_tr)}  Test rows: {len(y_te)}  Test trades: {len(test_ids)}")

    reg = LGBMRegressor(
        n_estimators=int(args.trees),
        learning_rate=float(args.learning_rate),
        max_depth=8,
        num_leaves=128,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    print("Training oracle-gap regressor...")
    t0 = time.time()
    reg.fit(X_tr, y_tr)
    dt = time.time() - t0
    print(f"Trained in {dt:.1f}s")

    pred_tr = reg.predict(X_tr)
    pred_te = reg.predict(X_te)

    print("=== Regression metrics (oracle_gap_pct) ===")
    print(f"Train RMSE: {rmse(pred_tr, y_tr):.4f}  MAE: {mae(pred_tr, y_tr):.4f}  Pearson: {pearson(pred_tr, y_tr):.4f}")
    print(f"Test  RMSE: {rmse(pred_te, y_te):.4f}  MAE: {mae(pred_te, y_te):.4f}  Pearson: {pearson(pred_te, y_te):.4f}")

    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip()]

    eval_df = policy_eval_gap(
        df.loc[test_mask].reset_index(drop=True),
        pred_te,
        hold_min=hold_min,
        taus=taus,
        min_exit_k=int(args.min_exit_k),
    )

    print("\n=== Policy evaluation (TEST trades) ===")
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(eval_df.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    if bool(args.save_dataset):
        out_ds = out_dir / f"exit_oracle_gap_dataset_hold{hold_min}_{ts}.parquet"
        df.to_parquet(out_ds, index=False)
        print(f"Saved gap dataset: {out_ds}")

    eval_path = out_dir / f"exit_oracle_gap_eval_hold{hold_min}_{ts}.csv"
    eval_df.to_csv(eval_path, index=False)

    model_path = out_dir / f"exit_oracle_gap_regressor_hold{hold_min}_{ts}.joblib"
    joblib.dump(
        {
            "model": reg,
            "feature_cols": feat_cols,
            "label": "oracle_gap_pct",
            "context": {
                "dataset_parquet": str(ds_path.resolve()),
                "hold_min": int(hold_min),
                "trees": int(args.trees),
                "learning_rate": float(args.learning_rate),
                "test_trade_frac": float(args.test_trade_frac),
                "min_exit_k": int(args.min_exit_k),
                "taus": list(taus),
            },
            "metrics": {
                "train_rmse": rmse(pred_tr, y_tr),
                "train_mae": mae(pred_tr, y_tr),
                "train_pearson": pearson(pred_tr, y_tr),
                "test_rmse": rmse(pred_te, y_te),
                "test_mae": mae(pred_te, y_te),
                "test_pearson": pearson(pred_te, y_te),
            },
            "created_utc": ts,
        },
        model_path,
    )

    print(f"Saved eval:  {eval_path}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
