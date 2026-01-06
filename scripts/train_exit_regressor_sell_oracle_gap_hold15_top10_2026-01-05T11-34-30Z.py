#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T11:34:30Z
"""Train a SELL exit regressor (500 trees) on the top-10% entry-selected oracle-exit dataset.

Dataset expectation
- Produced by build_exit_oracle_dataset_sell_entrytop10_ctx120_hold15_v2_*.py
- One row per (trade_id, k_rel=1..15)
- Key columns include:
  - trade_id, entry_time, k_rel
  - ret_if_exit_now_pct
  - oracle_ret_pct
  - oracle_gap_pct = oracle_ret_pct - ret_if_exit_now_pct
  - decision-time feature columns (entry model feature space)
  - rolling PnL features (delta_mark_*, drawdown_from_peak_pct, mins_in_trade, ...)

Model
- LightGBM regressor, n_estimators=500.
- Label: oracle_gap_pct (pct points to oracle from now).

Evaluation
- Split by trades, time-ordered by entry_time (last test_trade_frac as test).
- Policy sweep on test trades:
  exit at first k where pred_gap_pct <= tau and k>=min_exit_k, else exit at k=hold_min.

Outputs
- data/exit_oracle_sell/
  - exit_oracle_gap_regressor_sell_hold15_top10_<ts>.joblib
  - exit_oracle_gap_regressor_sell_hold15_top10_eval_<ts>.csv
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

from lightgbm import LGBMRegressor


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


def progress_eta_callback(*, period: int = 10):
    """LightGBM callback that prints eval metrics + elapsed time + ETA.

    Works with the sklearn API when passed via reg.fit(..., callbacks=[...])
    and an eval_set is provided.
    """

    t0 = time.time()

    def _cb(env):
        nonlocal t0
        it = int(env.iteration) + 1
        total = int(env.end_iteration)
        if it == 1:
            t0 = time.time()

        if it % int(period) != 0 and it != total:
            return

        elapsed = time.time() - t0
        rate = elapsed / max(1, it)
        eta = rate * max(0, total - it)

        parts = []
        for tup in env.evaluation_result_list or []:
            data_name, metric_name, val, _ = tup
            parts.append(f"{data_name} {metric_name}={float(val):.6f}")
        evals = "  ".join(parts)

        print(f"[iter {it:>4}/{total}] {evals}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m", flush=True)

    return _cb


def policy_eval_gap(
    df: pd.DataFrame,
    pred_gap: np.ndarray,
    *,
    hold_min: int,
    taus: List[float],
    min_exit_k: int,
) -> pd.DataFrame:
    work = df[["trade_id", "k_rel", "ret_if_exit_now_pct", "oracle_ret_pct", "oracle_k", "oracle_gap_pct"]].copy()
    work["pred_gap_pct"] = pred_gap.astype(np.float64)

    by_trade = work.groupby("trade_id", sort=False)

    # baselines
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
                r = float(g2.loc[g2["k_rel"] == int(hold_min), "ret_if_exit_now_pct"].iloc[0])
            chosen_rets.append(r)
            chosen_ks.append(k)

        arr = np.asarray(chosen_rets, dtype=np.float64)
        out.append(
            {
                "tau_gap": float(tau),
                "policy": f"pred_gap<={tau}",
                "n_trades": float(len(arr)),
                "mean_ret_pct": float(np.mean(arr)),
                "median_ret_pct": float(np.median(arr)),
                "win_rate_gt0_pct": float(np.mean(arr > 0.0) * 100.0),
                "mean_exit_k": float(np.mean(np.asarray(chosen_ks, dtype=np.float64))),
            }
        )

    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SELL oracle-gap exit regressor (500 trees)")
    ap.add_argument(
        "--dataset-parquet",
        default="data/exit_oracle_sell/exit_oracle_rows_sell_hold15_top10_2026-01-05T11-31-54Z.parquet",
        help="Per-minute dataset parquet",
    )
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--test-trade-frac", type=float, default=0.2)

    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.05)

    ap.add_argument("--min-exit-k", type=int, default=2)
    ap.add_argument(
        "--taus",
        default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,1.00,1.50,2.00",
    )
    ap.add_argument("--progress-period", type=int, default=10, help="Print training progress every N boosting rounds")

    ap.add_argument("--out-dir", default="data/exit_oracle_sell")

    args = ap.parse_args()

    ds_path = Path(args.dataset_parquet)
    if not ds_path.exists():
        raise SystemExit(f"dataset not found: {ds_path}")

    hold_min = int(args.hold_min)

    print(f"Loading dataset: {ds_path}")
    df = pd.read_parquet(ds_path)

    # label
    df["oracle_gap_pct"] = pd.to_numeric(df["oracle_gap_pct"], errors="coerce")
    df = df[np.isfinite(df["oracle_gap_pct"].to_numpy(np.float64))].reset_index(drop=True)

    # time
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")

    ignore = {
        "trade_id",
        "signal_i",
        "entry_idx",
        "entry_time",
        "entry_px",
        "entry_pred",
        "top_frac",
        "entry_pred_threshold",
        "hold_min",
        "oracle_k",
        "oracle_ret_pct",
        "k_rel",
        "decision_idx",
        "decision_time",
        "y_oracle_exit",
        "ret_if_exit_now_pct",
        "oracle_gap_pct",
    }

    feat_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise SystemExit("No numeric feature columns found")

    print(f"Rows: {len(df):,}  Trades: {df['trade_id'].nunique():,}  Features: {len(feat_cols):,}")

    # split by trade, time-ordered
    trade_order = df[["trade_id", "entry_time"]].drop_duplicates("trade_id").sort_values("entry_time")
    n_tr = len(trade_order)
    n_test = max(1, int(n_tr * float(args.test_trade_frac)))
    test_ids = set(trade_order.tail(n_test)["trade_id"].tolist())

    train_mask = ~df["trade_id"].isin(test_ids)
    test_mask = df["trade_id"].isin(test_ids)

    X_tr = df.loc[train_mask, feat_cols].to_numpy(np.float32)
    y_tr = df.loc[train_mask, "oracle_gap_pct"].to_numpy(np.float32)
    X_te = df.loc[test_mask, feat_cols].to_numpy(np.float32)
    y_te = df.loc[test_mask, "oracle_gap_pct"].to_numpy(np.float32)

    print(f"Train rows: {len(y_tr):,}  Test rows: {len(y_te):,}  Test trades: {len(test_ids):,}")

    reg = LGBMRegressor(
        n_estimators=int(args.trees),
        learning_rate=float(args.learning_rate),
        max_depth=8,
        num_leaves=128,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    print("Training exit gap regressor (with progress)...")
    t0 = time.time()
    reg.fit(
        X_tr,
        y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric="l2",
        callbacks=[progress_eta_callback(period=int(args.progress_period))],
    )
    print(f"Trained in {time.time()-t0:.1f}s")

    pred_tr = reg.predict(X_tr)
    pred_te = reg.predict(X_te)

    metrics = {
        "train_rmse": rmse(pred_tr, y_tr),
        "test_rmse": rmse(pred_te, y_te),
        "train_mae": mae(pred_tr, y_tr),
        "test_mae": mae(pred_te, y_te),
        "test_pearson": pearson(pred_te, y_te),
    }

    print("=== Regression metrics (oracle_gap_pct) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip()]
    eval_df = policy_eval_gap(
        df.loc[test_mask].reset_index(drop=True),
        pred_te,
        hold_min=hold_min,
        taus=taus,
        min_exit_k=int(args.min_exit_k),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    eval_path = out_dir / f"exit_oracle_gap_regressor_sell_hold{hold_min}_top10_eval_{ts}.csv"
    model_path = out_dir / f"exit_oracle_gap_regressor_sell_hold{hold_min}_top10_{ts}.joblib"

    eval_df.to_csv(eval_path, index=False)

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
                "taus": taus,
            },
            "metrics": metrics,
            "created_utc": ts,
        },
        model_path,
    )

    print(f"Saved eval:  {eval_path}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
