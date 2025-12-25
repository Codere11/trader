#!/usr/bin/env python3
"""
Offline evaluator for the exit regressor:
- Inputs: market CSV, selected entries CSV, exit regressor model.
- For each entry_time, predict y_hat at rel_min 1..10 and choose argmax; compute realized return and aggregate by day.
Outputs:
- data/exit_regression/eval_exit_regressor_daily_<ts>.csv
- data/exit_regression/eval_exit_regressor_trades_<ts>.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame, map_ts_to_index

_MKT_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def minute_bars(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, parse_dates=["timestamp"], usecols=_MKT_COLS)
    if not raw["timestamp"].is_monotonic_increasing:
        raw = raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    raw["ts_min"] = raw["timestamp"].dt.floor("min")
    g = (
        raw.groupby("ts_min")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
        .reset_index()
        .rename(columns={"ts_min": "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return g


def net_return_pct(entry_px_open: float, exit_px: float, fee_side: float) -> float:
    mult = (exit_px * (1.0 - fee_side)) / (entry_px_open * (1.0 + fee_side))
    return (mult - 1.0) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline evaluate exit regressor with argmax or causal (greedy) policy")
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--selected-entries", required=True)
    ap.add_argument("--exit-model", required=True)
    ap.add_argument("--hold-min", type=int, default=10)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--policy", choices=["argmax","greedy_causal"], default="greedy_causal", help="argmax uses future info; greedy_causal exits when current prediction reaches a new high (no look-ahead)")
    ap.add_argument("--out-dir", default="data/exit_regression")
    args = ap.parse_args()

    ts = _now_ts()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Market + features
    mkt = minute_bars(Path(args.market_csv))
    feats = compute_feature_frame(mkt.rename(columns={"timestamp": "ts_min"}))
    ts_sorted = feats["ts_min"].to_numpy(dtype="datetime64[ns]")

    feat_arrs = {f: feats[f].to_numpy(np.float64, copy=False) for f in FEATURES}
    close = mkt["close"].to_numpy(np.float64, copy=False)
    openp = mkt["open"].to_numpy(np.float64, copy=False)

    payload = joblib.load(args.exit_model)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    ent = pd.read_csv(args.selected_entries, parse_dates=["entry_time"]) if isinstance(args.selected_entries, str) else args.selected_entries
    ent["entry_min"] = pd.to_datetime(ent["entry_time"]).dt.floor("min")

    entry_idx = map_ts_to_index(ts_sorted, ent["entry_min"].to_numpy(dtype="datetime64[ns]"))
    ok = entry_idx >= 0
    ent = ent.loc[ok].reset_index(drop=True)
    entry_idx = entry_idx[ok]

    hold = int(args.hold_min)
    fee_side = float(args.fee)

    trades = []

    for i, idx0 in enumerate(entry_idx):
        e_time = ent.loc[i, "entry_min"]
        e_open = openp[idx0]
        best_k, best_pred = None, -1e9
        running_best = -1e9
        for k in range(1, hold + 1):
            idx_k = idx0 + k
            if idx_k >= len(close):
                break
            x = np.array([feat_arrs[f][idx_k] for f in FEATURES], dtype=np.float32)[None, :]
            yhat = float(model.predict(x)[0])
            if args.policy == "argmax":
                if yhat > best_pred:
                    best_pred, best_k = yhat, k
            else:  # greedy_causal
                if yhat >= running_best:
                    running_best = yhat
                    best_pred, best_k = yhat, k
        if best_k is None:
            continue
        realized = net_return_pct(e_open, float(close[idx0 + best_k]), fee_side)
        trades.append({
            "entry_time": e_time,
            "exit_rel_min": best_k,
            "predicted_ret_pct": best_pred,
            "realized_ret_pct": realized,
            "policy": args.policy,
        })

    if not trades:
        raise SystemExit("No trades evaluated. Check inputs.")

    tdf = pd.DataFrame(trades)
    tdf["date"] = pd.to_datetime(tdf["entry_time"]).dt.date
    daily = tdf.groupby("date", as_index=False).agg(
        n_trades=("realized_ret_pct", "size"),
        mean_daily_pct=("realized_ret_pct", "mean"),
        sum_daily_pct=("realized_ret_pct", "sum"),
        median_daily_pct=("realized_ret_pct", "median"),
        top_day_pct=("realized_ret_pct", "max"),
        worst_day_pct=("realized_ret_pct", "min"),
    )

    trades_path = out_dir / f"eval_exit_regressor_trades_{ts}.csv"
    daily_path = out_dir / f"eval_exit_regressor_daily_{ts}.csv"
    tdf.to_csv(trades_path, index=False)
    daily.to_csv(daily_path, index=False)

    print("Trades:", trades_path)
    print("Daily:", daily_path)


if __name__ == "__main__":
    main()
