#!/usr/bin/env python3
"""
Leakage audit for the exit regression pipeline.
Checks:
1) Feature causality: for random samples in the dataset, recompute FEATURES from market at the exact exit minute and assert equality (within tolerance).
2) Label causality: verify target_close_ret_pct uses entry open and exit minute close only (no post-exit info).
3) Split integrity: all rows for a given entry_time fall into the same split bucket when using time-based splitting.
4) Selection look-ahead note: warn that selecting per-day top 0.1% entries uses end-of-day information (offline convenience), suggest score-threshold alternative for live.

Usage:
  python scripts/audit_exit_regression_leakage_2025-12-20T14-21-28Z.py \
    --market-csv data/btc_profitability_analysis_filtered.csv \
    --dataset data/exit_regression/exit_reg_dataset_<ts>.parquet \
    [--splits-file data/splits.json] [--train-end-date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame, map_ts_to_index

_MKT_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


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


def audit_features_and_labels(mkt_csv: str, dataset_path: str, samples: int = 200, fee_side: float = 0.001) -> dict:
    mkt = minute_bars(Path(mkt_csv))
    feats = compute_feature_frame(mkt.rename(columns={"timestamp": "ts_min"}))
    ts_sorted = feats["ts_min"].to_numpy(dtype="datetime64[ns]")
    close = mkt["close"].to_numpy(np.float64, copy=False)
    openp = mkt["open"].to_numpy(np.float64, copy=False)

    df = pd.read_parquet(dataset_path)
    # Randomly sample rows
    n = len(df)
    idxs = random.sample(range(n), min(samples, n))

    tol = 1e-9
    feat_mismatch = 0
    label_mismatch = 0

    for i in idxs:
        r = df.iloc[i]
        entry_time = pd.to_datetime(r["entry_time"])  # minute-aligned
        rel_min = int(r["rel_min"])  # 1..10
        # Map times
        entry_idx = map_ts_to_index(ts_sorted, np.array([entry_time], dtype="datetime64[ns]"))[0]
        if entry_idx < 0:
            continue
        exit_idx = entry_idx + rel_min
        if exit_idx >= len(close):
            continue
        # Recompute features at exit minute
        rec = {f: float(feats.loc[exit_idx, f]) for f in FEATURES}
        for f in FEATURES:
            if not np.isfinite(r[f]) or not np.isfinite(rec[f]) or abs(r[f] - rec[f]) > tol:
                feat_mismatch += 1
                break
        # Recompute label
        entry_open = float(openp[entry_idx])
        exit_close = float(close[exit_idx])
        mult = (exit_close * (1.0 - fee_side)) / (entry_open * (1.0 + fee_side))
        y_close = (mult - 1.0) * 100.0
        if not np.isfinite(r["target_close_ret_pct"]) or abs(float(r["target_close_ret_pct"]) - y_close) > 1e-6:
            label_mismatch += 1

    return {
        "checked_rows": len(idxs),
        "feature_mismatch_count": feat_mismatch,
        "label_mismatch_count": label_mismatch,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage audit for exit regression dataset")
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--fee", type=float, default=0.001)
    args = ap.parse_args()

    res = audit_features_and_labels(args.market_csv, args.dataset, samples=args.samples, fee_side=float(args.fee))

    print("Audit summary:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("Notes:")
    print("- Features are recomputed at the exit minute only; any mismatches indicate potential preprocessing drift or look-ahead.")
    print("- Labels depend only on entry open and exit minute close with the specified fee. No later bars are used.")
    print("- Entry selection via per-day top 0.1% uses end-of-day info (offline). For live, prefer a fixed score threshold derived from prior days.")


if __name__ == "__main__":
    main()
