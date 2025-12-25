#!/usr/bin/env python3
"""
Build per-entry, per-minute (1..10) exit regression dataset using model-fired entries.
Outputs under data/exit_regression/:
- exit_reg_dataset_<ts>.parquet
- exit_reg_dataset_sample_<ts>.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

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
    # round-trip: buy at entry open with +fee_side, sell at exit price with -fee_side
    mult = (exit_px * (1.0 - fee_side)) / (entry_px_open * (1.0 + fee_side))
    return (mult - 1.0) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Build exit regression dataset from selected entries")
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--selected-entries", required=True, help="CSV from score_entries_with_regressor_*.py with entry_time column")
    ap.add_argument("--fee", type=float, default=0.001, help="Per-side fee")
    ap.add_argument("--hold-min", type=int, default=10, help="Max minutes to consider for exit")
    ap.add_argument("--out-dir", default="data/exit_regression")
    args = ap.parse_args()

    ts = _now_ts()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load market and features
    mkt = minute_bars(Path(args.market_csv))
    feats = compute_feature_frame(mkt.rename(columns={"timestamp": "ts_min"}))
    ts_sorted = feats["ts_min"].to_numpy(dtype="datetime64[ns]")

    # Load entries
    ent = pd.read_csv(args.selected_entries, parse_dates=["entry_time"]) if isinstance(args.selected_entries, str) else args.selected_entries
    ent["entry_min"] = pd.to_datetime(ent["entry_time"]).dt.floor("min")

    entry_idx = map_ts_to_index(ts_sorted, ent["entry_min"].to_numpy(dtype="datetime64[ns]"))
    ok = entry_idx >= 0
    ent = ent.loc[ok].reset_index(drop=True)
    entry_idx = entry_idx[ok]

    close = mkt["close"].to_numpy(np.float64, copy=False)
    openp = mkt["open"].to_numpy(np.float64, copy=False)

    feat_arrs = {f: feats[f].to_numpy(np.float64, copy=False) for f in FEATURES}

    rows = []
    hold = int(args.hold_min)
    fee_side = float(args.fee)

    for i, idx0 in enumerate(entry_idx):
        e_open = openp[idx0]
        for k in range(1, hold + 1):
            idx_k = idx0 + k
            if idx_k >= len(close):
                break
            # Features at the exit minute k (no look-ahead beyond this minute)
            vals = {f: float(feat_arrs[f][idx_k]) for f in FEATURES}
            y_close = net_return_pct(e_open, float(close[idx_k]), fee_side)
            y_open = net_return_pct(e_open, float(openp[idx_k]), fee_side)
            rows.append({
                "entry_time": ent.loc[i, "entry_min"],
                "rel_min": k,
                **vals,
                "target_close_ret_pct": y_close,
                "target_open_ret_pct": y_open,
            })

    if not rows:
        raise SystemExit("No dataset rows were produced. Check selected entries and market range overlap.")

    df = pd.DataFrame(rows)
    df.sort_values(["entry_time", "rel_min"], inplace=True)

    pq = out_dir / f"exit_reg_dataset_{ts}.parquet"
    df.to_parquet(pq, index=False)

    sample_path = out_dir / f"exit_reg_dataset_sample_{ts}.csv"
    df.head(10000).to_csv(sample_path, index=False)

    print("Dataset:", pq)
    print("Sample:", sample_path)


if __name__ == "__main__":
    main()
