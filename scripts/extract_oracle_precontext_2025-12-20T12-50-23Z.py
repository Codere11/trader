#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import (
    FEATURES,
    build_dyn5,
    compute_feature_frame,
    map_ts_to_index,
    net_return_flat_round_trip,
)


_MKT_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def minute_bars(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        usecols=_MKT_COLS,
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )

    if not raw["timestamp"].is_monotonic_increasing:
        raw = raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    raw["ts_min"] = raw["timestamp"].dt.floor("min")
    g = (
        raw.groupby("ts_min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values("ts_min")
        .reset_index(drop=True)
    )
    return g


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract oracle-oracle trades and their 5-minute pre-entry feature context (fast, vectorized).",
    )
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv", help="CSV with timestamp/open/high/low/close/volume")
    ap.add_argument("--oracle-trades", required=True, help="CSV with oracle trades containing entry_time and exit_time (ISO timestamps)")
    ap.add_argument("--pre-min", type=int, default=5, help="Minutes of precontext to capture (uses negative offsets up to -pre-min)")
    ap.add_argument("--fee-round-trip", type=float, default=0.001, help="Total round-trip fee used for flat return calc")
    ap.add_argument("--out-dir", default="data/oracle_precontext", help="Directory to write parquet/csv outputs")
    ap.add_argument("--tag", default="oracle_precontext", help="Prefix for output filenames")
    ap.add_argument("--which", default="oracle", help="Label for which actor produced trades (e.g., oracle, agent_bad)")
    ap.add_argument("--target-col", default=None, help="Optional column name to add with constant target value")
    ap.add_argument("--target-val", default=None, type=float, help="Constant target value if --target-col is set")
    args = ap.parse_args()

    mkt = minute_bars(Path(args.market_csv))

    feats = compute_feature_frame(mkt)
    ts_sorted = feats["ts_min"].to_numpy(dtype="datetime64[ns]")

    # Load oracle trades
    oracle = pd.read_csv(Path(args.oracle_trades), parse_dates=["entry_time", "exit_time"])
    oracle["entry_min"] = oracle["entry_time"].dt.floor("min")
    oracle["exit_min"] = oracle["exit_time"].dt.floor("min")

    entry_idx = map_ts_to_index(ts_sorted, oracle["entry_min"].to_numpy(dtype="datetime64[ns]"))
    exit_idx = map_ts_to_index(ts_sorted, oracle["exit_min"].to_numpy(dtype="datetime64[ns]"))

    ok = (entry_idx >= 0) & (exit_idx >= 0)
    oracle = oracle.loc[ok].reset_index(drop=True)
    entry_idx = entry_idx[ok]
    exit_idx = exit_idx[ok]

    close = mkt["close"].to_numpy(np.float64, copy=False)
    oracle["net_return_flat_pct"] = net_return_flat_round_trip(close[entry_idx], close[exit_idx], float(args.fee_round_trip))
    oracle["duration_min"] = (oracle["exit_min"] - oracle["entry_min"]).dt.total_seconds() / 60.0

    feat_arrs = {name: feats[name].to_numpy(np.float64, copy=False) for name in FEATURES}
    offsets = np.arange(-int(args.pre_min), 0, dtype=np.int64)

    dyn = build_dyn5(feat_arrs, entry_idx, offsets)
    dyn["which"] = args.which
    dyn["phase"] = "pre_entry"

    dyn = dyn.merge(
        oracle.reset_index().rename(columns={"index": "trade_row"}),
        on="trade_row",
        how="left",
    )

    if args.target_col:
        dyn[args.target_col] = args.target_val

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_parquet = out_dir / f"{args.tag}_dyn{int(args.pre_min)}m_{ts}.parquet"
    out_csv = out_dir / f"{args.tag}_dyn{int(args.pre_min)}m_{ts}.csv"

    dyn.to_parquet(out_parquet, index=False)
    dyn.to_csv(out_csv, index=False)

    print(f"Oracle trades matched: {len(oracle):,}")
    print(f"Precontext rows: {len(dyn):,} (rel_min in [{offsets.min()}, {offsets.max()}])")
    print("Outputs:")
    print("  ", out_parquet)
    print("  ", out_csv)


if __name__ == "__main__":
    main()
