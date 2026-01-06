#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-02T21:38:50Z
"""
Generate oracle entry training dataset for BTC-USD perp:
1. Find all "oracle entries" = minutes where best exit within next 10 minutes yields >= +0.17% (net, after 0.1% total fees)
2. For each oracle entry, store 5 minutes of pre-entry context:
   - For each of the 5 minutes before entry: open, close, and all features
3. Output: parquet with oracle entries (y=1) for training new entry model

This creates a dataset of "perfect entries" where there's actually profit opportunity.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def net_return_pct(entry_px: float, exit_px: float, fee_side: float) -> float:
    """Net return % at 1x (accounts for entry and exit fees)."""
    gross_mult = float(exit_px) / max(1e-12, float(entry_px))
    net_mult = gross_mult * (1.0 - fee_side) / (1.0 + fee_side)
    return float((net_mult - 1.0) * 100.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate oracle entry training data (entries with +0.17% opportunity)")
    ap.add_argument("--market-csv", required=True, help="CSV with timestamp, open, high, low, close, volume")
    ap.add_argument("--min-profit-pct", type=float, default=0.17, help="Minimum profit % for oracle entry (net, after fees)")
    ap.add_argument("--hold-min", type=int, default=10, help="Max holding period in minutes")
    ap.add_argument("--fee", type=float, default=0.001, help="Per-side fee (0.001 = 0.1%)")
    ap.add_argument("--pre-entry-min", type=int, default=5, help="Minutes of pre-entry context")
    ap.add_argument("--out-dir", default="data/entry_oracle")
    
    args = ap.parse_args()
    
    # Load market data
    print("Loading market data...")
    bars = pd.read_csv(args.market_csv, parse_dates=["timestamp"])
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")
    
    print(f"Loaded {len(bars)} bars from {bars['timestamp'].min()} to {bars['timestamp'].max()}")
    
    # Compute features
    print("Computing features...")
    base = compute_feature_frame(bars)
    src = base[[c for c in FEATURES if c in base.columns]]
    
    # For pre-entry context, we need features at each minute
    # Combine base features with bars data
    full_data = pd.concat([
        bars[["timestamp", "open", "close"]].rename(columns={"timestamp": "ts_min"}),
        src
    ], axis=1)
    
    print(f"Features computed: {len(src.columns)} features")
    
    # Find oracle entries
    print(f"\nScanning for oracle entries (min profit >= {args.min_profit_pct}%)...")
    
    close_arr = bars["close"].to_numpy(np.float64)
    open_arr = bars["open"].to_numpy(np.float64)
    
    oracle_entries = []
    hold_max = int(args.hold_min)
    fee_side = float(args.fee)
    min_profit = float(args.min_profit_pct)
    pre_min = int(args.pre_entry_min)
    
    # Start from pre_min (need 5 minutes of history) and leave room for hold_max
    for entry_idx in range(pre_min, len(bars) - hold_max):
        entry_px = open_arr[entry_idx]
        
        # Find best exit within hold_max minutes
        best_ret = -1e9
        best_k = None
        
        for k in range(1, hold_max + 1):
            exit_idx = entry_idx + k
            if exit_idx >= len(bars):
                break
            
            exit_px = close_arr[exit_idx]
            ret = net_return_pct(entry_px, exit_px, fee_side)
            
            if ret > best_ret:
                best_ret = ret
                best_k = k
        
        # Check if this is an oracle entry (profitable opportunity exists)
        if best_ret >= min_profit:
            # Build record with 5 minutes of pre-entry context
            record = {
                "entry_time": bars.iloc[entry_idx]["timestamp"],
                "entry_open": entry_px,
                "entry_close": bars.iloc[entry_idx]["close"],
                "oracle_best_ret_pct": best_ret,
                "oracle_best_k": best_k,
            }
            
            # Add 5 minutes of pre-entry context
            for pre_k in range(1, pre_min + 1):
                ctx_idx = entry_idx - pre_k
                
                # Prices
                record[f"pre{pre_k}_open"] = open_arr[ctx_idx]
                record[f"pre{pre_k}_close"] = close_arr[ctx_idx]
                
                # Features at that minute
                for feat in src.columns:
                    record[f"pre{pre_k}_{feat}"] = src.iloc[ctx_idx][feat]
            
            # Add current minute (entry minute) features
            for feat in src.columns:
                record[f"entry_{feat}"] = src.iloc[entry_idx][feat]
            
            oracle_entries.append(record)
    
    print(f"Found {len(oracle_entries)} oracle entries (out of {len(bars)} total minutes)")
    print(f"Oracle entry rate: {len(oracle_entries)/len(bars)*100:.4f}%")
    
    # Save output
    df = pd.DataFrame(oracle_entries)
    ts = _now_ts()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / f"oracle_entry_training_data_{ts}.parquet"
    df.to_parquet(out_path, index=False)
    
    print(f"\nSaved: {out_path}")
    print(f"\nOracle entry stats:")
    print(f"  Mean best return: {df['oracle_best_ret_pct'].mean():.4f}%")
    print(f"  Median best return: {df['oracle_best_ret_pct'].median():.4f}%")
    print(f"  Min best return: {df['oracle_best_ret_pct'].min():.4f}%")
    print(f"\n  Best exit timing distribution:")
    print(df['oracle_best_k'].value_counts().sort_index())


if __name__ == "__main__":
    main()
