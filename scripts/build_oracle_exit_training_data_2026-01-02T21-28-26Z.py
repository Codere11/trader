#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-02T21:28:26Z
"""
Generate training data for exit model retraining:
1. Use entry regressor to select entries (causal daily threshold)
2. For each entry, find the BEST possible exit within 10 minutes (oracle)
3. Store: entry features + 5 minutes of pre-exit features + oracle exit timing + realized return

Output: CSV with columns:
- entry_time
- entry_features (all entry model features)
- exit_rel_min (oracle best exit, 1-10)
- exit_features_k1 through exit_features_k10 (features at each minute)
- oracle_ret_pct (return at oracle exit)
- realized_ret_pct_k1 through realized_ret_pct_k10 (return if exited at each minute)

This will let us retrain the exit model to predict "best exit" properly.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
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


def _parse_pre_min(payload: Any) -> int:
    ctx = payload.get("context") if isinstance(payload, dict) else None
    if not isinstance(ctx, dict):
        return 0
    pm = ctx.get("pre_min")
    if pm is None:
        return 0
    try:
        return max(0, int(pm))
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate oracle exit training data from entry signals")
    ap.add_argument("--market-csv", required=True, help="CSV with timestamp, open, high, low, close, volume")
    ap.add_argument("--entry-model", default=str(REPO_ROOT / "models" / "entry_regressor_btcusdt_2025-12-20T13-47-55Z.joblib"))
    ap.add_argument("--target-frac", type=float, default=0.001, help="Entry threshold quantile (0.001 = 0.1% of minutes)")
    ap.add_argument("--hold-min", type=int, default=10, help="Max holding period in minutes")
    ap.add_argument("--fee", type=float, default=0.001, help="Per-side fee")
    ap.add_argument("--out-dir", default="data/exit_regression")
    
    args = ap.parse_args()
    
    # Load market data
    print("Loading market data...")
    bars = pd.read_csv(args.market_csv, parse_dates=["timestamp"])
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")
    
    print(f"Loaded {len(bars)} bars from {bars['timestamp'].min()} to {bars['timestamp'].max()}")
    
    # Load entry model
    print("Loading entry model...")
    entry_art = joblib.load(args.entry_model)
    if not isinstance(entry_art, dict) or "model" not in entry_art or "features" not in entry_art:
        raise ValueError("Entry model must have 'model' and 'features' keys")
    
    entry_model = entry_art["model"]
    entry_features = list(entry_art["features"])
    entry_pre_min = int(_parse_pre_min(entry_art))
    
    print(f"Entry model: {len(entry_features)} features, pre_min={entry_pre_min}")
    
    # Build features with rolling context
    print("Computing features...")
    # compute_feature_frame expects ts_min column
    base = compute_feature_frame(bars)
    src = base[[c for c in FEATURES if c in base.columns]]
    
    # Add rolling context means
    full = src.copy()
    if entry_pre_min > 0:
        ctx_mean = src.rolling(int(entry_pre_min), min_periods=int(entry_pre_min)).mean().add_suffix(f"_mean_{int(entry_pre_min)}m")
        full = pd.concat([full, ctx_mean], axis=1)
    
    # Check for missing columns
    missing_entry = [c for c in entry_features if c not in full.columns]
    if missing_entry:
        raise ValueError(f"Missing entry features: {missing_entry}")
    
    # Build final feature frame
    feats = pd.concat([base[["ts_min"]], full[entry_features]], axis=1).rename(columns={"ts_min": "timestamp"})
    feats["date"] = pd.to_datetime(feats["timestamp"]).dt.date
    
    # Score all minutes with entry model
    print("Scoring entries...")
    entry_scores = entry_model.predict(feats[entry_features].to_numpy(dtype=np.float32))
    feats["entry_score"] = entry_scores
    
    # Causal per-day threshold
    dates = sorted(feats["date"].unique())
    prior_scores: List[float] = []
    thresholds: Dict[Any, float] = {}
    
    for d in dates:
        if len(prior_scores) > 0:
            thresholds[d] = float(np.quantile(prior_scores, 1.0 - args.target_frac))
        else:
            thresholds[d] = float("-inf")
        
        day_scores = feats[feats["date"] == d]["entry_score"].tolist()
        prior_scores.extend([float(x) for x in day_scores if np.isfinite(float(x))])
    
    feats["entry_threshold"] = feats["date"].map(thresholds)
    
    # Select entries
    entries = feats[(feats["entry_score"] > feats["entry_threshold"]) & 
                    (feats["entry_threshold"].notna())].copy()
    
    print(f"Selected {len(entries)} entry signals")
    
    # For each entry, find oracle best exit and collect pre-exit features
    close_arr = bars["close"].to_numpy(np.float64)
    open_arr = bars["open"].to_numpy(np.float64)
    
    oracle_data = []
    hold_max = int(args.hold_min)
    fee_side = float(args.fee)
    
    print("Finding oracle exits...")
    for idx, row in entries.iterrows():
        entry_idx = idx + 1  # Entry at next minute open
        if entry_idx >= len(bars):
            continue
        
        entry_time = bars.iloc[entry_idx]["timestamp"]
        entry_px = bars.iloc[entry_idx]["open"]
        
        # Find best exit within hold_max minutes
        best_k = None
        best_ret = -1e9
        returns_by_k = {}
        
        for k in range(1, hold_max + 1):
            exit_idx = entry_idx + k
            if exit_idx >= len(bars):
                break
            
            exit_px = close_arr[exit_idx]
            ret = net_return_pct(entry_px, exit_px, fee_side)
            returns_by_k[k] = ret
            
            if ret > best_ret:
                best_ret = ret
                best_k = k
        
        if best_k is None:
            continue
        
        # Store entry + oracle exit + all k returns + features at each k
        record = {
            "entry_time": entry_time,
            "entry_score": row["entry_score"],
            "entry_threshold": row["entry_threshold"],
            "entry_price": entry_px,
            "oracle_exit_k": best_k,
            "oracle_ret_pct": best_ret,
        }
        
        # Add entry features
        for feat in entry_features:
            record[f"entry_{feat}"] = row[feat]
        
        # Add returns at each k
        for k in range(1, hold_max + 1):
            record[f"ret_at_k{k}"] = returns_by_k.get(k, np.nan)
        
        # Add features at each k (for pre-exit context analysis)
        for k in range(1, hold_max + 1):
            feat_idx = entry_idx + k
            if feat_idx < len(feats):
                for feat in FEATURES:
                    if feat in full.columns:
                        record[f"k{k}_{feat}"] = full.iloc[feat_idx][feat]
        
        oracle_data.append(record)
    
    print(f"Generated {len(oracle_data)} oracle exit records")
    
    # Save output
    df = pd.DataFrame(oracle_data)
    ts = _now_ts()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / f"oracle_exit_training_data_{ts}.csv"
    df.to_csv(out_path, index=False)
    
    print(f"\nSaved: {out_path}")
    print(f"\nOracle exit distribution:")
    print(df["oracle_exit_k"].value_counts().sort_index())
    print(f"\nMean oracle return: {df['oracle_ret_pct'].mean():.4f}%")
    print(f"Median oracle return: {df['oracle_ret_pct'].median():.4f}%")


if __name__ == "__main__":
    main()
