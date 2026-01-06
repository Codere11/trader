#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-06T18:48:00Z
"""Check if any ETH SELL scores crossed the threshold."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import joblib
from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame

# Load model
model_obj = joblib.load("data/entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib")
model = model_obj["model"]
feature_cols = model_obj["feature_cols"]
pre_min = int(model_obj.get("pre_min", 5))

print(f"Model loaded: pre_min={pre_min}, {len(feature_cols)} features")

# Load combined bars
bars = pd.read_csv("data/eth_combined_for_scoring.csv")
bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
bars["ts_min"] = bars["timestamp"].dt.floor("min")
print(f"Loaded {len(bars)} bars from {bars['timestamp'].min()} to {bars['timestamp'].max()}")

# Compute base features
base = compute_feature_frame(bars)

# Add ctx windows (30, 60, 120)
for win in [30, 60, 120]:
    for col in ["ret_1m_pct", "mom_3m_pct", "mom_5m_pct", "vol_std_5m", "range_5m", "range_norm_5m", "macd", "vwap_dev_5m"]:
        if col in base.columns:
            s = pd.to_numeric(base[col], errors="coerce")
            base[f"{col}_ctx{win}_mean"] = s.rolling(win, min_periods=win).mean()
            base[f"{col}_ctx{win}_std"] = s.rolling(win, min_periods=win).std()
            base[f"{col}_ctx{win}_min"] = s.rolling(win, min_periods=win).min()
            base[f"{col}_ctx{win}_max"] = s.rolling(win, min_periods=win).max()

# Score all bars (only those after pre_min warmup and ctx120 warmup)
threshold = 0.344
scores = []
timestamps = []

min_idx = max(pre_min, 120)  # Need ctx120 window
for i in range(min_idx, len(bars)):
    try:
        # Get features for this bar
        row = base.loc[i, feature_cols]
        vals = pd.to_numeric(row, errors="coerce").to_numpy(dtype=np.float32).reshape(1, -1)
        
        if not np.all(np.isfinite(vals)):
            continue
            
        score = float(model.predict(vals)[0])
        scores.append(score)
        timestamps.append(bars.loc[i, "timestamp"])
        
    except Exception as e:
        continue

print(f"\nScored {len(scores)} bars")
print(f"Threshold: {threshold:.6f}")

if scores:
    max_score = max(scores)
    max_idx = scores.index(max_score)
    max_ts = timestamps[max_idx]
    
    print(f"Max score: {max_score:.6f} at {max_ts}")
    print(f"Crossed threshold: {'YES' if max_score >= threshold else 'NO'}")
    
    # Count how many crossed
    crossed = sum(1 for s in scores if s >= threshold)
    print(f"\nBars crossing threshold: {crossed} / {len(scores)} ({100*crossed/len(scores):.2f}%)")
    
    if crossed > 0:
        print("\nTop 10 scores:")
        sorted_scores = sorted(zip(scores, timestamps), reverse=True)[:10]
        for s, ts in sorted_scores:
            print(f"  {s:.6f} at {ts}")
else:
    print("No valid scores computed")
