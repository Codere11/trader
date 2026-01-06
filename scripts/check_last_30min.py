#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-06T21:32:23Z
"""Quick backtest: check last 30 minutes for valid ETH SELL entries above threshold 0.39"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from dydx_v4_client.indexer.candles_resolution import CandlesResolution
from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame

# Indexer endpoint
INDEXER_URL = "https://indexer.dydx.trade"

# Model paths
ENTRY_MODEL_PATH = REPO_ROOT / "data" / "entry_regressor_oracle15m_sell_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z" / "models" / "entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib"

# Threshold from live agent
THRESHOLD = 0.39
TARGET_FRAC = 0.001

BASE_FEATS = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "macd",
    "vwap_dev_5m",
]

CTX_WINDOWS = [30, 60, 120]


def load_entry_model(path: Path):
    obj = joblib.load(path)
    return obj["model"], obj["feature_cols"]


async def fetch_candles(minutes: int = 30):
    """Fetch last N minutes of 1m candles from dYdX indexer"""
    client = IndexerClient(INDEXER_URL)
    
    # Need extra for context window (120 minutes)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=minutes + 120)
    
    print(f"Fetching candles from {start_time.isoformat()} to {end_time.isoformat()}")
    
    resp = await client.markets.get_perpetual_market_candles(
        market="ETH-USD",
        resolution="1MIN",
        from_iso=start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        to_iso=end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        limit=500,
    )
    
    if not resp or "candles" not in resp:
        raise SystemExit("Failed to fetch candles")
    
    candles = resp["candles"]
    if not candles:
        raise SystemExit("No candles returned")
    
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["startedAt"], utc=True)
    
    for col in ["open", "high", "low", "close", "volume"]:
        if col == "volume" and col not in df.columns:
            df[col] = df.get("baseTokenVolume", 0)
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Fetched {len(df)} candles")
    
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def compute_scores(bars: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    """Compute entry scores for all bars"""
    
    # Add ts_min column required by compute_feature_frame
    bars = bars.copy()
    bars["ts_min"] = bars["timestamp"]
    
    # Compute base features
    feat_df = compute_feature_frame(bars)
    
    # Add context windows
    for col in BASE_FEATS:
        if col not in feat_df.columns:
            continue
        for w in CTX_WINDOWS:
            feat_df[f"{col}_ctx{w}"] = feat_df[col].rolling(w, min_periods=1).mean()
    
    # Align with feature_cols
    X = feat_df[feature_cols].fillna(0).values
    
    # Predict
    scores = model.predict(X)
    
    result = bars[["timestamp", "close"]].copy()
    result["score"] = scores
    
    return result


async def main():
    print("=" * 60)
    print("ETH-USD SELL: Last 30 minute entry check")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading entry model from {ENTRY_MODEL_PATH.name}")
    model, feature_cols = load_entry_model(ENTRY_MODEL_PATH)
    
    # Fetch data
    bars = await fetch_candles(minutes=30)
    
    # Compute scores
    print("\nComputing scores...")
    result = compute_scores(bars, model, feature_cols)
    
    # Filter to last 30 minutes only
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    recent = result[result["timestamp"] >= cutoff].copy()
    
    print(f"\nLast 30 minutes: {len(recent)} bars")
    print(f"Threshold: {THRESHOLD:.4f}")
    print(f"Target frac: {TARGET_FRAC} (implies threshold ~0.39)")
    
    # Check for valid entries
    valid_entries = recent[recent["score"] >= THRESHOLD]
    
    print(f"\n{'='*60}")
    print(f"VALID ENTRIES (score >= {THRESHOLD}): {len(valid_entries)}")
    print(f"{'='*60}")
    
    if len(valid_entries) > 0:
        print("\n⚠️  WARNING: Found valid entries that should have fired!")
        print("\nDetails:")
        for idx, row in valid_entries.iterrows():
            print(f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC | Price: ${row['close']:.2f} | Score: {row['score']:.4f}")
        
        print("\n❌ ENTRY LOGIC IS BROKEN - trades should have fired but didn't!")
        return 1
    else:
        print("\n✅ No valid entries found in last 30 minutes")
        print("   (This is expected if market hasn't crossed threshold)")
        
        # Show top scores
        print(f"\nTop 5 scores in last 30 min:")
        top5 = recent.nlargest(5, "score")
        for idx, row in top5.iterrows():
            print(f"  {row['timestamp'].strftime('%H:%M:%S')} | Score: {row['score']:.4f} (need {THRESHOLD:.4f})")
        
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
