#!/usr/bin/env python3
"""
Timestamp (UTC): 2026-01-02T17:14:24Z

Fetch the last 4 hours of minute data and run through entry/exit models
to analyze what signals were generated and what the thresholds looked like.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame

BINANCE_FAPI = "https://fapi.binance.com"

def load_models(entry_path: Path, exit_path: Path):
    ep = joblib.load(entry_path)
    xp = joblib.load(exit_path)

    entry_model = ep["model"] if isinstance(ep, dict) and "model" in ep else ep
    exit_model = xp["model"] if isinstance(xp, dict) and "model" in xp else xp

    entry_features = list(ep["features"]) if isinstance(ep, dict) and "features" in ep else list(FEATURES)
    exit_features = list(xp["features"]) if isinstance(xp, dict) and "features" in xp else list(FEATURES)

    return {
        "entry_model": entry_model,
        "exit_model": exit_model,
        "entry_features": entry_features,
        "exit_features": exit_features,
    }


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": "1m", "startTime": start_ms, "endTime": end_ms, "limit": 1500}
    r = requests.get(f"{BINANCE_FAPI}/fapi/v1/klines", params=params, timeout=10)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        arr,
        columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def minute_bars_from_klines(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    out = []
    s = start
    while s < end:
        e = min(s + timedelta(minutes=1490), end)
        df = fetch_klines(symbol, int(s.timestamp() * 1000), int(e.timestamp() * 1000))
        if not df.empty:
            out.append(df)
        s = e
    if not out:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    bars = pd.concat(out, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return bars


def build_feature_frame(bars: pd.DataFrame, pre_min: int, feat_names: list[str]) -> pd.DataFrame:
    base = compute_feature_frame(bars.rename(columns={"timestamp": "ts_min"}))
    src = base[[c for c in FEATURES if c in base.columns]]
    
    # Add rolling mean features
    ctx_mean = src.rolling(pre_min, min_periods=pre_min).mean().add_suffix(f"_mean_{pre_min}m")
    full = pd.concat([src, ctx_mean], axis=1)
    
    missing = [c for c in feat_names if c not in full.columns]
    for c in missing:
        full[c] = np.nan
    
    out = pd.concat([base[["ts_min"]], full[feat_names]], axis=1).rename(columns={"ts_min": "timestamp"})
    return out


def main():
    symbol = "BTCUSDT"
    hours_back = 4
    
    # Load models
    entry_path = REPO_ROOT / "models" / "entry_regressor_btcusdt_2025-12-20T13-47-55Z.joblib"
    exit_path = REPO_ROOT / "models" / "exit_regressor_btcusdt_2025-12-20T14-41-31Z.joblib"
    
    print(f"Loading models from:")
    print(f"  Entry: {entry_path.name}")
    print(f"  Exit:  {exit_path.name}")
    
    models = load_models(entry_path, exit_path)
    
    # Fetch recent data
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours_back + 1)  # Extra buffer for feature calculation
    
    print(f"\nFetching {hours_back}h of minute data for {symbol}...")
    print(f"  Start: {start.isoformat()}")
    print(f"  End:   {end.isoformat()}")
    
    bars = minute_bars_from_klines(symbol, start, end)
    print(f"  Fetched {len(bars)} bars")
    
    if bars.empty:
        print("No data fetched!")
        return
    
    # Build features (using pre_min=5 as default)
    pre_min = 5
    print(f"\nBuilding features with pre_min={pre_min}...")
    entry_feats = build_feature_frame(bars, pre_min, models["entry_features"])
    
    print(f"  Feature frame shape: {entry_feats.shape}")
    
    # Score all bars
    print(f"\nScoring entries...")
    valid_mask = entry_feats[models["entry_features"]].notna().all(axis=1)
    valid_indices = entry_feats.index[valid_mask].tolist()
    
    if not valid_indices:
        print("No valid feature rows to score!")
        return
    
    scores = []
    for i in valid_indices:
        x = entry_feats.loc[i, models["entry_features"]].to_numpy(dtype=np.float32)[None, :]
        score = float(models["entry_model"].predict(x)[0])
        scores.append({"timestamp": entry_feats.loc[i, "timestamp"], "score": score, "index": i})
    
    scores_df = pd.DataFrame(scores)
    scores_df["timestamp"] = pd.to_datetime(scores_df["timestamp"], utc=True)
    
    # Focus on last 4 hours only
    cutoff = end - timedelta(hours=hours_back)
    scores_df = scores_df[scores_df["timestamp"] >= cutoff].copy()
    
    print(f"  Scored {len(scores_df)} bars in the last {hours_back} hours")
    
    # Calculate statistics
    if scores_df.empty:
        print("No scores in time window!")
        return
    
    scores_array = scores_df["score"].values
    
    print(f"\n{'='*60}")
    print(f"ENTRY SCORE ANALYSIS (Last {hours_back} hours)")
    print(f"{'='*60}")
    print(f"Total bars scored: {len(scores_df)}")
    print(f"Score range: [{scores_array.min():.4f}, {scores_array.max():.4f}]")
    print(f"Mean score: {scores_array.mean():.4f}")
    print(f"Median score: {np.median(scores_array):.4f}")
    print(f"Std dev: {scores_array.std():.4f}")
    
    print(f"\nPercentile thresholds:")
    for pct in [90, 95, 97, 99, 99.5]:
        threshold = np.percentile(scores_array, pct)
        n_above = (scores_array >= threshold).sum()
        print(f"  {pct:5.1f}%: {threshold:8.4f}  ({n_above:3d} bars above)")
    
    # Find top signals
    top_n = 10
    top_scores = scores_df.nlargest(top_n, "score")
    
    print(f"\n{'='*60}")
    print(f"TOP {top_n} ENTRY SIGNALS")
    print(f"{'='*60}")
    for idx, row in top_scores.iterrows():
        ts = row["timestamp"]
        score = row["score"]
        bar_idx = int(row["index"])
        bar = bars.iloc[bar_idx]
        print(f"{ts.strftime('%Y-%m-%d %H:%M')} UTC  |  Score: {score:7.4f}  |  Price: ${bar['close']:,.2f}")
    
    # Check if any would have triggered with common thresholds
    print(f"\n{'='*60}")
    print(f"SIGNAL TRIGGERS (example thresholds)")
    print(f"{'='*60}")
    
    for target_frac in [0.01, 0.02, 0.05]:
        threshold = np.percentile(scores_array, (1 - target_frac) * 100)
        triggered = scores_df[scores_df["score"] >= threshold]
        print(f"Target frac {target_frac:.2%} (threshold={threshold:.4f}): {len(triggered)} signals")
        
        if len(triggered) > 0 and len(triggered) <= 5:
            for _, row in triggered.iterrows():
                print(f"  - {row['timestamp'].strftime('%H:%M')} UTC: score={row['score']:.4f}")
    
    # Price movement analysis
    print(f"\n{'='*60}")
    print(f"PRICE MOVEMENT")
    print(f"{'='*60}")
    recent_bars = bars[bars["timestamp"] >= cutoff].copy()
    if len(recent_bars) > 0:
        start_price = recent_bars.iloc[0]["close"]
        end_price = recent_bars.iloc[-1]["close"]
        high_price = recent_bars["high"].max()
        low_price = recent_bars["low"].min()
        
        pct_change = ((end_price - start_price) / start_price) * 100
        volatility = ((high_price - low_price) / start_price) * 100
        
        print(f"Start: ${start_price:,.2f}")
        print(f"End:   ${end_price:,.2f}")
        print(f"Change: {pct_change:+.2f}%")
        print(f"High:  ${high_price:,.2f}")
        print(f"Low:   ${low_price:,.2f}")
        print(f"Range: {volatility:.2f}%")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
