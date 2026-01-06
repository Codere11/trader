#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-06T21:45:44Z
"""Compute ETH SELL entry scores for last 30 minutes using EXACT live agent logic"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame

INDEXER_URL = "https://indexer.dydx.trade"
ENTRY_MODEL_PATH = REPO_ROOT / "data" / "entry_regressor_oracle15m_sell_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z" / "models" / "entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib"

THRESHOLD = 0.39

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


def _slope5_window(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64)
    if v.shape != (5,):
        return float("nan")
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return float(np.nansum(v * xc) / 10.0)


def _safe_std(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64)
    m = np.nanmean(v)
    if not np.isfinite(m):
        return float("nan")
    return float(np.nanmean((v - m) ** 2) ** 0.5)


def _agg5_from_window(win5: np.ndarray) -> Dict[str, float]:
    w = np.asarray(win5, dtype=np.float64)
    return {
        "last": float(w[-1]),
        "mean5": float(np.nanmean(w)),
        "std5": float(_safe_std(w)),
        "min5": float(np.nanmin(w)),
        "max5": float(np.nanmax(w)),
        "range5": float(np.nanmax(w) - np.nanmin(w)),
        "slope5": float(_slope5_window(w)),
    }


async def fetch_candles():
    """Fetch last 30 min + 120 for context"""
    client = IndexerClient(INDEXER_URL)
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=150)
    
    print(f"Fetching {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} UTC")
    
    resp = await client.markets.get_perpetual_market_candles(
        market="ETH-USD",
        resolution="1MIN",
        from_iso=start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        to_iso=end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        limit=200,
    )
    
    if not resp or "candles" not in resp:
        raise SystemExit("Failed to fetch candles")
    
    df = pd.DataFrame(resp["candles"])
    df["timestamp"] = pd.to_datetime(df["startedAt"], utc=True)
    
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["volume"] = pd.to_numeric(df.get("baseTokenVolume", 0), errors="coerce").astype(float)
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Fetched {len(df)} candles")
    
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def compute_features(bars: pd.DataFrame):
    """Replicate EXACT live agent feature computation"""
    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"], utc=True)
    
    # Base features
    base_full = compute_feature_frame(bars)
    keep_base = [c for c in BASE_FEATS if c in base_full.columns]
    base = base_full[keep_base].copy()
    
    # Context features (CAUSAL - uses shifted data)
    close_prev = pd.to_numeric(bars["close"], errors="coerce").shift(1)
    high_prev = pd.to_numeric(bars["high"], errors="coerce").shift(1)
    low_prev = pd.to_numeric(bars["low"], errors="coerce").shift(1)
    vol_prev = pd.to_numeric(bars["volume"], errors="coerce").shift(1)
    
    ctx_cols = {}
    for w in CTX_WINDOWS:
        w = int(w)
        ctx_cols[f"mom_{w}m_pct"] = close_prev.pct_change(w) * 100.0
        ctx_cols[f"vol_std_{w}m"] = close_prev.rolling(w, min_periods=w).std(ddof=0)
        
        rng = (high_prev.rolling(w, min_periods=w).max() - low_prev.rolling(w, min_periods=w).min())
        ctx_cols[f"range_{w}m"] = rng
        ctx_cols[f"range_norm_{w}m"] = rng / close_prev.replace(0, np.nan)
        
        cmax = close_prev.rolling(w, min_periods=w).max()
        cmin = close_prev.rolling(w, min_periods=w).min()
        crng = cmax - cmin
        eps = 1e-9
        
        cp = close_prev
        ctx_cols[f"close_dd_from_{w}m_max_pct"] = (cmax / (cp.clip(lower=1e-9)) - 1.0) * 100.0
        ctx_cols[f"close_bounce_from_{w}m_min_pct"] = (cp / (cmin.clip(lower=1e-9)) - 1.0) * 100.0
        ctx_cols[f"close_pos_in_{w}m_range"] = (cp - cmin) / (crng + eps)
        
        v = pd.to_numeric(vol_prev, errors="coerce").fillna(0.0)
        cp2 = pd.to_numeric(cp, errors="coerce").fillna(0.0)
        sum_v = v.rolling(w, min_periods=w).sum()
        sum_pv = (v * cp2).rolling(w, min_periods=w).sum()
        vwap = sum_pv / (sum_v.clip(lower=1e-9))
        ctx_cols[f"vwap_dev_{w}m"] = np.where(sum_v > 0.0, ((cp2 - vwap) / (vwap.clip(lower=1e-9))) * 100.0, 0.0)
    
    ctx = pd.DataFrame(ctx_cols)
    
    return bars, base.reset_index(drop=True), ctx.reset_index(drop=True)


def feature_map_at(bars, base, ctx, feature_cols, i: int) -> Dict[str, float]:
    """Build feature map at index i (EXACT live agent logic)"""
    i = int(i)
    if i < 5:
        return {}
    
    j0 = i - 5
    j1 = i
    
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64)
    vol = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(np.float64)
    
    close_prev = close[j0:j1]
    if close_prev.shape != (5,):
        return {}
    
    close_last = float(close_prev[-1])
    if not (np.isfinite(close_last) and close_last > 0.0):
        return {}
    
    close_norm = (close_prev / close_last - 1.0) * 100.0
    px = _agg5_from_window(close_norm)
    
    px_ret5m = float(close_norm[-1] - close_norm[0])
    px_absret5m = float(abs(px_ret5m))
    
    vol_prev = vol[j0:j1]
    vol_log = np.log1p(np.maximum(0.0, vol_prev.astype(np.float64)))
    vv = _agg5_from_window(vol_log)
    
    out = {}
    
    # px_close_norm_pct
    for k, v in px.items():
        out[f"px_close_norm_pct__{k}"] = float(v)
    out["px_close_norm_pct__ret5m"] = float(px_ret5m)
    out["px_close_norm_pct__absret5m"] = float(px_absret5m)
    out["px_close_norm_pct__m5"] = float(close_norm[0])
    out["px_close_norm_pct__m4"] = float(close_norm[1])
    out["px_close_norm_pct__m3"] = float(close_norm[2])
    out["px_close_norm_pct__m2"] = float(close_norm[3])
    out["px_close_norm_pct__m1"] = float(close_norm[4])
    
    # vol_log1p
    for k, v in vv.items():
        out[f"vol_log1p__{k}"] = float(v)
    
    # base features
    for c in BASE_FEATS:
        if c not in base.columns:
            continue
        arr = pd.to_numeric(base[c], errors="coerce").to_numpy(np.float64)
        w = arr[j0:j1]
        if w.shape != (5,):
            continue
        agg = _agg5_from_window(w)
        for k, v in agg.items():
            out[f"{c}__{k}"] = float(v)
    
    # ctx features
    for c in ctx.columns:
        arr = pd.to_numeric(ctx[c], errors="coerce").to_numpy(np.float64)
        w = arr[j0:j1]
        if w.shape != (5,):
            continue
        agg = _agg5_from_window(w)
        for k, v in agg.items():
            out[f"{c}__{k}"] = float(v)
    
    # missing indicators
    missing_close_n = float(np.sum(~np.isfinite(close_prev)))
    out["missing_close_n"] = float(missing_close_n)
    
    miss_any = False
    if np.any(~np.isfinite(close_prev)):
        miss_any = True
    for c in BASE_FEATS:
        if c in base.columns:
            v = pd.to_numeric(base[c], errors="coerce").to_numpy(np.float64)[j0:j1]
            if np.any(~np.isfinite(v)):
                miss_any = True
                break
    if not miss_any:
        for c in ctx.columns:
            v = pd.to_numeric(ctx[c], errors="coerce").to_numpy(np.float64)[j0:j1]
            if np.any(~np.isfinite(v)):
                miss_any = True
                break
    out["missing_any"] = float(1.0 if miss_any else 0.0)
    
    return out


def entry_score_at(bars, base, ctx, model, feature_cols, i: int):
    """Compute entry score at index i"""
    feat = feature_map_at(bars, base, ctx, feature_cols, i)
    if not feat:
        return None
    x = np.asarray([float(feat.get(c, float("nan"))) for c in feature_cols], dtype=np.float32)
    try:
        s = float(model.predict(np.asarray([x], dtype=np.float32))[0])
    except Exception:
        return None
    return s if np.isfinite(s) else None


async def main():
    print("="*60)
    print("ETH-USD SELL: Last 30 min scores (EXACT live agent logic)")
    print("="*60)
    
    # Load model
    print(f"\nLoading: {ENTRY_MODEL_PATH.name}")
    obj = joblib.load(ENTRY_MODEL_PATH)
    model = obj["model"]
    feature_cols = obj["feature_cols"]
    
    # Fetch data
    bars_raw = await fetch_candles()
    
    # Compute features
    print("Computing features...")
    bars, base, ctx = compute_features(bars_raw)
    
    # Get last 30 minutes
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    
    results = []
    for i in range(len(bars)):
        ts = bars.iloc[i]["timestamp"]
        if ts < cutoff:
            continue
        
        score = entry_score_at(bars, base, ctx, model, feature_cols, i)
        if score is not None:
            results.append({
                "time": ts.strftime("%H:%M:%S"),
                "price": bars.iloc[i]["close"],
                "score": score,
                "above_thr": score >= THRESHOLD
            })
    
    print(f"\n{'='*60}")
    print(f"LAST 30 MINUTES: {len(results)} bars")
    print(f"Threshold: {THRESHOLD:.4f}")
    print(f"{'='*60}\n")
    
    valid_entries = [r for r in results if r["above_thr"]]
    
    if valid_entries:
        print(f"⚠️  {len(valid_entries)} VALID ENTRIES (should have fired):\n")
        for r in valid_entries:
            print(f"  {r['time']} | ${r['price']:.2f} | Score: {r['score']:.4f} ✓")
        print(f"\n❌ ENTRY LOGIC IS BROKEN!")
        return 1
    else:
        print("✅ No valid entries (scores below threshold)\n")
        print("Top 10 scores:")
        top10 = sorted(results, key=lambda x: x["score"], reverse=True)[:10]
        for r in top10:
            print(f"  {r['time']} | Score: {r['score']:.4f} (need {THRESHOLD:.4f})")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
