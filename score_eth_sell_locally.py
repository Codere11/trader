#!/usr/bin/env python3
"""Compute ETH SELL entry scores locally using the exact live agent feature engineering."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame

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


def _slope5_window(vals):
    v = np.asarray(vals, dtype=np.float64)
    if v.shape != (5,):
        return float("nan")
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return float(np.nansum(v * xc) / 10.0)


def _safe_std(vals):
    v = np.asarray(vals, dtype=np.float64)
    m = np.nanmean(v)
    if not np.isfinite(m):
        return float("nan")
    return float(np.nanmean((v - m) ** 2) ** 0.5)


def _agg5_from_window(win5):
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


def build_ctx_frame(bars_df):
    """Build context features from bars (causal, using shifted OHLCV)."""
    close_prev = pd.to_numeric(bars_df["close"], errors="coerce").shift(1)
    high_prev = pd.to_numeric(bars_df["high"], errors="coerce").shift(1)
    low_prev = pd.to_numeric(bars_df["low"], errors="coerce").shift(1)
    vol_prev = pd.to_numeric(bars_df["volume"], errors="coerce").shift(1)

    ctx_cols = {}
    for w in CTX_WINDOWS:
        w = int(w)
        ctx_cols[f"mom_{w}m_pct"] = close_prev.pct_change(w) * 100.0
        ctx_cols[f"vol_std_{w}m"] = close_prev.rolling(w, min_periods=w).std(ddof=0)

        rng = high_prev.rolling(w, min_periods=w).max() - low_prev.rolling(w, min_periods=w).min()
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

    return pd.DataFrame(ctx_cols)


def feature_map_at(i, bars_df, base_df, ctx_df):
    """Build feature map at index i (decision at close[i], uses i-5..i-1)."""
    i = int(i)
    if i < 5:
        return {}

    j0 = i - 5
    j1 = i

    close = pd.to_numeric(bars_df["close"], errors="coerce").to_numpy(np.float64)
    vol = pd.to_numeric(bars_df["volume"], errors="coerce").to_numpy(np.float64)

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
        if c not in base_df.columns:
            continue
        arr = pd.to_numeric(base_df[c], errors="coerce").to_numpy(np.float64)
        w = arr[j0:j1]
        if w.shape != (5,):
            continue
        agg = _agg5_from_window(w)
        for k, v in agg.items():
            out[f"{c}__{k}"] = float(v)

    # ctx features
    for c in ctx_df.columns:
        arr = pd.to_numeric(ctx_df[c], errors="coerce").to_numpy(np.float64)
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
        if c in base_df.columns:
            v = pd.to_numeric(base_df[c], errors="coerce").to_numpy(np.float64)[j0:j1]
            if np.any(~np.isfinite(v)):
                miss_any = True
                break
    if not miss_any:
        for c in ctx_df.columns:
            v = pd.to_numeric(ctx_df[c], errors="coerce").to_numpy(np.float64)[j0:j1]
            if np.any(~np.isfinite(v)):
                miss_any = True
                break
    out["missing_any"] = float(1.0 if miss_any else 0.0)

    return out


def main():
    # Load data
    data_path = REPO_ROOT / "data" / "eth_combined_for_scoring.csv"
    print(f"Loading data from {data_path}")
    bars = pd.read_csv(data_path)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"], utc=True)
    print(f"Loaded {len(bars)} bars")

    # Load model
    model_path = REPO_ROOT / "data" / "entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib"
    print(f"Loading model from {model_path}")
    model_obj = joblib.load(model_path)
    model = model_obj["model"]
    feature_cols = model_obj["feature_cols"]
    print(f"Model loaded: {len(feature_cols)} features")

    # Build base features
    print("Computing base features...")
    base_full = compute_feature_frame(bars)
    keep_base = [c for c in BASE_FEATS if c in base_full.columns]
    base = base_full[keep_base].copy().reset_index(drop=True)

    # Build ctx features
    print("Computing context features...")
    ctx = build_ctx_frame(bars).reset_index(drop=True)

    # Score all bars
    print("Computing entry scores...")
    scores = []
    timestamps = []
    
    for i in range(len(bars)):
        feat = feature_map_at(i, bars, base, ctx)
        if not feat:
            continue
        
        x = np.asarray([float(feat.get(c, float("nan"))) for c in feature_cols], dtype=np.float32)
        try:
            s = float(model.predict(np.asarray([x], dtype=np.float32))[0])
            if np.isfinite(s):
                scores.append(s)
                timestamps.append(bars.loc[i, "ts_min"])
        except Exception:
            pass

    print(f"\n{'='*80}")
    print(f"RESULTS: Computed {len(scores)} valid scores from {len(bars)} bars")
    print(f"{'='*80}")

    if len(scores) == 0:
        print("ERROR: No valid scores computed!")
        return

    scores_arr = np.array(scores)
    threshold = 0.344

    # Stats
    print(f"\nScore statistics:")
    print(f"  Min:    {scores_arr.min():.6f}")
    print(f"  Max:    {scores_arr.max():.6f}")
    print(f"  Mean:   {scores_arr.mean():.6f}")
    print(f"  Median: {np.median(scores_arr):.6f}")
    print(f"  Std:    {scores_arr.std():.6f}")

    # Threshold crossing
    above = scores_arr >= threshold
    n_above = above.sum()
    print(f"\nThreshold: {threshold}")
    print(f"Scores >= threshold: {n_above} ({100*n_above/len(scores):.2f}%)")

    if n_above > 0:
        print(f"\nTop 20 scores that crossed threshold:")
        idx_sorted = np.argsort(scores_arr)[::-1]
        for rank, idx in enumerate(idx_sorted[:20], 1):
            if scores_arr[idx] >= threshold:
                print(f"  {rank}. {timestamps[idx]} | Score: {scores_arr[idx]:.6f}")
    else:
        print("\nNo scores crossed the threshold.")
        print("\nTop 20 highest scores (all below threshold):")
        idx_sorted = np.argsort(scores_arr)[::-1]
        for rank, idx in enumerate(idx_sorted[:20], 1):
            print(f"  {rank}. {timestamps[idx]} | Score: {scores_arr[idx]:.6f}")


if __name__ == "__main__":
    main()
