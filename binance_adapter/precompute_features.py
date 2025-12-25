#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Feature sets needed
PREFILTER_FEATS = [
'mom_1m','mom_3m','mom_5m','vol_std_5m','range_5m','range_norm_5m',
    'macd','vwap_dev',
    'score','confidence','p_buy'
]

# Ranker core features (superset; we compute commonly used ones)
RANKER_FEATS = [
    'momentum_5m','volatility_5m','price_range_5m','volume_avg_5m',
    'momentum_15m','volatility_15m','price_range_15m',
    'price_change_lag_1','price_change_lag_3','price_change_lag_5',
    'ctx_close_mean','ctx_close_std','ctx_close_min','ctx_close_max','ctx_close_range','ctx_close_slope',
    'ctx_vol_mean','ctx_vol_std','ctx_range_norm','ctx_std_norm','ctx_skew_10m',
    'hour','hour_sin','hour_cos','dow',
    'ret_1m','ret_2m','ret_3m',
    'z_momentum_5m','z_price_range_5m',
    'vwap_dev','order_flow_proxy','acceleration','rejection_count',
    'ema_cross','macd','macd_signal',
    'vol_regime_high','volume_regime_high','price_near_extreme',
    'confidence'
]

ALL_FEATS = sorted(set(PREFILTER_FEATS + RANKER_FEATS))


def rolling_slope(arr: np.ndarray) -> float:
    n = arr.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, arr, rcond=None)[0]
    return float(m)


def precompute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('timestamp').reset_index(drop=True)
    # basic returns
    # Prefilter-compatible momentum names
    df['mom_1m'] = df['close'].pct_change(1) * 100
    df['mom_3m'] = df['close'].pct_change(3) * 100
    df['mom_5m'] = df['close'].pct_change(5) * 100
    # Also keep generic returns for ranker micro features
    df['ret_1m'] = df['mom_1m']
    df['ret_2m'] = df['close'].pct_change(2) * 100
    df['ret_3m'] = df['mom_3m']
    # last3_same_sign
    # last3_same_sign removed as low-signal
    # windows
    df['volatility_5m'] = df['close'].rolling(5, min_periods=1).std()
    df['vol_std_5m'] = df['volatility_5m']
    df['momentum_5m'] = df['close'].pct_change(5) * 100
    df['volume_avg_5m'] = df['volume'].rolling(5, min_periods=1).mean()
    df['price_range_5m'] = df['high'].rolling(5, min_periods=1).max() - df['low'].rolling(5, min_periods=1).min()
    df['range_5m'] = df['price_range_5m']
    df['range_norm_5m'] = df['range_5m'] / df['close']

    df['volatility_15m'] = df['close'].rolling(15, min_periods=1).std()
    df['momentum_15m'] = df['close'].pct_change(15) * 100
    df['price_range_15m'] = df['high'].rolling(15, min_periods=1).max() - df['low'].rolling(15, min_periods=1).min()

    for lag in [1,3,5]:
        df[f'price_change_lag_{lag}'] = df['close'].pct_change(lag) * 100

    win = 10
    df['ctx_close_mean'] = df['close'].rolling(win, min_periods=1).mean()
    df['ctx_close_std']  = df['close'].rolling(win, min_periods=1).std()
    df['ctx_close_min']  = df['close'].rolling(win, min_periods=1).min()
    df['ctx_close_max']  = df['close'].rolling(win, min_periods=1).max()
    df['ctx_close_range']= df['ctx_close_max'] - df['ctx_close_min']
    df['ctx_close_slope']= df['close'].rolling(win, min_periods=2).apply(lambda x: rolling_slope(x.values), raw=False)
    # slope_ols_5m removed as low-signal
    df['ctx_vol_mean']   = df['volume'].rolling(win, min_periods=1).mean()
    df['ctx_vol_std']    = df['volume'].rolling(win, min_periods=1).std()

    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['dow']  = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24.0)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24.0)
    df['ctx_range_norm'] = df['ctx_close_range'] / df['close']
    df['ctx_std_norm']   = df['ctx_close_std'] / df['ctx_close_mean'].replace(0, np.nan)
    df['ctx_skew_10m']   = df['close'].rolling(win, min_periods=3).skew()

    # expanding z-scores (approx causal)
    def zexp(s: pd.Series) -> pd.Series:
        m = s.expanding(min_periods=30).mean()
        sd= s.expanding(min_periods=30).std()
        return ((s - m) / sd.replace(0, np.nan)).fillna(0)
    df['z_momentum_5m'] = zexp(df['momentum_5m'])
    df['z_price_range_5m'] = zexp(df['price_range_5m'])

    # VWAP deviation and order flow
    cv = df['volume'].cumsum()
    pv = (df['close']*df['volume']).cumsum()
    vwap = (pv / cv.replace(0, np.nan)).ffill()
    df['vwap_dev'] = ((df['close'] - vwap) / vwap * 100).fillna(0)
    df['co_diff'] = df['close'] - df['open']
    df['vw_co'] = df['co_diff'] * df['volume']
    df['order_flow_proxy'] = df['vw_co'].rolling(win, min_periods=1).sum() / df['volume'].rolling(win, min_periods=1).sum().replace(0, np.nan)

    # acceleration
    slope3 = df['close'].rolling(3, min_periods=2).apply(lambda x: rolling_slope(x.values), raw=False)
    df['acceleration'] = (slope3 / df['ctx_close_slope'].replace(0, np.nan)).fillna(0)

    # rejection_count approx
    h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float)
    c = df['close'].to_numpy(float)
    out = np.zeros(len(c))
    for i in range(len(c)):
        s = max(0, i - win + 1)
        hs = h[s:i+1]; ls = l[s:i+1]; cs = c[s:i+1]
        cnt = 0
        for j in range(len(cs)-1):
            rng = hs[j] - ls[j]
            if rng <= 0: continue
            if abs(cs[j]-hs[j])/rng < 0.01 and cs[j+1] < cs[j]: cnt += 1
            if abs(cs[j]-ls[j])/rng < 0.01 and cs[j+1] > cs[j]: cnt += 1
        out[i] = cnt
    df['rejection_count'] = out

    # technicals
    ema9  = df['close'].ewm(span=9, adjust=False).mean()
    ema21 = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_cross'] = ema9 - ema21
    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    # rsi_14 removed as low-signal
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    # macd_hist removed as low-signal

    # success regime flags
    z_std = zexp(df['ctx_close_std'])
    z_rng = zexp(df['ctx_close_range'])
    df['vol_regime_high'] = ((z_std > 0.5) | (z_rng > 0.5)).astype(int)
    z_vol = zexp(df['ctx_vol_mean'])
    df['volume_regime_high'] = (z_vol > 0.5).astype(int)
    near_low  = (df['close'] - df['ctx_close_min']) / (df['ctx_close_range'] + 1e-6) < 0.2
    near_high = (df['ctx_close_max'] - df['close']) / (df['ctx_close_range'] + 1e-6) < 0.2
    df['price_near_extreme'] = (near_low | near_high).astype(int)

    # placeholders for ranker extras
    df['score'] = 0.0
    df['confidence'] = 0.5
    df['p_buy'] = 0.5

    # Keep only required columns
    keep = ['timestamp','open','high','low','close','volume'] + ALL_FEATS
    # Drop low-signal columns if present
    drop_cols = ['last3_same_sign','rsi_14','macd_hist','slope_ols_5m']
    keep = [c for c in keep if c not in drop_cols]
    return df[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-file', required=True, help='Input CSV with timestamp, open, high, low, close, volume')
    ap.add_argument('--out', default=None, help='Output Parquet path')
    args = ap.parse_args()

    inp = Path(args.cache_file)
    df = pd.read_csv(inp, parse_dates=['timestamp'])
    print(f'Loaded {len(df):,} bars from {inp}')
    feat = precompute(df)
    out = Path(args.out) if args.out else inp.with_suffix('.features.parquet')
    out.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out, index=False)
    print(f'Wrote features: {out} with {feat.shape[1]} columns')

if __name__ == '__main__':
    main()