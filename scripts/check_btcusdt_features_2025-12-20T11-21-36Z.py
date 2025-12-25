#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from binance_adapter.precompute_features import precompute, ALL_FEATS

PCT_LIKE = {
    # Heuristics: values are computed via pct_change()*100
    'mom_1m','mom_3m','mom_5m','ret_1m','ret_2m','ret_3m',
    'momentum_5m','momentum_15m',
    # vwap_dev is expressed in percent
    'vwap_dev',
    # macd_hist/macd are differences, not pct
}

RATIO_LIKE = {
    'range_norm_5m','ctx_range_norm','ctx_std_norm',
}


def infer_pct_like_columns(df: pd.DataFrame) -> list[str]:
    suspects = []
    for c in df.columns:
        if c in PCT_LIKE:
            suspects.append(c)
            continue
        # Heuristic: values usually within +/-50 for minute returns if percent
        if df[c].dtype.kind in 'fc' and df[c].abs().quantile(0.95) > 1.0 and df[c].abs().median() < 5.0:
            # Likely percent-scale feature; flag if no explicit suffix
            if not c.endswith('_pct') and c not in RATIO_LIKE:
                suspects.append(c)
    return sorted(set(suspects))


def main():
    ap = argparse.ArgumentParser(description='Compute features for BTCUSDT data and validate labels (no entry/exit logic).')
    ap.add_argument('--market-csv', default='data/btc_profitability_analysis_filtered.csv', help='CSV with timestamp, open, high, low, close, volume')
    ap.add_argument('--out', default=None, help='Optional path to save computed features as Parquet')
    ap.add_argument('--limit', type=int, default=200000, help='Max number of most-recent rows to process for quick checks (None = all)')
    args = ap.parse_args()

    src = Path(args.market_csv)
    if not src.exists():
        raise SystemExit(f'Missing market CSV: {src}')

    raw = pd.read_csv(src, parse_dates=['timestamp'])
    # Keep only the last N rows if limit specified
    if args.limit is not None and args.limit > 0 and len(raw) > args.limit:
        raw = raw.sort_values('timestamp').iloc[-args.limit:]

    # Normalize to minute bars if needed
    if 'open' not in raw.columns or raw.groupby(pd.to_datetime(raw['timestamp']).dt.floor('min')).size().max() > 1:
        g = (
            raw.assign(ts_min=pd.to_datetime(raw['timestamp']).dt.floor('min'))
               .groupby('ts_min')
               .agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'), volume=('volume','sum'))
               .reset_index()
               .rename(columns={'ts_min':'timestamp'})
        )
        mkt = g
    else:
        mkt = raw[['timestamp','open','high','low','close','volume']].copy()

    mkt = mkt.sort_values('timestamp').reset_index(drop=True)
    feats = precompute(mkt)

    # Basic schema checks
    missing = [c for c in ['timestamp','open','high','low','close','volume'] if c not in feats.columns]
    extra = [c for c in feats.columns if c not in (['timestamp','open','high','low','close','volume'] + list(ALL_FEATS))]

    # Label sanity checks
    pct_like = infer_pct_like_columns(feats)

    print('Input bars:', len(mkt))
    print('Feature columns:', len(feats.columns))
    print('\nFirst 5 rows:')
    print(feats.head(5).to_string(index=False))

    print('\nSchema check:')
    print('  Missing core cols:', missing)
    print('  Unexpected extra cols:', extra)

    print('\nPotential percent-like columns lacking explicit _pct suffix (for review):')
    print(' ', pct_like)

    # Optionally save
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(outp, index=False)
        print(f'Wrote features to {outp}')


if __name__ == '__main__':
    main()
