#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# repo path for feature function
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame  # type: ignore

FEATURES = [
    'ret_1m_pct','mom_3m_pct','mom_5m_pct','vol_std_5m','range_5m','range_norm_5m','macd','vwap_dev_5m'
]


def load_all(dir_path: Path) -> pd.DataFrame:
    paths = sorted(p for p in dir_path.glob('minute_bars_*.csv'))
    if not paths:
        raise SystemExit(f'No files like minute_bars_*.csv in {dir_path}')
    parts = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=['timestamp'])
        # Ensure required columns
        for c in ['open','high','low','close','volume']:
            if c not in df.columns:
                raise SystemExit(f'{p} missing column {c}')
        parts.append(df[['timestamp','open','high','low','close','volume']])
    all_df = pd.concat(parts, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return all_df


def compute_features(all_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure UTC tz on bars
    bars = all_df.copy()
    bars['ts_min'] = pd.to_datetime(bars['timestamp'], utc=True)
    bars = bars.drop(columns=['timestamp'])
    feats = compute_feature_frame(bars)
    feats = feats.rename(columns={'ts_min':'timestamp'})
    feats['timestamp'] = pd.to_datetime(feats['timestamp'], utc=True)
    # Keep only feature columns
    keep = ['timestamp'] + FEATURES
    return feats[keep].copy()


def write_back(dir_path: Path, merged: pd.DataFrame) -> None:
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    merged['date'] = pd.to_datetime(merged['timestamp']).dt.date
    for d, g in merged.groupby('date'):
        out = g.drop(columns=['date']).copy()
        p = dir_path / f'minute_bars_{d}.csv'
        # Write merged in place (overwrite) with features appended as columns
        out.to_csv(p, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description='Append standard features to ETH CSVs (per-minute rows)')
    ap.add_argument('--dir', default='data/eth_usd', help='Folder with minute_bars_YYYY-MM-DD.csv')
    args = ap.parse_args()

    path = Path(args.dir)
    all_df = load_all(path)
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'], utc=True)
    feats = compute_features(all_df)
    # Merge columns onto all_df
    merged = all_df.merge(feats, on='timestamp', how='left')
    write_back(path, merged)
    print('Updated CSVs with features:', path)

if __name__ == '__main__':
    main()
