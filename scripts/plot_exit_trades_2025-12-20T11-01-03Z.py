#!/usr/bin/env python3
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades-csv', required=True)
    ap.add_argument('--out-dir', default='data/plots')
    return ap.parse_args()


def main():
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.trades_csv, parse_dates=['entry_time','exit_time'])
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    else:
        df['date'] = df['entry_time'].dt.date

    # Basic fields
    df['hour'] = df['entry_time'].dt.hour
    df['success'] = df['trade_return_pct'] > 0

    # 1) Histogram of trade returns
    plt.figure(figsize=(8,5))
    plt.hist(df['trade_return_pct'], bins=40, color='#1f77b4', alpha=0.8)
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Trade return % distribution')
    plt.xlabel('trade_return_pct')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(out_dir / f"exit_returns_hist_{ts}.png", dpi=140)
    plt.close()

    # 2) Scatter: duration vs return
    if 'duration_min' in df.columns:
        plt.figure(figsize=(7,5))
        plt.scatter(df['duration_min'], df['trade_return_pct'], s=10, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.title('Return vs duration (min)')
        plt.xlabel('duration_min')
        plt.ylabel('trade_return_pct')
        plt.tight_layout()
        plt.savefig(out_dir / f"exit_return_vs_duration_{ts}.png", dpi=140)
        plt.close()

    # 3) Hourly win rate bar
    hr = df.groupby('hour')['success'].mean().reindex(range(24)).fillna(0)
    plt.figure(figsize=(9,4))
    plt.bar(hr.index.astype(int), (hr.values*100.0), color='#2ca02c')
    plt.title('Win rate by UTC hour')
    plt.xlabel('hour')
    plt.ylabel('win_rate %')
    plt.tight_layout()
    plt.savefig(out_dir / f"exit_winrate_by_hour_{ts}.png", dpi=140)
    plt.close()

    # 4) Daily cumulative equity curve (from per-day sum of trade pct)
    daily = df.groupby('date')['trade_return_pct'].sum().sort_index()
    curve = (1.0 + daily/100.0).cumprod()
    plt.figure(figsize=(9,4))
    plt.plot(curve.index, curve.values, color='#ff7f0e')
    plt.title('Daily cumulative equity (unit=1 at start)')
    plt.xlabel('date')
    plt.ylabel('equity (unit)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir / f"exit_daily_equity_curve_{ts}.png", dpi=140)
    plt.close()

    # 5) Liquidations breakdown
    if 'liquidated' in df.columns:
        liq_counts = df[df['liquidated']].groupby('liquidation_reason').size()
        if not liq_counts.empty:
            plt.figure(figsize=(6,6))
            plt.pie(liq_counts.values, labels=liq_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Liquidations by reason')
            plt.tight_layout()
            plt.savefig(out_dir / f"exit_liquidations_pie_{ts}.png", dpi=140)
            plt.close()

    print('Saved plots to', out_dir)

if __name__ == '__main__':
    main()
